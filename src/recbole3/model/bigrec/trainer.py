"""BIGRec trainer: LoRA SFT fine-tuning + embedding-grounding evaluation.

Two-step BIGRec paradigm (arXiv:2308.08434):
  Step 1 (fit):      Fine-tune a CausalLM with LoRA to generate item titles
                     given a user's interaction history.
  Step 2 (evaluate): Generate a title via beam-search, embed it, and rank all
                     candidate items by L2 distance to the oracle embedding,
                     optionally reweighted by popularity / CF signals (Eq. 3).

Grounding weight injection (Eq. 3)
-----------------------------------
When ``config.grounding_mode`` is not ``'none'``, raw L2 distances are first
min-max normalised per row, then multiplied by a per-item weight factor:

    D̂ᵢ = (Dᵢ − min_j Dⱼ) / (max_j Dⱼ − min_j Dⱼ)
    D̃ᵢ = D̂ᵢ × (1 + Wᵢ)^(−γ)

where Wᵢ ∈ [0, 1] is the grounding weight (popularity or CF score) and γ is a
hyperparameter.  A higher Wᵢ decreases D̃ᵢ, promoting the item in the ranking.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer as HFTrainer,
    TrainingArguments,
)

from recbole3.dataset.utils import CANDIDATE_ITEM_IDS, ITEM_ID
from recbole3.evaluation.metric import NDCGMetric, RecallMetric, RetrievalEvalData
from recbole3.model.bigrec.config import BIGRecConfig
from recbole3.model.bigrec.data import (
    BIGRecSFTDataset,
    batchify,
    build_eval_prompts,
    build_item_text_lookup,
)

logger = logging.getLogger(__name__)


class BIGRecTrainer:
    """BIGRec trainer: LoRA SFT fine-tuning and embedding-grounding evaluation.

    The trainer is intentionally self-contained — it does not inherit from
    RecBole3.0's ``Trainer`` class, mirroring the LCRecTrainer pattern.  It
    delegates the inner optimization loop to the HuggingFace ``Trainer`` while
    owning model loading, checkpointing, embedding pre-computation, and the
    final Recall/NDCG metric reporting.

    Args:
        config: Fully resolved :class:`BIGRecConfig` dataclass.
    """

    def __init__(self, config: BIGRecConfig) -> None:
        self.config = config

    # ── Utility helpers ────────────────────────────────────────────────────────

    def _is_main_process(self) -> bool:
        """Return True on rank-0 (single-GPU or DDP master process)."""
        return int(os.environ.get("RANK", "0")) == 0

    def _log(self, msg: str, *args: Any, level: str = "info") -> None:
        """Emit a log message on rank-0 only.

        Args:
            msg: %-style format string.
            *args: Format arguments.
            level: Python logging level name.
        """
        if self._is_main_process():
            getattr(logger, level)(msg, *args)

    def _get_device_map(self) -> dict[str, int]:
        """Resolve ``device_map`` for ``from_pretrained``.

        Returns ``{"": local_rank}`` under DDP (torchrun sets ``LOCAL_RANK``),
        or ``{"": 0}`` for single-process runs.

        In single-process mode, :meth:`fit` and :meth:`evaluate` set
        ``CUDA_VISIBLE_DEVICES=device_id`` *before* the CUDA context is
        initialised, so the target physical GPU always appears as logical
        GPU 0.  Returning ``{"": 0}`` therefore always resolves to the
        correct physical device.

        ``device_map="auto"`` is intentionally avoided: it shards the model
        across all visible GPUs, which triggers ``CUDA peer mapping resources
        exhausted`` errors during training.  HF Trainer handles multi-GPU
        training via DDP instead.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            return {"": local_rank}
        # CUDA_VISIBLE_DEVICES is set to device_id before CUDA context init,
        # so the target physical GPU is always visible as logical GPU 0.
        return {"": 0}

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def _load_tokenizer(self, padding_side: str = "right") -> AutoTokenizer:
        """Load the HuggingFace tokenizer from ``config.llm_path``.

        Args:
            padding_side: ``'right'`` during SFT training;
                          ``'left'`` for batch beam-search generation and
                          embedding extraction (aligns the real last token to
                          index ``-1``).

        Returns:
            Loaded tokenizer with ``pad_token_id`` guaranteed non-None.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_path,
            use_fast=False,
            padding_side=padding_side,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        return tokenizer

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(
        self,
        device_map: str | dict[str, int],
        *,
        inference_mode: bool = False,
    ) -> Any:
        """Load the CausalLM from ``config.llm_path`` and wrap with LoRA if configured.

        Args:
            device_map: Placement map from :meth:`_get_device_map`.
            inference_mode: When ``True`` the LoRA adapter is frozen and
                            ``use_cache`` is enabled for fast generation.

        Returns:
            Loaded model (optionally wrapped in ``PeftModel``).
        """
        dtype = getattr(torch, self.config.torch_dtype)

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": self.config.attn_implementation,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
        }
        # INT8 quantization via bitsandbytes (requires: pip install bitsandbytes).
        # Official BIGRec trains with load_in_8bit=True by default.
        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(self.config.llm_path, **load_kwargs)
        # Disable KV cache during training to allow gradient checkpointing.
        model.config.use_cache = inference_mode

        if self.config.use_lora:
            # When INT8 quantisation is active, peft requires an additional
            # preparation step before LoRA adapters can be attached.
            if self.config.load_in_8bit and not inference_mode:
                try:
                    from peft import prepare_model_for_kbit_training  # peft ≥ 0.4
                    model = prepare_model_for_kbit_training(
                        model,
                        use_gradient_checkpointing=self.config.gradient_checkpointing,
                    )
                except ImportError:
                    from peft import prepare_model_for_int8_training  # peft < 0.4
                    model = prepare_model_for_int8_training(model)

            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=list(self.config.lora_target_modules),
                lora_dropout=self.config.lora_dropout,
                bias="none",
                inference_mode=inference_mode,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            if not inference_mode and self._is_main_process():
                model.print_trainable_parameters()

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log("Model: %d total params, %d trainable", n_params, n_trainable)
        return model

    def _load_trained_model(self, checkpoint_path: str) -> Any:
        """Load a saved model from *checkpoint_path* for inference.

        When LoRA is configured, the base model is loaded first and the saved
        adapter weights are merged on top.

        Args:
            checkpoint_path: Directory containing saved LoRA adapter weights
                             (or the full fine-tuned model if LoRA is off).

        Returns:
            Model in ``eval()`` mode.
        """
        device_map = self._get_device_map()
        dtype = getattr(torch, self.config.torch_dtype)

        if self.config.use_lora:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_path,
                dtype=dtype,
                attn_implementation=self.config.attn_implementation,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                dtype=dtype,
                is_trainable=False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                dtype=dtype,
                attn_implementation=self.config.attn_implementation,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )

        model.eval()
        self._log("Trained model loaded from %s", checkpoint_path)
        return model

    def _load_base_model_for_embedding(
        self,
        device_map: str | dict[str, int],
    ) -> Any:
        """Load the base CausalLM (no LoRA) for embedding extraction.

        When ``config.embedding_use_base_model=True`` (official BIGRec default),
        both item embeddings and oracle embeddings are computed in the same vector
        space — that of the original pre-trained model, not the fine-tuned one.

        Args:
            device_map: Placement map from :meth:`_get_device_map`.

        Returns:
            Base CausalLM in ``eval()`` mode (no LoRA adapter attached).
        """
        dtype = getattr(torch, self.config.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_path,
            dtype=dtype,
            attn_implementation=self.config.attn_implementation,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.config.use_cache = True
        model.eval()
        self._log("Base model loaded for embedding extraction from %s", self.config.llm_path)
        return model

    # ── Embedding extraction ───────────────────────────────────────────────────

    def _extract_embeddings(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        texts: list[str],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode *texts* and return last-token hidden states from the last layer.

        Uses left-padding so that the last real token of each sequence always
        falls at position ``-1`` of the padded batch, making pooling trivial.

        Args:
            model: Loaded CausalLM (with or without LoRA).
            tokenizer: Tokenizer; ``padding_side`` will be temporarily set to
                       ``'left'`` for the duration of this call.
            texts: Text strings to encode.
            batch_size: Forward-pass batch size.
            device: Inference device.

        Returns:
            Float32 CPU tensor of shape ``[len(texts), hidden_size]``.
        """
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        all_embs: list[torch.Tensor] = []

        for batch_texts in batchify(texts, batch_size):
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_length,
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                )

            # hidden_states: tuple of (num_layers+1) tensors, each [B, seq_len, H].
            last_layer: torch.Tensor = outputs.hidden_states[-1]  # [B, seq_len, H]
            # Left-padding guarantees the last real token is at index -1.
            batch_emb = last_layer[:, -1, :].float().cpu()  # [B, H]
            all_embs.append(batch_emb)

        tokenizer.padding_side = orig_padding_side
        return torch.cat(all_embs, dim=0)  # [len(texts), H]

    def _precompute_item_embeddings(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        item_texts: list[str],
        cache_path: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Return item embeddings, loading from disk cache when available.

        The on-disk cache is a ``.pt`` file with a float32 tensor of shape
        ``[num_items, hidden_size]``.

        Args:
            model: CausalLM used for embedding extraction.
            tokenizer: Tokenizer.
            item_texts: Title strings indexed by framework ``item_id``.
                        Index 0 is the reserved placeholder item.
            cache_path: Full path to the ``.pt`` cache file.
            device: Device for the forward pass.

        Returns:
            CPU tensor of shape ``[num_items, hidden_size]``.
        """
        if not self.config.refresh_embedding_cache and os.path.isfile(cache_path):
            self._log("Loading item embeddings from cache: %s", cache_path)
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        self._log("Pre-computing embeddings for %d items …", len(item_texts))
        embeddings = self._extract_embeddings(
            model,
            tokenizer,
            item_texts,
            batch_size=self.config.embedding_batch_size,
            device=device,
        )  # [num_items, H], on CPU

        if self._is_main_process():
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(embeddings, cache_path)
            self._log("Item embeddings saved to %s", cache_path)

        return embeddings

    # ── Grounding weight injection (Eq. 3) ────────────────────────────────────

    def _compute_popularity_weights(
        self,
        task_data: Any,
        num_items: int,
    ) -> torch.Tensor:
        """Compute min-max normalised item popularity from training interactions.

        Popularity Cᵢ = Nᵢ / Σⱼ Nⱼ where Nᵢ is the interaction count of item i
        in the training split.  Cᵢ is then min-max normalised to Pᵢ ∈ [0, 1].

        Args:
            task_data: Prepared ``BIGRecModelDataset`` with a training split.
            num_items: Total number of items (length of the returned tensor).

        Returns:
            Float tensor of shape ``[num_items]`` with values in ``[0, 1]``.
        """
        train_frame: pd.DataFrame = task_data.get_train_dataset().frame  # type: ignore[attr-defined]
        counts = torch.zeros(num_items, dtype=torch.float32)

        for item_id, n in train_frame[ITEM_ID].value_counts().items():
            idx = int(item_id)
            if 0 <= idx < num_items:
                counts[idx] = float(n)

        total = counts.sum()
        if total <= 0.0:
            self._log("Popularity: all interaction counts are zero; weights default to 0.", level="warning")
            return counts

        ci = counts / total                           # raw frequency Cᵢ
        c_min, c_max = ci.min(), ci.max()
        if c_max > c_min:
            pi = (ci - c_min) / (c_max - c_min)      # min-max normalise → [0, 1]
        else:
            pi = torch.zeros_like(ci)

        self._log(
            "Popularity weights — mean=%.4f, max=%.4f, min=%.4f",
            pi.mean().item(), pi.max().item(), pi.min().item(),
        )
        return pi  # [num_items]

    def _load_cf_weights(self, num_items: int) -> torch.Tensor:
        """Load and normalise pre-computed CF model scores from disk.

        The file must be a ``.pt`` file containing a 1-D float tensor of shape
        ``[num_items]``.  Scores are min-max normalised to ``[0, 1]`` so they
        are on the same scale as the popularity weights.

        Args:
            num_items: Expected number of items (for shape validation).

        Returns:
            Float tensor of shape ``[num_items]`` with values in ``[0, 1]``.

        Raises:
            FileNotFoundError: If ``config.cf_score_path`` is unset or missing.
            ValueError: If the loaded tensor's size does not match *num_items*.
        """
        path = self.config.cf_score_path
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"CF score file not found: '{path}'. "
                "Set BIGRecConfig.cf_score_path to a .pt file with shape [num_items]."
            )

        scores: torch.Tensor = torch.load(path, weights_only=True, map_location="cpu").float()

        if scores.ndim != 1 or scores.shape[0] != num_items:
            raise ValueError(
                f"CF score tensor shape {tuple(scores.shape)} does not match "
                f"num_items={num_items}. Expected a 1-D tensor of length {num_items}."
            )

        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = torch.zeros_like(scores)

        self._log(
            "CF weights — mean=%.4f, max=%.4f, min=%.4f",
            scores.mean().item(), scores.max().item(), scores.min().item(),
        )
        return scores  # [num_items]

    def _build_grounding_weights(
        self,
        task_data: Any,
        num_items: int,
    ) -> torch.Tensor | None:
        """Build the combined grounding weight vector for Eq. 3.

        Depending on ``config.grounding_mode``:

        * ``'none'``:           Returns ``None`` (pure L2, no reweighting).
        * ``'popularity'``:     Returns min-max normalised popularity Pᵢ.
        * ``'cf'``:             Returns min-max normalised CF scores.
        * ``'popularity+cf'``:  Sums both signals, then re-normalises to [0, 1].

        Args:
            task_data: Prepared ``BIGRecModelDataset``.
            num_items: Number of items (weight vector length).

        Returns:
            Float CPU tensor of shape ``[num_items]`` in ``[0, 1]``, or ``None``
            when no reweighting is configured.
        """
        mode = self.config.grounding_mode.strip().lower()
        if mode == "none":
            return None

        weights = torch.zeros(num_items, dtype=torch.float32)

        if "popularity" in mode:
            pop = self._compute_popularity_weights(task_data, num_items)
            weights = weights + pop

        if "cf" in mode:
            cf = self._load_cf_weights(num_items)
            weights = weights + cf

        # When both signals are summed, re-normalise the combined vector so it
        # stays in [0, 1] — required for Eq. 3 to behave consistently.
        if "popularity" in mode and "cf" in mode:
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                weights = (weights - w_min) / (w_max - w_min)
            self._log("Combined popularity+CF weights built (re-normalised).")

        return weights  # [num_items], CPU

    @staticmethod
    def _apply_grounding_weights(
        dist: torch.Tensor,
        weights: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Apply Eq. 3 to reweight L2 distances by popularity / CF signal.

        Steps:

        1. Per-row min-max normalise the raw L2 distances → D̂ ∈ [0, 1].
        2. Multiply by the inverse weight factor: D̃ᵢ = D̂ᵢ × (1 + Wᵢ)^(−γ).

        A higher Wᵢ → smaller D̃ᵢ → item ranks higher.  When γ=0 the weights
        have no effect and D̃ = D̂.

        Args:
            dist: Raw L2 distances, shape ``[B, num_items]``, on any device.
            weights: Per-item grounding weights, shape ``[num_items]``,
                     values in ``[0, 1]``.  Must be on the same device as *dist*.
            gamma: Exponent γ ≥ 0.  Larger values amplify the weight effect.

        Returns:
            Reweighted distances of shape ``[B, num_items]`` on the same device
            as *dist*.
        """
        # Per-row min-max normalisation (eps guards against zero-range rows).
        dist_min = dist.min(dim=1, keepdim=True)[0]  # [B, 1]
        dist_max = dist.max(dim=1, keepdim=True)[0]  # [B, 1]
        dist_hat = (dist - dist_min) / (dist_max - dist_min + 1e-8)  # [B, num_items]

        # Eq. 3: D̃ = D̂ × (1 + W)^(−γ)
        multiplier = torch.pow(1.0 + weights.unsqueeze(0), -gamma)  # [1, num_items]
        return dist_hat * multiplier  # [B, num_items]

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, task_data: Any, output_dir: str) -> dict[str, Any]:
        """Fine-tune the CausalLM backbone with LoRA on (history, target) pairs.

        Builds :class:`~recbole3.model.bigrec.data.BIGRecSFTDataset` objects from
        both the training and validation splits (which already contain
        ``history_item_ids`` injected by
        :class:`~recbole3.model.bigrec.data.BIGRecModelDataset`), then delegates
        the optimization to HuggingFace ``Trainer``.

        Early stopping monitors HF Trainer's built-in LM validation loss
        (``EarlyStoppingCallback``), matching the official BIGRec training
        procedure exactly — recommendation metrics are computed separately in
        :meth:`evaluate` after training completes.

        Args:
            task_data: A prepared :class:`~recbole3.model.bigrec.data.BIGRecModelDataset`.
            output_dir: Directory where model checkpoints and the tokenizer will
                        be saved.

        Returns:
            ``{"checkpoint_path": output_dir}`` on success.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Restrict visible GPUs *before* the CUDA context is initialised.
        # HF Trainer calls torch.cuda.device_count() when constructing
        # TrainingArguments and wraps the model with nn.DataParallel when
        # count > 1.  DataParallel scatters inputs to every visible GPU,
        # triggering "CUDA error: peer mapping resources exhausted" on
        # multi-GPU nodes.  Setting CUDA_VISIBLE_DEVICES to a single device
        # makes device_count() return 1 and suppresses DataParallel entirely.
        # (In DDP mode LOCAL_RANK is set by torchrun; skip this path.)
        if int(os.environ.get("LOCAL_RANK", "-1")) == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device_id)
            self._log("Single-GPU mode: CUDA_VISIBLE_DEVICES=%s", self.config.device_id)

        # 1. Tokenizer (right-padding during SFT).
        tokenizer = self._load_tokenizer(padding_side="right")
        if self._is_main_process():
            tokenizer.save_pretrained(output_dir)

        # 2. Item text lookup: list[str] indexed by framework item_id.
        item_text_lookup = build_item_text_lookup(task_data, self.config)

        # 3. Tokenize training and validation (history, target) pairs.
        #    The validation SFT dataset drives EarlyStoppingCallback via LM val
        #    loss (official BIGRec approach; recommendation metrics are computed
        #    post-training by evaluate()).
        train_frame: pd.DataFrame = task_data.get_train_dataset().frame  # type: ignore[attr-defined]

        # When max_steps > 0, cap the training frame to avoid tokenising rows
        # that will never be reached during training.  This cuts startup time
        # from O(total_samples) to O(max_steps × effective_batch_size).
        if self.config.max_steps > 0:
            max_samples: int = (
                self.config.max_steps
                * self.config.train_batch_size
                * self.config.gradient_accumulation_steps
            )
            if len(train_frame) > max_samples:
                train_frame = train_frame.head(max_samples).reset_index(drop=True)
                self._log(
                    "max_steps=%d: capped training frame to %d samples "
                    "(full dataset: %d rows).",
                    self.config.max_steps,
                    max_samples,
                    len(task_data.get_train_dataset().frame),  # type: ignore[attr-defined]
                )

        sft_train = BIGRecSFTDataset(
            records=train_frame,
            tokenizer=tokenizer,
            item_text_lookup=item_text_lookup,
            config=self.config,
        )
        valid_frame: pd.DataFrame = task_data.get_eval_dataset("valid").frame  # type: ignore[attr-defined]

        # Mirror the training cap on the validation set so that eval time stays
        # proportional to training time.  Without this, max_steps=500 trains on
        # 16 k samples but evaluates on ~57 k (14 360 eval forward-passes on
        # LLaMA-8B), which takes far longer than the training pass itself.
        # Cap: max_steps × eval_batch_size gives the same number of forward
        # passes during eval as optimizer steps during training (eval has no
        # gradient accumulation).
        if self.config.max_steps > 0:
            max_eval_samples: int = self.config.max_steps * self.config.eval_batch_size
            if len(valid_frame) > max_eval_samples:
                valid_frame = valid_frame.head(max_eval_samples).reset_index(drop=True)
                self._log(
                    "max_steps=%d: capped validation frame to %d samples "
                    "(full validation: %d rows).",
                    self.config.max_steps,
                    max_eval_samples,
                    len(task_data.get_eval_dataset("valid").frame),  # type: ignore[attr-defined]
                )

        sft_val = BIGRecSFTDataset(
            records=valid_frame,
            tokenizer=tokenizer,
            item_text_lookup=item_text_lookup,
            config=self.config,
        )

        # 4. Load model for training (single model, no base model during training).
        device_map = self._get_device_map()
        model = self._load_model(device_map, inference_mode=False)

        # 5. Collator pads to a multiple of 8 for memory-efficient CUDA kernels.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            padding="longest",
            return_tensors="pt",
        )

        # 6. HuggingFace TrainingArguments — mirror BIGRecConfig fields directly.
        #    warmup_steps takes precedence over warmup_ratio (official BIGRec uses
        #    a fixed 20-step warm-up rather than a ratio).
        #    evaluation_strategy="epoch" + load_best_model_at_end=True matches
        #    the official BIGRec train.py configuration exactly.
        warmup_kwargs: dict[str, Any] = (
            {"warmup_steps": self.config.warmup_steps}
            if self.config.warmup_steps is not None
            else {"warmup_ratio": self.config.warmup_ratio}
        )
        # When max_steps > 0 it overrides num_train_epochs in HF Trainer.
        # eval_strategy must stay "epoch" so EarlyStoppingCallback can run;
        # HF Trainer handles the max_steps / epoch boundary automatically.
        hf_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **warmup_kwargs,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            eval_strategy="epoch",
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            deepspeed=self.config.deepspeed,
            report_to="none",
            remove_unused_columns=False,
        )

        # 7. Construct and run the HF Trainer.
        #    EarlyStoppingCallback monitors LM val loss (official BIGRec default).
        hf_trainer = HFTrainer(
            model=model,
            args=hf_args,
            train_dataset=sft_train,
            eval_dataset=sft_val,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ],
        )

        self._log("Starting BIGRec LoRA fine-tuning …")
        hf_trainer.train()
        hf_trainer.save_state()
        hf_trainer.save_model(output_dir)
        self._log("Checkpoint saved to %s", output_dir)

        return {"checkpoint_path": output_dir}

    # ── Gamma-search helpers ──────────────────────────────────────────────────

    @staticmethod
    def _default_gamma_search_values() -> tuple[float, ...]:
        """Return the official BIGRec 199-value gamma grid.

        Mirrors the official implementation: 0.00, 0.01, …, 0.99 (fine-grained)
        followed by 1, 2, …, 99 (coarse-grained).

        Returns:
            Tuple of 199 float gamma candidates.
        """
        fine: tuple[float, ...] = tuple(round(i * 0.01, 2) for i in range(100))   # 0.00 … 0.99
        coarse: tuple[float, ...] = tuple(float(i) for i in range(1, 100))         # 1.0  … 99.0
        return fine + coarse  # 199 values

    def _generate_all_titles(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        eval_frame: pd.DataFrame,
        item_text_lookup: list[str],
        device: torch.device,
    ) -> tuple[list[str], list[int], list[list[int] | None]]:
        """Run beam-search over a full eval frame and collect generated titles.

        Used by the gamma-search path to decouple generation from distance
        computation and the gamma grid-search loop.

        Args:
            model: Generation model (fine-tuned LoRA) in ``eval()`` mode.
            tokenizer: Left-padding tokenizer.
            eval_frame: Full evaluation DataFrame with ``history_item_ids`` and
                        ``item_id`` columns; optionally ``candidate_item_ids``.
            item_text_lookup: ``item_id → title string`` mapping.
            device: Inference device.

        Returns:
            Tuple of:

            - ``clean_texts``:  Decoded (stripped) generated titles, one per row.
            - ``target_ids``:   Ground-truth ``item_id`` per row.
            - ``cand_lists``:   Candidate ``item_id`` lists per row, or ``None``
                                when the row has no pre-defined candidates
                                (→ full-ranking).
        """
        model.eval()
        tokenizer.padding_side = "left"

        clean_texts: list[str] = []
        target_ids: list[int] = []
        cand_lists: list[list[int] | None] = []

        batch_size: int = self.config.eval_batch_size
        num_rows: int = len(eval_frame)

        pbar = tqdm(
            range(0, num_rows, batch_size),
            desc="BIGRec generation",
            disable=not self._is_main_process(),
        )

        for start in pbar:
            batch_df = eval_frame.iloc[start : start + batch_size].reset_index(drop=True)
            actual_bs: int = len(batch_df)

            prompts: list[str] = build_eval_prompts(batch_df, item_text_lookup, self.config)
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_length,
            ).to(device)
            prompt_length: int = encoded["input_ids"].shape[1]

            with torch.no_grad():
                output_ids: torch.Tensor = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    num_return_sequences=1,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[:, prompt_length:]
            generated_texts: list[str] = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for idx, t in enumerate(generated_texts):
                clean_texts.append(t.strip().strip('"').strip() or f"[empty_{start + idx}]")

            for i in range(actual_bs):
                row = batch_df.iloc[i]
                target_ids.append(int(row[ITEM_ID]))
                cand_val = row.get(CANDIDATE_ITEM_IDS, None)
                if cand_val is None or (
                    not hasattr(cand_val, "__len__")
                    and isinstance(cand_val, float)
                    and np.isnan(cand_val)
                ):
                    cand_lists.append(None)
                else:
                    cand_lists.append(list(cand_val))

        return clean_texts, target_ids, cand_lists

    def _run_gamma_search(
        self,
        dist: torch.Tensor,
        grounding_weights: torch.Tensor,
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        device: torch.device,
        gamma_values: tuple[float, ...],
    ) -> dict[str, float]:
        """Grid-search for the best gamma per metric×K on a validation split.

        For each candidate gamma, applies Eq. 3, ranks all items, and computes
        all configured metrics.  The best-performing gamma is tracked
        *independently* per metric×K combination (official BIGRec behaviour).

        Args:
            dist: Pre-computed raw L2 distances of shape ``[N, num_items]``
                  already on *device*.
            grounding_weights: Per-item grounding weights ``[num_items]`` in
                               ``[0, 1]`` on *device*.
            target_ids: Ground-truth item_ids, length N.
            cand_lists: Per-row candidate item_id lists (``None`` → full ranking).
            device: Inference device.
            gamma_values: Candidate γ values to evaluate.

        Returns:
            Dict mapping ``"metric@K"`` → best γ float.
        """
        maxk: int = max(self.config.eval_topk)
        is_sampled: bool = self.config.eval_protocol == "sampled"
        n: int = len(target_ids)

        target_arr = np.array(target_ids, dtype=np.int64).reshape(n, 1)
        mask_arr = np.ones((n, 1), dtype=bool)

        # Initialise best-score / best-gamma trackers keyed by "metric@K".
        metric_keys: list[str] = [
            f"{m.lower()}@{k}"
            for m in self.config.eval_metrics
            for k in self.config.eval_topk
        ]
        best_scores: dict[str, float] = {key: -1.0 for key in metric_keys}
        best_gammas: dict[str, float] = {key: gamma_values[0] for key in metric_keys}

        ks: tuple[int, ...] = tuple(self.config.eval_topk)

        for gamma in tqdm(
            gamma_values,
            desc="Gamma search",
            disable=not self._is_main_process(),
        ):
            eff_dist: torch.Tensor = self._apply_grounding_weights(
                dist, grounding_weights, gamma
            )  # [N, num_items]

            pred_list: list[np.ndarray] = []
            for i in range(n):
                cand = cand_lists[i]
                if is_sampled and cand is not None:
                    cand_t = torch.tensor(cand, dtype=torch.long, device=device)
                    cand_dists = eff_dist[i, cand_t]
                    sorted_idx = torch.argsort(cand_dists)[:maxk]
                    top_k = cand_t[sorted_idx].cpu().numpy()
                else:
                    top_k = torch.argsort(eff_dist[i])[:maxk].cpu().numpy()

                if len(top_k) < maxk:
                    pad = np.full(maxk - len(top_k), -1, dtype=np.int64)
                    top_k = np.concatenate([top_k, pad])
                pred_list.append(top_k.reshape(1, maxk))

            pred_arr = np.concatenate(pred_list, axis=0)  # [N, maxk]
            eval_data = RetrievalEvalData(
                pred_item_ids=pred_arr,
                target_item_ids=target_arr,
                target_mask=mask_arr,
            )

            for metric_name in self.config.eval_metrics:
                name = metric_name.strip().lower()
                if name == "recall":
                    scores = RecallMetric(ks).compute(eval_data)
                elif name == "ndcg":
                    scores = NDCGMetric(ks).compute(eval_data)
                else:
                    continue
                for key, val in scores.items():
                    if val > best_scores.get(key, -1.0):
                        best_scores[key] = val
                        best_gammas[key] = gamma

        self._log("Gamma search complete — best γ per metric@K:")
        for key in metric_keys:
            self._log("  %s → γ=%.3f  (val=%.4f)", key, best_gammas[key], best_scores[key])

        return best_gammas

    def _evaluate_from_dist_per_k_gammas(
        self,
        dist: torch.Tensor,
        grounding_weights: torch.Tensor | None,
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        best_gammas: dict[str, float],
        device: torch.device,
    ) -> dict[str, float]:
        """Evaluate a split using the best gamma independently per metric×K.

        For each ``(metric, K)`` pair the corresponding γ from gamma search
        (found on the validation split) is applied to re-rank items, then only
        that specific metric@K is computed.

        Args:
            dist: Pre-computed raw L2 distances ``[N, num_items]`` on *device*.
            grounding_weights: Per-item grounding weights ``[num_items]`` on
                               *device*, or ``None`` for pure L2.
            target_ids: Ground-truth item_ids, length N.
            cand_lists: Per-row candidate item_id lists (``None`` → full ranking).
            best_gammas: Dict ``"metric@K"`` → best γ from :meth:`_run_gamma_search`.
            device: Inference device.

        Returns:
            Dict of metric scores ``{"recall@K": float, "ndcg@K": float, …}``.
        """
        maxk: int = max(self.config.eval_topk)
        is_sampled: bool = self.config.eval_protocol == "sampled"
        n: int = len(target_ids)
        target_arr = np.array(target_ids, dtype=np.int64).reshape(n, 1)
        mask_arr = np.ones((n, 1), dtype=bool)

        results: dict[str, float] = {}

        for metric_name in self.config.eval_metrics:
            name = metric_name.strip().lower()
            if name not in ("recall", "ndcg"):
                self._log("Unknown eval metric '%s' — skipping.", metric_name, level="warning")
                continue

            for k in self.config.eval_topk:
                key = f"{name}@{k}"
                gamma = best_gammas.get(key, self.config.grounding_gamma)

                # Apply the per-K optimal gamma.
                if grounding_weights is not None:
                    eff_dist: torch.Tensor = self._apply_grounding_weights(
                        dist, grounding_weights, gamma
                    )  # [N, num_items]
                else:
                    eff_dist = dist

                pred_list: list[np.ndarray] = []
                for i in range(n):
                    cand = cand_lists[i]
                    if is_sampled and cand is not None:
                        cand_t = torch.tensor(cand, dtype=torch.long, device=device)
                        cand_dists = eff_dist[i, cand_t]
                        sorted_idx = torch.argsort(cand_dists)[:k]
                        top_k = cand_t[sorted_idx].cpu().numpy()
                    else:
                        top_k = torch.argsort(eff_dist[i])[:k].cpu().numpy()

                    if len(top_k) < maxk:
                        pad = np.full(maxk - len(top_k), -1, dtype=np.int64)
                        top_k = np.concatenate([top_k, pad])
                    pred_list.append(top_k.reshape(1, maxk))

                pred_arr = np.concatenate(pred_list, axis=0)  # [N, maxk]
                eval_data = RetrievalEvalData(
                    pred_item_ids=pred_arr,
                    target_item_ids=target_arr,
                    target_mask=mask_arr,
                )

                if name == "recall":
                    scores = RecallMetric((k,)).compute(eval_data)
                else:  # ndcg
                    scores = NDCGMetric((k,)).compute(eval_data)

                if key in scores:
                    results[key] = scores[key]
                    self._log("  %s (γ=%.3f) = %.4f", key, gamma, scores[key])

        return results

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        task_data: Any,
        checkpoint_path: str,
        split: Literal["valid", "test"] = "test",
    ) -> dict[str, Any]:
        """Evaluate a trained BIGRec model using embedding grounding.

        Workflow:

        1. Load model from *checkpoint_path* (base + LoRA adapter).
        2. Pre-compute item embeddings or load them from the disk cache.
        3. For each eval row: build prompt → beam-search → decode generated
           title → extract oracle embedding → L2 distance ranking.
        4. Compute Recall@K and/or NDCG@K averaged over all eval rows.

        Args:
            task_data: A prepared :class:`~recbole3.model.bigrec.data.BIGRecModelDataset`.
            checkpoint_path: Directory with the saved LoRA adapter (or full
                             fine-tuned model).
            split: Which split to evaluate — ``'valid'`` or ``'test'``.

        Returns:
            Dict mapping ``"recall@K"`` / ``"ndcg@K"`` to scalar floats.
        """
        # Same single-GPU restriction as in fit() — needed when evaluate() is
        # invoked standalone (pipeline_stage='evaluation') without a prior fit().
        if int(os.environ.get("LOCAL_RANK", "-1")) == -1:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.device_id))

        # 1. Load the fine-tuned generation model.
        gen_model = self._load_trained_model(checkpoint_path)
        tokenizer = self._load_tokenizer(padding_side="left")
        device: torch.device = next(gen_model.parameters()).device

        # 2. Optionally load a separate base model for embedding extraction.
        #    Official BIGRec uses the BASE model (no LoRA) so that item embeddings
        #    and oracle embeddings share the same vector space.
        if self.config.embedding_use_base_model:
            emb_model: Any = self._load_base_model_for_embedding(self._get_device_map())
        else:
            emb_model = None  # _evaluate_split falls back to gen_model

        item_text_lookup = build_item_text_lookup(task_data, self.config)

        # Derive a deterministic cache file name from dataset name and split.
        dataset_name = getattr(getattr(task_data, "config", None), "name", "dataset")
        cache_filename = f"{dataset_name}_{split}_item_embs.pt"
        cache_path = os.path.join(self.config.embedding_cache_dir, cache_filename)

        # 3. Pre-compute item embeddings using emb_model (or gen_model).
        _model_for_emb = emb_model if emb_model is not None else gen_model
        item_embeddings: torch.Tensor = self._precompute_item_embeddings(
            _model_for_emb, tokenizer, item_text_lookup, cache_path, device
        )  # [num_items, H] on CPU

        item_emb_device = item_embeddings.to(device)  # [num_items, H]

        # Build grounding weights for Eq. 3 (None when grounding_mode='none').
        num_items: int = item_embeddings.shape[0]
        grounding_weights: torch.Tensor | None = self._build_grounding_weights(
            task_data, num_items
        )  # [num_items] CPU, or None

        # ── Gamma-search path (official BIGRec per-K optimal gamma) ──────────
        if self.config.grounding_gamma_search and grounding_weights is not None:
            weights_device = grounding_weights.to(device)  # [num_items]
            gamma_values: tuple[float, ...] = (
                tuple(self.config.grounding_gamma_search_values)
                if self.config.grounding_gamma_search_values
                else self._default_gamma_search_values()
            )

            # Step A: Find best gamma per metric@K on the validation split.
            self._log("Gamma search: running beam-search on validation split …")
            valid_frame: pd.DataFrame = task_data.get_eval_dataset("valid").frame  # type: ignore[attr-defined]
            valid_texts, valid_targets, valid_cands = self._generate_all_titles(
                gen_model, tokenizer, valid_frame, item_text_lookup, device
            )
            valid_oracle_embs = self._extract_embeddings(
                _model_for_emb, tokenizer, valid_texts,
                batch_size=self.config.embedding_batch_size, device=device,
            )  # [N_valid, H] CPU
            valid_oracle_device = valid_oracle_embs.to(device)
            valid_dist: torch.Tensor = torch.cdist(
                valid_oracle_device, item_emb_device, p=2.0
            )  # [N_valid, num_items]

            best_gammas = self._run_gamma_search(
                valid_dist, weights_device, valid_targets, valid_cands, device, gamma_values
            )

            # Step B: Evaluate target split with per-K best gammas.
            self._log("Gamma search: running beam-search on %s split …", split)
            test_frame: pd.DataFrame = task_data.get_eval_dataset(split).frame  # type: ignore[attr-defined]
            test_texts, test_targets, test_cands = self._generate_all_titles(
                gen_model, tokenizer, test_frame, item_text_lookup, device
            )
            test_oracle_embs = self._extract_embeddings(
                _model_for_emb, tokenizer, test_texts,
                batch_size=self.config.embedding_batch_size, device=device,
            )  # [N_test, H] CPU
            test_oracle_device = test_oracle_embs.to(device)
            test_dist: torch.Tensor = torch.cdist(
                test_oracle_device, item_emb_device, p=2.0
            )  # [N_test, num_items]

            return self._evaluate_from_dist_per_k_gammas(
                test_dist, weights_device, test_targets, test_cands, best_gammas, device
            )

        # ── Standard evaluation path (no gamma search) ────────────────────────
        eval_frame: pd.DataFrame = task_data.get_eval_dataset(split).frame  # type: ignore[attr-defined]

        return self._evaluate_split(
            model=gen_model,
            tokenizer=tokenizer,
            item_emb_device=item_emb_device,
            item_text_lookup=item_text_lookup,
            eval_frame=eval_frame,
            device=device,
            grounding_weights=grounding_weights,
            emb_model=emb_model,
        )

    def _evaluate_split(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        item_emb_device: torch.Tensor,
        item_text_lookup: list[str],
        eval_frame: pd.DataFrame,
        device: torch.device,
        grounding_weights: torch.Tensor | None = None,
        emb_model: Any = None,
    ) -> dict[str, Any]:
        """Core evaluation loop: beam-search → oracle embedding → L2 ranking → metrics.

        Args:
            model: Loaded fine-tuned model in ``eval()`` mode (used for generation).
            tokenizer: Left-padding tokenizer.
            item_emb_device: Pre-computed item embeddings on *device*,
                             shape ``[num_items, hidden_size]``.
            item_text_lookup: Mapping ``item_id → title string``.
            eval_frame: DataFrame with at least ``history_item_ids`` and
                        ``item_id`` columns; optionally ``candidate_item_ids``.
            device: Inference device.
            grounding_weights: Optional per-item grounding weights of shape
                               ``[num_items]`` in ``[0, 1]`` (CPU tensor).
                               When provided, Eq. 3 is applied after computing
                               raw L2 distances.  ``None`` → pure L2 ranking.
            emb_model: Optional base CausalLM (no LoRA) used for oracle embedding
                       extraction.  When ``None``, *model* is used instead.
                       Supplying the base model ensures oracle embeddings live in
                       the same space as the pre-computed item embeddings.

        Returns:
            Dict of metric scores averaged over all evaluation rows.
        """
        # Move grounding weights to the inference device once (avoids per-batch transfers).
        weights_device: torch.Tensor | None = (
            grounding_weights.to(device) if grounding_weights is not None else None
        )
        model.eval()
        tokenizer.padding_side = "left"

        maxk: int = max(self.config.eval_topk)
        batch_size: int = self.config.eval_batch_size
        num_rows: int = len(eval_frame)
        is_sampled: bool = self.config.eval_protocol == "sampled"

        # Accumulators — each element is shape [1, …] for easy concatenation.
        all_pred_item_ids: list[np.ndarray] = []   # [1, maxk] per row
        all_target_item_ids: list[np.ndarray] = [] # [1, 1]    per row
        all_target_masks: list[np.ndarray] = []    # [1, 1]    per row

        pbar = tqdm(
            range(0, num_rows, batch_size),
            desc="BIGRec eval",
            disable=not self._is_main_process(),
        )

        for start in pbar:
            batch_df = eval_frame.iloc[start : start + batch_size].reset_index(drop=True)
            actual_bs: int = len(batch_df)

            # ── Step 1: Build Alpaca-format prompts from history ───────────────
            prompts: list[str] = build_eval_prompts(
                batch_df, item_text_lookup, self.config
            )

            # ── Step 2: Tokenize prompts (left-padded) ─────────────────────────
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_length,
            ).to(device)
            prompt_length: int = encoded["input_ids"].shape[1]

            # ── Step 3: Beam-search generation ────────────────────────────────
            with torch.no_grad():
                output_ids: torch.Tensor = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    num_return_sequences=1,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # output_ids: [B, prompt_len + gen_len]
            generated_ids = output_ids[:, prompt_length:]  # [B, gen_len]

            # ── Step 4: Decode → strip surrounding quotes ──────────────────────
            generated_texts: list[str] = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # BIGRec wraps target titles in double-quotes in the SFT response.
            clean_texts: list[str] = [
                t.strip().strip('"').strip() or f"[empty_{i}]"
                for i, t in enumerate(generated_texts)
            ]

            # ── Step 5: Extract oracle embeddings from decoded titles ──────────
            # Use the base model (emb_model) when available so that oracle
            # embeddings are in the same vector space as item embeddings.
            _emb_model = emb_model if emb_model is not None else model
            oracle_embeddings: torch.Tensor = self._extract_embeddings(
                _emb_model,
                tokenizer,
                clean_texts,
                batch_size=actual_bs,
                device=device,
            )  # [B, H] on CPU
            oracle_emb_device = oracle_embeddings.to(device)  # [B, H]

            # ── Step 6: L2 distance to all item embeddings ─────────────────────
            # distances[i, j] = ||oracle_i - item_j||_2  →  [B, num_items]
            distances: torch.Tensor = torch.cdist(
                oracle_emb_device,  # [B, H]
                item_emb_device,    # [num_items, H]
                p=2.0,
            )  # [B, num_items]

            # ── Step 6b: Optional Eq. 3 grounding weight injection ────────────
            # Apply popularity / CF reweighting when grounding_mode != 'none'.
            # effective_dist is still small-is-better (lower → higher rank).
            if weights_device is not None:
                effective_dist: torch.Tensor = self._apply_grounding_weights(
                    distances, weights_device, self.config.grounding_gamma
                )  # [B, num_items]
            else:
                effective_dist = distances  # unchanged; pure L2 ranking

            # ── Step 7: Rank and collect top-K predictions ─────────────────────
            for i in range(actual_bs):
                row = batch_df.iloc[i]
                target_id = int(row[ITEM_ID])

                if is_sampled:
                    # Restrict ranking to pre-defined candidate_item_ids.
                    cand_val = row.get(CANDIDATE_ITEM_IDS, None)
                    if cand_val is None or (
                        not hasattr(cand_val, "__len__")
                        and isinstance(cand_val, float)
                        and np.isnan(cand_val)
                    ):
                        # No candidates available; fall back to full ranking.
                        cand_ids_list: list[int] = list(range(item_emb_device.shape[0]))
                    else:
                        cand_ids_list = list(cand_val)

                    cand_tensor = torch.tensor(
                        cand_ids_list, dtype=torch.long, device=device
                    )
                    cand_dists = effective_dist[i, cand_tensor]  # [num_cands]
                    sorted_cand_idx = torch.argsort(cand_dists)[:maxk]  # [≤ maxk]
                    top_k_ids = cand_tensor[sorted_cand_idx].cpu().numpy()  # [≤ maxk]

                    # Pad with -1 sentinels when fewer candidates than maxk.
                    if len(top_k_ids) < maxk:
                        pad = np.full(maxk - len(top_k_ids), -1, dtype=np.int64)
                        top_k_ids = np.concatenate([top_k_ids, pad])
                else:
                    # Full protocol: rank all items.
                    sorted_idx = torch.argsort(effective_dist[i])[:maxk]  # [maxk]
                    top_k_ids = sorted_idx.cpu().numpy()  # [maxk]

                all_pred_item_ids.append(top_k_ids.reshape(1, maxk))
                all_target_item_ids.append(np.array([[target_id]], dtype=np.int64))
                all_target_masks.append(np.array([[True]], dtype=bool))

        # ── Aggregate and compute metrics ──────────────────────────────────────
        pred_arr = np.concatenate(all_pred_item_ids, axis=0)       # [N, maxk]
        target_arr = np.concatenate(all_target_item_ids, axis=0)   # [N, 1]
        mask_arr = np.concatenate(all_target_masks, axis=0)        # [N, 1]

        eval_data = RetrievalEvalData(
            pred_item_ids=pred_arr,
            target_item_ids=target_arr,
            target_mask=mask_arr,
        )
        return self._compute_metrics(eval_data)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, eval_data: RetrievalEvalData) -> dict[str, float]:
        """Compute Recall@K and NDCG@K from a :class:`RetrievalEvalData` object.

        Args:
            eval_data: Aggregated retrieval evaluation data over all users.

        Returns:
            Dict mapping ``"recall@K"`` / ``"ndcg@K"`` to scalar floats,
            for every K in ``config.eval_topk``.
        """
        results: dict[str, float] = {}
        ks: tuple[int, ...] = tuple(self.config.eval_topk)

        for metric_name in self.config.eval_metrics:
            name = metric_name.strip().lower()
            if name == "recall":
                scores = RecallMetric(ks).compute(eval_data)
            elif name == "ndcg":
                scores = NDCGMetric(ks).compute(eval_data)
            else:
                self._log(
                    "Unknown eval metric '%s' — skipping.", metric_name, level="warning"
                )
                continue
            results.update(scores)

        for key, val in results.items():
            self._log("  %s = %.4f", key, val)

        return results


__all__ = ["BIGRecTrainer"]
