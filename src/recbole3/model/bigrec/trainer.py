"""BIGRec LoRA SFT trainer.

This module owns model/tokenizer loading and the HuggingFace ``Trainer`` based
fine-tuning loop. Generation and grounding evaluation live in ``generator.py``
and ``grounding.py``; thin wrappers remain here for compatibility with existing
tests and call sites.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from importlib.util import find_spec
from typing import Any, Literal

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer as HFTrainer,
    TrainingArguments,
)

from recbole3.dataset.utils import ITEM_ID
from recbole3.evaluation.metric import RetrievalEvalData
from recbole3.model.bigrec.config import BIGRecConfig
from recbole3.model.bigrec.data import (
    BIGRecSFTDataset,
    build_item_text_lookup,
    select_sft_training_records,
)
from recbole3.model.bigrec.generator import BIGRecGenerator
from recbole3.model.bigrec.grounding import BIGRecGrounder

logger = logging.getLogger(__name__)

# Fallback token IDs used when the tokenizer does not expose them explicitly.
# Both values are the standard LLaMA / LLaMA-2 defaults.
_DEFAULT_PAD_TOKEN_ID: int = 0   # unk_token_id; used when pad_token_id is None
_DEFAULT_EOS_TOKEN_ID: int = 2   # </s>; used when eos_token_id is None

class BIGRecTrainer:
    """BIGRec trainer for LoRA SFT fine-tuning.

    The trainer is intentionally self-contained: it does not inherit from
    RecBole3.0's ``Trainer`` class, mirroring the LCRecTrainer pattern. It
    delegates the optimization loop to HuggingFace ``Trainer`` and delegates
    generation / grounding evaluation to dedicated helper components.

    Args:
        config: Fully resolved :class:`BIGRecConfig` dataclass.
    """

    def __init__(self, config: BIGRecConfig) -> None:
        self.config = config

    def _generator(self) -> BIGRecGenerator:
        return BIGRecGenerator(
            self.config,
            is_main_process=self._is_main_process,
            log=self._log,
        )

    def _grounder(self) -> BIGRecGrounder:
        return BIGRecGrounder(
            self.config,
            is_main_process=self._is_main_process,
            log=self._log,
        )

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

    def _get_device_map(self) -> str | dict[str, int]:
        """Resolve ``device_map`` for ``from_pretrained``.

        * DDP (torchrun / LOCAL_RANK set): returns ``{"": local_rank}``.
        * Pipeline-parallel mode (``config.pipeline_parallel=True``): returns
          ``"auto"`` so the model is sharded across the GPUs exposed by
          ``CUDA_VISIBLE_DEVICES`` (set to at most ``pipeline_parallel_gpus``
          consecutive GPUs before CUDA initialises).
        * Single-GPU mode (default): returns ``{"": 0}``.

        The P2P peer-mapping exhaustion error caused by ``device_map="auto"``
        on an 8-GPU server is avoided by restricting ``CUDA_VISIBLE_DEVICES``
        to exactly ``pipeline_parallel_gpus`` GPUs (default 2), which limits
        GPU-pair P2P mappings to C(2,2)=1 instead of C(8,2)=28.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            return {"": local_rank}
        if self.config.pipeline_parallel:
            return "auto"
        # Single-GPU: CUDA_VISIBLE_DEVICES is already restricted to device_id,
        # so the target physical GPU always appears as logical GPU 0.
        return {"": 0}

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def _load_tokenizer(self, padding_side: str = "left") -> AutoTokenizer:
        """Load the HuggingFace tokenizer from ``config.llm_path``.

        Args:
            padding_side: Padding direction for tokenizer outputs. BIGRec uses
                          ``'left'`` during SFT training to match the official
                          implementation, and also during batch beam-search
                          generation / embedding extraction so the real last
                          token aligns to index ``-1``.

        Returns:
            Loaded tokenizer with ``pad_token_id`` guaranteed non-None.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_path,
            use_fast=False,
            padding_side=padding_side,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = _DEFAULT_PAD_TOKEN_ID
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
        if self.config.load_in_8bit:
            if find_spec("bitsandbytes") is None:
                raise ImportError(
                    "BIGRecConfig.load_in_8bit=True requires bitsandbytes. "
                    "Install a CUDA-compatible bitsandbytes build in the training "
                    "environment, or set load_in_8bit=False."
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "BIGRecConfig.load_in_8bit=True requires a Transformers version "
                    "that provides BitsAndBytesConfig. Please upgrade transformers "
                    "or set load_in_8bit=False."
                ) from exc

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # Distribute model evenly across visible GPUs (~90% of each GPU's memory).
        if device_map == "auto" and torch.cuda.is_available():
            n_visible = torch.cuda.device_count()
            per_gpu_bytes = int(torch.cuda.get_device_properties(0).total_memory * 0.9)
            per_gpu_str = f"{per_gpu_bytes // (1024 ** 3)}GiB"
            load_kwargs["max_memory"] = {i: per_gpu_str for i in range(n_visible)}

        model = AutoModelForCausalLM.from_pretrained(self.config.llm_path, **load_kwargs)
        model.config.use_cache = inference_mode

        if self.config.use_lora:
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
        """Encode text strings as BIGRec grounding embeddings."""
        return self._grounder().extract_embeddings(model, tokenizer, texts, batch_size, device)

    def _precompute_item_embeddings(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        item_texts: list[str],
        cache_path: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Return item embeddings, loading from disk cache when available."""
        return self._grounder().precompute_item_embeddings(
            model,
            tokenizer,
            item_texts,
            cache_path,
            device,
            extract_embeddings=self._extract_embeddings,
        )

    def _compute_popularity_weights(
        self,
        task_data: Any,
        num_items: int,
    ) -> torch.Tensor:
        """Compute min-max normalised item popularity from training interactions."""
        return self._grounder().compute_popularity_weights(task_data, num_items)

    def _load_cf_weights(self, num_items: int) -> torch.Tensor:
        """Load and normalise pre-computed CF model scores from disk."""
        return self._grounder().load_cf_weights(num_items)

    def _build_grounding_weights(
        self,
        task_data: Any,
        num_items: int,
    ) -> torch.Tensor | None:
        """Build the combined grounding weight vector for Eq. 3."""
        return self._grounder().build_grounding_weights(task_data, num_items)

    @staticmethod
    def _apply_grounding_weights(
        dist: torch.Tensor,
        weights: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Apply Eq. 3 to reweight L2 distances by popularity / CF signal."""
        return BIGRecGrounder.apply_grounding_weights(dist, weights, gamma)

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

        # Set CUDA_VISIBLE_DEVICES BEFORE the CUDA context is initialised.
        # HF Trainer wraps the model with nn.DataParallel when device_count()>1
        # and LOCAL_RANK==-1, causing P2P peer-mapping errors on multi-GPU nodes.
        # (In DDP mode LOCAL_RANK is set by torchrun; skip this path.)
        if int(os.environ.get("LOCAL_RANK", "-1")) == -1:
            if self.config.pipeline_parallel:
                # Limit to pipeline_parallel_gpus GPUs starting at device_id;
                # keeps P2P mappings at C(2,2)=1 instead of C(8,2)=28.
                gpu_ids = ",".join(
                    str(self.config.device_id + i)
                    for i in range(self.config.pipeline_parallel_gpus)
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                self._log(
                    "Pipeline-parallel mode: CUDA_VISIBLE_DEVICES=%s "
                    "(%d GPUs, device_map=auto)",
                    gpu_ids, self.config.pipeline_parallel_gpus,
                )
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device_id)
                self._log("Single-GPU mode: CUDA_VISIBLE_DEVICES=%s", self.config.device_id)

        # 1. Tokenizer (left-padding during SFT, matching official BIGRec).
        tokenizer = self._load_tokenizer(padding_side="left")
        if self._is_main_process():
            tokenizer.save_pretrained(output_dir)

        # 2. Item text lookup.
        item_text_lookup = build_item_text_lookup(task_data, self.config)

        # 3. Keep only full sliding-window histories. A user whose available
        #    history never reaches the configured length contributes one latest
        #    (therefore longest) row. Filter before sampling so sample_num does
        #    not preferentially waste its budget on short prefixes.
        train_frame: pd.DataFrame = task_data.get_train_dataset().frame  # type: ignore[attr-defined]
        unfiltered_train_size = len(train_frame)
        train_frame = select_sft_training_records(
            train_frame,
            self.config.history_max_length,
        )
        self._log(
            "BIGRec SFT history selection: retained %d/%d rows (history length=%s).",
            len(train_frame),
            unfiltered_train_size,
            self.config.history_max_length,
        )

        if self.config.sample_num > 0 and len(train_frame) > self.config.sample_num:
            train_frame = (
                train_frame
                .sample(n=self.config.sample_num, random_state=42)
                .reset_index(drop=True)
            )
            self._log(
                "sample_num=%d: subsampled %d training rows (full dataset: %d rows).",
                self.config.sample_num, self.config.sample_num,
                unfiltered_train_size,
            )

        # 4. Resolve effective_max_steps: when num_train_epochs finishes within
        #    max_steps, let epochs control training naturally (pass -1 to HF Trainer).
        effective_batch_size: int = (
            self.config.train_batch_size * self.config.gradient_accumulation_steps
        )
        epoch_steps: int = (
            math.ceil(len(train_frame) / effective_batch_size)
            * self.config.num_train_epochs
        )
        if self.config.max_steps > 0 and epoch_steps <= self.config.max_steps:
            effective_max_steps: int = -1
            actual_train_steps: int = epoch_steps
            self._log(
                "epoch_steps=%d ≤ max_steps=%d: disabling max_steps cap — "
                "num_train_epochs=%d controls training.",
                epoch_steps, self.config.max_steps, self.config.num_train_epochs,
            )
        elif self.config.max_steps > 0:
            effective_max_steps = self.config.max_steps
            actual_train_steps = self.config.max_steps
        else:
            effective_max_steps = -1
            actual_train_steps = epoch_steps

        # Cap training frame to rows actually reached (speeds up tokenisation).
        if effective_max_steps > 0:
            max_samples: int = effective_max_steps * effective_batch_size
            if len(train_frame) > max_samples:
                train_frame = train_frame.head(max_samples).reset_index(drop=True)
                self._log(
                    "max_steps=%d: capped training frame to %d samples.",
                    effective_max_steps, max_samples,
                )

        sft_train = BIGRecSFTDataset(
            records=train_frame,
            tokenizer=tokenizer,
            item_text_lookup=item_text_lookup,
            config=self.config,
        )

        # Validation SFT dataset drives EarlyStoppingCallback via LM val loss
        # (official BIGRec approach; recommendation metrics are computed post-training).
        valid_frame: pd.DataFrame = task_data.get_eval_dataset("valid").frame  # type: ignore[attr-defined]
        max_eval_samples: int = actual_train_steps * self.config.eval_batch_size
        if len(valid_frame) > max_eval_samples:
            valid_frame = valid_frame.head(max_eval_samples).reset_index(drop=True)
            self._log(
                "Capped validation frame to %d samples (actual_train_steps=%d).",
                max_eval_samples, actual_train_steps,
            )

        sft_val = BIGRecSFTDataset(
            records=valid_frame,
            tokenizer=tokenizer,
            item_text_lookup=item_text_lookup,
            config=self.config,
        )

        # 5. Load model for training.
        device_map = self._get_device_map()
        model = self._load_model(device_map, inference_mode=False)

        # 6. Collator; pad_to_multiple_of=8 for memory-efficient CUDA kernels.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            padding="longest",
            return_tensors="pt",
        )

        # 7. HuggingFace TrainingArguments.
        #    warmup_steps takes precedence over warmup_ratio (official BIGRec uses
        #    a fixed 20-step warm-up).
        warmup_kwargs: dict[str, Any] = (
            {"warmup_steps": self.config.warmup_steps}
            if self.config.warmup_steps is not None
            else {"warmup_ratio": self.config.warmup_ratio}
        )
        hf_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=effective_max_steps,
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

        # Explicitly release the training model from GPU VRAM before returning.
        # Python's reference-counting GC does not free CUDA memory immediately
        # when local variables go out of scope; torch.cuda.empty_cache() is
        # required.  Without this, the training model (~18 GB) remains resident
        # when the pipeline subsequently calls evaluate(), which loads the
        # generation model (~17 GB) plus the base embedding model (~16 GB),
        # pushing total VRAM usage to ~51 GB and causing OOM on a 40 GB A100.
        #
        # HF Trainer internally keeps multiple references to the wrapped model
        # (self.model, self.model_wrapped, self.optimizer.param_groups,
        # self.lr_scheduler, self.accelerator, self.deepspeed_engine, ...).
        # Cyclic references between Trainer/optimizer/model mean that plain
        # `del` alone leaves the objects reachable via the cyclic GC generation
        # until it runs on its own schedule.  Force a full collection cycle
        # BEFORE calling empty_cache() so the CUDA allocator sees the tensors
        # as truly free.
        try:
            hf_trainer.model = None            # break Trainer → model ref
            hf_trainer.model_wrapped = None    # break Trainer → DDP/DP wrapper
            hf_trainer.optimizer = None        # break Trainer → optimizer → params
            hf_trainer.lr_scheduler = None
            hf_trainer.accelerator = None
            hf_trainer.deepspeed = None
            hf_trainer.callback_handler = None
        except AttributeError:
            # HF Trainer's attribute set differs across versions; ignore missing.
            pass
        del hf_trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._log(
            "Training model freed from GPU memory (allocated=%.2f GB, reserved=%.2f GB).",
            torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
        )

        return {"checkpoint_path": output_dir}

    # ── Gamma-search helpers ──────────────────────────────────────────────────

    @staticmethod
    def _default_gamma_search_values() -> tuple[float, ...]:
        """Return the official BIGRec 199-value gamma grid."""
        return BIGRecGrounder.default_gamma_search_values()

    # ── vLLM server management ────────────────────────────────────────────────

    def _resolve_vllm_python(self) -> str:
        """Return the Python executable to use for the vLLM subprocess."""
        return self._generator().resolve_vllm_python()

    def _collect_eval_targets(
        self,
        eval_frame: pd.DataFrame,
    ) -> tuple[list[int], list[list[int] | None]]:
        """Extract target item ids and per-row candidate lists from an eval split."""
        return BIGRecGenerator.collect_eval_targets(eval_frame)

    def _collect_eval_exclusions(
        self,
        eval_frame: pd.DataFrame,
    ) -> list[list[int]]:
        """Extract per-row interaction histories used by full ranking."""
        return BIGRecGenerator.collect_eval_exclusions(eval_frame)

    def _run_vllm_offline(
        self,
        prompts: list[str],
        checkpoint_path: str,
        work_dir: str,
    ) -> list[str]:
        """Run vLLM ``LLM.beam_search`` in a one-shot subprocess."""
        return self._generator().run_vllm_offline(prompts, checkpoint_path, work_dir)

    def _generate_all_titles_vllm(
        self,
        eval_frame: pd.DataFrame,
        item_text_lookup: list[str],
        checkpoint_path: str,
        work_dir: str,
    ) -> tuple[list[str], list[int], list[list[int] | None]]:
        """Generate item titles for an eval split via the offline vLLM subprocess."""
        return self._generator().generate_all_titles_vllm(
            eval_frame, item_text_lookup, checkpoint_path, work_dir
        )

    def _write_generation_debug_sample(
        self,
        generated_texts: list[str],
        target_ids: list[int],
        item_text_lookup: list[str],
        checkpoint_path: str,
        split: str,
        sample_size: int = 20,
    ) -> None:
        """Write a small random sample of raw LLM generations for inspection."""
        if not self._is_main_process() or not generated_texts:
            return

        n = min(sample_size, len(generated_texts))
        sample_indices = (
            pd.Series(range(len(generated_texts)))
            .sample(n=n, random_state=42)
            .astype(int)
            .tolist()
        )
        rows: list[dict[str, Any]] = []
        for index in sample_indices:
            target_id = int(target_ids[index])
            target_title = (
                item_text_lookup[target_id]
                if 0 <= target_id < len(item_text_lookup)
                else f"[invalid_target_id:{target_id}]"
            )
            rows.append(
                {
                    "row_index": index,
                    "generated_text": generated_texts[index],
                    "target_item_id": target_id,
                    "target_title": target_title,
                }
            )

        os.makedirs(checkpoint_path, exist_ok=True)
        debug_path = os.path.join(checkpoint_path, f"bigrec_generation_debug_{split}.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        self._log("Wrote BIGRec generation debug sample to %s", debug_path)

    def _run_gamma_search(
        self,
        dist: torch.Tensor,
        grounding_weights: torch.Tensor,
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        device: torch.device,
        gamma_values: tuple[float, ...],
        excluded_item_ids: list[list[int]] | None = None,
    ) -> dict[str, float]:
        """Grid-search for the best gamma per metric@K on a validation split."""
        return self._grounder().run_gamma_search(
            dist,
            grounding_weights,
            target_ids,
            cand_lists,
            device,
            gamma_values,
            excluded_item_ids,
            compute_metrics=self._compute_metrics,
        )

    def _evaluate_from_dist_per_k_gammas(
        self,
        dist: torch.Tensor,
        grounding_weights: torch.Tensor | None,
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        best_gammas: dict[str, float],
        device: torch.device,
        excluded_item_ids: list[list[int]] | None = None,
    ) -> dict[str, float]:
        """Evaluate a split using the best gamma independently per metric@K."""
        return self._grounder().evaluate_from_dist_per_k_gammas(
            dist,
            grounding_weights,
            target_ids,
            cand_lists,
            best_gammas,
            device,
            excluded_item_ids,
        )

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
        # Same GPU visibility setup as fit() — needed when evaluate() is
        # invoked standalone (pipeline_stage='evaluation') without a prior fit().
        if int(os.environ.get("LOCAL_RANK", "-1")) == -1:
            if self.config.pipeline_parallel:
                gpu_ids = ",".join(
                    str(self.config.device_id + i)
                    for i in range(self.config.pipeline_parallel_gpus)
                )
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_ids)
            else:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.device_id))

        item_text_lookup = build_item_text_lookup(task_data, self.config)

        # ── Phase 1: Generation ───────────────────────────────────────────────
        self._log("Eval phase 1: generation on %s split …", split)
        eval_frame: pd.DataFrame = task_data.get_eval_dataset(split).frame  # type: ignore[attr-defined]

        # eval_user_num: subsample a fixed user set (mirrors official BIGRec 5k test set).
        if 0 < self.config.eval_user_num < len(eval_frame):
            eval_frame = (
                eval_frame
                .sample(n=self.config.eval_user_num, random_state=42)
                .reset_index(drop=True)
            )
            self._log(
                "eval_user_num=%d: sampled %d users from %s split.",
                self.config.eval_user_num, self.config.eval_user_num, split,
            )

        if self.config.max_steps > 0:
            max_eval_users: int = self.config.max_steps * self.config.eval_batch_size
            if len(eval_frame) > max_eval_users:
                eval_frame = eval_frame.head(max_eval_users).reset_index(drop=True)
                self._log(
                    "max_steps=%d: capped %s evaluation to %d users.",
                    self.config.max_steps, split, max_eval_users,
                )

        valid_texts: list[str] | None = None
        valid_targets: list[int] | None = None
        valid_cands: list[list[int] | None] | None = None
        valid_excluded_item_ids: list[list[int]] | None = None

        excluded_item_ids = self._collect_eval_exclusions(eval_frame)

        # Generation runs in a one-shot vLLM subprocess (LLM.beam_search):
        # vLLM ≥ 0.6.4 dropped use_beam_search from the OpenAI-compatible
        # completions endpoint, so an HTTP server cannot produce true width-N
        # beam search anymore.  The subprocess exits before Phase 2 runs, so
        # its VRAM is reclaimed naturally — no manual server lifecycle needed.
        gen_work_dir = os.path.join(checkpoint_path, "vllm_gen_io")

        eval_texts, target_ids, cand_lists = self._generate_all_titles_vllm(
            eval_frame, item_text_lookup, checkpoint_path, gen_work_dir,
        )
        self._write_generation_debug_sample(
            eval_texts,
            target_ids,
            item_text_lookup,
            checkpoint_path,
            split,
        )

        if self.config.grounding_gamma_search:
            self._log("Eval phase 1b: valid split for gamma-search …")
            valid_frame: pd.DataFrame = task_data.get_eval_dataset("valid").frame  # type: ignore[attr-defined]
            if self.config.max_steps > 0:
                max_valid_users: int = self.config.max_steps * self.config.eval_batch_size
                if len(valid_frame) > max_valid_users:
                    valid_frame = valid_frame.head(max_valid_users).reset_index(drop=True)
            valid_excluded_item_ids = self._collect_eval_exclusions(valid_frame)
            valid_texts, valid_targets, valid_cands = self._generate_all_titles_vllm(
                valid_frame, item_text_lookup, checkpoint_path, gen_work_dir,
            )

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._log("vLLM offline generation complete, VRAM reclaimed.")

        tokenizer = self._load_tokenizer(padding_side="left")
        device: torch.device = torch.device(
            f"cuda:{self.config.device_id}" if torch.cuda.is_available() else "cpu"
        )

        # ── Phase 2: Embedding extraction ────────────────────────────────────
        if self.config.embedding_use_base_model:
            emb_model: Any = self._load_base_model_for_embedding(self._get_device_map())
        else:
            emb_model = self._load_trained_model(checkpoint_path)
        device = next(emb_model.parameters()).device

        dataset_name = getattr(getattr(task_data, "config", None), "name", "dataset")
        cache_filename = f"{dataset_name}_{split}_item_embs.pt"
        cache_path = os.path.join(self.config.embedding_cache_dir, cache_filename)

        self._log("Eval phase 2: pre-computing item embeddings …")
        item_embeddings: torch.Tensor = self._precompute_item_embeddings(
            emb_model, tokenizer, item_text_lookup, cache_path, device
        )  # [num_items, H] CPU
        item_emb_device = item_embeddings.to(device)  # [num_items, H]

        num_items: int = item_embeddings.shape[0]
        grounding_weights: torch.Tensor | None = self._build_grounding_weights(
            task_data, num_items
        )  # [num_items] CPU, or None

        # ── Phase 3: Ranking ──────────────────────────────────────────────────
        if self.config.grounding_gamma_search and grounding_weights is not None:
            weights_device = grounding_weights.to(device)
            gamma_values: tuple[float, ...] = (
                tuple(self.config.grounding_gamma_search_values)
                if self.config.grounding_gamma_search_values
                else self._default_gamma_search_values()
            )

            self._log("Eval phase 3a: gamma-search on valid split …")
            valid_oracle_embs = self._extract_embeddings(
                emb_model, tokenizer, valid_texts,  # type: ignore[arg-type]
                batch_size=self.config.embedding_batch_size, device=device,
            )  # [N_valid, H] CPU
            valid_dist: torch.Tensor = torch.cdist(
                valid_oracle_embs.to(device), item_emb_device, p=2.0
            )  # [N_valid, num_items]
            best_gammas = self._run_gamma_search(
                valid_dist, weights_device,
                valid_targets, valid_cands,  # type: ignore[arg-type]
                device, gamma_values,
                valid_excluded_item_ids,
            )

            self._log("Eval phase 3b: evaluating %s with per-K best gammas …", split)
            eval_oracle_embs = self._extract_embeddings(
                emb_model, tokenizer, eval_texts,
                batch_size=self.config.embedding_batch_size, device=device,
            )  # [N_eval, H] CPU
            eval_dist: torch.Tensor = torch.cdist(
                eval_oracle_embs.to(device), item_emb_device, p=2.0
            )  # [N_eval, num_items]
            return self._evaluate_from_dist_per_k_gammas(
                eval_dist,
                weights_device,
                target_ids,
                cand_lists,
                best_gammas,
                device,
                excluded_item_ids,
            )

        return self._rank_from_texts(
            emb_model=emb_model,
            tokenizer=tokenizer,
            item_emb_device=item_emb_device,
            generated_texts=eval_texts,
            target_ids=target_ids,
            cand_lists=cand_lists,
            grounding_weights=grounding_weights,
            device=device,
            excluded_item_ids=excluded_item_ids,
        )

    def _rank_from_texts(
        self,
        emb_model: Any,
        tokenizer: AutoTokenizer,
        item_emb_device: torch.Tensor,
        generated_texts: list[str],
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        grounding_weights: torch.Tensor | None,
        device: torch.device,
        excluded_item_ids: list[list[int]] | None = None,
    ) -> dict[str, float]:
        """Batch-wise oracle embedding extraction + L2 ranking + metrics."""
        return self._grounder().rank_from_texts(
            emb_model,
            tokenizer,
            item_emb_device,
            generated_texts,
            target_ids,
            cand_lists,
            grounding_weights,
            device,
            excluded_item_ids,
            extract_embeddings=self._extract_embeddings,
            compute_metrics=self._compute_metrics,
        )

    def _compute_metrics(self, eval_data: RetrievalEvalData) -> dict[str, float]:
        """Compute Recall@K and NDCG@K from retrieval evaluation data."""
        return self._grounder().compute_metrics(eval_data)


__all__ = ["BIGRecTrainer"]
