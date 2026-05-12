"""ReaRec: inference-time computing framework for sequential recommendation.

Paper: "Think Before Recommend: Unleashing the Latent Reasoning Power for
Sequential Recommendation" (Tang et al., 2025).

This module implements ReaRec with a SASRec-style Transformer backbone and
supports two learning strategies: ERL and PRL.

Design notes
------------
* Left-padding is used for item sequences, matching the official codebase.
  Position IDs are assigned via cumsum of the non-padding mask (0-indexed from
  the first real item), and padding positions receive a sentinel position index
  that maps to the zero vector via padding_idx.

* The PADDING item ID is `num_items` (one beyond valid IDs 0..num_items-1).
  The item embedding table has shape [num_items+1, D] with padding_idx=num_items.

* Parameters are lazily initialised in ensure_initialized() once num_items is known.

* KV cache: SASRec backbone caches Keys/Values from the initial sequence encode
  so that each reasoning step only processes one new token, avoiding O(K*L)
  redundant computation. HSTU backbone is reserved via a stub.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.dataset import ITEM_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.model.rearec.config import ReaRecConfig
from recbole3.model.rearec.data import (
    ReaRecEvalCollator,
    ReaRecTrainCollator,
)
from recbole3.model.rearec.layers import (
    HSTUBackbone,
    ReaRecAutoRegressiveWrapper,
    SASRecBackbone,
    TransformerEncoder,
)


class ReaRecModel(BaseRetrievalModel):
    """ReaRec sequential retrieval model.

    Inherits from BaseRetrievalModel (RecBole3 interface).
    Supports ERL and PRL learning strategies on a SASRec backbone.
    """

    def __init__(self, config: ReaRecConfig) -> None:
        super().__init__(config)
        backbone_name = str(config.backbone).lower()
        if backbone_name not in ("sasrec", "hstu"):
            raise ValueError(
                f"ReaRec backbone '{config.backbone}' is not supported. "
                "Choose 'sasrec' or 'hstu'."
            )
        self.config: ReaRecConfig  # narrow type for mypy

        # Lazy-initialised parameters (populated in ensure_initialized)
        self._num_items: int | None = None
        self._item_emb: nn.Embedding | None = None
        self._pos_emb: nn.Embedding | None = None
        self._ar_wrapper: ReaRecAutoRegressiveWrapper | None = None

        # Loss function (ignore_index is overwritten after init when num_items known)
        self._loss_fct = nn.CrossEntropyLoss()

        # Step-based epoch tracking for PRL warmup gating.
        # _train_steps is incremented once per forward() call (training only).
        # _steps_per_epoch is inferred from train-dataset size on the first
        # forward pass; until then warmup_epochs=0 (always-on) works without it.
        self._train_steps: int = 0
        self._train_size: int | None = None
        self._steps_per_epoch: int | None = None

    # ------------------------------------------------------------------
    # Framework lifecycle hooks
    # ------------------------------------------------------------------

    def ensure_initialized(self, prepared_data: Any) -> None:
        num_items = int(prepared_data.get_num_items())
        print(
            f"[rearec] initializing — num_items={num_items}, "
            f"embedding_dim={self.config.embedding_dim}, "
            f"strategy={self.config.learning_strategy}, "
            f"reason_step={self.config.reason_step}, "
            f"backbone={self.config.backbone}",
            flush=True,
        )
        self._init_params(num_items)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[rearec] trainable parameters: {total_params:,}", flush=True)

    def build_train_collator(self, prepared_data: Any) -> BaseCollator:
        self._init_params(int(prepared_data.get_num_items()))
        self._train_size = len(prepared_data.get_train_dataset())
        L = int(self.config.history_max_length or 50)
        print(
            f"[rearec] building train collator — "
            f"backbone={self.config.backbone}, "
            f"history_max_length={L}, "
            f"warmup_epochs={self.config.warmup_epochs}, "
            f"train_size={self._train_size}",
            flush=True,
        )
        if str(self.config.backbone).lower() == "hstu":
            from recbole3.model.rearec.data import ReaRecHSTUTrainCollator
            return ReaRecHSTUTrainCollator(
                self.config,
                prepared_data,
                history_max_length=L,
            )
        return ReaRecTrainCollator(
            self.config,
            prepared_data,
            num_items=self._num_items,  # type: ignore[arg-type]
            history_max_length=L,
        )

    def build_eval_collator(self, prepared_data: Any) -> BaseCollator:
        self._init_params(int(prepared_data.get_num_items()))
        L = int(self.config.history_max_length or 50)
        if str(self.config.backbone).lower() == "hstu":
            from recbole3.model.rearec.data import ReaRecHSTUEvalCollator
            return ReaRecHSTUEvalCollator(
                self.config,
                prepared_data,
                history_max_length=L,
            )
        return ReaRecEvalCollator(
            self.config,
            prepared_data,
            num_items=self._num_items,  # type: ignore[arg-type]
            history_max_length=L,
        )

    # ------------------------------------------------------------------
    # Core forward / loss / predict
    # ------------------------------------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Encode the sequence with K reasoning steps.

        Called only during training. Returns intermediate tensors consumed by
        compute_loss().

        Returns dict with keys:
            model_output:    [B*(1+noise), K+1, D] reasoning-step hidden states.
            item_emb_weight: [num_items, D] or [num_items+1, D] item embeddings;
                             compute_loss slices ``[:num_items]`` to strip any
                             padding row, which is a no-op when HSTU backbone
                             already returns ``[num_items, D]``.
        """
        self._require_initialized()
        ar_wrapper = self._ar_wrapper_module()

        history_item_ids = batch[HISTORY_ITEM_IDS].to(dtype=torch.long)   # [B, L]
        history_lengths = batch["history_lengths"].to(dtype=torch.long)    # [B]

        # Count this training step and, on the first call, infer steps_per_epoch
        # from the training-dataset size stored by build_train_collator.
        self._train_steps += 1
        if self._steps_per_epoch is None and self._train_size is not None:
            batch_size = int(history_item_ids.shape[0])
            self._steps_per_epoch = max(1, -(-self._train_size // batch_size))

        backbone_name = str(self.config.backbone).lower()
        if backbone_name == "hstu":
            from recbole3.model.hstu.data import HISTORY_TIMESTAMPS
            timestamps = batch[HISTORY_TIMESTAMPS].to(dtype=torch.float32)
            raw_context: dict[str, torch.Tensor] | None = {
                "item_ids": history_item_ids,
                "timestamps": timestamps,
            }
            B, L = history_item_ids.shape
            D = int(self.config.embedding_dim)
            # Dummy input_embs — ignored by HSTUBackbone.initial_encode
            input_embs = history_item_ids.new_zeros(B, L, D, dtype=torch.float32)
        else:
            input_embs = self._build_input_embeddings(history_item_ids, history_lengths)
            raw_context = None

        noise_factor = self._effective_noise_factor()
        model_output = ar_wrapper(
            input_embs, history_lengths,
            noise_factor=noise_factor,
            raw_context=raw_context,
        )  # [B*(1 or 2), K+1, D]

        return {
            "model_output": model_output,
            "item_emb_weight": self._scoring_embs(),  # [num_items, D] or [num_items+1, D]
        }

    def compute_loss(
        self, batch: Mapping[str, torch.Tensor], outputs: dict[str, Any]
    ) -> torch.Tensor:
        """Compute ERL or PRL training loss."""
        self._require_initialized()
        target_item_ids = batch[ITEM_ID].to(dtype=torch.long)  # [B]
        model_output: torch.Tensor = outputs["model_output"]   # [B*(1 or 2), K+1, D]
        item_emb_weight: torch.Tensor = outputs["item_emb_weight"]  # [num_items, D] (or [num_items+1, D] for SASRec)

        # For SASRec the weight includes a padding row at index num_items; slice it off.
        # For HSTU the weight is already [num_items, D], so this is a no-op.
        scoring_embs = item_emb_weight[: self._num_items]  # [num_items, D]

        strategy = str(self.config.learning_strategy).lower()
        if strategy == "erl":
            return self._compute_erl_loss(model_output, target_item_ids, scoring_embs)
        elif strategy == "prl":
            return self._compute_prl_loss(model_output, target_item_ids, scoring_embs)
        else:
            raise ValueError(
                f"Unknown learning strategy '{strategy}'. Choose 'erl' or 'prl'."
            )

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-k item IDs for each user in the batch.

        Uses the final reasoning step hidden state as the user representation.
        """
        self._require_initialized()
        user_embs = self._encode_user_embeddings(model_inputs)  # [B, D]
        scoring_embs = self._scoring_embs()                     # [num_items, D]

        if candidate_item_ids is not None:
            candidate_item_ids = candidate_item_ids.to(
                device=user_embs.device, dtype=torch.long
            )
            cand_embs = scoring_embs[candidate_item_ids]        # [B, C, D]
            scores = torch.einsum("bd,bcd->bc", user_embs, cand_embs) / self.config.temperature
            topk_local = torch.topk(scores, k=k, dim=1).indices
            return torch.gather(candidate_item_ids, 1, topk_local)

        scores = torch.matmul(user_embs, scoring_embs.t()) / self.config.temperature
        # [B, num_items]

        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
            history_mask = torch.zeros_like(scores, dtype=torch.bool)
            history_mask.scatter_(
                1,
                exclude_item_ids.to(device=scores.device, dtype=torch.long),
                exclude_mask.to(device=scores.device, dtype=torch.bool),
            )
            scores = scores.masked_fill(history_mask, float("-inf"))

        return torch.topk(scores, k=k, dim=1).indices.to(dtype=torch.long)

    # ------------------------------------------------------------------
    # ERL loss
    # ------------------------------------------------------------------

    def _compute_erl_loss(
        self,
        model_output: torch.Tensor,   # [B, K+1, D]
        target_ids: torch.Tensor,     # [B]
        scoring_embs: torch.Tensor,   # [num_items, D]
    ) -> torch.Tensor:
        """Ensemble Reasoning Learning loss.

        L_ERL = CE(mean_of_K+1_steps) - lambda * KL_diversity

        CE variant (full-vocab or sampled softmax) is controlled by config.loss_type.
        The KL term uses the same item set as the CE to keep the distribution support
        consistent; for sampled_softmax, one shared negative set is drawn for all T steps.
        """
        B = target_ids.shape[0]
        thinking_embs = model_output[:B]        # [B, K+1, D]
        T = thinking_embs.shape[1]              # K+1
        temperature = float(self.config.temperature)

        ensemble_embs = thinking_embs.mean(dim=1)   # [B, D]
        loss = self._item_ce_loss(ensemble_embs, target_ids, scoring_embs, temperature)

        if self.config.kl_weight > 0 and T > 1:
            if self._effective_loss_type() == "sampled_softmax":
                # Share one negative set across all T steps for a consistent support.
                num_neg = int(self.config.num_negatives)
                neg_ids = torch.randint(
                    0, self._num_items, (B, num_neg),
                    device=thinking_embs.device, dtype=torch.long,
                )
                pos_embs = scoring_embs[target_ids]   # [B, D]
                neg_embs = scoring_embs[neg_ids]       # [B, num_neg, D]
                pos_logits = torch.einsum("btd,bd->bt", thinking_embs, pos_embs).unsqueeze(-1) / temperature
                neg_logits = torch.einsum("btd,bnd->btn", thinking_embs, neg_embs) / temperature
                step_logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, T, num_neg+1]
            else:
                step_logits = torch.matmul(thinking_embs, scoring_embs.t()) / temperature
                # [B, T, num_items]
            step_probs = F.softmax(step_logits, dim=-1)                   # [B, T, N]
            step_log_probs = F.log_softmax(step_logits.detach(), dim=-1)  # [B, T, N]
            cross = torch.bmm(step_probs, step_log_probs.transpose(1, 2)) # [B, T, T]
            self_ent = (step_probs * step_log_probs).sum(-1, keepdim=True) # [B, T, 1]
            kl_div = self_ent - cross  # KL(p_t || p_s),  [B, T, T]
            off_diag = ~torch.eye(T, device=thinking_embs.device, dtype=torch.bool)
            kl_loss = kl_div[:, off_diag].mean()
            loss = loss - self.config.kl_weight * kl_loss

        return loss

    # ------------------------------------------------------------------
    # PRL loss
    # ------------------------------------------------------------------

    def _compute_prl_loss(
        self,
        model_output: torch.Tensor,   # [B*(1 or 2), K+1, D]
        target_ids: torch.Tensor,     # [B]
        scoring_embs: torch.Tensor,   # [num_items, D]
    ) -> torch.Tensor:
        """Progressive Reasoning Learning loss.

        L_PRL = CE(final_step) + pl_weight * progressive_CE + cl_weight * contrastive_CE

        Progressive CE uses decreasing temperature across steps 0..K-1.
        Contrastive CE aligns clean and noisy reasoning trajectories (batch-level, not item-level).
        CE variant (full-vocab or sampled softmax) is controlled by config.loss_type.
        """
        B = target_ids.shape[0]
        repeat_times = model_output.shape[0] // B  # 1 or 2

        clean_output = model_output[:B]             # [B, K+1, D]
        K1 = clean_output.shape[1]                  # K+1
        temperature = float(self.config.temperature)

        # --- 1. Primary CE on the final reasoning step ---
        final_embs = clean_output[:, -1, :]         # [B, D]
        loss = self._item_ce_loss(final_embs, target_ids, scoring_embs, temperature)

        # --- 2. Progressive learning CE on steps 0..K-1 ---
        if K1 > 1:
            prior_embs = clean_output[:, :-1, :]    # [B, K, D]
            K = prior_embs.shape[1]
            exponents = torch.arange(K, 0, -1, device=clean_output.device, dtype=torch.float32)
            temps = self.config.temperature * (self.config.temp_scale ** exponents)  # [K]
            pl_loss_acc = clean_output.new_tensor(0.0)
            for k in range(K):
                pl_loss_acc = pl_loss_acc + self._item_ce_loss(
                    prior_embs[:, k, :], target_ids, scoring_embs, float(temps[k])
                )
            pl_loss = pl_loss_acc / K
            loss = loss + self.config.pl_weight * pl_loss

        # --- 3. Reasoning-aware contrastive loss (clean vs noisy, steps 1..K) ---
        # This is batch-level contrastive (not item-level), so it is unaffected by loss_type.
        if repeat_times > 1 and self.config.cl_weight > 0 and K1 > 1:
            noisy_output = model_output[B:]         # [B, K+1, D]
            view1 = clean_output[:, 1:, :]          # [B, K, D]
            view2 = noisy_output[:, 1:, :]          # [B, K, D]
            K = view1.shape[1]
            sim = torch.einsum("bkd,jkd->bjk", view2, view1) / temperature
            # sim[b, j, k] → [B, B, K]; for each (b, k) predict j=b
            cl_labels = (
                torch.arange(B, device=clean_output.device)
                .unsqueeze(1)
                .expand(B, K)
            )  # [B, K]
            cl_loss = self._loss_fct(sim, cl_labels)
            loss = loss + self.config.cl_weight * cl_loss

        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_user_embeddings(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode batch histories and return the final-step user embeddings [B, D]."""
        history_item_ids = batch[HISTORY_ITEM_IDS].to(dtype=torch.long)   # [B, L]
        history_lengths = batch["history_lengths"].to(dtype=torch.long)    # [B]

        backbone_name = str(self.config.backbone).lower()
        if backbone_name == "hstu":
            from recbole3.model.hstu.data import HISTORY_TIMESTAMPS
            timestamps = batch[HISTORY_TIMESTAMPS].to(dtype=torch.float32)
            raw_context: dict[str, torch.Tensor] | None = {
                "item_ids": history_item_ids,
                "timestamps": timestamps,
            }
            B, L = history_item_ids.shape
            D = int(self.config.embedding_dim)
            input_embs = history_item_ids.new_zeros(B, L, D, dtype=torch.float32)
        else:
            input_embs = self._build_input_embeddings(history_item_ids, history_lengths)
            raw_context = None

        # During eval, run full reasoning without noise
        model_output = self._ar_wrapper_module()(
            input_embs, history_lengths,
            noise_factor=0.0,
            raw_context=raw_context,
        )  # [B, K+1, D]
        return model_output[:, -1, :]  # [B, D] — final reasoning step

    def _build_input_embeddings(
        self,
        history_item_ids: torch.Tensor,  # [B, L]
        history_lengths: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """Compute item + position embeddings with left-padding position IDs.

        For a left-padded sequence [PAD, PAD, item1, item2, item3]:
          - Padded positions → position index = history_max_length (zero embedding)
          - Real positions → 0-indexed from the first real item via cumsum
        """
        item_emb = self._item_emb_module()
        pos_emb = self._pos_emb_module()
        L = self.config.history_max_length or 50

        padding_id = self._num_items  # type: ignore[assignment]
        padding_mask = (history_item_ids != padding_id)  # [B, L] True = real item

        # Cumsum gives 1-indexed counts; subtract 1 for 0-indexed positions
        valid_pos_ids = torch.cumsum(padding_mask.long(), dim=1) - 1   # [B, L]
        # Padding positions get sentinel index L (maps to zero pos_emb via padding_idx)
        pos_ids = torch.where(padding_mask, valid_pos_ids, torch.full_like(valid_pos_ids, L))

        item_embs = item_emb(history_item_ids)  # [B, L, D]
        pos_embs = pos_emb(pos_ids)             # [B, L, D]
        return item_embs + pos_embs             # [B, L, D]

    def _effective_noise_factor(self) -> float:
        """Return the noise factor for PRL, respecting warmup_steps.

        Noise is only meaningful when reason_step >= 1 (there are reasoning tokens
        to perturb). With reason_step=0, no reasoning tokens exist so noise is skipped.
        """
        if not self.training:
            return 0.0
        if str(self.config.learning_strategy).lower() != "prl":
            return 0.0
        if int(self.config.reason_step) < 1:
            return 0.0
        # Mirror official condition: `if epoch > self.warmup_epoch`.
        # Derive current epoch from training-step counter when steps_per_epoch
        # is known (set on the first forward pass if build_train_collator was
        # called).  Formula: ceil(train_steps / steps_per_epoch), floored at 1.
        warmup = int(self.config.warmup_epochs)
        if warmup == 0:
            return float(self.config.noise_factor)
        if self._steps_per_epoch is None:
            # steps_per_epoch not yet known → stay in warmup conservatively
            return 0.0
        current_epoch = max(1, (self._train_steps - 1) // self._steps_per_epoch + 1)
        if current_epoch <= warmup:
            return 0.0
        return float(self.config.noise_factor)

    def _init_params(self, num_items: int) -> None:
        """Initialise lazily-created parameters once num_items is known."""
        if self._num_items is not None:
            return
        self._num_items = int(num_items)
        cfg = self.config
        L = int(cfg.history_max_length or 50)
        D = int(cfg.embedding_dim)
        K = int(cfg.reason_step)

        backbone_name = str(cfg.backbone).lower()

        if backbone_name == "sasrec":
            # Item embedding: valid IDs 0..num_items-1, padding ID = num_items
            self._item_emb = nn.Embedding(
                self._num_items + 1, D, padding_idx=self._num_items
            )
            # Position embedding: valid positions 0..L-1, padding sentinel = L
            self._pos_emb = nn.Embedding(L + 1, D, padding_idx=L)

            encoder = TransformerEncoder(
                n_layers=int(cfg.num_layers),
                n_heads=int(cfg.num_heads),
                hidden_size=D,
                inner_size=int(cfg.inner_size),
                hidden_dropout_prob=float(cfg.dropout),
                attn_dropout_prob=float(cfg.dropout),
                hidden_act=str(cfg.hidden_act),
                layer_norm_eps=float(cfg.layer_norm_eps),
            )
            backbone = SASRecBackbone(encoder)

        elif backbone_name == "hstu":
            # No _item_emb / _pos_emb for HSTU — the inner HSTUModel owns them.
            # Set hstu_config.history_max_length so that max_encoder_length = L + K,
            # providing enough capacity for K reasoning steps beyond the history.
            from recbole3.model.hstu.config import HSTUConfig
            from recbole3.model.hstu.model import HSTUModel

            hstu_history_max_length = L + max(K, 1) - 1
            hstu_cfg = HSTUConfig(
                name="hstu",
                history_max_length=hstu_history_max_length,
                embedding_dim=D,
                num_layers=int(cfg.num_layers),
                num_heads=int(cfg.num_heads),
                attention_dim=int(cfg.attention_dim),
                linear_hidden_dim=int(cfg.linear_hidden_dim),
                input_dropout_rate=float(cfg.input_dropout_rate),
                attn_dropout_rate=float(cfg.attn_dropout_rate),
                linear_dropout_rate=float(cfg.linear_dropout_rate),
                num_time_buckets=int(cfg.num_time_buckets),
                temperature=float(cfg.temperature),
                normalize_embeddings=False,  # ReaRec handles scoring independently
                num_negatives=128,           # unused by ReaRec's CE loss
            )
            hstu_model = HSTUModel(hstu_cfg)
            hstu_model._ensure_initialized(self._num_items)
            backbone = HSTUBackbone(hstu_model)

        else:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. Choose 'sasrec' or 'hstu'."
            )

        self._ar_wrapper = ReaRecAutoRegressiveWrapper(
            backbone=backbone,
            hidden_size=D,
            reason_step=K,
        )

        self._loss_fct = nn.CrossEntropyLoss(ignore_index=self._num_items)
        self._init_weights()

    def _init_weights(self) -> None:
        std = float(self.config.initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _scoring_embs(self) -> torch.Tensor:
        """Return item scoring embeddings ``[num_items, D]``.

        For SASRec backbone: slices the model's own ``_item_emb`` table
        (``[num_items+1, D]``) to exclude the padding row.
        For HSTU backbone: delegates to ``HSTUBackbone.get_item_embs()`` which
        already returns ``[num_items, D]`` (ITEM_ID_OFFSET slot excluded).
        """
        from recbole3.model.rearec.layers import HSTUBackbone
        backbone = self._ar_wrapper_module().backbone
        if isinstance(backbone, HSTUBackbone):
            return backbone.get_item_embs()  # [num_items, D]
        return self._item_emb_module().weight[: self._num_items]  # [num_items, D]

    # ------------------------------------------------------------------
    # Loss-type helpers
    # ------------------------------------------------------------------

    def _effective_loss_type(self) -> str:
        """Resolve 'auto' to the concrete loss type based on backbone."""
        loss_type = str(self.config.loss_type).lower()
        if loss_type == "auto":
            return "sampled_softmax" if str(self.config.backbone).lower() == "hstu" else "ce"
        if loss_type not in ("ce", "sampled_softmax"):
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. Choose 'auto', 'ce', or 'sampled_softmax'."
            )
        return loss_type

    def _item_ce_loss(
        self,
        user_embs: torch.Tensor,     # [B, D]
        target_ids: torch.Tensor,    # [B]
        scoring_embs: torch.Tensor,  # [num_items, D]
        temperature: float,
    ) -> torch.Tensor:
        """Dispatch to full-vocab CE or sampled softmax depending on config.loss_type."""
        if self._effective_loss_type() == "sampled_softmax":
            return self._sampled_softmax_loss(user_embs, target_ids, scoring_embs, temperature)
        logits = torch.matmul(user_embs, scoring_embs.t()) / temperature  # [B, num_items]
        return self._loss_fct(logits, target_ids)

    def _sampled_softmax_loss(
        self,
        user_embs: torch.Tensor,     # [B, D]
        target_ids: torch.Tensor,    # [B]
        scoring_embs: torch.Tensor,  # [num_items, D]
        temperature: float,
    ) -> torch.Tensor:
        """InfoNCE-style sampled softmax: 1 positive + num_negatives random negatives."""
        B = user_embs.shape[0]
        num_neg = int(self.config.num_negatives)
        neg_ids = torch.randint(
            0, self._num_items, (B, num_neg),  # type: ignore[arg-type]
            device=user_embs.device, dtype=torch.long,
        )
        pos_embs = scoring_embs[target_ids]             # [B, D]
        neg_embs = scoring_embs[neg_ids]                # [B, num_neg, D]
        pos_logits = (user_embs * pos_embs).sum(-1, keepdim=True) / temperature  # [B, 1]
        neg_logits = torch.bmm(
            user_embs.unsqueeze(1), neg_embs.transpose(1, 2)
        ).squeeze(1) / temperature                       # [B, num_neg]
        neg_logits = neg_logits.masked_fill(
            neg_ids == target_ids.unsqueeze(1), -5e4
        )
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, num_neg+1]
        labels = torch.zeros(B, dtype=torch.long, device=user_embs.device)
        return F.cross_entropy(logits, labels)

    def _require_initialized(self) -> None:
        if self._num_items is None:
            raise RuntimeError(
                "ReaRecModel must be initialized via ensure_initialized() or "
                "build_train_collator() before use."
            )

    def _item_emb_module(self) -> nn.Embedding:
        if self._item_emb is None:
            raise RuntimeError("ReaRecModel not yet initialized.")
        return self._item_emb

    def _pos_emb_module(self) -> nn.Embedding:
        if self._pos_emb is None:
            raise RuntimeError("ReaRecModel not yet initialized.")
        return self._pos_emb

    def _ar_wrapper_module(self) -> ReaRecAutoRegressiveWrapper:
        if self._ar_wrapper is None:
            raise RuntimeError("ReaRecModel not yet initialized.")
        return self._ar_wrapper


__all__ = ["ReaRecModel"]
