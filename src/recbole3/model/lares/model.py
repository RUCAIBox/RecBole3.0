from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.dataset import ITEM_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.lares.config import LARESConfig
from recbole3.model.lares.data import LARESEvalCollator, LARESTrainCollator
from recbole3.model.sequential import HISTORY_ITEM_IDS


# ---------------------------------------------------------------------------
# SASRec Transformer Encoder
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        attn_dropout_prob: float,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, H = query.shape
        Q = self.query(query).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_scores = attn_scores + attn_mask.unsqueeze(1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        return self.dense(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        if hidden_act == "gelu":
            self.act = F.gelu
        elif hidden_act == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense_2(self.dropout(self.act(self.dense_1(x)))))


class TransformerLayer(nn.Module):
    """One SASRec transformer block: attention + FFN, post-norm style."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, hidden_size, attn_dropout_prob)
        self.feed_forward = FeedForward(hidden_size, inner_size, hidden_dropout_prob, hidden_act)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.attn_layer_norm(x)
        h = self.attention(h, h, h, attn_mask)
        h = self.dropout(h)
        x = x + h

        h = self.ffn_layer_norm(x)
        h = self.feed_forward(h)
        h = self.dropout(h)
        x = x + h
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerLayer modules."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                n_heads=n_heads,
                hidden_size=hidden_size,
                inner_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


# ---------------------------------------------------------------------------
# Contrastive Loss
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss."""

    def __init__(self, tau: float, sem_func: str) -> None:
        super().__init__()
        self.tau = tau
        self.sem_func = sem_func

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.sem_func == "cos":
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        B = x.shape[0]
        logits = torch.matmul(x, y.transpose(0, 1)) / self.tau
        labels = torch.arange(B, device=x.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# LARES Model
# ---------------------------------------------------------------------------

class LARESModel(BaseRetrievalModel):
    """LARES (Learnable Recurrent State) retrieval model with SASRec backbone."""

    def __init__(self, config: LARESConfig) -> None:
        super().__init__(config)
        self._eval_recurrence_override: int | None = None
        self._num_items: int | None = None
        self._item_embeddings: nn.Embedding | None = None
        self._position_embeddings: nn.Embedding | None = None
        self._pre_encoder: TransformerEncoder | None = None
        self._core_encoder: TransformerEncoder | None = None
        self._layernorm_1: nn.LayerNorm | None = None
        self._dropout: nn.Dropout | None = None
        self._layernorm_2: nn.LayerNorm | None = None
        self._adapter: nn.Module | None = None
        self._empty_history_embedding: nn.Parameter | None = None
        self._loss_fct: nn.CrossEntropyLoss | None = None
        self._contrastive_loss: ContrastiveLoss | None = None

    # ---- Lazy initialization ----

    def ensure_initialized(self, prepared_data) -> None:
        self._ensure_initialized(int(prepared_data.get_num_items()))

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return LARESTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return LARESEvalCollator(self.config, prepared_data=prepared_data)

    def _ensure_initialized(self, num_items: int) -> None:
        if self._num_items is not None:
            if self._num_items != int(num_items):
                raise ValueError(
                    f"LARESModel initialized for num_items={self._num_items}, got {num_items}."
                )
            return

        self._num_items = int(num_items)
        hidden_size = self.config.hidden_size
        max_seq_length = int(self.config.history_max_length)

        self._item_embeddings = nn.Embedding(
            self._num_items + 1, hidden_size, padding_idx=0
        )
        self._position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        self._layernorm_1 = nn.LayerNorm(hidden_size, eps=self.config.layer_norm_eps)
        self._dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self._layernorm_2 = nn.LayerNorm(hidden_size, eps=self.config.layer_norm_eps)

        self._pre_encoder = TransformerEncoder(
            n_layers=self.config.n_pre_layers,
            n_heads=self.config.n_heads,
            hidden_size=hidden_size,
            inner_size=self.config.inner_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attn_dropout_prob=self.config.attn_dropout_prob,
            hidden_act=self.config.hidden_act,
            layer_norm_eps=self.config.layer_norm_eps,
        )
        self._core_encoder = TransformerEncoder(
            n_layers=self.config.n_core_layers,
            n_heads=self.config.n_heads,
            hidden_size=hidden_size,
            inner_size=self.config.inner_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attn_dropout_prob=self.config.attn_dropout_prob,
            hidden_act=self.config.hidden_act,
            layer_norm_eps=self.config.layer_norm_eps,
        )

        # Adapter
        adapter_type = self.config.adapter_type
        if adapter_type == "concat":
            self._adapter = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        elif adapter_type == "linear":
            self._adapter = nn.Parameter(torch.tensor(0.0))
        else:
            self._adapter = nn.Identity()

        self._empty_history_embedding = nn.Parameter(torch.empty(hidden_size))
        self._loss_fct = nn.CrossEntropyLoss()
        self._contrastive_loss = ContrastiveLoss(
            tau=self.config.tau, sem_func=self.config.sem_func
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init_range = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=init_range)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        item_embeddings = self._item_embedding_module()
        with torch.no_grad():
            item_embeddings.weight[0].zero_()

        empty_param = self._empty_history_parameter()
        empty_param.data.normal_(mean=0.0, std=init_range)

    # ---- Module accessors ----

    def _item_embedding_module(self) -> nn.Embedding:
        if self._item_embeddings is None:
            raise RuntimeError("LARESModel must be initialized before use.")
        return self._item_embeddings

    def _position_embedding_module(self) -> nn.Embedding:
        if self._position_embeddings is None:
            raise RuntimeError("LARESModel must be initialized before use.")
        return self._position_embeddings

    def _pre_encoder_module(self) -> TransformerEncoder:
        if self._pre_encoder is None:
            raise RuntimeError("LARESModel must be initialized before use.")
        return self._pre_encoder

    def _core_encoder_module(self) -> TransformerEncoder:
        if self._core_encoder is None:
            raise RuntimeError("LARESModel must be initialized before use.")
        return self._core_encoder

    def _empty_history_parameter(self) -> nn.Parameter:
        if self._empty_history_embedding is None:
            raise RuntimeError("LARESModel must be initialized before use.")
        return self._empty_history_embedding

    # ---- Recurrence step sampling ----

    @torch.no_grad()
    def _sample_recurrence_steps(self) -> int:
        if not self.training:
            override = self._eval_recurrence_override
            if override is not None:
                return override
            return int(self.config.mean_recurrence)

        scheme = self.config.sampling_scheme
        mean_T = self.config.mean_recurrence

        if "uniform" in scheme:
            t = torch.randint(low=1, high=int(mean_T * 2) + 1, size=(1,))
        elif "poisson-lognormal" in scheme:
            sigma = 0.5
            mu = math.log(mean_T) - (sigma ** 2 / 2)
            rate = torch.zeros(1).log_normal_(mean=mu, std=sigma)
            t = torch.poisson(rate) + 1
            t = torch.clamp(t, max=int(3 * mean_T))
        elif "poisson-unbounded" in scheme:
            t = torch.poisson(torch.full((1,), mean_T))
        elif "poisson-bounded" in scheme:
            t = torch.poisson(torch.full((1,), mean_T))
            t = torch.clamp(t, max=int(2 * mean_T))
        elif "non-recurrent" in scheme:
            t = torch.tensor([1])
        elif "constant" in scheme:
            t = torch.full((1,), int(mean_T), dtype=torch.long)
        else:
            t = torch.full((1,), int(mean_T), dtype=torch.long)

        return max(1, int(t.item()))

    # ---- State initialization ----

    def _initialize_state(self, pre_output: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(pre_output)
        method = self.config.state_init_method
        if method == "normal":
            nn.init.trunc_normal_(
                x, mean=0.0, std=self.config.state_std,
                a=-3 * self.config.state_std, b=3 * self.config.state_std,
            )
        elif method == "normal_zero":
            if self.training:
                nn.init.trunc_normal_(
                    x, mean=0.0, std=self.config.state_std,
                    a=-3 * self.config.state_std, b=3 * self.config.state_std,
                )
        return self.config.state_scale * x

    # ---- Attention mask ----

    def _build_causal_attention_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        device = lengths.device
        batch_size = lengths.shape[0]
        positions = torch.arange(max_len, device=device)
        # Key-side validity: can only attend to positions with real items
        valid_key_mask = positions.view(1, 1, max_len) < lengths.view(batch_size, 1, 1)
        # Causal: position i can attend to j <= i
        causal_mask = torch.tril(
            torch.ones((max_len, max_len), dtype=torch.bool, device=device)
        ).unsqueeze(0)
        mask = valid_key_mask & causal_mask
        return mask.to(dtype=torch.float32).masked_fill(~mask, float("-inf"))

    # ---- Encoding ----

    def _encode(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_all_states: bool = False,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode item sequences through pre_encoder + recurrence loop.

        Returns:
            user_embeddings: (B, hidden_size)
            all_step_outputs: (B, T, hidden_size) if return_all_states else None
        """
        item_embeddings = self._item_embedding_module()
        layernorm_1 = self._layernorm_1
        dropout = self._dropout
        layernorm_2 = self._layernorm_2
        pre_encoder = self._pre_encoder_module()
        core_encoder = self._core_encoder_module()

        B, L = item_ids.shape
        device = item_embeddings.weight.device

        # Item + position embeddings
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        item_emb = item_embeddings(item_ids.to(device=device))
        pos_emb = self._position_embedding_module()(position_ids)
        input_emb = item_emb + pos_emb
        input_emb = layernorm_1(input_emb)
        input_emb = dropout(input_emb)

        attn_mask = self._build_causal_attention_mask(lengths.to(device=device), L)

        # Pre-encoder
        pre_output = pre_encoder(input_emb, attn_mask)  # [B, L, H]

        # Initialize state
        states = self._initialize_state(pre_output)  # [B, L, H]

        # Recurrence loop
        if num_steps is None:
            num_steps = self._sample_recurrence_steps()

        all_states: list[torch.Tensor] = []
        for step in range(num_steps):
            # Adapter: fuse states with pre_output
            adapter_type = self.config.adapter_type
            if adapter_type == "concat":
                adapter = self._adapter
                if isinstance(adapter, nn.Linear):
                    states = adapter(torch.cat([states, pre_output], dim=-1))
                else:
                    states = states
            elif adapter_type == "linear":
                param = self._adapter
                if isinstance(param, nn.Parameter):
                    gate = torch.sigmoid(param)
                    states = gate * states + (1 - gate) * pre_output
                else:
                    states = (states + pre_output) / 2
            else:
                # add (default)
                states = (states + pre_output) / 2

            states = layernorm_2(states)
            states = core_encoder(states, attn_mask)

            all_states.append(states)

        # Gather last position from final state
        empty_history_embedding = self._empty_history_parameter()
        user_emb = empty_history_embedding.unsqueeze(0).expand(B, -1).clone()
        non_empty = lengths > 0
        if torch.any(non_empty):
            gather_idx = (
                (lengths[non_empty].to(device=device) - 1)
                .view(-1, 1, 1)
                .expand(-1, 1, pre_output.shape[-1])
            )
            user_emb[non_empty] = states[non_empty].gather(1, gather_idx).squeeze(1)

        all_step_outputs = None
        if return_all_states:
            step_outputs = []
            for s in all_states:
                out = empty_history_embedding.unsqueeze(0).expand(B, -1).clone()
                if torch.any(non_empty):
                    out[non_empty] = s[non_empty].gather(1, gather_idx).squeeze(1)
                step_outputs.append(out)
            all_step_outputs = torch.stack(step_outputs, dim=1)  # [B, T, H]

        return user_emb, all_step_outputs

    # ---- Public API ----

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        history_ids = batch[HISTORY_ITEM_IDS]
        history_lengths = batch["history_lengths"]

        user_emb, _ = self._encode(history_ids, history_lengths)

        result: dict[str, torch.Tensor] = {"user_embeddings": user_emb}

        if "aug_history_item_ids" in batch:
            aug_ids = batch["aug_history_item_ids"]
            aug_lengths = batch["aug_history_lengths"]
            aug_emb, _ = self._encode(aug_ids, aug_lengths)
            result["aug_user_embeddings"] = aug_emb

        return result

    def compute_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._num_items is None:
            raise RuntimeError("LARESModel must be initialized before computing loss.")

        history_ids = batch[HISTORY_ITEM_IDS]
        history_lengths = batch["history_lengths"]
        pos_items = batch[ITEM_ID]
        item_embeddings = self._item_embedding_module()

        # 1. CE loss from forward's first encode
        user_emb = outputs["user_embeddings"]
        if self.config.sem_func == "cos":
            user_emb_norm = F.normalize(user_emb, dim=-1)
            item_weights_norm = F.normalize(item_embeddings.weight[1:], dim=-1)
            logits = torch.matmul(user_emb_norm, item_weights_norm.transpose(0, 1)) / self.config.tau
        else:
            logits = torch.matmul(user_emb, item_embeddings.weight[1:].transpose(0, 1))
        ce_loss = self._loss_fct(logits, pos_items.to(device=logits.device))

        # 2. Re-encode original sequence with return_all_states for contrastive
        n_step = self._sample_recurrence_steps() if self.config.same_step else None
        aug_seq_output1, all_step_outputs = self._encode(
            history_ids, history_lengths,
            return_all_states=True,
            num_steps=n_step,
        )

        # 3. Encode augmented sequence (from forward output)
        aug_emb = outputs.get("aug_user_embeddings")
        if aug_emb is None:
            aug_emb, _ = self._encode(history_ids, history_lengths)

        # 4. Contrastive losses
        cl_func = self._contrastive_loss

        # Inter-sequence: original vs augmented
        inter_loss = (
            cl_func(aug_seq_output1, aug_emb) + cl_func(aug_emb, aug_seq_output1)
        ) / 2

        loss = ce_loss + self.config.alpha * inter_loss

        # Intra-sequence: original vs random intermediate step
        if all_step_outputs is not None:
            B_val, T = all_step_outputs.shape[0], all_step_outputs.shape[1]
            if T > 1:
                idx = torch.randint(0, T, (1,), device=all_step_outputs.device)[0]
                selected_output = all_step_outputs[:, idx, :]
                intra_loss = (
                    cl_func(aug_seq_output1, selected_output)
                    + cl_func(selected_output, aug_seq_output1)
                ) / 2
                loss = loss + self.config.gamma * intra_loss

        return loss

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.forward(model_inputs)
        user_embeddings = outputs["user_embeddings"]

        if candidate_item_ids is not None:
            return self._predict_from_candidates(user_embeddings, candidate_item_ids, k=k)

        scores = self._score_all_items(user_embeddings)
        if (
            exclude_item_ids is not None
            and exclude_mask is not None
            and exclude_item_ids.numel() > 0
        ):
            history_mask = torch.zeros_like(scores, dtype=torch.bool)
            history_mask.scatter_(
                1,
                exclude_item_ids.to(device=scores.device, dtype=torch.long),
                exclude_mask.to(device=scores.device, dtype=torch.bool),
            )
            scores = scores.masked_fill(history_mask, float("-inf"))
        return self._topk_item_ids(scores, k=k)

    def _predict_from_candidates(
        self,
        user_embeddings: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        *,
        k: int,
    ) -> torch.Tensor:
        if k <= 0:
            return torch.empty(
                (user_embeddings.shape[0], 0),
                dtype=torch.long,
                device=user_embeddings.device,
            )
        candidate_item_ids = candidate_item_ids.to(
            device=user_embeddings.device, dtype=torch.long
        )
        candidate_embeddings = self._item_embedding_module()(candidate_item_ids + 1)
        scores = self._score_embeddings(user_embeddings, candidate_embeddings)
        topk_indices = torch.topk(scores, k=k, dim=1).indices
        return torch.gather(candidate_item_ids, 1, topk_indices)

    def _topk_item_ids(self, scores: torch.Tensor, *, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty((scores.shape[0], 0), dtype=torch.long, device=scores.device)
        return torch.topk(scores, k=k, dim=1).indices.to(dtype=torch.long)

    def _score_all_items(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        item_embeddings = self._item_embedding_module()
        return self._score_embeddings(user_embeddings, item_embeddings.weight[1:])

    def _score_embeddings(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.config.sem_func == "cos":
            user_embeddings = F.normalize(user_embeddings, dim=-1)
            item_embeddings = F.normalize(item_embeddings, dim=-1)
            if item_embeddings.ndim == 3:
                return torch.einsum("bd,bkd->bk", user_embeddings, item_embeddings) / self.config.tau
            return torch.matmul(user_embeddings, item_embeddings.transpose(0, 1)) / self.config.tau
        if item_embeddings.ndim == 3:
            return torch.einsum("bd,bkd->bk", user_embeddings, item_embeddings)
        return torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))


__all__ = [
    "LARESModel",
    "TransformerEncoder",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerLayer",
    "ContrastiveLoss",
]
