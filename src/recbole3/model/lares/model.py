from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.dataset import ITEM_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.lares.config import LARESConfig, ITEM_ID_OFFSET
from recbole3.model.lares.data import LARESEvalCollator, LARESTrainCollator
from recbole3.model.sequential import HISTORY_ITEM_IDS


# ---------------------------------------------------------------------------
# SASRec Transformer
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, attn_dropout_prob: float):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, H = query.shape
        Q = self.query(query).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_mask.unsqueeze(1)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, L, H)
        return self.dense(context)


class TransformerLayer(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, inner_size: int, hidden_dropout_prob: float, attn_dropout_prob: float, hidden_act: str, layer_norm_eps: float):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, hidden_size, attn_dropout_prob)
        self.attn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU() if hidden_act == "gelu" else nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attn_ln(x), self.attn_ln(x), self.attn_ln(x), attn_mask))
        x = x + self.ffn(self.ffn_ln(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, hidden_size: int, inner_size: int, hidden_dropout_prob: float, attn_dropout_prob: float, hidden_act: str, layer_norm_eps: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, tau: float, sem_func: str):
        super().__init__()
        self.tau = tau
        self.sem_func = sem_func

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.sem_func == "cos":
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        B = x.shape[0]
        logits = torch.matmul(x, y.transpose(0, 1)) / self.tau
        return F.cross_entropy(logits, torch.arange(B, device=x.device, dtype=torch.long))


class LARESModel(BaseRetrievalModel):
    def __init__(self, config: LARESConfig) -> None:
        super().__init__(config)
        self._eval_recurrence_override: int | None = None
        self._num_items: int | None = None
        self._item_emb: nn.Embedding | None = None
        self._pos_emb: nn.Embedding | None = None
        self._pre_encoder: TransformerEncoder | None = None
        self._core_encoder: TransformerEncoder | None = None

    def ensure_initialized(self, prepared_data) -> None:
        self._init_modules(int(prepared_data.get_num_items()))

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._init_modules(int(prepared_data.get_num_items()))
        return LARESTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._init_modules(int(prepared_data.get_num_items()))
        return LARESEvalCollator(self.config, prepared_data=prepared_data)

    def _init_modules(self, num_items: int) -> None:
        if self._num_items is not None:
            if self._num_items != num_items:
                raise ValueError(f"LARESModel initialized for num_items={self._num_items}, got {num_items}.")
            return

        self._num_items = num_items
        cfg = self.config
        H = cfg.hidden_size
        max_L = int(cfg.history_max_length)

        self._item_emb = nn.Embedding(num_items + ITEM_ID_OFFSET, H, padding_idx=0)
        self._pos_emb = nn.Embedding(max_L, H)
        self._ln1 = nn.LayerNorm(H, eps=cfg.layer_norm_eps)
        self._dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self._ln2 = nn.LayerNorm(H, eps=cfg.layer_norm_eps)

        enc_kwargs = dict(n_heads=cfg.n_heads, hidden_size=H, inner_size=cfg.inner_size,
                          hidden_dropout_prob=cfg.hidden_dropout_prob, attn_dropout_prob=cfg.attn_dropout_prob,
                          hidden_act=cfg.hidden_act, layer_norm_eps=cfg.layer_norm_eps)
        self._pre_encoder = TransformerEncoder(n_layers=cfg.n_pre_layers, **enc_kwargs)
        self._core_encoder = TransformerEncoder(n_layers=cfg.n_core_layers, **enc_kwargs)

        at = cfg.adapter_type
        self._adapter: nn.Linear | nn.Parameter | None
        if at == "concat":
            self._adapter = nn.Linear(2 * H, H, bias=True)
        elif at == "linear":
            self._adapter = nn.Parameter(torch.tensor(0.0))
        else:
            self._adapter = None

        # weight init
        rng = cfg.initializer_range
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=rng)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        with torch.no_grad():
            self._item_emb.weight[0].zero_()

    @torch.no_grad()
    def _sample_T(self) -> int:
        if not self.training:
            return self._eval_recurrence_override or int(self.config.mean_recurrence)

        scheme = self.config.sampling_scheme
        mean_T = self.config.mean_recurrence
        if "uniform" in scheme:
            t = torch.randint(low=1, high=int(mean_T * 2) + 1, size=(1,))
        elif "poisson-lognormal" in scheme:
            mu = math.log(mean_T) - 0.125  # sigma=0.5: sigma^2/2 = 0.125
            rate = torch.zeros(1).log_normal_(mean=mu, std=0.5)
            t = torch.clamp(torch.poisson(rate) + 1, max=int(3 * mean_T))
        elif "poisson-unbounded" in scheme:
            t = torch.poisson(torch.full((1,), mean_T))
        elif "poisson-bounded" in scheme:
            t = torch.clamp(torch.poisson(torch.full((1,), mean_T)), max=int(2 * mean_T))
        elif "non-recurrent" in scheme:
            t = torch.tensor([1])
        else:  # constant / fallback
            t = torch.full((1,), int(mean_T), dtype=torch.long)
        return max(1, int(t.item()))

    def _init_state(self, pre_output: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(pre_output)
        method = self.config.state_init_method
        std = self.config.state_std
        if method == "normal":
            nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif method == "normal_zero" and self.training:
            nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return self.config.state_scale * x

    def _to_model_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        return item_ids + ITEM_ID_OFFSET

    def _encode(self, item_ids: torch.Tensor, lengths: torch.Tensor, *, return_all_states: bool = False, num_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, L = item_ids.shape
        dev = self._item_emb.weight.device

        # item + position embeddings
        pos_ids = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        x = self._dropout(self._ln1(self._item_emb(item_ids.to(dev)) + self._pos_emb(pos_ids)))

        # causal attention mask (key-side validity only to avoid all-masked softmax rows)
        pos = torch.arange(L, device=dev)
        valid_key = (pos.view(1, 1, L) < lengths.view(B, 1, 1).to(dev))
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=dev)).unsqueeze(0)
        attn_mask = torch.zeros(B, L, L, device=dev).masked_fill(~(valid_key & causal), float("-inf"))

        # pre-encoder (once)
        pre_output = self._pre_encoder(x, attn_mask)

        # recurrence loop
        states = self._init_state(pre_output)
        T = num_steps if num_steps is not None else self._sample_T()

        all_states: list[torch.Tensor] = []
        for _ in range(T):
            at = self.config.adapter_type
            if at == "concat" and isinstance(self._adapter, nn.Linear):
                states = self._adapter(torch.cat([states, pre_output], dim=-1))
            elif at == "linear" and isinstance(self._adapter, nn.Parameter):
                g = torch.sigmoid(self._adapter)
                states = g * states + (1 - g) * pre_output
            else:
                states = (states + pre_output) / 2
            states = self._core_encoder(self._ln2(states), attn_mask)
            all_states.append(states)

        # gather last valid position
        idx = (lengths.to(dev) - 1).view(-1, 1, 1).expand(-1, 1, x.shape[-1])
        user_emb = states.gather(1, idx).squeeze(1)

        if not return_all_states:
            return user_emb, None, None

        step_outputs = torch.stack([s.gather(1, idx).squeeze(1) for s in all_states], dim=1)
        per_step_logits = torch.matmul(step_outputs, self._item_emb.weight.T)
        per_step_logps = torch.log_softmax(per_step_logits, dim=-1)
        return user_emb, step_outputs, per_step_logps

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user_emb, _, _ = self._encode(batch[HISTORY_ITEM_IDS], batch["history_lengths"])
        result = {"user_embeddings": user_emb}
        if "aug_history_item_ids" in batch:
            aug_emb, _, _ = self._encode(batch["aug_history_item_ids"], batch["aug_history_lengths"])
            result["aug_user_embeddings"] = aug_emb
        return result

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        history_ids = batch[HISTORY_ITEM_IDS]
        history_lengths = batch["history_lengths"]
        pos_items = batch[ITEM_ID]

        # CE loss (next-item prediction)
        user_emb = outputs["user_embeddings"]
        logits = self._score(user_emb, self._item_emb.weight)
        ce_loss = F.cross_entropy(logits, pos_items.to(device=logits.device))

        # re-encode original for contrastive
        n_step = self._sample_T() if self.config.same_step else None
        aug1, all_steps, _ = self._encode(history_ids, history_lengths, return_all_states=True, num_steps=n_step)

        aug2 = outputs.get("aug_user_embeddings")
        if aug2 is None:
            aug2, _, _ = self._encode(history_ids, history_lengths)

        cl = ContrastiveLoss(self.config.tau, self.config.sem_func)
        loss = ce_loss + self.config.alpha * (cl(aug1, aug2) + cl(aug2, aug1)) / 2

        if all_steps is not None and all_steps.shape[1] > 1:
            pick = all_steps[:, torch.randint(0, all_steps.shape[1], (1,), device=all_steps.device)[0], :]
            loss = loss + self.config.gamma * (cl(aug1, pick) + cl(pick, aug1)) / 2

        return loss

    def predict(self, model_inputs: Mapping[str, torch.Tensor], *, k: int, candidate_item_ids: torch.Tensor | None = None, exclude_item_ids: torch.Tensor | None = None, exclude_mask: torch.Tensor | None = None) -> torch.Tensor:
        user_emb = self.forward(model_inputs)["user_embeddings"]

        if candidate_item_ids is not None:
            cand = self._to_model_item_ids(candidate_item_ids).to(device=user_emb.device, dtype=torch.long)
            scores = self._score(user_emb, self._item_emb(cand))
            return torch.gather(cand, 1, torch.topk(scores, k=k, dim=1).indices) - ITEM_ID_OFFSET

        scores = self._score(user_emb, self._item_emb.weight[ITEM_ID_OFFSET:])
        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, exclude_item_ids.to(device=scores.device, dtype=torch.long),
                          exclude_mask.to(device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(mask, float("-inf"))
        return torch.topk(scores, k=k, dim=1).indices.to(dtype=torch.long)

    def _score(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.config.sem_func == "cos":
            user_emb = F.normalize(user_emb, dim=-1)
            item_emb = F.normalize(item_emb, dim=-1)
            if item_emb.ndim == 3:
                return torch.einsum("bd,bkd->bk", user_emb, item_emb) / self.config.tau
            return user_emb @ item_emb.T / self.config.tau
        if item_emb.ndim == 3:
            return torch.einsum("bd,bkd->bk", user_emb, item_emb)
        return user_emb @ item_emb.T


__all__ = [
    "ContrastiveLoss",
    "LARESModel",
    "MultiHeadAttention",
    "TransformerEncoder",
    "TransformerLayer",
]
