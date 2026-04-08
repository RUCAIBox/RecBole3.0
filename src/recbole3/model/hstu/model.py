from __future__ import annotations

import importlib
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.hstu.config import HSTUConfig
from recbole3.model.hstu.data import HSTUEvalCollator, HSTUTrainCollator


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        indices = valid.max(-1, keepdim=True)[1]
        x.copy_(tmp.gather(-1, indices).squeeze(-1))
        x.mul_(std).add_(mean)
        return x


def l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(nn.Module):
    """Add learnable position embeddings to padded history embeddings."""

    def __init__(self, *, max_sequence_length: int, embedding_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.max_sequence_length = int(max_sequence_length)
        self.position_embeddings = nn.Embedding(self.max_sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        truncated_normal(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]]:
        del past_ids
        batch_size, sequence_length, embedding_dim = past_embeddings.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"HSTU received sequence length {sequence_length}, which exceeds max_sequence_length={self.max_sequence_length}."
            )
        position_ids = torch.arange(sequence_length, device=past_embeddings.device, dtype=torch.long)
        position_embeddings = self.position_embeddings(position_ids).view(1, sequence_length, embedding_dim)
        valid_mask = position_ids.view(1, sequence_length) < past_lengths.view(batch_size, 1)
        processed = torch.where(valid_mask.unsqueeze(-1), past_embeddings + position_embeddings, torch.zeros_like(past_embeddings))
        return past_lengths, self.dropout(processed), past_payloads


class RelativeAttentionBiasModule(nn.Module):
    """Base class for relative attention bias modules."""

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """Learnable relative bias from position deltas and bucketed time deltas."""

    def __init__(self, *, max_sequence_length: int, num_time_buckets: int) -> None:
        super().__init__()
        self.max_sequence_length = int(max_sequence_length)
        self.num_time_buckets = int(num_time_buckets)
        self.position_bias = nn.Embedding(self.max_sequence_length * 2 - 1, 1)
        self.time_bias = nn.Embedding(self.num_time_buckets, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.position_bias.weight)
        nn.init.zeros_(self.time_bias.weight)

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        sequence_length = int(all_timestamps.shape[1])
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"HSTU received sequence length {sequence_length}, which exceeds max_sequence_length={self.max_sequence_length}."
            )
        positions = torch.arange(sequence_length, device=all_timestamps.device, dtype=torch.long)
        relative_positions = positions.view(sequence_length, 1) - positions.view(1, sequence_length)
        position_indices = (relative_positions + self.max_sequence_length - 1).clamp(
            min=0,
            max=self.max_sequence_length * 2 - 2,
        )
        position_bias = self.position_bias(position_indices).squeeze(-1)
        time_deltas = torch.abs(all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1))
        time_bucket_ids = self._bucketize_time_deltas(time_deltas)
        time_bias = self.time_bias(time_bucket_ids).squeeze(-1)
        return position_bias.unsqueeze(0) + time_bias

    def _bucketize_time_deltas(self, time_deltas: torch.Tensor) -> torch.Tensor:
        buckets = torch.log(time_deltas.clamp(min=1.0)) / 0.301
        return buckets.to(dtype=torch.long).clamp(min=0, max=self.num_time_buckets - 1)


class SequentialTransductionUnitJagged(nn.Module):
    """One HSTU block operating on jagged history embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        relative_attention_bias_module: RelativeAttentionBiasModule,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.linear_hidden_dim = int(linear_hidden_dim)
        self.attention_dim = int(attention_dim)
        self.dropout_ratio = float(dropout_ratio)
        self.attn_dropout_ratio = float(attn_dropout_ratio)
        self.num_heads = int(num_heads)
        self.relative_attention_bias = relative_attention_bias_module
        self.epsilon = float(epsilon)
        output_dim = self.linear_hidden_dim * self.num_heads * 2 + self.attention_dim * self.num_heads * 2
        self.uvqk = nn.Parameter(torch.empty((self.embedding_dim, output_dim)))
        self.output = nn.Linear(self.linear_hidden_dim * self.num_heads, self.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        truncated_normal(self.uvqk, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        *,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(x_offsets.shape[0] - 1)
        sequence_length = int(invalid_attn_mask.shape[-1])
        normed_x = F.layer_norm(x, normalized_shape=[self.embedding_dim], eps=self.epsilon)
        uvqk = torch.mm(normed_x, self.uvqk)
        uvqk = F.silu(uvqk)
        u, v, q, k = torch.split(
            uvqk,
            [
                self.linear_hidden_dim * self.num_heads,
                self.linear_hidden_dim * self.num_heads,
                self.attention_dim * self.num_heads,
                self.attention_dim * self.num_heads,
            ],
            dim=1,
        )
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
            values=q,
            offsets=[x_offsets],
            max_lengths=[sequence_length],
            padding_value=0.0,
        ).view(batch_size, sequence_length, self.num_heads, self.attention_dim)
        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k,
            offsets=[x_offsets],
            max_lengths=[sequence_length],
            padding_value=0.0,
        ).view(batch_size, sequence_length, self.num_heads, self.attention_dim)
        padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
            values=v,
            offsets=[x_offsets],
            max_lengths=[sequence_length],
            padding_value=0.0,
        ).view(batch_size, sequence_length, self.num_heads, self.linear_hidden_dim)
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k)
        qk_attn = qk_attn + self.relative_attention_bias(all_timestamps).unsqueeze(1)
        qk_attn = F.silu(qk_attn) / max(sequence_length, 1)
        qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)
        qk_attn = F.dropout(qk_attn, p=self.attn_dropout_ratio, training=self.training)
        attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, padded_v).reshape(
            batch_size,
            sequence_length,
            self.num_heads * self.linear_hidden_dim,
        )
        jagged_attn_output = torch.ops.fbgemm.dense_to_jagged(attn_output, [x_offsets])[0]
        o_input = u * F.layer_norm(
            jagged_attn_output,
            normalized_shape=[self.linear_hidden_dim * self.num_heads],
            eps=self.epsilon,
        )
        return self.output(F.dropout(o_input, p=self.dropout_ratio, training=self.training)) + x


class HSTUJagged(nn.Module):
    """Stack of HSTU blocks executed on jagged sequences."""

    def __init__(self, *, layers: list[SequentialTransductionUnitJagged]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        *,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim == 3:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
        for layer in self.layers:
            x = layer(
                x=x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
            )
        return torch.ops.fbgemm.jagged_to_padded_dense(
            values=x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.shape[-1]],
            padding_value=0.0,
        )


class HSTUModel(BaseRetrievalModel):
    """HSTU retrieval model adapted to the RecBole3 model interface."""

    def __init__(self, config: HSTUConfig):
        super().__init__(config)
        self._require_runtime_support()
        self._num_items: int | None = None
        self._item_embeddings: nn.Embedding | None = None
        self._input_preprocessor: LearnablePositionalEmbeddingInputFeaturesPreprocessor | None = None
        self._encoder: HSTUJagged | None = None
        self._empty_history_embedding: nn.Parameter | None = None

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return HSTUTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return HSTUEvalCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"user_embeddings": self._encode_user_embeddings(batch)}

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self._score_all_items(outputs["user_embeddings"])
        return F.cross_entropy(logits, batch["item_id"].to(dtype=torch.long))

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_embeddings = self._encode_user_embeddings(model_inputs)
        if candidate_item_ids is not None:
            return self._predict_from_candidates(user_embeddings, candidate_item_ids, k=k)

        scores = self._score_all_items(user_embeddings)
        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
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
            return torch.empty((user_embeddings.shape[0], 0), dtype=torch.long, device=user_embeddings.device)
        candidate_item_ids = candidate_item_ids.to(device=user_embeddings.device, dtype=torch.long)
        candidate_embeddings = self._item_embedding_module()(candidate_item_ids + 1)
        scores = self._score_embeddings(user_embeddings, candidate_embeddings)
        topk_width = min(k, int(candidate_item_ids.shape[1]))
        topk_indices = torch.topk(scores, k=topk_width, dim=1).indices
        pred_item_ids = torch.gather(candidate_item_ids, 1, topk_indices)
        if topk_width == k:
            return pred_item_ids
        padded = torch.full((candidate_item_ids.shape[0], k), -1, dtype=torch.long, device=user_embeddings.device)
        padded[:, :topk_width] = pred_item_ids
        return padded

    def _topk_item_ids(self, scores: torch.Tensor, *, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty((scores.shape[0], 0), dtype=torch.long, device=scores.device)
        topk_width = min(k, int(scores.shape[1]))
        pred_item_ids = torch.topk(scores, k=topk_width, dim=1).indices.to(dtype=torch.long)
        if topk_width == k:
            return pred_item_ids
        padded = torch.full((scores.shape[0], k), -1, dtype=torch.long, device=scores.device)
        padded[:, :topk_width] = pred_item_ids
        return padded

    def _score_all_items(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        item_embeddings = self._item_embedding_module().weight[1:]
        return self._score_embeddings(user_embeddings, item_embeddings)

    def _score_embeddings(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_embeddings:
            user_embeddings = l2_normalize(user_embeddings, eps=self.config.l2_norm_eps)
            item_embeddings = l2_normalize(item_embeddings, eps=self.config.l2_norm_eps)
        if item_embeddings.ndim == 2:
            return torch.matmul(user_embeddings, item_embeddings.transpose(0, 1)) / self.config.temperature
        return torch.einsum("bd,bkd->bk", user_embeddings, item_embeddings) / self.config.temperature

    def _encode_user_embeddings(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        item_embeddings = self._item_embedding_module()
        input_preprocessor = self._input_preprocessor_module()
        encoder = self._encoder_module()
        empty_history_embedding = self._empty_history_parameter()
        history_item_ids = batch["history_item_ids"].to(dtype=torch.long, device=item_embeddings.weight.device)
        history_timestamps = batch["history_timestamps"].to(dtype=torch.float32, device=item_embeddings.weight.device)
        history_lengths = batch["history_lengths"].to(dtype=torch.long, device=item_embeddings.weight.device)
        batch_size = int(history_lengths.shape[0])
        sequence_length = int(history_item_ids.shape[1]) if history_item_ids.ndim == 2 else 0
        if sequence_length == 0:
            return empty_history_embedding.unsqueeze(0).expand(batch_size, -1)
        if sequence_length > int(self.config.history_max_length):
            raise ValueError(
                f"HSTU received batch sequence length {sequence_length}, which exceeds history_max_length={self.config.history_max_length}."
            )
        positions = torch.arange(sequence_length, device=history_lengths.device, dtype=torch.long)
        valid_mask = positions.view(1, sequence_length) < history_lengths.view(batch_size, 1)
        shifted_history_item_ids = torch.where(valid_mask, history_item_ids + 1, torch.zeros_like(history_item_ids))
        dense_embeddings = item_embeddings(shifted_history_item_ids)
        _, processed_embeddings, _ = input_preprocessor(
            past_lengths=history_lengths,
            past_ids=shifted_history_item_ids,
            past_embeddings=dense_embeddings,
            past_payloads={"timestamps": history_timestamps},
        )
        encoded_embeddings = encoder(
            x=processed_embeddings,
            x_offsets=self._complete_cumsum(history_lengths),
            all_timestamps=history_timestamps,
            invalid_attn_mask=self._build_attention_mask(history_lengths, sequence_length, dtype=processed_embeddings.dtype),
        )
        user_embeddings = empty_history_embedding.unsqueeze(0).expand(batch_size, -1).clone()
        non_empty = history_lengths > 0
        if torch.any(non_empty):
            non_empty_lengths = history_lengths[non_empty]
            gather_index = (non_empty_lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded_embeddings.shape[-1])
            user_embeddings[non_empty] = encoded_embeddings[non_empty].gather(1, gather_index).squeeze(1)
        return user_embeddings

    def _build_attention_mask(self, history_lengths: torch.Tensor, sequence_length: int, *, dtype: torch.dtype) -> torch.Tensor:
        valid_positions = torch.arange(sequence_length, device=history_lengths.device).view(1, sequence_length) < history_lengths.view(-1, 1)
        causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=history_lengths.device))
        return (valid_positions.unsqueeze(1) & valid_positions.unsqueeze(2) & causal_mask.unsqueeze(0)).to(dtype=dtype)

    def _complete_cumsum(self, history_lengths: torch.Tensor) -> torch.Tensor:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(history_lengths.to(dtype=torch.int32))

    def _ensure_initialized(self, num_items: int) -> None:
        if self._num_items is not None:
            if self._num_items != int(num_items):
                raise ValueError(f"HSTUModel was initialized for num_items={self._num_items}, got {num_items}.")
            return

        self._num_items = int(num_items)
        self._item_embeddings = nn.Embedding(self._num_items + 1, self.config.embedding_dim, padding_idx=0)
        self._input_preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_length=int(self.config.history_max_length),
            embedding_dim=self.config.embedding_dim,
            dropout_rate=self.config.linear_dropout_rate,
        )
        self._encoder = HSTUJagged(
            layers=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self.config.embedding_dim,
                    linear_hidden_dim=self.config.linear_hidden_dim,
                    attention_dim=self.config.attention_dim,
                    dropout_ratio=self.config.linear_dropout_rate,
                    attn_dropout_ratio=self.config.attn_dropout_rate,
                    num_heads=self.config.num_heads,
                    relative_attention_bias_module=RelativeBucketedTimeAndPositionBasedBias(
                        max_sequence_length=int(self.config.history_max_length),
                        num_time_buckets=self.config.num_time_buckets,
                    ),
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self._empty_history_embedding = nn.Parameter(torch.empty(self.config.embedding_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        item_embeddings = self._item_embedding_module()
        truncated_normal(item_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            item_embeddings.weight[0].zero_()
        truncated_normal(self._empty_history_parameter(), mean=0.0, std=0.02)

    def _item_embedding_module(self) -> nn.Embedding:
        if self._item_embeddings is None:
            raise RuntimeError("HSTUModel must be initialized with prepared_data before it can be used.")
        return self._item_embeddings

    def _input_preprocessor_module(self) -> LearnablePositionalEmbeddingInputFeaturesPreprocessor:
        if self._input_preprocessor is None:
            raise RuntimeError("HSTUModel must be initialized with prepared_data before it can be used.")
        return self._input_preprocessor

    def _encoder_module(self) -> HSTUJagged:
        if self._encoder is None:
            raise RuntimeError("HSTUModel must be initialized with prepared_data before it can be used.")
        return self._encoder

    def _empty_history_parameter(self) -> nn.Parameter:
        if self._empty_history_embedding is None:
            raise RuntimeError("HSTUModel must be initialized with prepared_data before it can be used.")
        return self._empty_history_embedding

    def _require_runtime_support(self) -> None:
        try:
            importlib.import_module("fbgemm_gpu")
        except ImportError as exc:
            raise RuntimeError(
                "HSTU requires optional dependency `fbgemm-gpu`. Install it with `pip install .[hstu]` "
                "or `pip install fbgemm-gpu` before using `model.name=hstu`."
            ) from exc
        required_ops = (
            "asynchronous_complete_cumsum",
            "dense_to_jagged",
            "jagged_to_padded_dense",
        )
        missing_ops = [name for name in required_ops if not hasattr(torch.ops.fbgemm, name)]
        if missing_ops:
            raise RuntimeError(
                "HSTU requires jagged FBGEMM operators provided by `fbgemm-gpu`. "
                f"Missing torch.ops.fbgemm ops: {', '.join(missing_ops)}."
            )


__all__ = [
    "HSTUModel",
]
