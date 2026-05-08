from __future__ import annotations

import importlib
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.dataset import ITEM_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.hstu.config import HSTUConfig, HSTU_PADDING_ITEM_ID, ITEM_ID_OFFSET
from recbole3.model.hstu.data import (
    HISTORY_TIMESTAMPS,
    HSTUEvalCollator,
    HSTUTrainCollator,
)
from recbole3.model.sequential import HISTORY_ITEM_IDS


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
        self.embedding_dim = embedding_dim
        self.position_embeddings = nn.Embedding(self.max_sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        truncated_normal(self.position_embeddings.weight, mean=0.0, std=math.sqrt(1.0 / self.embedding_dim))

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
        position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)
        input_embeddings = past_embeddings * (self.embedding_dim ** 0.5) + position_embeddings
        input_embeddings = self.dropout(input_embeddings)

        valid_mask = position_ids.view(1, sequence_length) < past_lengths.view(batch_size, 1)
        input_embeddings = input_embeddings * valid_mask.unsqueeze(-1)

        return past_lengths, input_embeddings, past_payloads


class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    """Learnable relative bias from position deltas and bucketed time deltas."""

    def __init__(self, *, max_sequence_length: int, num_time_buckets: int) -> None:
        super().__init__()
        self.max_sequence_length = int(max_sequence_length)
        self.num_time_buckets = int(num_time_buckets)
        self.position_bias = nn.Parameter(
            torch.empty(2 * self.max_sequence_length - 1).normal_(mean=0, std=0.02),
        )
        self.time_bias = nn.Parameter(
            torch.empty(self.num_time_buckets + 1).normal_(mean=0, std=0.02),
        )

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        sequence_length = int(all_timestamps.shape[1])
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"HSTU received sequence length {sequence_length}, which exceeds max_sequence_length={self.max_sequence_length}."
            )

        B = all_timestamps.size(0)
        N = sequence_length
        t = F.pad(self.position_bias[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        rel_pos_bias = t[:, :, r:-r]

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketize_time_deltas(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self.num_time_buckets,
        ).detach()
        rel_ts_bias = torch.index_select(
            self.time_bias, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)

        return rel_pos_bias + rel_ts_bias

    def _bucketize_time_deltas(self, time_deltas: torch.Tensor) -> torch.Tensor:
        buckets = torch.log(time_deltas.abs().clamp(min=1.0)) / 0.301
        return buckets.to(dtype=torch.long)


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
        relative_attention_bias_module: RelativeBucketedTimeAndPositionBasedBias,
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
        nn.init.normal_(self.uvqk, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)

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

    def ensure_initialized(self, prepared_data) -> None:
        self._ensure_initialized(int(prepared_data.get_num_items()))

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return HSTUTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return HSTUEvalCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sequence_embeddings": self._encode_sequence_embeddings(batch)}

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._num_items is None:
            raise RuntimeError("HSTUModel must be initialized with prepared_data before it can compute loss.")
        num_negatives = int(self.config.num_negatives)
        if num_negatives <= 0:
            raise ValueError("HSTU requires config.num_negatives to be a positive integer.")

        sequence_embeddings = outputs["sequence_embeddings"]
        history_item_ids = batch[HISTORY_ITEM_IDS].to(
            dtype=torch.long,
            device=sequence_embeddings.device,
        )
        history_lengths = batch["history_lengths"].to(dtype=torch.long, device=sequence_embeddings.device)
        if sequence_embeddings.ndim != 3:
            raise ValueError(f"HSTU loss expects sequence_embeddings to be 3D, got shape {tuple(sequence_embeddings.shape)}.")
        if history_item_ids.shape[:2] != sequence_embeddings.shape[:2]:
            raise ValueError(
                "HSTU loss expects history_item_ids and sequence_embeddings to have matching batch and sequence dimensions."
            )
        if sequence_embeddings.shape[1] <= 1:
            return sequence_embeddings.sum() * 0.0

        prediction_embeddings = sequence_embeddings[:, :-1, :]
        next_item_ids = history_item_ids[:, 1:]
        positions = torch.arange(prediction_embeddings.shape[1], device=sequence_embeddings.device)
        supervision_mask = positions.view(1, -1) < history_lengths.view(-1, 1)
        if not torch.any(supervision_mask):
            return prediction_embeddings.sum() * 0.0

        flat_prediction_embeddings = prediction_embeddings[supervision_mask]
        positive_item_ids = next_item_ids[supervision_mask]
        negative_item_ids = torch.randint(
            0,
            self._num_items,
            (int(positive_item_ids.shape[0]), num_negatives),
            device=sequence_embeddings.device,
            dtype=torch.long,
        )

        item_embeddings = self._item_embedding_module()
        positive_item_embeddings = item_embeddings(self._to_model_item_ids(positive_item_ids))
        negative_item_embeddings = item_embeddings(self._to_model_item_ids(negative_item_ids))
        positive_logits = self._score_embeddings(flat_prediction_embeddings, positive_item_embeddings.unsqueeze(1)).squeeze(1)
        negative_logits = self._score_embeddings(flat_prediction_embeddings, negative_item_embeddings)
        negative_logits = negative_logits.masked_fill(negative_item_ids == positive_item_ids.unsqueeze(1), -5e4)
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

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
        if k > int(candidate_item_ids.shape[1]):
            raise ValueError(
                f"HSTU cannot return k={k} sampled predictions from {int(candidate_item_ids.shape[1])} candidates."
            )
        candidate_embeddings = self._item_embedding_module()(self._to_model_item_ids(candidate_item_ids))
        scores = self._score_embeddings(user_embeddings, candidate_embeddings)
        topk_indices = torch.topk(scores, k=k, dim=1).indices
        pred_item_ids = torch.gather(candidate_item_ids, 1, topk_indices)
        return pred_item_ids

    def _topk_item_ids(self, scores: torch.Tensor, *, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty((scores.shape[0], 0), dtype=torch.long, device=scores.device)
        available_items = int(scores.shape[1])
        if k > available_items:
            raise ValueError(f"HSTU cannot return k={k} full predictions from {available_items} real items.")
        return torch.topk(scores, k=k, dim=1).indices.to(dtype=torch.long)

    def _score_all_items(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        return self._score_embeddings(user_embeddings, self._item_embedding_module().weight[ITEM_ID_OFFSET:])

    def _score_embeddings(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_embeddings:
            user_embeddings = l2_normalize(user_embeddings, eps=self.config.l2_norm_eps)
            item_embeddings = l2_normalize(item_embeddings, eps=self.config.l2_norm_eps)
        if item_embeddings.ndim == 2:
            return torch.matmul(user_embeddings, item_embeddings.transpose(0, 1)) / self.config.temperature
        return torch.einsum("bd,bkd->bk", user_embeddings, item_embeddings) / self.config.temperature

    def _encode_user_embeddings(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        sequence_embeddings = self._encode_sequence_embeddings(batch)
        empty_history_embedding = self._empty_history_parameter()
        history_lengths = batch["history_lengths"].to(dtype=torch.long, device=sequence_embeddings.device)
        batch_size = int(history_lengths.shape[0])
        user_embeddings = empty_history_embedding.unsqueeze(0).expand(batch_size, -1).clone()
        if sequence_embeddings.shape[1] == 0:
            return user_embeddings
        non_empty = history_lengths > 0
        if torch.any(non_empty):
            non_empty_lengths = history_lengths[non_empty]
            gather_index = (non_empty_lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence_embeddings.shape[-1])
            user_embeddings[non_empty] = sequence_embeddings[non_empty].gather(1, gather_index).squeeze(1)
        return user_embeddings

    def _encode_sequence_embeddings(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        item_embeddings = self._item_embedding_module()
        input_preprocessor = self._input_preprocessor_module()
        encoder = self._encoder_module()
        history_item_ids = batch[HISTORY_ITEM_IDS].to(dtype=torch.long, device=item_embeddings.weight.device)
        history_timestamps = batch[HISTORY_TIMESTAMPS].to(dtype=torch.float32, device=item_embeddings.weight.device)
        history_lengths = batch["history_lengths"].to(dtype=torch.long, device=item_embeddings.weight.device)
        batch_size = int(history_lengths.shape[0])
        sequence_length = int(history_item_ids.shape[1]) if history_item_ids.ndim == 2 else 0
        if sequence_length == 0:
            return item_embeddings.weight.new_zeros((batch_size, 0, self.config.embedding_dim))
        max_encoder_length = int(self.config.history_max_length) + 1
        if sequence_length > max_encoder_length:
            raise ValueError(
                f"HSTU received batch sequence length {sequence_length}, which exceeds history_max_length + 1={max_encoder_length}."
            )
        positions = torch.arange(sequence_length, device=history_lengths.device, dtype=torch.long)
        max_lengths = torch.full_like(history_lengths, sequence_length)
        sequence_lengths = torch.minimum(history_lengths + 1, max_lengths)
        item_token_lengths = history_lengths + (1 if ITEM_ID in batch else 0)
        item_token_lengths = torch.minimum(item_token_lengths, sequence_lengths)
        item_token_mask = positions.view(1, sequence_length) < item_token_lengths.view(batch_size, 1)
        masked_history_item_ids = torch.full_like(history_item_ids, HSTU_PADDING_ITEM_ID)
        if torch.any(item_token_mask):
            masked_history_item_ids[item_token_mask] = self._to_model_item_ids(history_item_ids[item_token_mask])
        dense_embeddings = item_embeddings(masked_history_item_ids)
        _, processed_embeddings, _ = input_preprocessor(
            past_lengths=sequence_lengths,
            past_ids=masked_history_item_ids,
            past_embeddings=dense_embeddings,
            past_payloads={"timestamps": history_timestamps},
        )
        encoded_embeddings = encoder(
            x=processed_embeddings,
            x_offsets=self._complete_cumsum(sequence_lengths),
            all_timestamps=history_timestamps,
            invalid_attn_mask=self._build_attention_mask(sequence_lengths, sequence_length, dtype=processed_embeddings.dtype),
        )
        return encoded_embeddings

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
        self._item_embeddings = nn.Embedding(
            self._num_items + ITEM_ID_OFFSET,
            self.config.embedding_dim,
            padding_idx=HSTU_PADDING_ITEM_ID,
        )
        max_encoder_length = int(self.config.history_max_length) + 1
        self._input_preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_length=max_encoder_length,
            embedding_dim=self.config.embedding_dim,
            dropout_rate=self.config.input_dropout_rate,
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
                        max_sequence_length=max_encoder_length,
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
            item_embeddings.weight[HSTU_PADDING_ITEM_ID].zero_()

        truncated_normal(self._empty_history_parameter(), mean=0.0, std=0.02)

        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
            except:
                pass

    def _to_model_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self._num_items is None:
            raise RuntimeError("HSTUModel must be initialized with prepared_data before it can map item ids.")
        if item_ids.numel() > 0:
            invalid = (item_ids < 0) | (item_ids >= self._num_items)
            if torch.any(invalid):
                raise ValueError(
                    f"HSTU received item ids outside dataset range [0, {self._num_items - 1}]."
                )
        return item_ids + ITEM_ID_OFFSET

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
