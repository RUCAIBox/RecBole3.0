# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
import math
import torch
from typing import Dict, Tuple, Callable
import torch.nn.functional as F
from recbole.models.utils import truncated_normal


class InputFeaturesPreprocessorModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            item_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask


class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]
    

class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # # [B, N + 1] to simplify tensor manipulations.
        # ext_timestamps = torch.cat(
        #     [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        # )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                all_timestamps[:, 1:].unsqueeze(2) - all_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias

