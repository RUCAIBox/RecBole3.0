from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.dataset import LABEL
from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.base import BaseCollator, BaseRankingModel
from recbole3.model.rankmixer.config import RankMixerConfig
from recbole3.model.rankmixer.data import (
    RANKMIXER_FEATURES,
    RankMixerEvalCollator,
    RankMixerTrainCollator,
)


class TokenMix(nn.Module):
    """Rearrange token channels across heads, optionally followed by residual layernorm."""

    def __init__(
        self,
        *,
        num_features: int,
        dim_multiplier: int,
        num_heads: int | None,
        use_add_norm: bool,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.output_dim = int(num_features) * int(dim_multiplier)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_features)
        self.use_add_norm = bool(use_add_norm)
        if self.num_features <= 0:
            raise ValueError("TokenMix requires num_features to be positive.")
        if self.num_heads <= 0:
            raise ValueError("TokenMix requires num_heads to be positive.")
        if self.output_dim % self.num_heads != 0:
            raise ValueError(
                f"TokenMix requires num_features * dim_multiplier to be divisible by num_heads, "
                f"got output_dim={self.output_dim} and num_heads={self.num_heads}."
            )
        self.head_dim = self.output_dim // self.num_heads
        if self.use_add_norm:
            self.layer_norm = nn.LayerNorm(self.output_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.shape[0])
        reshaped = x.view(batch_size, self.num_features, self.output_dim)
        mixed = (
            reshaped.view(batch_size, self.num_features, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size, self.num_features * self.output_dim)
        )
        if not self.use_add_norm:
            return mixed

        residual = mixed + x
        normalized = self.layer_norm(residual.view(batch_size, self.num_features, self.output_dim))
        return normalized.reshape(batch_size, self.num_features * self.output_dim)


class PerTokenFFN(nn.Module):
    """Apply an independent FFN to each token after token mixing."""

    def __init__(
        self,
        *,
        num_features: int,
        dim_multiplier: int,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.output_dim = int(num_features) * int(dim_multiplier)
        hidden_dim = int(ffn_multiplier) * self.output_dim
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.output_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.output_dim),
                )
                for _ in range(self.num_features)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.shape[0])
        tokens = x.view(batch_size, self.num_features, self.output_dim)
        outputs = [layer(tokens[:, index, :]) for index, layer in enumerate(self.ffn_layers)]
        return torch.stack(outputs, dim=1).reshape(batch_size, self.num_features * self.output_dim)


class RankMixerLayer(nn.Module):
    """One RankMixer block composed of token mixing and per-token FFN."""

    def __init__(
        self,
        *,
        num_features: int,
        dim_multiplier: int,
        num_heads: int | None,
        use_add_norm: bool,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        self.token_mix = TokenMix(
            num_features=num_features,
            dim_multiplier=dim_multiplier,
            num_heads=num_heads,
            use_add_norm=use_add_norm,
        )
        self.per_token_ffn = PerTokenFFN(
            num_features=num_features,
            dim_multiplier=dim_multiplier,
            ffn_multiplier=ffn_multiplier,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.per_token_ffn(self.token_mix(x))


class RankMixerModel(BaseRankingModel):
    """RankMixer CTR model adapted from MLCC into the RecBole3 ranking API."""

    def __init__(self, config: RankMixerConfig):
        super().__init__(config)
        self.num_features = int(config.num_features)
        self.embedding_dim = int(config.embedding_dim)
        self.output_dim = self.num_features * int(config.dim_multiplier)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(config.hash_size), self.embedding_dim) for _ in range(self.num_features)]
        )
        self.tokenization_layers = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.output_dim) for _ in range(self.num_features)]
        )
        self.rankmixer_layers = nn.ModuleList(
            [
                RankMixerLayer(
                    num_features=self.num_features,
                    dim_multiplier=int(config.dim_multiplier),
                    num_heads=config.num_heads,
                    use_add_norm=bool(config.use_add_norm),
                    ffn_multiplier=int(config.ffn_multiplier),
                )
                for _ in range(int(config.num_layers))
            ]
        )

        mlp_layers: list[nn.Module] = []
        previous_dim = self.num_features * self.output_dim
        for hidden_dim in config.mix_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(previous_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(config.dropout)),
                ]
            )
            previous_dim = int(hidden_dim)
        mlp_layers.append(nn.Linear(previous_dim, 1))
        self.prediction_mlp = nn.Sequential(*mlp_layers)

        self._reset_parameters()

    def build_train_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        self._validate_dataset_compatibility(prepared_data)
        return RankMixerTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        self._validate_dataset_compatibility(prepared_data)
        return RankMixerEvalCollator(self.config, prepared_data)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = batch[RANKMIXER_FEATURES].to(dtype=torch.long)
        if features.ndim != 2 or int(features.shape[1]) != self.num_features:
            raise ValueError(
                f"RankMixer expects features with shape [batch, {self.num_features}], got {tuple(features.shape)}."
            )

        token_embeddings = [
            projection(embedding(features[:, feature_index]))
            for feature_index, (embedding, projection) in enumerate(
                zip(self.embeddings, self.tokenization_layers, strict=True)
            )
        ]
        mixed = torch.cat(token_embeddings, dim=1)
        for layer in self.rankmixer_layers:
            mixed = layer(mixed)

        logits = self.prediction_mlp(mixed).reshape(-1)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch[LABEL].to(dtype=torch.float32).reshape(-1)
        return F.binary_cross_entropy_with_logits(outputs["logits"], labels)

    def predict(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        return self.forward(model_inputs)["probs"]

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def _validate_dataset_compatibility(self, prepared_data: BaseTaskDataset) -> None:
        dataset_hash_size = getattr(prepared_data.config, "hash_size", None)
        if dataset_hash_size is not None and int(dataset_hash_size) != int(self.config.hash_size):
            raise ValueError(
                f"RankMixer hash_size={self.config.hash_size} does not match dataset hash_size={dataset_hash_size}."
            )


__all__ = [
    "PerTokenFFN",
    "RankMixerLayer",
    "RankMixerModel",
    "TokenMix",
]
