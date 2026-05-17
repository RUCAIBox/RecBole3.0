from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.base import ModelConfig


@dataclass(slots=True)
class RankMixerConfig(ModelConfig):
    """Configuration for the RankMixer CTR model."""

    name: str = field(default="rankmixer", metadata={"help": "Registered RankMixer model name."})
    num_features: int = field(default=22, metadata={"help": "Number of input feature fields."})
    feature_columns: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Optional explicit feature column names. Defaults to feature_0..feature_{num_features-1}."},
    )
    label_column: str = field(default="label", metadata={"help": "Label column used for point-wise CTR training."})
    user_id_column: str | None = field(
        default="user_id",
        metadata={"help": "Optional user id column to preserve when present in prepared frames."},
    )
    item_id_column: str | None = field(
        default="item_id",
        metadata={"help": "Optional item id column to preserve when present in prepared frames."},
    )
    timestamp_column: str | None = field(
        default="timestamp",
        metadata={"help": "Optional timestamp column to preserve when present in prepared frames."},
    )
    embedding_dim: int = field(default=16, metadata={"help": "Embedding dimension used for each feature field."})
    dim_multiplier: int = field(
        default=2,
        metadata={"help": "Multiplier used to build the per-token hidden width: num_features * dim_multiplier."},
    )
    num_layers: int = field(default=2, metadata={"help": "Number of stacked RankMixer blocks."})
    num_heads: int | None = field(
        default=None,
        metadata={"help": "Token mixing head count. Defaults to num_features when unset."},
    )
    use_add_norm: bool = field(
        default=True,
        metadata={"help": "Whether token mixing uses residual add-and-layernorm."},
    )
    ffn_multiplier: int = field(
        default=4,
        metadata={"help": "Expansion ratio used by each per-token feed-forward block."},
    )
    mix_hidden_dims: tuple[int, ...] = field(
        default=(256, 128, 64, 32),
        metadata={"help": "Hidden widths of the final CTR prediction MLP."},
    )
    dropout: float = field(default=0.2, metadata={"help": "Dropout rate used in the final prediction MLP."})
    hash_size: int = field(
        default=1_000_000,
        metadata={"help": "Hash space size that must match the Avazu parser hash size."},
    )


__all__ = [
    "RankMixerConfig",
]
