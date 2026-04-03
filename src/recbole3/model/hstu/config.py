from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class HSTUConfig(SequentialModelConfig):
    """Configuration for the HSTU retrieval model."""

    name: str = field(default="hstu", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=50,
        metadata={"help": "Maximum number of recent history events visible to HSTU."},
    )
    embedding_dim: int = field(default=64, metadata={"help": "Item and hidden embedding dimension."})
    num_layers: int = field(default=2, metadata={"help": "Number of HSTU blocks."})
    num_heads: int = field(default=2, metadata={"help": "Number of attention heads per HSTU block."})
    attention_dim: int = field(default=16, metadata={"help": "Per-head attention projection dimension."})
    linear_hidden_dim: int = field(default=16, metadata={"help": "Per-head value projection dimension."})
    linear_dropout_rate: float = field(default=0.1, metadata={"help": "Dropout applied to HSTU linear outputs."})
    attn_dropout_rate: float = field(default=0.0, metadata={"help": "Dropout applied to attention weights."})
    temperature: float = field(default=0.05, metadata={"help": "Temperature used for retrieval logits."})
    l2_norm_eps: float = field(default=1e-6, metadata={"help": "Epsilon used by L2 normalization."})
    normalize_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to L2-normalize user and item embeddings before scoring."},
    )
    num_time_buckets: int = field(default=128, metadata={"help": "Bucket count used by relative time bias."})
