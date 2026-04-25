from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class TIGERConfig(SequentialModelConfig):
    """Configuration for the TIGER generative retrieval model."""

    name: str = field(default="tiger", metadata={"help": "Registered model name."})
    sid_file: str = field(default="", metadata={"help": "Path to item_sids.json using remapped item ids as keys."})
    history_max_length: int | None = field(
        default=20,
        metadata={"help": "Maximum number of recent history items used by TIGER."},
    )
    n_user_tokens: int = field(default=1, metadata={"help": "Number of hash buckets used for user tokens."})
    num_beams: int = field(default=50, metadata={"help": "Beam width used by T5 generation."})

    num_layers: int = field(default=4, metadata={"help": "Number of T5 encoder layers."})
    num_decoder_layers: int = field(default=4, metadata={"help": "Number of T5 decoder layers."})
    d_model: int = field(default=128, metadata={"help": "T5 hidden size."})
    d_ff: int = field(default=1024, metadata={"help": "T5 feed-forward hidden size."})
    num_heads: int = field(default=6, metadata={"help": "Number of T5 attention heads."})
    d_kv: int = field(default=64, metadata={"help": "T5 key/value projection size per head."})
    dropout_rate: float = field(default=0.1, metadata={"help": "T5 dropout rate."})
    activation_function: str = field(default="relu", metadata={"help": "T5 activation function."})
    feed_forward_proj: str = field(default="relu", metadata={"help": "T5 feed-forward projection type."})


__all__ = ["TIGERConfig"]
