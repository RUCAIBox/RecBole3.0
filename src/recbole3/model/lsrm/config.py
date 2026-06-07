from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class LSRMConfig(SequentialModelConfig):
    """Configuration for the LSRM (SASRec-based) retrieval model."""

    name: str = field(default="lsrm", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=50,
        metadata={"help": "Maximum number of recent history items visible to LSRM."},
    )
    n_embd: int = field(default=64, metadata={"help": "GPT-2 embedding / hidden dimension."})
    n_layer: int = field(default=2, metadata={"help": "Number of GPT-2 transformer layers."})
    n_head: int = field(default=2, metadata={"help": "Number of GPT-2 attention heads."})
    n_inner: int = field(default=256, metadata={"help": "GPT-2 feed-forward inner dimension."})
    activation_function: str = field(
        default="gelu_new", metadata={"help": "GPT-2 activation function."}
    )
    resid_pdrop: float = field(default=0.0, metadata={"help": "Residual dropout rate."})
    embd_pdrop: float = field(default=0.0, metadata={"help": "Embedding dropout rate."})
    attn_pdrop: float = field(default=0.5, metadata={"help": "Attention dropout rate."})
    layer_norm_epsilon: float = field(default=1e-12, metadata={"help": "Layer norm epsilon."})
    initializer_range: float = field(default=0.02, metadata={"help": "Weight initializer range."})


__all__ = ["LSRMConfig"]
