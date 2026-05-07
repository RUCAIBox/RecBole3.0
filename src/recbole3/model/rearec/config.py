from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class ReaRecConfig(SequentialModelConfig):
    """Configuration for the ReaRec inference-time reasoning sequential recommendation model.

    Supports two learning strategies built on a SASRec-style Transformer backbone:
      - ERL (Ensemble Reasoning Learning): ensemble average of K+1 steps + KL diversity.
      - PRL (Progressive Reasoning Learning): progressive temperature CE + noise contrastive.

    The HSTU backbone is reserved for future work (backbone='hstu' raises NotImplementedError).
    """

    name: str = field(default="rearec")

    # --- backbone ---
    backbone: str = field(
        default="sasrec",
        metadata={"help": "Sequence encoder backbone. Currently only 'sasrec' is supported."},
    )

    # --- SASRec Transformer architecture ---
    embedding_dim: int = field(default=256, metadata={"help": "Item and hidden embedding dimension."})
    num_layers: int = field(default=2, metadata={"help": "Number of Transformer encoder layers."})
    num_heads: int = field(default=2, metadata={"help": "Number of multi-head attention heads."})
    inner_size: int = field(default=300, metadata={"help": "Feed-forward inner (intermediate) dimension."})
    hidden_act: str = field(
        default="gelu",
        metadata={"help": "Activation function in feed-forward layers. Options: gelu, relu, swish, tanh, sigmoid."},
    )
    layer_norm_eps: float = field(default=1e-12, metadata={"help": "Epsilon for LayerNorm stability."})
    dropout: float = field(default=0.5, metadata={"help": "Dropout probability applied to embeddings and attention."})
    initializer_range: float = field(
        default=0.02, metadata={"help": "Std of truncated normal weight initialization."}
    )

    # --- ReaRec reasoning ---
    learning_strategy: str = field(
        default="prl",
        metadata={"help": "Reasoning learning strategy. Options: 'erl' (Ensemble) or 'prl' (Progressive)."},
    )
    reason_step: int = field(default=2, metadata={"help": "Number of autoregressive reasoning steps K."})
    temperature: float = field(default=0.07, metadata={"help": "Softmax temperature for retrieval logits."})

    # --- ERL-specific ---
    kl_weight: float = field(
        default=0.05,
        metadata={"help": "Weight lambda for KL divergence diversity regularization (ERL only)."},
    )

    # --- PRL-specific ---
    pl_weight: float = field(default=1.0, metadata={"help": "Weight for progressive learning CE loss (PRL only)."})
    temp_scale: float = field(
        default=5.0,
        metadata={"help": "Progressive temperature decay base alpha; tau_k = tau * alpha^(K-k) (PRL only)."},
    )
    noise_factor: float = field(
        default=0.01,
        metadata={"help": "Gaussian noise scale injected into reasoning tokens for contrastive learning (PRL only)."},
    )
    cl_weight: float = field(default=1.0, metadata={"help": "Weight for contrastive reasoning loss (PRL only)."})
    warmup_epochs: int = field(
        default=0,
        metadata={"help": "Noise injection is enabled only when current_epoch > warmup_epochs (PRL only). 0 = always on, matching official default behaviour."},
    )

    # --- HSTU backbone hyperparameters (only used when backbone='hstu') ---
    attention_dim: int = field(
        default=32,
        metadata={"help": "Per-head attention projection dimension (HSTU backbone only)."},
    )
    linear_hidden_dim: int = field(
        default=32,
        metadata={"help": "Per-head value/output projection dimension (HSTU backbone only)."},
    )
    input_dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout on input embeddings after position encoding (HSTU backbone only)."},
    )
    attn_dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout on SiLU attention weights (HSTU backbone only)."},
    )
    linear_dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout on HSTU linear outputs (HSTU backbone only)."},
    )
    num_time_buckets: int = field(
        default=128,
        metadata={"help": "Number of time-delta buckets for relative time bias (HSTU backbone only)."},
    )


__all__ = ["ReaRecConfig"]
