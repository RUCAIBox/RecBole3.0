from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig

LARES_PADDING_ITEM_ID = 0
ITEM_ID_OFFSET = 1

@dataclass(slots=True)
class LARESConfig(SequentialModelConfig):
    """Configuration for the LARES (Learnable Recurrent State) retrieval model."""

    name: str = field(default="lares", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=50,
        metadata={"help": "Maximum number of recent history items visible to the model."},
    )

    # Model architecture
    n_pre_layers: int = field(default=1, metadata={"help": "Number of transformer layers in pre_encoder."})
    n_core_layers: int = field(default=1, metadata={"help": "Number of transformer layers in core_encoder."})
    n_heads: int = field(default=2, metadata={"help": "Number of attention heads."})
    hidden_size: int = field(default=64, metadata={"help": "Embedding and hidden dimension."})
    inner_size: int = field(default=256, metadata={"help": "Feed-forward inner dimension."})
    hidden_dropout_prob: float = field(default=0.5, metadata={"help": "Dropout rate on hidden states."})
    attn_dropout_prob: float = field(default=0.5, metadata={"help": "Dropout rate on attention weights."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function for feed-forward layers."})
    layer_norm_eps: float = field(default=1e-12, metadata={"help": "Epsilon for layer normalization."})
    initializer_range: float = field(default=0.02, metadata={"help": "Standard deviation for weight initialization."})

    # Recurrence
    mean_recurrence: float = field(default=4.0, metadata={"help": "Mean number of recurrence steps."})
    state_init_method: str = field(
        default="normal",
        metadata={"help": "State initialization method: zero, normal, normal_zero."},
    )
    state_std: float = field(default=1.0, metadata={"help": "Standard deviation for normal state initialization."})
    state_scale: float = field(default=3.0, metadata={"help": "Scale factor applied to initialized state."})
    sampling_scheme: str = field(
        default="poisson-lognormal",
        metadata={"help": "Training recurrence sampling: poisson-lognormal, uniform, poisson-unbounded, poisson-bounded, non-recurrent, constant."},
    )
    adapter_type: str = field(
        default="add",
        metadata={"help": "State-pre_output fusion method: concat, add, linear."},
    )
    test_recurrence_ratios: tuple[float, ...] = field(
        default=(1.0,),
        metadata={"help": "Ratios of mean_recurrence to test at evaluation time."},
    )

    # Contrastive loss
    tau: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss."})
    alpha: float = field(default=0.1, metadata={"help": "Weight for inter-sequence contrastive loss."})
    gamma: float = field(default=0.1, metadata={"help": "Weight for intra-sequence (step-wise) contrastive loss."})
    sem_func: str = field(
        default="cos",
        metadata={"help": "Similarity function for scoring: cos or dot."},
    )
    same_step: bool = field(
        default=True,
        metadata={"help": "Whether augmented sequences use the same recurrence steps as original."},
    )

    # Training stage
    stage: str = field(
        default="SL",
        metadata={"help": "Training stage: SL (supervised) or RL (reinforcement)."},
    )

    # RL (GRPO) parameters
    k: int = field(default=10, metadata={"help": "Top-k for RL reward computation."})
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient in GRPO loss."})
    group_num: int = field(default=8, metadata={"help": "Group size for GRPO advantage normalization."})
    reward_metric: str = field(
        default="recall",
        metadata={"help": "RL reward metric: recall or ndcg."},
    )
    pretrain_model_path: str = field(
        default="",
        metadata={"help": "Path to pretrained SL checkpoint for RL training."},
    )
