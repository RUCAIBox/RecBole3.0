from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recbole3.evaluation.config import EvalConfig


SchedulerInterval = Literal["step", "epoch"]


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Configuration for validation-driven early stopping."""

    enabled: bool = field(default=False, metadata={"help": "Whether fit() stops early when the monitor stops improving."})
    patience: int = field(default=3, metadata={"help": "Number of non-improving epochs tolerated before fit() stops."})
    min_delta: float = field(default=0.0, metadata={"help": "Minimum monitor improvement required to reset early stopping."})


@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model-weight checkpoint persistence."""

    save_best: bool = field(default=False, metadata={"help": "Whether fit() writes the best monitored model weights."})
    save_last: bool = field(default=False, metadata={"help": "Whether fit() writes the most recent model weights each epoch."})


@dataclass(slots=True)
class OptimizerConfig:
    """Configuration for one torch.optim optimizer."""

    name: str = field(default="Adam", metadata={"help": "Name of one torch.optim optimizer class."})
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3},
        metadata={"help": "Keyword arguments passed into the optimizer constructor."},
    )


@dataclass(slots=True)
class SchedulerConfig:
    """Configuration for one torch.optim.lr_scheduler scheduler."""

    name: str = field(metadata={"help": "Name of one torch.optim.lr_scheduler scheduler class."})
    interval: SchedulerInterval = field(
        default="step",
        metadata={"help": "Whether fit() steps the scheduler every optimizer step or every epoch."},
    )
    kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments passed into the scheduler constructor."},
    )


@dataclass(slots=True)
class TrainerConfig:
    """Convenience trainer config template with the framework's standard fields."""

    batch_size: int = field(default=256, metadata={"help": "Default batch size used by trainer dataloaders."})
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle train samples in the dataloader."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Worker count used by dataloaders."})
    pin_memory: bool = field(default=False, metadata={"help": "Whether dataloaders pin host memory."})
    mixed_precision: Literal["no", "fp16", "bf16"] = field(
        default="no",
        metadata={"help": "Accelerate mixed precision mode."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of optimizer accumulation steps handled by accelerate."},
    )
    max_epochs: int = field(default=1, metadata={"help": "Number of epochs executed by fit()."})
    eval_step: int = field(
        default=1,
        metadata={
            "help": (
                "Run validation (and best-checkpointing) every `eval_step` epochs. "
                "The last epoch is always evaluated. Set to 2000 to reproduce the original LETTER schedule."
            ),
        },
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig,
        metadata={"help": "Optimizer settings used during fit()."},
    )
    scheduler: SchedulerConfig | None = field(
        default=None,
        metadata={"help": "Optional learning-rate scheduler settings used during fit()."},
    )
    monitor: str | None = field(
        default=None,
        metadata={"help": "Validation metric key used by early stopping, best checkpointing, and metric-driven schedulers."},
    )
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig,
        metadata={"help": "Early stopping policy driven by validation metrics."},
    )
    checkpoint: CheckpointConfig = field(
        default_factory=CheckpointConfig,
        metadata={"help": "Checkpoint save policy."},
    )
    eval: EvalConfig = field(kw_only=True, metadata={"help": "Evaluation protocol settings."})


__all__ = [
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SchedulerInterval",
    "TrainerConfig",
]
