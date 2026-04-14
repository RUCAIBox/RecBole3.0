from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class SplitConfig:
    """Task-level split configuration shared by all datasets."""

    strategy: Literal["ratio", "leave_one_out"] = field(
        default="ratio",
        metadata={"help": "Dataset split strategy."},
    )
    order: Literal["chronological", "random"] = field(
        default="chronological",
        metadata={"help": "Record order used before splitting."},
    )
    per_user: bool = field(
        default=True,
        metadata={"help": "Whether to split each user's interactions independently."},
    )
    train_ratio: float = field(default=0.8, metadata={"help": "Training split ratio for ratio-based splitting."})
    valid_ratio: float = field(default=0.1, metadata={"help": "Validation split ratio for ratio-based splitting."})
    test_ratio: float = field(default=0.1, metadata={"help": "Test split ratio for ratio-based splitting."})
    valid_holdout_num: int = field(
        default=1,
        metadata={"help": "Number of interactions held out for validation in leave-one-out splitting."},
    )
    test_holdout_num: int = field(
        default=1,
        metadata={"help": "Number of interactions held out for test in leave-one-out splitting."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed used by random split ordering."})


@dataclass(slots=True)
class DatasetConfig:
    """Convenience dataset config template with the framework's standard fields."""

    name: str = field(default="", metadata={"help": "Registered dataset name."})
    split: SplitConfig = field(default_factory=SplitConfig, metadata={"help": "Dataset split configuration."})


__all__ = [
    "DatasetConfig",
    "SplitConfig",
]
