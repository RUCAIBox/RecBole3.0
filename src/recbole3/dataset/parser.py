from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from recbole3.dataset.config import DatasetConfig, SplitConfig


@dataclass(slots=True)
class ParsedData:
    """Parser output consumed by task-level dataset builders.

    `interactions` must be a pandas DataFrame with raw `user_id` and `item_id`
    columns. `timestamp` and `label` are optional, and extra columns are
    preserved. `BaseTaskDataset.prepare(...)` remaps raw ids into framework ids:
    users start at 0, item 0 is padding, and real items start at 1.

    `user_table` and `item_table` are optional metadata DataFrames. When
    provided, their `user_id` / `item_id` columns are raw keys.
    """

    interactions: pd.DataFrame
    user_table: pd.DataFrame | None = None
    item_table: pd.DataFrame | None = None


class BaseDatasetParser(ABC):
    """Dataset-specific parser that hides raw download, cache, and normalization details."""

    config_cls: type[DatasetConfig] = DatasetConfig

    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def parse(self) -> ParsedData:
        """Return raw interaction and optional entity metadata DataFrames."""


__all__ = [
    "BaseDatasetParser",
    "DatasetConfig",
    "ParsedData",
    "SplitConfig",
]
