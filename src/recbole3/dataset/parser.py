from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recbole3.dataset.config import DatasetConfig, SplitConfig


@dataclass(slots=True)
class ParsedData:
    """Parser output consumed by task-level dataset builders.

    `interactions` must be a pandas DataFrame with raw `user_id` and `item_id`
    columns. `timestamp` and `label` are optional, and extra columns are
    preserved. `BaseTaskDataset.prepare(...)` remaps raw ids into framework ids:
    users and items both start at 0. Any padding ids are owned by the model.

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

    @property
    def data_dir(self) -> Path:
        """Return the root directory for processed data files.

        Subclasses that manage local data files should override this property.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement data_dir. "
            "Subclasses that manage local data files must override this property."
        )

    @abstractmethod
    def parse(self) -> ParsedData:
        """Return raw interaction and optional entity metadata DataFrames."""


__all__ = [
    "BaseDatasetParser",
    "DatasetConfig",
    "ParsedData",
    "SplitConfig",
]
