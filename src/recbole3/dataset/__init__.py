from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recbole3.dataset.amazon2023 import Amazon2023Parser, Amazon2023RetrievalConfig, Amazon2023RetrievalDataset
from recbole3.dataset.base import (
    BaseDatasetParser,
    BaseTaskDataset,
    DatasetConfig,
    DatasetTask,
    Interaction,
    ParsedData,
    RankingDataset,
    RecordsDataset,
    RetrievalDataset,
    RetrievalEvalRequest,
    SplitConfig,
    SplitName,
    leave_one_out_boundaries,
    ratio_boundaries,
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Static dataset table entry."""

    dataset_cls: type[BaseTaskDataset[Any, Any]]
    config_cls: type[DatasetConfig]


DATASET_TABLE: dict[str, DatasetSpec] = {
    "amazon2023_retrieval": DatasetSpec(
        dataset_cls=Amazon2023RetrievalDataset,
        config_cls=Amazon2023RetrievalConfig,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_TABLE[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_TABLE)) or "<empty>"
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc


__all__ = [
    "Amazon2023Parser",
    "Amazon2023RetrievalConfig",
    "Amazon2023RetrievalDataset",
    "BaseDatasetParser",
    "BaseTaskDataset",
    "DATASET_TABLE",
    "DatasetConfig",
    "DatasetSpec",
    "DatasetTask",
    "Interaction",
    "ParsedData",
    "RankingDataset",
    "RecordsDataset",
    "RetrievalDataset",
    "RetrievalEvalRequest",
    "SplitConfig",
    "SplitName",
    "get_dataset_spec",
    "leave_one_out_boundaries",
    "ratio_boundaries",
]
