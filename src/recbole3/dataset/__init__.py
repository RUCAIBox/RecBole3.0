from __future__ import annotations

from dataclasses import dataclass

from recbole3.dataset.amazon2014 import Amazon2014RetrievalParser, Amazon2014RetrievalConfig, Amazon2014RetrievalDataset
from recbole3.dataset.amazon2023 import Amazon2023RetrievalParser, Amazon2023RetrievalConfig, Amazon2023RetrievalDataset
from recbole3.dataset.config import DatasetConfig, SplitConfig
from recbole3.dataset.base import (
    BaseTaskDataset,
    DatasetTask,
    FrameDataset,
    PARSER_INTERACTIONS_SCHEMA,
    PREPARED_INTERACTIONS_SCHEMA,
    RETRIEVAL_EVAL_SCHEMA,
    TaskDataset,
)

from recbole3.dataset.parser import (
    BaseDatasetParser,
    ParsedData,
)
from recbole3.dataset.utils import (
    FrameSchema,
    require_columns, USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS, CANDIDATE_ITEM_IDS,
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Static dataset table entry."""

    dataset_cls: type[BaseTaskDataset]
    config_cls: type[DatasetConfig]


DATASET_TABLE: dict[str, DatasetSpec] = {
    "amazon2014_retrieval": DatasetSpec(
        dataset_cls=Amazon2014RetrievalDataset,
        config_cls=Amazon2014RetrievalConfig,
    ),
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
    "Amazon2014RetrievalParser",
    "Amazon2014RetrievalConfig",
    "Amazon2014RetrievalDataset",
    "Amazon2023RetrievalParser",
    "Amazon2023RetrievalConfig",
    "Amazon2023RetrievalDataset",
    "BaseDatasetParser",
    "BaseTaskDataset",
    "CANDIDATE_ITEM_IDS",
    "DATASET_TABLE",
    "DatasetConfig",
    "DatasetSpec",
    "DatasetTask",
    "FrameDataset",
    "FrameSchema",
    "ITEM_ID",
    "LABEL",
    "PARSER_INTERACTIONS_SCHEMA",
    "PREPARED_INTERACTIONS_SCHEMA",
    "ParsedData",
    "RETRIEVAL_EVAL_SCHEMA",
    "TaskDataset",
    "SEEN_ITEM_IDS",
    "SplitConfig",
    "TIMESTAMP",
    "USER_ID",
    "get_dataset_spec",
    "require_columns",
]
