from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from recbole3.dataset.base import RetrievalDataset
from recbole3.dataset.utils import ITEM_ID, LABEL, TIMESTAMP, USER_ID

from .base import Amazon2023BaseConfig, Amazon2023BaseParser
from .utils import numeric_or_none


@dataclass(slots=True)
class Amazon2023RetrievalConfig(Amazon2023BaseConfig):
    name: str = field(default="amazon2023_retrieval", metadata={"help": "Amazon 2023 retrieval dataset name."})


class Amazon2023RetrievalParser(Amazon2023BaseParser):
    """Amazon Reviews 2023 parser for retrieval tasks."""

    config_cls = Amazon2023RetrievalConfig
    config: Amazon2023RetrievalConfig

    def _build_interactions(self, reviews: pd.DataFrame) -> pd.DataFrame:
        timestamps = tuple(numeric_or_none(value) for value in reviews["timestamp"].tolist())
        return pd.DataFrame(
            {
                USER_ID: reviews["user_id"].astype(object).tolist(),
                ITEM_ID: reviews["parent_asin"].astype(object).tolist(),
                TIMESTAMP: timestamps,
                LABEL: None,
            }
        )


class Amazon2023RetrievalDataset(RetrievalDataset):
    """Amazon Reviews 2023 retrieval dataset implementation."""

    config_cls = Amazon2023RetrievalConfig
    parser_cls = Amazon2023RetrievalParser


__all__ = [
    "Amazon2023RetrievalConfig",
    "Amazon2023RetrievalDataset",
    "Amazon2023RetrievalParser",
]
