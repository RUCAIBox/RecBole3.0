from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from recbole3.dataset.base import TaskDataset
from recbole3.dataset.utils import ITEM_ID, LABEL, TIMESTAMP, USER_ID

from .base import Amazon2014BaseConfig, Amazon2014BaseParser
from .utils import numeric_or_none


@dataclass(slots=True)
class Amazon2014RetrievalConfig(Amazon2014BaseConfig):
    name: str = field(default="amazon2014_retrieval", metadata={"help": "Amazon 2014 retrieval dataset name."})


class Amazon2014RetrievalParser(Amazon2014BaseParser):
    """Amazon Reviews 2014 parser for retrieval tasks."""

    config_cls = Amazon2014RetrievalConfig
    config: Amazon2014RetrievalConfig

    def _build_interactions(self, reviews: pd.DataFrame) -> pd.DataFrame:
        timestamps = tuple(numeric_or_none(value) for value in reviews["unixReviewTime"].tolist())
        if any(value is None for value in timestamps):
            raise ValueError("Amazon 2014 canonical reviews contain invalid unixReviewTime values.")
        return pd.DataFrame(
            {
                USER_ID: reviews["reviewerID"].astype(object).tolist(),
                ITEM_ID: reviews["asin"].astype(object).tolist(),
                TIMESTAMP: timestamps,
                LABEL: None,
            }
        )


class Amazon2014RetrievalDataset(TaskDataset):
    """Amazon Reviews 2014 retrieval dataset implementation."""

    config_cls = Amazon2014RetrievalConfig
    parser_cls = Amazon2014RetrievalParser


__all__ = [
    "Amazon2014RetrievalConfig",
    "Amazon2014RetrievalDataset",
    "Amazon2014RetrievalParser",
]
