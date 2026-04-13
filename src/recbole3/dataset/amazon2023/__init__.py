from __future__ import annotations

from . import utils as _utils
from .base import Amazon2023BaseConfig, Amazon2023BaseParser
from .retrieval import (
    Amazon2023RetrievalConfig,
    Amazon2023RetrievalDataset,
    Amazon2023RetrievalParser,
)
from .utils import (
    AMAZON2023_AVAILABLE_CATEGORIES,
    AMAZON2023_DATASET_ID,
    AMAZON2023_META_COLUMNS,
    AMAZON2023_REVIEW_COLUMNS,
    AMAZON2023_UNSUPPORTED_5CORE_CATEGORIES,
)


def __getattr__(name: str) -> object:
    if name in {"load_huggingface_dataset", "load_modelscope_dataset"}:
        return getattr(_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AMAZON2023_AVAILABLE_CATEGORIES",
    "AMAZON2023_DATASET_ID",
    "AMAZON2023_META_COLUMNS",
    "AMAZON2023_REVIEW_COLUMNS",
    "AMAZON2023_UNSUPPORTED_5CORE_CATEGORIES",
    "Amazon2023BaseConfig",
    "Amazon2023BaseParser",
    "Amazon2023RetrievalConfig",
    "Amazon2023RetrievalDataset",
    "Amazon2023RetrievalParser",
]
