from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

PAD_ITEM_ID = 0
USER_ID = "user_id"
ITEM_ID = "item_id"
TIMESTAMP = "timestamp"
LABEL = "label"
SEEN_ITEM_IDS = "seen_item_ids"
CANDIDATE_ITEM_IDS = "candidate_item_ids"


@dataclass(frozen=True, slots=True)
class FrameSchema:
    """Lightweight schema hint for DataFrame contracts."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


def require_columns(frame: pd.DataFrame, schema: FrameSchema, *, name: str) -> None:
    """Require only the schema's mandatory columns and allow all extra fields."""

    missing = [column for column in schema.required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns {missing}. "
            f"Expected required={schema.required}, optional={schema.optional}; "
            f"got columns={tuple(frame.columns)}."
        )


def load_huggingface_dataset(
    dataset_id: str,
    subset_name: str,
    *,
    split: str,
    cache_dir: str,
    trust_remote_code: bool,
) -> Any:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`datasets` is required when download_source='huggingface'. Install recbole3[huggingface]."
        ) from exc
    return load_dataset(
        dataset_id,
        subset_name,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )


def load_modelscope_dataset(
    dataset_id: str,
    subset_name: str,
    *,
    split: str,
    trust_remote_code: bool,
) -> Any:
    try:
        from modelscope.msdatasets import MsDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`modelscope` is required when download_source='modelscope'. Install recbole3[modelscope]."
        ) from exc
    return MsDataset.load(
        dataset_id,
        subset_name=subset_name,
        split=split,
        trust_remote_code=trust_remote_code,
    )


__all__ = [
    "CANDIDATE_ITEM_IDS",
    "FrameSchema",
    "ITEM_ID",
    "LABEL",
    "PAD_ITEM_ID",
    "SEEN_ITEM_IDS",
    "TIMESTAMP",
    "USER_ID",
    "load_huggingface_dataset",
    "load_modelscope_dataset",
    "require_columns",
]
