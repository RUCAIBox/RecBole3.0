from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID, LABEL, TIMESTAMP, USER_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModelDataset, ModelDatasets
from recbole3.model.hstu.config import HSTUConfig, HSTU_PADDING_ITEM_ID
from recbole3.model.sequential import HISTORY_ITEM_IDS


HISTORY_TIMESTAMPS = "history_timestamps"

HistoryEntry = tuple[int, float]
HistoryState = dict[int, tuple[HistoryEntry, ...]]


def build_hstu_histories(
    records: pd.DataFrame,
    *,
    history_max_length: int,
    initial_histories: Mapping[int, tuple[HistoryEntry, ...]] | None = None,
    include_target_item: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[list[tuple[int, ...]], list[tuple[float, ...]], HistoryState]:
    """Build per-record HSTU prefix histories of items and timestamps."""

    if history_max_length <= 0:
        raise ValueError("HSTU requires history_max_length to be a positive integer.")

    include_target = include_target_item or _default_include_target_item
    history_state: dict[int, list[HistoryEntry]] = {
        int(user_id): list(entries)[-history_max_length:]
        for user_id, entries in (initial_histories or {}).items()
    }
    history_item_ids: list[tuple[int, ...]] = []
    history_timestamps: list[tuple[float, ...]] = []
    for record in records.to_dict("records"):
        if pd.isna(record.get(TIMESTAMP)):
            raise ValueError("HSTU requires timestamp on every interaction used for sequence construction.")
        user_id = int(record[USER_ID])
        user_history = history_state.setdefault(user_id, [])
        history_item_ids.append(tuple(item_id for item_id, _ in user_history))
        history_timestamps.append(tuple(timestamp for _, timestamp in user_history))
        if include_target(record):
            user_history.append((int(record[ITEM_ID]), float(record[TIMESTAMP])))
            if len(user_history) > history_max_length:
                del user_history[:-history_max_length]
    return (
        history_item_ids,
        history_timestamps,
        {user_id: tuple(entries) for user_id, entries in history_state.items()},
    )


class HSTUModelDataset(
    BaseRetrievalModelDataset[pd.DataFrame, pd.DataFrame],
):
    """Model-side retrieval dataset that adds HSTU item and timestamp histories."""

    def _build_model_datasets(
        self,
        *,
        model_config: HSTUConfig,
    ) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        history_max_length = _require_history_max_length(model_config)
        train_frame, history_state = self._build_hstu_frame(
            _dataset_frame(self.get_train_dataset()),
            history_max_length=history_max_length,
        )
        valid_frame, history_state = self._build_hstu_frame(
            _dataset_frame(self.get_eval_dataset("valid")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        test_frame, _ = self._build_hstu_frame(
            _dataset_frame(self.get_eval_dataset("test")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        return ModelDatasets(
            train_dataset=FrameDataset(train_frame),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )

    def _build_hstu_frame(
        self,
        records: pd.DataFrame,
        *,
        history_max_length: int,
        initial_histories: Mapping[int, tuple[HistoryEntry, ...]] | None = None,
    ) -> tuple[pd.DataFrame, HistoryState]:
        history_item_ids, history_timestamps, history_state = build_hstu_histories(
            records,
            history_max_length=history_max_length,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
        )
        hstu_records = records.copy()
        hstu_records[HISTORY_ITEM_IDS] = history_item_ids
        hstu_records[HISTORY_TIMESTAMPS] = history_timestamps
        return hstu_records, history_state

    def _include_target_item_in_history(self, record: Mapping[str, Any]) -> bool:
        return _default_include_target_item(record)


class HSTUTrainCollator(BaseCollator):
    """Collate HSTU training records into padded history tensors."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        batch = _build_hstu_history_batch(feature_records)
        batch[ITEM_ID] = torch.as_tensor(feature_records[ITEM_ID].to_numpy(), dtype=torch.long)
        return batch


class HSTUEvalCollator(BaseCollator):
    """Collate HSTU evaluation records into padded history tensors."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _build_hstu_history_batch(feature_records)


def _build_hstu_history_batch(records: pd.DataFrame) -> dict[str, torch.Tensor]:
    history_items = [tuple(values) for values in records[HISTORY_ITEM_IDS].tolist()]
    history_times = [tuple(values) for values in records[HISTORY_TIMESTAMPS].tolist()]
    history_lengths = torch.tensor([len(values) for values in history_items], dtype=torch.long)
    batch_size = len(records)
    max_length = int(torch.max(history_lengths).item()) if batch_size > 0 else 0
    history_item_ids = torch.full((batch_size, max_length), HSTU_PADDING_ITEM_ID, dtype=torch.long)
    history_timestamps = torch.zeros((batch_size, max_length), dtype=torch.float32)
    for row_index, (item_history, time_history) in enumerate(zip(history_items, history_times, strict=True)):
        row_length = len(item_history)
        if row_length == 0:
            continue
        history_item_ids[row_index, :row_length] = torch.tensor(item_history, dtype=torch.long)
        history_timestamps[row_index, :row_length] = torch.tensor(time_history, dtype=torch.float32)
    return {
        HISTORY_ITEM_IDS: history_item_ids,
        HISTORY_TIMESTAMPS: history_timestamps,
        "history_lengths": history_lengths,
    }


def _require_history_max_length(model_config: HSTUConfig) -> int:
    history_max_length = getattr(model_config, "history_max_length", None)
    if history_max_length is None or int(history_max_length) <= 0:
        raise ValueError("HSTU requires model_config.history_max_length to be a positive integer.")
    return int(history_max_length)


def _default_include_target_item(record: Mapping[str, Any]) -> bool:
    label = record.get(LABEL)
    return label is None or pd.isna(label) or float(label) > 0


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"HSTU model datasets require FrameDataset, got {type(dataset).__name__}.")
    return dataset.frame.copy()


__all__ = [
    "HISTORY_TIMESTAMPS",
    "HSTUEvalCollator",
    "HSTUModelDataset",
    "HSTUTrainCollator",
    "build_hstu_histories",
]
