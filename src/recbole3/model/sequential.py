from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID, LABEL, USER_ID
from recbole3.model.base import BaseRankingModelDataset, BaseRetrievalModelDataset, ModelConfig, ModelDatasets


HISTORY_ITEM_IDS = "history_item_ids"


@dataclass(slots=True)
class SequentialModelConfig(ModelConfig):
    """Shared config fields for models that consume sequential histories."""

    history_max_length: int | None = field(
        default=None,
        metadata={"help": "Optional maximum number of most recent items kept in history_item_ids."},
    )


def build_history_item_ids(
    records: pd.DataFrame,
    *,
    initial_histories: Mapping[int, tuple[int, ...]] | None = None,
    include_target_item: Callable[[Mapping[str, Any]], bool] | None = None,
    history_max_length: int | None = None,
) -> tuple[list[tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Build one prefix history for each record and return the final user-history state."""

    include_target = include_target_item or _default_include_target_item
    normalized_history_max_length = _normalize_history_max_length(history_max_length)
    history_state: dict[int, list[int]] = {
        int(user_id): _truncate_history(list(item_ids), history_max_length=normalized_history_max_length)
        for user_id, item_ids in (initial_histories or {}).items()
    }
    history_item_ids: list[tuple[int, ...]] = []
    for record in records.to_dict("records"):
        user_id = int(record[USER_ID])
        user_history = history_state.setdefault(user_id, [])
        history_item_ids.append(tuple(user_history))
        if include_target(record):
            user_history.append(int(record[ITEM_ID]))
            if normalized_history_max_length is not None and len(user_history) > normalized_history_max_length:
                del user_history[:-normalized_history_max_length]
    return history_item_ids, {user_id: tuple(item_ids) for user_id, item_ids in history_state.items()}


def _default_include_target_item(record: Mapping[str, Any]) -> bool:
    """Only positive or unlabeled interactions contribute new items to history."""

    label = record.get(LABEL)
    return label is None or pd.isna(label) or float(label) > 0


def _normalize_history_max_length(history_max_length: int | None) -> int | None:
    if history_max_length is None:
        return None
    if history_max_length <= 0:
        raise ValueError("history_max_length must be None or a positive integer.")
    return int(history_max_length)


def _truncate_history(item_ids: list[int], *, history_max_length: int | None) -> list[int]:
    if history_max_length is None or len(item_ids) <= history_max_length:
        return item_ids
    return item_ids[-history_max_length:]


class BaseSequentialRankingModelDataset(BaseRankingModelDataset[pd.DataFrame, pd.DataFrame], ABC):
    """Model-side ranking dataset that adds history_item_ids to every split."""

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        history_max_length = _get_history_max_length(model_config)
        train_frame, history_state = self._build_sequential_frame(
            _dataset_frame(self.get_train_dataset()),
            history_max_length=history_max_length,
        )
        valid_frame, history_state = self._build_sequential_frame(
            _dataset_frame(self.get_eval_dataset("valid")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        test_frame, _ = self._build_sequential_frame(
            _dataset_frame(self.get_eval_dataset("test")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        return ModelDatasets(
            train_dataset=FrameDataset(train_frame),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )

    def _build_sequential_frame(
        self,
        records: pd.DataFrame,
        *,
        initial_histories: Mapping[int, tuple[int, ...]] | None = None,
        history_max_length: int | None = None,
    ) -> tuple[pd.DataFrame, dict[int, tuple[int, ...]]]:
        history_item_ids, history_state = build_history_item_ids(
            records,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
            history_max_length=history_max_length,
        )
        sequential_records = records.copy()
        sequential_records[HISTORY_ITEM_IDS] = history_item_ids
        return sequential_records, history_state

    def _include_target_item_in_history(self, record: Mapping[str, Any]) -> bool:
        return _default_include_target_item(record)


class BaseSequentialRetrievalModelDataset(BaseRetrievalModelDataset[pd.DataFrame, pd.DataFrame], ABC):
    """Model-side retrieval dataset that adds history_item_ids to train and eval splits."""

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        history_max_length = _get_history_max_length(model_config)
        train_frame, history_state = self._build_sequential_frame(
            _dataset_frame(self.get_train_dataset()),
            history_max_length=history_max_length,
        )
        valid_frame, history_state = self._build_sequential_frame(
            _dataset_frame(self.get_eval_dataset("valid")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        test_frame, _ = self._build_sequential_frame(
            _dataset_frame(self.get_eval_dataset("test")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        return ModelDatasets(
            train_dataset=FrameDataset(train_frame),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )

    def _build_sequential_frame(
        self,
        records: pd.DataFrame,
        *,
        initial_histories: Mapping[int, tuple[int, ...]] | None = None,
        history_max_length: int | None = None,
    ) -> tuple[pd.DataFrame, dict[int, tuple[int, ...]]]:
        history_item_ids, history_state = build_history_item_ids(
            records,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
            history_max_length=history_max_length,
        )
        sequential_records = records.copy()
        sequential_records[HISTORY_ITEM_IDS] = history_item_ids
        return sequential_records, history_state

    def _include_target_item_in_history(self, record: Mapping[str, Any]) -> bool:
        return _default_include_target_item(record)


def _get_history_max_length(model_config: ModelConfig) -> int | None:
    return _normalize_history_max_length(getattr(model_config, "history_max_length", None))


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"Sequential model datasets require FrameDataset, got {type(dataset).__name__}.")
    return dataset.frame.copy()


__all__ = [
    "BaseSequentialRankingModelDataset",
    "BaseSequentialRetrievalModelDataset",
    "HISTORY_ITEM_IDS",
    "SequentialModelConfig",
    "build_history_item_ids",
]
