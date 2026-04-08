from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from recbole3.dataset import Interaction, RecordsDataset, RetrievalEvalRequest
from recbole3.model.base import BaseCollator, BaseRetrievalModelDataset, ModelDatasets
from recbole3.model.hstu.config import HSTUConfig


@dataclass(frozen=True, slots=True)
class HSTUInteraction(Interaction):
    """Training interaction augmented with sequence history for HSTU."""

    history_item_ids: tuple[int, ...] = ()
    history_timestamps: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class HSTURetrievalEvalRequest(RetrievalEvalRequest):
    """Retrieval evaluation request augmented with HSTU sequence history."""

    history_item_ids: tuple[int, ...] = ()
    history_timestamps: tuple[float, ...] = ()


HistoryEntry = tuple[int, float]
HistoryState = dict[int, tuple[HistoryEntry, ...]]


def build_hstu_histories(
    records: Sequence[Interaction],
    *,
    history_max_length: int,
    initial_histories: Mapping[int, tuple[HistoryEntry, ...]] | None = None,
    include_target_item: Callable[[Interaction], bool] | None = None,
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
    for record in records:
        if record.timestamp is None:
            raise ValueError("HSTU requires timestamp on every interaction used for sequence construction.")
        user_id = int(record.user_id)
        user_history = history_state.setdefault(user_id, [])
        history_item_ids.append(tuple(item_id for item_id, _ in user_history))
        history_timestamps.append(tuple(timestamp for _, timestamp in user_history))
        if include_target(record):
            user_history.append((int(record.item_id), float(record.timestamp)))
            if len(user_history) > history_max_length:
                del user_history[:-history_max_length]
    return (
        history_item_ids,
        history_timestamps,
        {user_id: tuple(entries) for user_id, entries in history_state.items()},
    )


class HSTUModelDataset(
    BaseRetrievalModelDataset[HSTUInteraction, HSTURetrievalEvalRequest],
):
    """Model-side retrieval dataset that adds HSTU item and timestamp histories."""

    def _build_model_datasets(
        self,
        *,
        model_config: HSTUConfig,
    ) -> ModelDatasets[HSTUInteraction, HSTURetrievalEvalRequest]:
        history_max_length = _require_history_max_length(model_config)
        train_records, history_state = self._build_hstu_interactions(
            list(self.get_train_dataset()),
            history_max_length=history_max_length,
        )
        valid_records, history_state = self._build_hstu_eval_requests(
            list(self.get_eval_dataset("valid")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        test_records, _ = self._build_hstu_eval_requests(
            list(self.get_eval_dataset("test")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        return ModelDatasets(
            train_dataset=RecordsDataset(train_records),
            valid_dataset=RecordsDataset(valid_records),
            test_dataset=RecordsDataset(test_records),
        )

    def _build_hstu_interactions(
        self,
        records: Sequence[Interaction],
        *,
        history_max_length: int,
        initial_histories: Mapping[int, tuple[HistoryEntry, ...]] | None = None,
    ) -> tuple[list[HSTUInteraction], HistoryState]:
        history_item_ids, history_timestamps, history_state = build_hstu_histories(
            records,
            history_max_length=history_max_length,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
        )
        hstu_records = [
            HSTUInteraction(
                user_id=int(record.user_id),
                item_id=int(record.item_id),
                timestamp=record.timestamp,
                label=record.label,
                history_item_ids=item_history,
                history_timestamps=time_history,
            )
            for record, item_history, time_history in zip(records, history_item_ids, history_timestamps, strict=True)
        ]
        return hstu_records, history_state

    def _build_hstu_eval_requests(
        self,
        records: Sequence[RetrievalEvalRequest],
        *,
        history_max_length: int,
        initial_histories: Mapping[int, tuple[HistoryEntry, ...]] | None = None,
    ) -> tuple[list[HSTURetrievalEvalRequest], HistoryState]:
        history_item_ids, history_timestamps, history_state = build_hstu_histories(
            records,
            history_max_length=history_max_length,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
        )
        hstu_records = [
            HSTURetrievalEvalRequest(
                user_id=int(record.user_id),
                item_id=int(record.item_id),
                timestamp=record.timestamp,
                label=record.label,
                seen_item_ids=record.seen_item_ids,
                candidate_item_ids=record.candidate_item_ids,
                history_item_ids=item_history,
                history_timestamps=time_history,
            )
            for record, item_history, time_history in zip(records, history_item_ids, history_timestamps, strict=True)
        ]
        return hstu_records, history_state

    def _include_target_item_in_history(self, record: Interaction) -> bool:
        return _default_include_target_item(record)


class HSTUTrainCollator(BaseCollator):
    """Collate HSTU training records into padded history tensors."""

    def __call__(self, feature_records: Sequence[HSTUInteraction]) -> dict[str, torch.Tensor]:
        batch = _build_hstu_history_batch(feature_records)
        batch["item_id"] = torch.tensor([int(record.item_id) for record in feature_records], dtype=torch.long)
        return batch


class HSTUEvalCollator(BaseCollator):
    """Collate HSTU evaluation records into padded history tensors."""

    def __call__(self, feature_records: Sequence[HSTURetrievalEvalRequest]) -> dict[str, torch.Tensor]:
        return _build_hstu_history_batch(feature_records)


def _build_hstu_history_batch(records: Sequence[HSTUInteraction | HSTURetrievalEvalRequest]) -> dict[str, torch.Tensor]:
    history_lengths = torch.tensor([len(record.history_item_ids) for record in records], dtype=torch.long)
    batch_size = len(records)
    max_length = int(torch.max(history_lengths).item()) if batch_size > 0 else 0
    history_item_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    history_timestamps = torch.zeros((batch_size, max_length), dtype=torch.float32)
    for row_index, record in enumerate(records):
        row_length = len(record.history_item_ids)
        if row_length == 0:
            continue
        history_item_ids[row_index, :row_length] = torch.tensor(record.history_item_ids, dtype=torch.long)
        history_timestamps[row_index, :row_length] = torch.tensor(record.history_timestamps, dtype=torch.float32)
    return {
        "history_item_ids": history_item_ids,
        "history_timestamps": history_timestamps,
        "history_lengths": history_lengths,
    }


def _require_history_max_length(model_config: HSTUConfig) -> int:
    history_max_length = getattr(model_config, "history_max_length", None)
    if history_max_length is None or int(history_max_length) <= 0:
        raise ValueError("HSTU requires model_config.history_max_length to be a positive integer.")
    return int(history_max_length)


def _default_include_target_item(record: Interaction) -> bool:
    return record.label is None or float(record.label) > 0


__all__ = [
    "HSTUEvalCollator",
    "HSTUInteraction",
    "HSTUModelDataset",
    "HSTURetrievalEvalRequest",
    "HSTUTrainCollator",
    "build_hstu_histories",
]
