from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from recbole3.dataset import Interaction, RecordsDataset, RetrievalEvalRequest
from recbole3.model.base import BaseRankingModelDataset, BaseRetrievalModelDataset, ModelConfig, ModelDatasets


@dataclass(slots=True)
class SequentialModelConfig(ModelConfig):
    """Shared config fields for models that consume sequential histories."""

    history_max_length: int | None = field(
        default=None,
        metadata={"help": "Optional maximum number of most recent items kept in history_item_ids."},
    )


@dataclass(frozen=True, slots=True)
class SequentialInteraction(Interaction):
    """Interaction record augmented with one prefix history sequence."""

    history_item_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SequentialRetrievalEvalRequest(RetrievalEvalRequest):
    """Retrieval evaluation request augmented with one prefix history sequence."""

    history_item_ids: tuple[int, ...] = ()


def build_history_item_ids(
    records: Sequence[Interaction],
    *,
    initial_histories: Mapping[int, tuple[int, ...]] | None = None,
    include_target_item: Callable[[Interaction], bool] | None = None,
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
    for record in records:
        user_id = int(record.user_id)
        user_history = history_state.setdefault(user_id, [])
        history_item_ids.append(tuple(user_history))
        if include_target(record):
            user_history.append(int(record.item_id))
            if normalized_history_max_length is not None and len(user_history) > normalized_history_max_length:
                del user_history[:-normalized_history_max_length]
    return history_item_ids, {user_id: tuple(item_ids) for user_id, item_ids in history_state.items()}


def _default_include_target_item(record: Interaction) -> bool:
    """Only positive or unlabeled interactions contribute new items to history."""

    return record.label is None or float(record.label) > 0


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


class BaseSequentialRankingModelDataset(
    BaseRankingModelDataset[SequentialInteraction, SequentialInteraction],
    ABC,
):
    """Model-side ranking dataset that adds history_item_ids to every split."""

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[SequentialInteraction, SequentialInteraction]:
        history_max_length = _get_history_max_length(model_config)
        train_records, history_state = self._build_sequential_interactions(
            list(self.get_train_dataset()),
            history_max_length=history_max_length,
        )
        valid_records, history_state = self._build_sequential_interactions(
            list(self.get_eval_dataset("valid")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        test_records, _ = self._build_sequential_interactions(
            list(self.get_eval_dataset("test")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        return ModelDatasets(
            train_dataset=RecordsDataset(train_records),
            valid_dataset=RecordsDataset(valid_records),
            test_dataset=RecordsDataset(test_records),
        )

    def _build_sequential_interactions(
        self,
        records: Sequence[Interaction],
        *,
        initial_histories: Mapping[int, tuple[int, ...]] | None = None,
        history_max_length: int | None = None,
    ) -> tuple[list[SequentialInteraction], dict[int, tuple[int, ...]]]:
        history_item_ids, history_state = build_history_item_ids(
            records,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
            history_max_length=history_max_length,
        )
        sequential_records = [
            SequentialInteraction(
                user_id=int(record.user_id),
                item_id=int(record.item_id),
                timestamp=record.timestamp,
                label=record.label,
                history_item_ids=history,
            )
            for record, history in zip(records, history_item_ids, strict=True)
        ]
        return sequential_records, history_state

    def _include_target_item_in_history(self, record: Interaction) -> bool:
        return _default_include_target_item(record)


class BaseSequentialRetrievalModelDataset(
    BaseRetrievalModelDataset[SequentialInteraction, SequentialRetrievalEvalRequest],
    ABC,
):
    """Model-side retrieval dataset that adds history_item_ids to train and eval splits."""

    def _build_model_datasets(
        self,
        *,
        model_config: ModelConfig,
    ) -> ModelDatasets[SequentialInteraction, SequentialRetrievalEvalRequest]:
        history_max_length = _get_history_max_length(model_config)
        train_records, history_state = self._build_sequential_interactions(
            list(self.get_train_dataset()),
            history_max_length=history_max_length,
        )
        valid_records, history_state = self._build_sequential_eval_requests(
            list(self.get_eval_dataset("valid")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        test_records, _ = self._build_sequential_eval_requests(
            list(self.get_eval_dataset("test")),
            initial_histories=history_state,
            history_max_length=history_max_length,
        )
        return ModelDatasets(
            train_dataset=RecordsDataset(train_records),
            valid_dataset=RecordsDataset(valid_records),
            test_dataset=RecordsDataset(test_records),
        )

    def _build_sequential_interactions(
        self,
        records: Sequence[Interaction],
        *,
        initial_histories: Mapping[int, tuple[int, ...]] | None = None,
        history_max_length: int | None = None,
    ) -> tuple[list[SequentialInteraction], dict[int, tuple[int, ...]]]:
        history_item_ids, history_state = build_history_item_ids(
            records,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
            history_max_length=history_max_length,
        )
        sequential_records = [
            SequentialInteraction(
                user_id=int(record.user_id),
                item_id=int(record.item_id),
                timestamp=record.timestamp,
                label=record.label,
                history_item_ids=history,
            )
            for record, history in zip(records, history_item_ids, strict=True)
        ]
        return sequential_records, history_state

    def _build_sequential_eval_requests(
        self,
        records: Sequence[RetrievalEvalRequest],
        *,
        initial_histories: Mapping[int, tuple[int, ...]] | None = None,
        history_max_length: int | None = None,
    ) -> tuple[list[SequentialRetrievalEvalRequest], dict[int, tuple[int, ...]]]:
        history_item_ids, history_state = build_history_item_ids(
            records,
            initial_histories=initial_histories,
            include_target_item=self._include_target_item_in_history,
            history_max_length=history_max_length,
        )
        sequential_records = [
            SequentialRetrievalEvalRequest(
                user_id=int(record.user_id),
                item_id=int(record.item_id),
                timestamp=record.timestamp,
                label=record.label,
                seen_item_ids=record.seen_item_ids,
                candidate_item_ids=record.candidate_item_ids,
                history_item_ids=history,
            )
            for record, history in zip(records, history_item_ids, strict=True)
        ]
        return sequential_records, history_state

    def _include_target_item_in_history(self, record: Interaction) -> bool:
        return _default_include_target_item(record)


def _get_history_max_length(model_config: ModelConfig) -> int | None:
    return _normalize_history_max_length(getattr(model_config, "history_max_length", None))


__all__ = [
    "BaseSequentialRankingModelDataset",
    "BaseSequentialRetrievalModelDataset",
    "SequentialModelConfig",
    "SequentialInteraction",
    "SequentialRetrievalEvalRequest",
    "build_history_item_ids",
]
