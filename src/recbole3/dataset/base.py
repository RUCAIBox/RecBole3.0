from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from recbole3.evaluation.config import EvalConfig


SplitName = Literal["train", "valid", "test"]
DatasetTask = Literal["ranking", "retrieval"]
TRecord = TypeVar("TRecord")
TTrain = TypeVar("TTrain")
TEval = TypeVar("TEval")


@dataclass(slots=True)
class SplitConfig:
    """Task-level split configuration shared by all datasets."""

    strategy: Literal["ratio", "leave_one_out"] = field(
        default="ratio",
        metadata={"help": "Dataset split strategy."},
    )
    order: Literal["chronological", "random"] = field(
        default="chronological",
        metadata={"help": "Record order used before splitting."},
    )
    per_user: bool = field(
        default=True,
        metadata={"help": "Whether to split each user's interactions independently."},
    )
    train_ratio: float = field(default=0.8, metadata={"help": "Training split ratio for ratio-based splitting."})
    valid_ratio: float = field(default=0.1, metadata={"help": "Validation split ratio for ratio-based splitting."})
    test_ratio: float = field(default=0.1, metadata={"help": "Test split ratio for ratio-based splitting."})
    valid_holdout_num: int = field(
        default=1,
        metadata={"help": "Number of interactions held out for validation in leave-one-out splitting."},
    )
    test_holdout_num: int = field(
        default=1,
        metadata={"help": "Number of interactions held out for test in leave-one-out splitting."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed used by random split ordering."})


@dataclass(slots=True)
class DatasetConfig:
    """Convenience dataset config template with the framework's standard fields."""

    name: str = field(default="", metadata={"help": "Registered dataset name."})
    split: SplitConfig = field(default_factory=SplitConfig, metadata={"help": "Dataset split configuration."})


@dataclass(frozen=True, slots=True)
class Interaction:
    """Unified interaction record shared by parsers and ranking datasets."""

    user_id: int
    item_id: int
    timestamp: int | float | None = None
    label: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalEvalRequest(Interaction):
    """Typed request view for retrieval evaluation splits."""

    seen_item_ids: tuple[int, ...] = ()
    candidate_item_ids: tuple[int, ...] | None = None


@dataclass(slots=True)
class ParsedData:
    """Parser output consumed by task-level dataset builders."""

    interactions: list[Interaction]
    user_table: pd.DataFrame
    item_table: pd.DataFrame


class RecordsDataset(Dataset[TRecord], Generic[TRecord]):
    """Simple in-memory Dataset backed by one sequence of records."""

    def __init__(self, records: list[TRecord] | tuple[TRecord, ...]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TRecord:
        return self.records[index]


class BaseDatasetParser(ABC):
    """Dataset-specific parser that hides raw download, cache, and normalization details."""

    config_cls: type[DatasetConfig] = DatasetConfig

    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def parse(self) -> ParsedData:
        """Return normalized interactions and entity tables for one dataset source."""


class BaseTaskDataset(ABC, Generic[TTrain, TEval]):
    """Prepare task-aware split datasets from one dataset parser."""

    task: DatasetTask
    config_cls: type[DatasetConfig] = DatasetConfig
    parser_cls: type[BaseDatasetParser] | None = None

    def __init__(self, config: DatasetConfig):
        parser_cls = self._require_parser_cls()
        self.config = config
        self._parser = parser_cls(config)
        self._eval_config: EvalConfig | None = None
        self._is_prepared = False
        self._interactions: list[Interaction] = []
        self._user_table = pd.DataFrame()
        self._item_table = pd.DataFrame()
        self._num_users = 0
        self._num_items = 0
        self._train_dataset: Dataset[TTrain] = RecordsDataset([])
        self._valid_dataset: Dataset[TEval] = RecordsDataset([])
        self._test_dataset: Dataset[TEval] = RecordsDataset([])

    def prepare(self, *, eval_config: EvalConfig) -> Self:
        self._reset_prepared_state()
        self._eval_config = eval_config
        self._load_parsed_data(self._parser.parse())
        self._build_prepared_datasets()
        self._is_prepared = True
        return self

    def get_train_dataset(self) -> Dataset[TTrain]:
        self._require_prepared()
        return self._train_dataset

    def get_eval_dataset(self, split: Literal["valid", "test"]) -> Dataset[TEval]:
        self._require_prepared()
        return self._valid_dataset if split == "valid" else self._test_dataset

    def get_interactions(self) -> list[Interaction]:
        self._require_prepared()
        return self._interactions

    def get_user_table(self) -> pd.DataFrame:
        self._require_prepared()
        return self._user_table

    def get_item_table(self) -> pd.DataFrame:
        self._require_prepared()
        return self._item_table

    def get_num_users(self) -> int:
        self._require_prepared()
        return self._num_users

    def get_num_items(self) -> int:
        self._require_prepared()
        return self._num_items

    @abstractmethod
    def _build_split_records(
        self,
        ordered_interactions: list[Interaction],
    ) -> tuple[list[TTrain], list[TEval], list[TEval]]:
        ...

    def _load_parsed_data(self, parsed: ParsedData) -> None:
        self._load_entity_tables(parsed)
        self._load_interactions(parsed.interactions)

    def _load_entity_tables(self, parsed: ParsedData) -> None:
        self._user_table = self._normalize_entity_table(parsed.user_table, required_column="user_id")
        self._item_table = self._normalize_entity_table(parsed.item_table, required_column="item_id")
        self._num_users = int(len(self._user_table))
        self._num_items = int(len(self._item_table))

    def _load_interactions(self, interactions: list[Interaction]) -> None:
        ordered_interactions = self._order_interactions(interactions)
        self._validate_interactions(ordered_interactions)
        self._interactions = ordered_interactions

    def _build_prepared_datasets(self) -> None:
        train_records, valid_records, test_records = self._build_split_records(self._interactions)
        self._train_dataset = RecordsDataset(train_records)
        self._valid_dataset = RecordsDataset(valid_records)
        self._test_dataset = RecordsDataset(test_records)

    def _split_interactions(
        self,
        ordered_interactions: list[Interaction],
    ) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
        if not ordered_interactions:
            return [], [], []
        if self.config.split.per_user:
            return self._split_interactions_per_user(ordered_interactions)
        return self._split_interactions_group(ordered_interactions)

    def _split_interactions_per_user(
        self,
        ordered_interactions: list[Interaction],
    ) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
        train_records: list[Interaction] = []
        valid_records: list[Interaction] = []
        test_records: list[Interaction] = []
        for user_records in self._group_by_user(ordered_interactions).values():
            train_slice, valid_slice, test_slice = self._split_interactions_group(user_records)
            train_records.extend(train_slice)
            valid_records.extend(valid_slice)
            test_records.extend(test_slice)
        return train_records, valid_records, test_records

    def _split_interactions_group(
        self,
        interaction_group: list[Interaction],
    ) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
        train_end, valid_end = self._split_boundaries(len(interaction_group))
        return self._slice_interaction_group(interaction_group, train_end=train_end, valid_end=valid_end)

    @staticmethod
    def _slice_interaction_group(
        interaction_group: list[Interaction],
        *,
        train_end: int,
        valid_end: int,
    ) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
        return (
            interaction_group[:train_end],
            interaction_group[train_end:valid_end],
            interaction_group[valid_end:],
        )

    def _order_interactions(self, interactions: list[Interaction]) -> list[Interaction]:
        if not interactions:
            return []
        rng = np.random.default_rng(self.config.split.seed)
        if self.config.split.per_user:
            return self._order_interactions_per_user(interactions, rng)
        return self._sort_interaction_group(interactions, rng)

    def _order_interactions_per_user(
        self,
        interactions: list[Interaction],
        rng: np.random.Generator,
    ) -> list[Interaction]:
        ordered: list[Interaction] = []
        for group in self._group_by_user(interactions).values():
            ordered.extend(self._sort_interaction_group(group, rng))
        return ordered

    def _sort_interaction_group(
        self,
        interactions: list[Interaction],
        rng: np.random.Generator,
    ) -> list[Interaction]:
        order = self.config.split.order
        if order == "random":
            return self._shuffle_interaction_group(interactions, rng)
        if order == "chronological":
            return self._chronological_or_original_group(interactions)
        raise ValueError(f"Unsupported split order '{order}'.")

    @staticmethod
    def _shuffle_interaction_group(
        interactions: list[Interaction],
        rng: np.random.Generator,
    ) -> list[Interaction]:
        if len(interactions) <= 1:
            return list(interactions)
        indices = rng.permutation(len(interactions))
        return [interactions[int(index)] for index in indices]

    def _chronological_or_original_group(self, interactions: list[Interaction]) -> list[Interaction]:
        if self._has_complete_timestamps(interactions):
            return sorted(interactions, key=lambda record: record.timestamp)
        return list(interactions)

    def _split_boundaries(self, size: int) -> tuple[int, int]:
        strategy = self.config.split.strategy
        if strategy == "ratio":
            return ratio_boundaries(
                size,
                train_ratio=self.config.split.train_ratio,
                valid_ratio=self.config.split.valid_ratio,
                test_ratio=self.config.split.test_ratio,
            )
        if strategy == "leave_one_out":
            return leave_one_out_boundaries(
                size,
                valid_holdout_num=self.config.split.valid_holdout_num,
                test_holdout_num=self.config.split.test_holdout_num,
            )
        raise ValueError(f"Unsupported split strategy '{strategy}'.")

    def _validate_interactions(self, interactions: list[Interaction]) -> None:
        if not interactions:
            return
        max_user_id = max(record.user_id for record in interactions)
        max_item_id = max(record.item_id for record in interactions)
        if max_user_id >= self._num_users:
            raise ValueError(
                f"Interaction user_id range exceeds user_table size: max user_id={max_user_id}, num_users={self._num_users}."
            )
        if max_item_id >= self._num_items:
            raise ValueError(
                f"Interaction item_id range exceeds item_table size: max item_id={max_item_id}, num_items={self._num_items}."
            )

    @staticmethod
    def _normalize_entity_table(table: pd.DataFrame, *, required_column: str) -> pd.DataFrame:
        normalized = table.reset_index(drop=True).copy()
        if required_column not in normalized.columns:
            raise ValueError(f"Parsed data must include '{required_column}' in the corresponding entity table.")
        return normalized

    def _require_prepared(self) -> None:
        if not self._is_prepared:
            raise RuntimeError(f"{type(self).__name__} must be prepared before data can be accessed.")

    def _require_eval_config(self) -> EvalConfig:
        if self._eval_config is None:
            raise RuntimeError(f"{type(self).__name__} must receive eval_config before preparing.")
        return self._eval_config

    def _reset_prepared_state(self) -> None:
        self._eval_config = None
        self._is_prepared = False
        self._interactions = []
        self._user_table = pd.DataFrame()
        self._item_table = pd.DataFrame()
        self._num_users = 0
        self._num_items = 0
        self._train_dataset = RecordsDataset([])
        self._valid_dataset = RecordsDataset([])
        self._test_dataset = RecordsDataset([])

    @classmethod
    def _require_parser_cls(cls) -> type[BaseDatasetParser]:
        parser_cls = cls.parser_cls
        if parser_cls is None:
            raise TypeError(f"{cls.__name__} must define parser_cls.")
        return parser_cls

    @staticmethod
    def _group_by_user(interactions: list[Interaction]) -> dict[int, list[Interaction]]:
        grouped: dict[int, list[Interaction]] = {}
        for interaction in interactions:
            grouped.setdefault(int(interaction.user_id), []).append(interaction)
        return grouped

    @staticmethod
    def _has_complete_timestamps(interactions: list[Interaction]) -> bool:
        return all(record.timestamp is not None for record in interactions)


class RankingDataset(BaseTaskDataset[Interaction, Interaction]):
    """Task dataset that keeps row-based records for all splits."""

    task: DatasetTask = "ranking"

    def _build_split_records(
        self,
        ordered_interactions: list[Interaction],
    ) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
        protocol = self._require_eval_config().protocol
        if protocol != "labeled":
            raise ValueError(f"Ranking datasets only support eval protocol 'labeled', got '{protocol}'.")
        return self._split_interactions(ordered_interactions)


class RetrievalDataset(BaseTaskDataset[Interaction, RetrievalEvalRequest]):
    """Task dataset that uses request-level records for retrieval evaluation."""

    task: DatasetTask = "retrieval"

    def _build_split_records(
        self,
        ordered_interactions: list[Interaction],
    ) -> tuple[list[Interaction], list[RetrievalEvalRequest], list[RetrievalEvalRequest]]:
        self._require_retrieval_protocol()
        train_records, valid_interactions, test_interactions = self._split_interactions(ordered_interactions)
        valid_records = self._build_eval_records(
            valid_interactions,
            seen_history_interactions=train_records,
            split="valid",
        )
        test_records = self._build_eval_records(
            test_interactions,
            seen_history_interactions=train_records + self._positive_interactions(valid_interactions),
            split="test",
        )
        return train_records, valid_records, test_records

    def _require_retrieval_protocol(self) -> None:
        protocol = self._require_eval_config().protocol
        if protocol not in {"full", "sampled"}:
            raise ValueError(
                f"Retrieval datasets only support eval protocols 'full' and 'sampled', got '{protocol}'."
            )

    def _build_eval_records(
        self,
        interactions: list[Interaction],
        *,
        seen_history_interactions: list[Interaction],
        split: Literal["valid", "test"],
    ) -> list[RetrievalEvalRequest]:
        positive_interactions = self._positive_interactions(interactions)
        requests = self._build_retrieval_eval_requests(
            positive_interactions,
            seen_history_interactions=self._positive_interactions(seen_history_interactions),
        )
        return self._maybe_attach_sampled_candidates(requests, split=split)

    def _maybe_attach_sampled_candidates(
        self,
        requests: list[RetrievalEvalRequest],
        *,
        split: Literal["valid", "test"],
    ) -> list[RetrievalEvalRequest]:
        protocol = self._require_eval_config().protocol
        if protocol != "sampled":
            return requests
        return self._attach_sampled_candidates(requests, split=split)

    @staticmethod
    def _positive_interactions(interactions: list[Interaction]) -> list[Interaction]:
        positive: list[Interaction] = []
        for interaction in interactions:
            if interaction.label is not None and float(interaction.label) <= 0:
                continue
            positive.append(interaction)
        return positive

    def _build_retrieval_eval_requests(
        self,
        positive_interactions: list[Interaction],
        *,
        seen_history_interactions: list[Interaction],
    ) -> list[RetrievalEvalRequest]:
        if not positive_interactions:
            return []
        seen_histories = self._group_unique_item_sequences(seen_history_interactions)
        positive_groups = self._group_by_user(positive_interactions)
        requests: list[RetrievalEvalRequest] = []
        for user_id, user_interactions in positive_groups.items():
            requests.extend(
                self._build_user_eval_requests(
                    positive_interactions=user_interactions,
                    seen_item_ids=seen_histories.get(user_id, ()),
                )
            )
        return requests

    @staticmethod
    def _build_user_eval_requests(
        *,
        positive_interactions: list[Interaction],
        seen_item_ids: tuple[int, ...],
    ) -> list[RetrievalEvalRequest]:
        if not positive_interactions:
            return []
        history = list(seen_item_ids)
        seen_item_set = set(history)
        requests: list[RetrievalEvalRequest] = []
        for interaction in positive_interactions:
            item_id = int(interaction.item_id)
            requests.append(
                RetrievalEvalRequest(
                    user_id=int(interaction.user_id),
                    item_id=item_id,
                    timestamp=interaction.timestamp,
                    label=interaction.label,
                    seen_item_ids=tuple(history),
                )
            )
            if item_id not in seen_item_set:
                history.append(item_id)
                seen_item_set.add(item_id)
        return requests

    def _attach_sampled_candidates(
        self,
        requests: list[RetrievalEvalRequest],
        *,
        split: Literal["valid", "test"],
    ) -> list[RetrievalEvalRequest]:
        return [
            self._attach_sampled_candidates_to_request(request, split=split, record_index=index)
            for index, request in enumerate(requests)
        ]

    def _attach_sampled_candidates_to_request(
        self,
        request: RetrievalEvalRequest,
        *,
        split: Literal["valid", "test"],
        record_index: int,
    ) -> RetrievalEvalRequest:
        return RetrievalEvalRequest(
            user_id=request.user_id,
            item_id=request.item_id,
            timestamp=request.timestamp,
            label=request.label,
            seen_item_ids=request.seen_item_ids,
            candidate_item_ids=(int(request.item_id),)
            + self._sample_negative_item_ids(
                user_id=request.user_id,
                target_item_id=int(request.item_id),
                split=split,
                record_index=record_index,
            ),
        )

    def _sample_negative_item_ids(
        self,
        *,
        user_id: int,
        target_item_id: int,
        split: Literal["valid", "test"],
        record_index: int,
    ) -> tuple[int, ...]:
        negative_pool = self._build_negative_pool(target_item_id)
        sample_size = self._negative_sample_size(len(negative_pool))
        if sample_size == 0:
            return ()
        if sample_size == len(negative_pool):
            return tuple(int(item_id) for item_id in negative_pool.tolist())

        sampled_negative_item_ids = np.random.default_rng(
            self._sample_seed(user_id=user_id, split=split, record_index=record_index)
        ).choice(negative_pool, size=sample_size, replace=False)
        return tuple(int(item_id) for item_id in sampled_negative_item_ids.tolist())

    def _build_negative_pool(self, target_item_id: int) -> np.ndarray:
        available_count = max(0, self._num_items - 1)
        return np.fromiter(
            (item_id for item_id in range(self._num_items) if item_id != target_item_id),
            dtype=np.int64,
            count=available_count,
        )

    def _negative_sample_size(self, available_count: int) -> int:
        eval_config = self._require_eval_config()
        return min(max(0, int(eval_config.neg_sampling_num)), available_count)

    def _sample_seed(
        self,
        *,
        user_id: int,
        split: Literal["valid", "test"],
        record_index: int,
    ) -> int:
        eval_config = self._require_eval_config()
        split_offset = 0 if split == "valid" else 10_000
        return int(eval_config.candidate_seed) + int(user_id) + split_offset + int(record_index)

    @staticmethod
    def _group_item_sequences(interactions: list[Interaction]) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for interaction in interactions:
            grouped.setdefault(int(interaction.user_id), []).append(int(interaction.item_id))
        return grouped

    @classmethod
    def _group_unique_item_sequences(cls, interactions: list[Interaction]) -> dict[int, tuple[int, ...]]:
        grouped = cls._group_item_sequences(interactions)
        unique_sequences: dict[int, tuple[int, ...]] = {}
        for user_id, item_ids in grouped.items():
            unique_sequences[user_id] = cls._unique_item_sequence(item_ids)
        return unique_sequences

    @staticmethod
    def _unique_item_sequence(item_ids: list[int]) -> tuple[int, ...]:
        seen_item_ids: list[int] = []
        seen_item_set: set[int] = set()
        for item_id in item_ids:
            if item_id in seen_item_set:
                continue
            seen_item_ids.append(item_id)
            seen_item_set.add(item_id)
        return tuple(seen_item_ids)


def ratio_boundaries(
    size: int,
    *,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    ratios = np.asarray([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = float(np.sum(ratios))
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")

    expected_counts = ratios / ratio_sum * float(size)
    split_counts = np.floor(expected_counts).astype(np.int64)
    remainder = int(size - int(np.sum(split_counts)))
    if remainder > 0:
        fractional = expected_counts - split_counts
        for split_index in np.argsort(-fractional, kind="mergesort")[:remainder]:
            split_counts[int(split_index)] += 1

    train_count = int(split_counts[0])
    valid_count = int(split_counts[1])
    return train_count, train_count + valid_count


def leave_one_out_boundaries(
    size: int,
    *,
    valid_holdout_num: int,
    test_holdout_num: int,
) -> tuple[int, int]:
    test_size = min(size, int(test_holdout_num))
    valid_size = min(size - test_size, int(valid_holdout_num))
    train_end = max(0, size - valid_size - test_size)
    return train_end, train_end + valid_size


__all__ = [
    "BaseDatasetParser",
    "BaseTaskDataset",
    "DatasetConfig",
    "DatasetTask",
    "Interaction",
    "ParsedData",
    "RankingDataset",
    "RecordsDataset",
    "RetrievalDataset",
    "RetrievalEvalRequest",
    "SplitConfig",
    "SplitName",
    "leave_one_out_boundaries",
    "ratio_boundaries",
]
