from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from recbole3.dataset.config import DatasetConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import (
    CANDIDATE_ITEM_IDS,
    ITEM_ID,
    LABEL,
    SEEN_ITEM_IDS,
    TIMESTAMP,
    USER_ID,
    FrameSchema,
    require_columns,
)

if TYPE_CHECKING:
    try:
        from typing import Self
    except ImportError:  # pragma: no cover - Python < 3.11
        from typing_extensions import Self

    from recbole3.evaluation.config import EvalConfig


DatasetTask = Literal["ranking", "retrieval"]

PARSER_INTERACTIONS_SCHEMA = FrameSchema(
    required=(USER_ID, ITEM_ID),
    optional=(TIMESTAMP, LABEL),
)
PREPARED_INTERACTIONS_SCHEMA = FrameSchema(
    required=(USER_ID, ITEM_ID),
    optional=(TIMESTAMP, LABEL),
)
RETRIEVAL_EVAL_SCHEMA = FrameSchema(
    required=(USER_ID, ITEM_ID, SEEN_ITEM_IDS),
    optional=(TIMESTAMP, LABEL, CANDIDATE_ITEM_IDS),
)


class FrameDataset(Dataset[pd.DataFrame | dict[str, Any]]):
    """Map-style Dataset backed by a DataFrame."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int | list[int] | tuple[int, ...] | np.ndarray) -> dict[str, Any] | pd.DataFrame:
        if isinstance(index, (list, tuple, np.ndarray)):
            return self.__getitems__(index)
        return self.frame.iloc[int(index)].to_dict()

    def __getitems__(self, indices: list[int] | tuple[int, ...] | np.ndarray) -> pd.DataFrame:
        return self.frame.take([int(index) for index in indices]).reset_index(drop=True)


class BaseTaskDataset(ABC):
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
        self._interactions = pd.DataFrame()
        self._user_table = pd.DataFrame()
        self._item_table = pd.DataFrame()
        self._num_users = 0
        self._num_items = 0
        self._train_dataset: Dataset[Any] = FrameDataset(pd.DataFrame())
        self._valid_dataset: Dataset[Any] = FrameDataset(pd.DataFrame())
        self._test_dataset: Dataset[Any] = FrameDataset(pd.DataFrame())

    def prepare(self, *, eval_config: EvalConfig) -> Self:
        self._reset_prepared_state()
        self._eval_config = eval_config
        self._load_parsed_data(self._parser.parse())
        self._build_prepared_datasets()
        self._is_prepared = True
        return self

    def get_train_dataset(self) -> Dataset[Any]:
        self._require_prepared()
        return self._train_dataset

    def get_eval_dataset(self, split: Literal["valid", "test"]) -> Dataset[Any]:
        self._require_prepared()
        return self._valid_dataset if split == "valid" else self._test_dataset

    def get_interactions(self) -> pd.DataFrame:
        self._require_prepared()
        return self._interactions.copy()

    def get_user_table(self) -> pd.DataFrame:
        self._require_prepared()
        return self._user_table.copy()

    def get_item_table(self) -> pd.DataFrame:
        self._require_prepared()
        return self._item_table.copy()

    def get_num_users(self) -> int:
        self._require_prepared()
        return self._num_users

    def get_num_items(self) -> int:
        self._require_prepared()
        return self._num_items

    @abstractmethod
    def _build_split_frames(
        self,
        ordered_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ...

    @staticmethod
    def _normalize_parser_interactions(
        interactions: pd.DataFrame,
        *,
        schema: FrameSchema,
        user_column: str,
        item_column: str,
        optional_columns: Sequence[str],
        name: str,
    ) -> pd.DataFrame:
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame, got {type(interactions).__name__}.")
        normalized = interactions.reset_index(drop=True).copy()
        require_columns(normalized, schema, name=name)
        if normalized[[user_column, item_column]].isna().any().any():
            raise ValueError(f"{name} requires non-null {user_column} and {item_column} values.")
        for column in optional_columns:
            if column not in normalized.columns:
                normalized[column] = None
        return normalized

    def _normalize_entity_table(
        self,
        table: pd.DataFrame | None,
        *,
        key_column: str,
        fallback_values: pd.Series,
        name: str,
    ) -> pd.DataFrame:
        fallback_keys = self._unique_values(fallback_values)
        if table is None:
            return pd.DataFrame({key_column: fallback_keys})
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame, got {type(table).__name__}.")
        if table.empty:
            return pd.DataFrame({key_column: fallback_keys})
        normalized = table.reset_index(drop=True).copy()
        require_columns(normalized, FrameSchema(required=(key_column,)), name=name)
        if normalized[key_column].isna().any():
            raise ValueError(f"{name} requires non-null {key_column} values.")
        if normalized[key_column].duplicated().any():
            raise ValueError(f"{name} requires unique {key_column} values.")
        known_keys = set(normalized[key_column].tolist())
        missing_keys = [key for key in fallback_keys if key not in known_keys]
        if not missing_keys:
            return normalized
        missing_rows = pd.DataFrame({key_column: missing_keys})
        return pd.concat([normalized, missing_rows], ignore_index=True, sort=False)

    @staticmethod
    def _build_id_map(keys: pd.Series, *, start: int, name: str) -> dict[Any, int]:
        if keys.duplicated().any():
            raise ValueError(f"Cannot build id map from duplicate {name} values.")
        return {key: index for index, key in enumerate(keys.tolist(), start=start)}

    @staticmethod
    def _unique_values(values: pd.Series) -> list[Any]:
        return list(pd.unique(values))

    def _load_parsed_data(self, parsed: ParsedData) -> None:
        if not isinstance(parsed, ParsedData):
            raise TypeError(f"Parser must return ParsedData, got {type(parsed).__name__}.")
        raw_interactions = self._normalize_parser_interactions(
            parsed.interactions,
            schema=PARSER_INTERACTIONS_SCHEMA,
            user_column=USER_ID,
            item_column=ITEM_ID,
            optional_columns=(TIMESTAMP, LABEL),
            name="ParsedData.interactions",
        )
        self._load_entity_tables(parsed, raw_interactions=raw_interactions)
        self._load_interactions(raw_interactions)

    def _load_entity_tables(self, parsed: ParsedData, *, raw_interactions: pd.DataFrame) -> None:
        raw_user_table = self._normalize_entity_table(
            parsed.user_table,
            key_column=USER_ID,
            fallback_values=raw_interactions[USER_ID],
            name="ParsedData.user_table",
        )
        raw_item_table = self._normalize_entity_table(
            parsed.item_table,
            key_column=ITEM_ID,
            fallback_values=raw_interactions[ITEM_ID],
            name="ParsedData.item_table",
        )
        user_id_map = self._build_id_map(raw_user_table[USER_ID], start=0, name=USER_ID)
        item_id_map = self._build_id_map(raw_item_table[ITEM_ID], start=0, name=ITEM_ID)

        self._user_table = raw_user_table.copy()
        self._user_table[USER_ID] = self._user_table[USER_ID].map(user_id_map).astype("int64")

        self._item_table = raw_item_table.copy()
        self._item_table[ITEM_ID] = self._item_table[ITEM_ID].map(item_id_map).astype("int64")

        self._num_users = int(len(self._user_table))
        self._num_items = int(len(self._item_table))
        self._user_id_map = user_id_map
        self._item_id_map = item_id_map

    def _load_interactions(self, raw_interactions: pd.DataFrame) -> None:
        interactions = raw_interactions.copy()
        interactions[USER_ID] = interactions[USER_ID].map(self._user_id_map)
        interactions[ITEM_ID] = interactions[ITEM_ID].map(self._item_id_map)
        if interactions[[USER_ID, ITEM_ID]].isna().any().any():
            raise ValueError("Parsed interactions contain ids that could not be mapped.")
        interactions[USER_ID] = interactions[USER_ID].astype("int64")
        interactions[ITEM_ID] = interactions[ITEM_ID].astype("int64")
        ordered_interactions = self._order_interactions(interactions)
        self._validate_interactions(ordered_interactions)
        self._interactions = ordered_interactions.reset_index(drop=True)

    def _build_prepared_datasets(self) -> None:
        train_frame, valid_frame, test_frame = self._build_split_frames(self._interactions)
        self._train_dataset = FrameDataset(train_frame)
        self._valid_dataset = FrameDataset(valid_frame)
        self._test_dataset = FrameDataset(test_frame)

    def _split_interactions(self, ordered_interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if ordered_interactions.empty:
            empty = self._empty_like(ordered_interactions)
            return empty, empty, empty
        if self.config.split.per_user:
            return self._split_interactions_per_user(ordered_interactions)
        return self._split_interactions_group(ordered_interactions)

    def _split_interactions_per_user(
        self,
        ordered_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_frames: list[pd.DataFrame] = []
        valid_frames: list[pd.DataFrame] = []
        test_frames: list[pd.DataFrame] = []
        for _, user_records in ordered_interactions.groupby(USER_ID, sort=False):
            train_slice, valid_slice, test_slice = self._split_interactions_group(user_records)
            train_frames.append(train_slice)
            valid_frames.append(valid_slice)
            test_frames.append(test_slice)
        return (
            self._concat_like(train_frames, ordered_interactions),
            self._concat_like(valid_frames, ordered_interactions),
            self._concat_like(test_frames, ordered_interactions),
        )

    def _split_interactions_group(
        self,
        interaction_group: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_end, valid_end = self._split_boundaries(len(interaction_group))
        return self._slice_interaction_group(interaction_group, train_end=train_end, valid_end=valid_end)

    @staticmethod
    def _slice_interaction_group(
        interaction_group: pd.DataFrame,
        *,
        train_end: int,
        valid_end: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            interaction_group.iloc[:train_end].copy(),
            interaction_group.iloc[train_end:valid_end].copy(),
            interaction_group.iloc[valid_end:].copy(),
        )

    def _order_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if interactions.empty:
            return interactions.reset_index(drop=True)
        rng = np.random.default_rng(self.config.split.seed)
        if self.config.split.per_user:
            return self._order_interactions_per_user(interactions, rng)
        return self._sort_interaction_group(interactions, rng).reset_index(drop=True)

    def _order_interactions_per_user(
        self,
        interactions: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        groups = [self._sort_interaction_group(group, rng) for _, group in interactions.groupby(USER_ID, sort=False)]
        return self._concat_like(groups, interactions)

    def _sort_interaction_group(
        self,
        interactions: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if self.config.split.order == "random":
            return self._shuffle_interaction_group(interactions, rng)
        if self.config.split.order == "chronological":
            return self._chronological_or_original_group(interactions)
        raise ValueError(f"Unsupported split order '{self.config.split.order}'.")

    @staticmethod
    def _shuffle_interaction_group(interactions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if len(interactions) <= 1:
            return interactions.copy()
        indices = rng.permutation(len(interactions))
        return interactions.iloc[indices].copy()

    def _chronological_or_original_group(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if self._has_complete_timestamps(interactions):
            return interactions.sort_values(TIMESTAMP, kind="mergesort").copy()
        return interactions.copy()

    def _split_boundaries(self, size: int) -> tuple[int, int]:
        if self.config.split.strategy == "ratio":
            return self._ratio_boundaries(
                size,
                train_ratio=self.config.split.train_ratio,
                valid_ratio=self.config.split.valid_ratio,
                test_ratio=self.config.split.test_ratio,
            )
        if self.config.split.strategy == "leave_one_out":
            return self._leave_one_out_boundaries(
                size,
                valid_holdout_num=self.config.split.valid_holdout_num,
                test_holdout_num=self.config.split.test_holdout_num,
            )
        raise ValueError(f"Unsupported split strategy '{self.config.split.strategy}'.")

    @staticmethod
    def _ratio_boundaries(
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

    @staticmethod
    def _leave_one_out_boundaries(
        size: int,
        *,
        valid_holdout_num: int,
        test_holdout_num: int,
    ) -> tuple[int, int]:
        test_size = min(size, int(test_holdout_num))
        valid_size = min(size - test_size, int(valid_holdout_num))
        train_end = max(0, size - valid_size - test_size)
        return train_end, train_end + valid_size

    def _validate_interactions(self, interactions: pd.DataFrame) -> None:
        if interactions.empty:
            return
        min_user_id = int(interactions[USER_ID].min())
        max_user_id = int(interactions[USER_ID].max())
        min_item_id = int(interactions[ITEM_ID].min())
        max_item_id = int(interactions[ITEM_ID].max())
        if min_user_id < 0 or max_user_id >= self._num_users:
            raise ValueError(
                f"Prepared interactions user_id range exceeds user_table size: min user_id={min_user_id}, "
                f"max user_id={max_user_id}, num_users={self._num_users}."
            )
        if min_item_id < 0 or max_item_id >= self._num_items:
            raise ValueError(
                f"Prepared interactions item_id range must be real item ids in [0, {self._num_items - 1}], "
                f"got min item_id={min_item_id}, max item_id={max_item_id}."
            )

    @staticmethod
    def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.iloc[0:0].copy()

    @classmethod
    def _concat_like(cls, frames: list[pd.DataFrame], like: pd.DataFrame) -> pd.DataFrame:
        non_empty = [frame for frame in frames if not frame.empty]
        if not non_empty:
            return cls._empty_like(like)
        return pd.concat(non_empty, ignore_index=True, sort=False)

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
        self._interactions = pd.DataFrame()
        self._user_table = pd.DataFrame()
        self._item_table = pd.DataFrame()
        self._num_users = 0
        self._num_items = 0
        self._train_dataset = FrameDataset(pd.DataFrame())
        self._valid_dataset = FrameDataset(pd.DataFrame())
        self._test_dataset = FrameDataset(pd.DataFrame())

    @classmethod
    def _require_parser_cls(cls) -> type[BaseDatasetParser]:
        parser_cls = cls.parser_cls
        if parser_cls is None:
            raise TypeError(f"{cls.__name__} must define parser_cls.")
        return parser_cls

    @staticmethod
    def _has_complete_timestamps(interactions: pd.DataFrame) -> bool:
        return TIMESTAMP in interactions.columns and bool(interactions[TIMESTAMP].notna().all())


class RankingDataset(BaseTaskDataset):
    """Task dataset that keeps row-based interaction frames for all splits."""

    task: DatasetTask = "ranking"

    def _build_split_frames(
        self,
        ordered_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        protocol = self._require_eval_config().protocol
        if protocol != "labeled":
            raise ValueError(f"Ranking datasets only support eval protocol 'labeled', got '{protocol}'.")
        return self._split_interactions(ordered_interactions)


class RetrievalDataset(BaseTaskDataset):
    """Task dataset that uses request-level frames for retrieval evaluation."""

    task: DatasetTask = "retrieval"

    def _build_split_frames(
        self,
        ordered_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._require_retrieval_protocol()
        train_frame, valid_interactions, test_interactions = self._split_interactions(ordered_interactions)
        valid_frame = self._build_eval_frame(
            valid_interactions,
            seen_history_interactions=train_frame,
            split="valid",
        )
        test_frame = self._build_eval_frame(
            test_interactions,
            seen_history_interactions=self._concat_like(
                [train_frame, self._positive_interactions(valid_interactions)],
                ordered_interactions,
            ),
            split="test",
        )
        return train_frame, valid_frame, test_frame

    def _require_retrieval_protocol(self) -> None:
        protocol = self._require_eval_config().protocol
        if protocol not in {"full", "sampled"}:
            raise ValueError(
                f"Retrieval datasets only support eval protocols 'full' and 'sampled', got '{protocol}'."
            )

    def _build_eval_frame(
        self,
        interactions: pd.DataFrame,
        *,
        seen_history_interactions: pd.DataFrame,
        split: Literal["valid", "test"],
    ) -> pd.DataFrame:
        positive_interactions = self._positive_interactions(interactions)
        requests = self._build_retrieval_eval_frame(
            positive_interactions,
            seen_history_interactions=self._positive_interactions(seen_history_interactions),
        )
        return self._maybe_attach_sampled_candidates(requests, split=split)

    def _maybe_attach_sampled_candidates(
        self,
        requests: pd.DataFrame,
        *,
        split: Literal["valid", "test"],
    ) -> pd.DataFrame:
        if self._require_eval_config().protocol != "sampled":
            return requests
        return self._attach_sampled_candidates(requests, split=split)

    @staticmethod
    def _positive_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
        if interactions.empty:
            return interactions.copy()
        labels = pd.to_numeric(interactions[LABEL], errors="coerce")
        positive_mask = interactions[LABEL].isna() | (labels > 0)
        return interactions.loc[positive_mask].copy()

    def _build_retrieval_eval_frame(
        self,
        positive_interactions: pd.DataFrame,
        *,
        seen_history_interactions: pd.DataFrame,
    ) -> pd.DataFrame:
        if positive_interactions.empty:
            columns = list(positive_interactions.columns)
            if SEEN_ITEM_IDS not in columns:
                columns.append(SEEN_ITEM_IDS)
            return pd.DataFrame(columns=columns)
        result = positive_interactions.copy()
        seen_histories = self._group_unique_item_sequences(seen_history_interactions)
        item_ids = result[ITEM_ID].to_numpy()
        seen_item_ids: list[tuple[int, ...]] = [()] * len(result)
        ordered_positions: list[int] = []
        for user_id, positions in result.groupby(USER_ID, sort=False).indices.items():
            history = list(seen_histories.get(int(user_id), ()))
            seen_item_set = set(history)
            for position in positions:
                row_position = int(position)
                seen_item_ids[row_position] = tuple(history)
                item_id = int(item_ids[row_position])
                if item_id not in seen_item_set:
                    history.append(item_id)
                    seen_item_set.add(item_id)
                ordered_positions.append(row_position)
        result[SEEN_ITEM_IDS] = seen_item_ids
        return result.take(ordered_positions).reset_index(drop=True)

    def _attach_sampled_candidates(
        self,
        requests: pd.DataFrame,
        *,
        split: Literal["valid", "test"],
    ) -> pd.DataFrame:
        if requests.empty:
            result = requests.copy()
            result[CANDIDATE_ITEM_IDS] = pd.Series(dtype=object)
            return result
        result = requests.copy()
        result[CANDIDATE_ITEM_IDS] = [
            (int(item_id),)
            + self._sample_negative_item_ids(
                user_id=int(user_id),
                target_item_id=int(item_id),
                split=split,
                record_index=index,
            )
            for index, (user_id, item_id) in enumerate(zip(result[USER_ID].to_numpy(), result[ITEM_ID].to_numpy()))
        ]
        self._validate_sampled_candidates(result)
        return result

    def _sample_negative_item_ids(
        self,
        *,
        user_id: int,
        target_item_id: int,
        split: Literal["valid", "test"],
        record_index: int,
    ) -> tuple[int, ...]:
        target_item_id = int(target_item_id)
        available_count = max(0, self._num_items - 1)
        sample_size = self._negative_sample_size(available_count)
        if sample_size == 0:
            return ()
        if sample_size == available_count:
            return self._all_negative_item_ids(target_item_id)

        sampled_offsets = np.random.default_rng(
            self._sample_seed(user_id=user_id, split=split, record_index=record_index)
        ).choice(available_count, size=sample_size, replace=False)
        sampled_negative_item_ids = sampled_offsets
        sampled_negative_item_ids[sampled_negative_item_ids >= target_item_id] += 1
        return tuple(int(item_id) for item_id in sampled_negative_item_ids.tolist())

    def _all_negative_item_ids(self, target_item_id: int) -> tuple[int, ...]:
        return tuple(range(0, target_item_id)) + tuple(range(target_item_id + 1, self._num_items))

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
    def _group_unique_item_sequences(interactions: pd.DataFrame) -> dict[int, tuple[int, ...]]:
        if interactions.empty:
            return {}
        unique_interactions = interactions.loc[:, [USER_ID, ITEM_ID]].drop_duplicates([USER_ID, ITEM_ID], keep="first")
        grouped_item_ids = unique_interactions.groupby(USER_ID, sort=False)[ITEM_ID].agg(tuple)
        return {
            int(user_id): tuple(int(item_id) for item_id in item_ids)
            for user_id, item_ids in grouped_item_ids.items()
        }

    @staticmethod
    def _validate_sampled_candidates(requests: pd.DataFrame) -> None:
        if requests.empty:
            return
        counts = requests[CANDIDATE_ITEM_IDS].map(len)
        if counts.nunique(dropna=False) > 1:
            raise ValueError("Sampled evaluation requires each candidate_item_ids row to have the same length.")


__all__ = [
    "BaseDatasetParser",
    "BaseTaskDataset",
    "DatasetConfig",
    "DatasetTask",
    "FrameDataset",
    "FrameSchema",
    "PARSER_INTERACTIONS_SCHEMA",
    "PREPARED_INTERACTIONS_SCHEMA",
    "ParsedData",
    "RankingDataset",
    "RETRIEVAL_EVAL_SCHEMA",
    "RetrievalDataset",
    "require_columns",
]
