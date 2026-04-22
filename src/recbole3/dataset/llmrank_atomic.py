from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from recbole3.config import project_root
from recbole3.dataset.base import FrameDataset, RetrievalDataset
from recbole3.dataset.config import DatasetConfig, SplitConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import CANDIDATE_ITEM_IDS, ITEM_ID, LABEL, TIMESTAMP, USER_ID


LLMRankAtomicSource = Literal["single_inter", "pre_split_test"]


@dataclass(slots=True)
class LLMRankAtomicDatasetConfig(DatasetConfig):
    """Dataset config for atomic RecBole-style files used by the original LLMRank project."""

    root_dir: str = field(
        default="../llmrank/data",
        metadata={"help": "Root directory that stores the sibling llmrank data project."},
    )
    dataset_dir_name: str = field(default="", metadata={"help": "Dataset subdirectory inside root_dir."})
    atomic_subdir: str = field(default="atomic", metadata={"help": "Subdirectory that stores extracted atomic files."})
    source_format: LLMRankAtomicSource = field(
        default="single_inter",
        metadata={"help": "Whether the source uses one .inter file or pre-split test records."},
    )
    interactions_filename: str | None = field(
        default=None,
        metadata={"help": "Atomic interactions filename for single_inter datasets."},
    )
    train_filename: str | None = field(
        default=None,
        metadata={"help": "Optional train split filename for pre_split_test datasets."},
    )
    valid_filename: str | None = field(
        default=None,
        metadata={"help": "Optional valid split filename for pre_split_test datasets."},
    )
    test_filename: str | None = field(
        default=None,
        metadata={"help": "Test split filename for pre_split_test datasets."},
    )
    item_filename: str = field(default="", metadata={"help": "Atomic item metadata filename."})
    candidate_filename: str | None = field(
        default=None,
        metadata={"help": "Optional external candidate ranking file such as *.bm25."},
    )
    candidate_budget: int = field(
        default=20,
        metadata={"help": "Maximum number of external candidates kept per user from the candidate file."},
    )
    use_external_candidates_for_valid: bool = field(
        default=False,
        metadata={"help": "Whether valid split should also use the external candidate lists."},
    )
    drop_missing_candidate_users: bool = field(
        default=True,
        metadata={"help": "Whether eval users without external candidates are removed from eval splits."},
    )
    split: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            strategy="leave_one_out",
            order="chronological",
            per_user=True,
            valid_holdout_num=1,
            test_holdout_num=1,
        ),
        metadata={"help": "Dataset split configuration."},
    )


@dataclass(slots=True)
class ML1MLLMRankDatasetConfig(LLMRankAtomicDatasetConfig):
    name: str = field(default="ml1m_llmrank", metadata={"help": "ML-1M dataset name."})
    dataset_dir_name: str = field(default="ml-1m", metadata={"help": "Dataset subdirectory name."})
    source_format: LLMRankAtomicSource = field(default="single_inter", metadata={"help": "ML-1M source format."})
    interactions_filename: str = field(default="ml-1m.inter", metadata={"help": "ML-1M atomic interactions file."})
    item_filename: str = field(default="ml-1m.item", metadata={"help": "ML-1M atomic item file."})
    candidate_filename: str = field(default="ml-1m.bm25", metadata={"help": "ML-1M BM25 candidate file."})


@dataclass(slots=True)
class GamesLLMRankDatasetConfig(LLMRankAtomicDatasetConfig):
    name: str = field(default="games_llmrank", metadata={"help": "Amazon Games dataset name."})
    dataset_dir_name: str = field(default="Games", metadata={"help": "Dataset subdirectory name."})
    source_format: LLMRankAtomicSource = field(default="pre_split_test", metadata={"help": "Games source format."})
    train_filename: str = field(default="Games.train.inter", metadata={"help": "Games train split file."})
    valid_filename: str = field(default="Games.valid.inter", metadata={"help": "Games valid split file."})
    test_filename: str = field(default="Games.test.inter", metadata={"help": "Games test split file."})
    item_filename: str = field(default="Games.item", metadata={"help": "Games atomic item file."})
    candidate_filename: str = field(default="Games.bm25", metadata={"help": "Games BM25 candidate file."})


class LLMRankAtomicParser(BaseDatasetParser):
    """Parser for RecBole-style atomic files used by the original LLMRank project."""

    config_cls = LLMRankAtomicDatasetConfig
    config: LLMRankAtomicDatasetConfig

    @property
    def data_dir(self) -> Path:
        root = Path(self.config.root_dir)
        if not root.is_absolute():
            root = (project_root() / root).resolve()
        return root / self.config.dataset_dir_name / self.config.atomic_subdir

    def parse(self) -> ParsedData:
        interactions = self._load_interactions()
        item_frame = self._load_item_frame()
        user_table = self._build_user_table(interactions)
        item_table = self._build_item_table(item_frame, interactions)
        return ParsedData(
            interactions=interactions,
            user_table=user_table,
            item_table=item_table,
        )

    def _load_interactions(self) -> pd.DataFrame:
        if self.config.source_format == "single_inter":
            return self._load_single_interactions()
        if self.config.source_format == "pre_split_test":
            return self._reconstruct_interactions_from_test_split()
        raise ValueError(f"Unsupported source_format '{self.config.source_format}'.")

    def _load_single_interactions(self) -> pd.DataFrame:
        frame = _read_atomic_frame(self._require_atomic_path(self.config.interactions_filename))
        return pd.DataFrame(
            {
                USER_ID: [_normalize_raw_token(value) for value in frame["user_id"].tolist()],
                ITEM_ID: [_normalize_raw_token(value) for value in frame["item_id"].tolist()],
                TIMESTAMP: [_optional_numeric(value) for value in frame.get("timestamp", pd.Series([None] * len(frame))).tolist()],
                LABEL: [1.0] * len(frame),
            }
        )

    def _reconstruct_interactions_from_test_split(self) -> pd.DataFrame:
        frame = _read_atomic_frame(self._require_atomic_path(self.config.test_filename))
        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            raw_user_id = _normalize_raw_token(getattr(row, "user_id"))
            history_item_ids = _parse_token_seq(getattr(row, "item_id_list", ""))
            target_item_id = _normalize_raw_token(getattr(row, "item_id"))
            full_sequence = [*history_item_ids, target_item_id]
            for position, raw_item_id in enumerate(full_sequence, start=1):
                rows.append(
                    {
                        USER_ID: raw_user_id,
                        ITEM_ID: raw_item_id,
                        TIMESTAMP: position,
                        LABEL: 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def _load_item_frame(self) -> pd.DataFrame:
        return _read_atomic_frame(self._require_atomic_path(self.config.item_filename))

    @staticmethod
    def _build_user_table(interactions: pd.DataFrame) -> pd.DataFrame:
        raw_user_ids = pd.Index(pd.unique(interactions[USER_ID]), name=USER_ID)
        return pd.DataFrame(
            {
                USER_ID: raw_user_ids.astype(str),
                "raw_user_id": raw_user_ids.astype(str),
            }
        )

    def _build_item_table(self, item_frame: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
        raw_item_ids = [_normalize_raw_token(value) for value in item_frame.get("item_id", pd.Series(dtype=object)).tolist()]
        if raw_item_ids:
            item_index = pd.Index(raw_item_ids, name=ITEM_ID)
        else:
            item_index = pd.Index(pd.unique(interactions[ITEM_ID]), name=ITEM_ID)

        item_rows = {
            _normalize_raw_token(row["item_id"]): row
            for row in item_frame.to_dict(orient="records")
        }
        rows: list[dict[str, Any]] = []
        for raw_item_id in item_index.astype(str):
            raw_row = item_rows.get(raw_item_id, {})
            normalized_row: dict[str, Any] = {
                ITEM_ID: raw_item_id,
                "raw_item_id": raw_item_id,
            }
            if "movie_title" in raw_row:
                title = _clean_ml1m_title(str(raw_row.get("movie_title", "")).strip())
                normalized_row["title"] = title
                normalized_row["metadata_text"] = title
                if "release_year" in raw_row:
                    normalized_row["release_year"] = _normalize_raw_token(raw_row["release_year"])
                if "genre" in raw_row:
                    normalized_row["genre"] = str(raw_row["genre"]).strip()
            elif "title" in raw_row:
                title = str(raw_row.get("title", "")).strip()
                normalized_row["title"] = title
                normalized_row["metadata_text"] = title
            else:
                normalized_row["metadata_text"] = raw_item_id
            rows.append(normalized_row)
        return pd.DataFrame(rows)

    def _require_atomic_path(self, filename: str | None) -> Path:
        if not filename:
            raise ValueError(f"{type(self.config).__name__} requires this atomic filename to be configured.")
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected atomic dataset file at {path}.")
        return path


class LLMRankAtomicRetrievalDataset(RetrievalDataset):
    """Retrieval dataset that can replace eval candidates using external LLMRank candidate files."""

    config_cls = LLMRankAtomicDatasetConfig
    parser_cls = LLMRankAtomicParser
    config: LLMRankAtomicDatasetConfig

    def prepare(self, *, eval_config):
        super().prepare(eval_config=eval_config)
        self._apply_external_candidates()
        return self

    def _apply_external_candidates(self) -> None:
        candidate_path = self._candidate_file_path()
        if candidate_path is None:
            return
        candidate_map = self._load_external_candidate_map(candidate_path)
        test_frame = self._require_frame_dataset(self.get_eval_dataset("test")).frame
        self._test_dataset = FrameDataset(self._frame_with_external_candidates(test_frame, candidate_map))
        valid_frame = self._require_frame_dataset(self.get_eval_dataset("valid")).frame
        if self.config.use_external_candidates_for_valid:
            self._valid_dataset = FrameDataset(self._frame_with_external_candidates(valid_frame, candidate_map))
        else:
            self._valid_dataset = FrameDataset(self._empty_eval_frame_like(valid_frame))

    def _frame_with_external_candidates(
        self,
        frame: pd.DataFrame,
        candidate_map: dict[int, tuple[int, ...]],
    ) -> pd.DataFrame:
        if frame.empty:
            result = frame.copy()
            if CANDIDATE_ITEM_IDS not in result.columns:
                result[CANDIDATE_ITEM_IDS] = pd.Series(dtype=object)
            return result

        rows: list[dict[str, Any]] = []
        for record in frame.to_dict(orient="records"):
            normalized_user_id = int(record[USER_ID])
            candidate_item_ids = candidate_map.get(normalized_user_id)
            if candidate_item_ids is None:
                if self.config.drop_missing_candidate_users:
                    continue
                candidate_item_ids = ()
            updated = dict(record)
            updated[CANDIDATE_ITEM_IDS] = candidate_item_ids
            rows.append(updated)

        if not rows:
            return self._empty_eval_frame_like(frame)
        return pd.DataFrame(rows, columns=tuple(frame.columns))

    @staticmethod
    def _empty_eval_frame_like(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.iloc[0:0].copy()
        if CANDIDATE_ITEM_IDS not in result.columns:
            result[CANDIDATE_ITEM_IDS] = pd.Series(dtype=object)
        return result

    def _load_external_candidate_map(self, candidate_path: Path) -> dict[int, tuple[int, ...]]:
        user_table = self.get_user_table()
        item_table = self.get_item_table()
        user_id_map = {
            str(raw_user_id): int(user_id)
            for user_id, raw_user_id in zip(
                user_table[USER_ID].tolist(),
                user_table["raw_user_id"].astype(str).tolist(),
                strict=True,
            )
        }
        item_id_map = {
            str(raw_item_id): int(item_id)
            for item_id, raw_item_id in zip(
                item_table[ITEM_ID].tolist(),
                item_table["raw_item_id"].astype(str).tolist(),
                strict=True,
            )
        }
        candidate_budget = max(0, int(self.config.candidate_budget))
        candidate_map: dict[int, tuple[int, ...]] = {}
        with candidate_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                raw_user_id, raw_candidate_ids = stripped.split("\t", 1)
                normalized_user_id = user_id_map.get(_normalize_raw_token(raw_user_id))
                if normalized_user_id is None:
                    continue
                candidate_ids: list[int] = []
                seen_item_ids: set[int] = set()
                for raw_item_id in _parse_token_seq(raw_candidate_ids):
                    normalized_item_id = item_id_map.get(_normalize_raw_token(raw_item_id))
                    if normalized_item_id is None or normalized_item_id in seen_item_ids:
                        continue
                    candidate_ids.append(normalized_item_id)
                    seen_item_ids.add(normalized_item_id)
                    if candidate_budget and len(candidate_ids) >= candidate_budget:
                        break
                candidate_map[normalized_user_id] = tuple(candidate_ids)
        return candidate_map

    def _candidate_file_path(self) -> Path | None:
        if not self.config.candidate_filename:
            return None
        path = self._parser.data_dir / self.config.candidate_filename
        if not path.exists():
            raise FileNotFoundError(f"Expected external candidate file at {path}.")
        return path

    @staticmethod
    def _require_frame_dataset(dataset: Any) -> FrameDataset:
        if not isinstance(dataset, FrameDataset):
            raise TypeError(f"LLMRank external candidates require FrameDataset, got {type(dataset).__name__}.")
        return dataset


class ML1MLLMRankRetrievalDataset(LLMRankAtomicRetrievalDataset):
    config_cls = ML1MLLMRankDatasetConfig
    parser_cls = LLMRankAtomicParser


class GamesLLMRankRetrievalDataset(LLMRankAtomicRetrievalDataset):
    config_cls = GamesLLMRankDatasetConfig
    parser_cls = LLMRankAtomicParser


def _read_atomic_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [_strip_type_suffix(column) for column in frame.columns]
    return frame


def _strip_type_suffix(name: str) -> str:
    return str(name).split(":", 1)[0]


def _normalize_raw_token(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _parse_token_seq(raw_value: Any) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    return [_normalize_raw_token(token) for token in text.split()]


def _optional_numeric(value: Any) -> int | float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    numeric_value = float(value)
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def _clean_ml1m_title(title: str) -> str:
    normalized = title.strip()
    if normalized.endswith(", The"):
        normalized = f"The {normalized[:-5].strip()}"
    elif normalized.endswith(", A"):
        normalized = f"A {normalized[:-2].strip()}"
    elif normalized.endswith(", An"):
        normalized = f"An {normalized[:-3].strip()}"
    return normalized


__all__ = [
    "GamesLLMRankDatasetConfig",
    "GamesLLMRankRetrievalDataset",
    "LLMRankAtomicDatasetConfig",
    "LLMRankAtomicParser",
    "LLMRankAtomicRetrievalDataset",
    "ML1MLLMRankDatasetConfig",
    "ML1MLLMRankRetrievalDataset",
]
