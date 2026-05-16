from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from recbole3.dataset.cache import DatasetCache
from recbole3.dataset.config import DatasetConfig, SplitConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import ITEM_ID, LABEL, USER_ID

from . import utils as avazu_utils


@dataclass(slots=True)
class AvazuBaseConfig(DatasetConfig):
    """
    Base configuration for the Avazu CTR dataset.

    Expects pre-split CSV files containing `label` plus 22 feature columns.
    Manual download reference:
    https://huggingface.co/datasets/reczoo/Avazu_x1/resolve/main/Avazu_x1.zip
    """

    name: str = field(default="", metadata={"help": "Registered Avazu dataset name."})
    train_path: str = field(
        default="data/avazu/train.csv",
        metadata={"help": "Path to the Avazu training split CSV."},
    )
    valid_path: str = field(
        default="data/avazu/valid.csv",
        metadata={"help": "Path to the Avazu validation split CSV."},
    )
    test_path: str = field(
        default="data/avazu/test.csv",
        metadata={"help": "Path to the Avazu test split CSV."},
    )
    processed_dir: str = field(
        default="data/processed",
        metadata={"help": "Directory used for parsed Avazu cache files."},
    )
    refresh_cache: bool = field(
        default=False,
        metadata={"help": "Whether to rebuild the parsed Avazu cache from raw split files."},
    )
    hash_size: int = field(
        default=1_000_000,
        metadata={"help": "Hash space size used to mirror MLCC feature ID processing."},
    )
    split: SplitConfig = field(
        default_factory=lambda: SplitConfig(strategy="ratio", order="chronological", per_user=False),
        metadata={"help": "Unused by the pre-split Avazu dataset but kept for API compatibility."},
    )


class AvazuBaseParser(BaseDatasetParser):
    """Shared Avazu pre-split CSV parsing and cache flow."""

    config_cls = AvazuBaseConfig
    config: AvazuBaseConfig

    def parse(self) -> ParsedData:
        cache = self._parsed_cache()
        if not self.config.refresh_cache and cache.parsed_exists():
            return cache.read_parsed()

        parsed = self._build_parsed_data()
        cache.write_parsed(parsed)
        return parsed

    @property
    def data_dir(self) -> Path:
        return self._parsed_root_dir()

    def _build_parsed_data(self) -> ParsedData:
        frames = [
            self._load_split_frame(self.config.train_path, split_name="train"),
            self._load_split_frame(self.config.valid_path, split_name="valid"),
            self._load_split_frame(self.config.test_path, split_name="test"),
        ]
        interactions = pd.concat(frames, ignore_index=True, sort=False)
        user_table = interactions.loc[:, [USER_ID]].drop_duplicates().reset_index(drop=True)
        item_table = interactions.loc[:, [ITEM_ID]].drop_duplicates().reset_index(drop=True)
        return ParsedData(
            interactions=interactions,
            user_table=user_table,
            item_table=item_table,
        )

    def _load_split_frame(self, split_path: str, *, split_name: str) -> pd.DataFrame:
        path = self._resolve_split_path(split_path)
        if not path.exists():
            raise FileNotFoundError(f"Avazu {split_name} split not found at {path}.")
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Avazu {split_name} split must be a CSV file, got '{path}'. "
                "Please provide pre-split CSV files via dataset yaml."
            )
        return self._load_preprocessed_csv(path, split_name=split_name)

    def _load_preprocessed_csv(self, path: Path, *, split_name: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        feature_columns = [f"feat_{index}" for index in range(1, avazu_utils.AVAZU_NUM_FEATURES + 1)]
        missing = [column for column in ["label", *feature_columns] if column not in frame.columns]
        if missing:
            raise ValueError(f"Preprocessed Avazu CSV at {path} is missing required columns: {missing}.")

        records = pd.DataFrame(
            {
                USER_ID: [f"{split_name}_user_{index}" for index in range(len(frame))],
                ITEM_ID: [f"{split_name}_item_{index}" for index in range(len(frame))],
                LABEL: pd.to_numeric(frame["label"], errors="coerce").fillna(0.0).astype("float32"),
                avazu_utils.AVAZU_SPLIT_COLUMN: split_name,
            }
        )
        for feature_index, source_column in enumerate(feature_columns):
            source_values = pd.to_numeric(frame[source_column], errors="coerce").fillna(0).astype("int64")
            records[avazu_utils.AVAZU_FEATURE_COLUMNS[feature_index]] = source_values.map(
                lambda value, field_id=feature_index: avazu_utils.hash_feature(
                    field_id=field_id,
                    value=str(int(value)),
                    hash_size=self.config.hash_size,
                )
            ).astype("int64")
        return records

    def _parsed_root_dir(self) -> Path:
        return Path(self.config.processed_dir) / self.config.name / f"hash_{int(self.config.hash_size)}"

    def _parsed_cache(self) -> DatasetCache:
        return DatasetCache(self._parsed_root_dir())

    def _resolve_split_path(self, split_path: str) -> Path:
        configured_path = Path(split_path)
        if configured_path.is_absolute():
            return configured_path

        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / configured_path


__all__ = [
    "AvazuBaseConfig",
    "AvazuBaseParser",
]