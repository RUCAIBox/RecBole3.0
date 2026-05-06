from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from recbole3.dataset.cache import DatasetCache
from recbole3.dataset.config import DatasetConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import ITEM_ID, USER_ID

from . import utils as amazon2014_utils


@dataclass(slots=True)
class Amazon2014BaseConfig(DatasetConfig):
    name: str = field(default="", metadata={"help": "Amazon 2014 dataset name."})
    download_dir: str = field(default="data/raw", metadata={"help": "Raw Amazon 2014 cache root."})
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed Amazon 2014 cache root."})
    refresh_cache: bool = field(default=False, metadata={"help": "Whether to rebuild parser-managed cache files."})
    category: str = field(default="Beauty", metadata={"help": "Amazon 2014 category name."})
    metadata_mode: Literal["none", "sentence"] = field(
        default="sentence",
        metadata={"help": "How to materialize item metadata in the item table."},
    )
    download_source: Literal["snap"] = field(
        default="snap",
        metadata={"help": "Remote source used when canonical raw cache and local gz files are missing."},
    )


class Amazon2014BaseParser(BaseDatasetParser):
    """Shared Amazon Reviews 2014 raw, cache, and metadata parser flow."""

    config_cls = Amazon2014BaseConfig
    config: Amazon2014BaseConfig

    def parse(self) -> ParsedData:
        self._validate_source_config()
        if not self.config.refresh_cache and self._parsed_cache_exists():
            return self._load_parsed_data()

        self._ensure_raw_snapshot(force=self.config.refresh_cache)
        parsed = self._build_parsed_data()
        self._write_parsed_data(parsed)
        return parsed

    @abstractmethod
    def _build_interactions(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """Build task-specific parser interactions from canonical Amazon 2014 reviews."""

    def _ensure_raw_snapshot(self, *, force: bool) -> None:
        raw_cache = self._raw_cache()
        raw_cache.get_or_create_frame("reviews.jsonl", self._download_reviews_frame, force=force)
        if self._metadata_enabled():
            raw_cache.get_or_create_frame("meta.jsonl", self._download_metadata_frame, force=force)

    def _build_parsed_data(self) -> ParsedData:
        reviews = self._load_raw_reviews_frame()
        user_index, item_index = self._build_entity_indexes(reviews)
        interactions = self._build_interactions(reviews)
        user_table = self._build_user_table(user_index)
        item_table = self._build_item_table(item_index)
        return ParsedData(interactions=interactions, user_table=user_table, item_table=item_table)

    def _build_entity_indexes(self, reviews: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
        user_index = pd.Index(pd.unique(reviews["reviewerID"]), name=USER_ID)
        item_index = pd.Index(pd.unique(reviews["asin"]), name=ITEM_ID)
        return user_index, item_index

    def _build_user_table(self, user_index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({USER_ID: user_index.astype(object)})

    def _build_item_table(self, item_index: pd.Index) -> pd.DataFrame:
        item_table = pd.DataFrame({ITEM_ID: item_index.astype(object)})
        if not self._metadata_enabled():
            return item_table
        metadata = self._load_raw_metadata_frame()
        if metadata.empty:
            item_table["metadata_text"] = ""
            return item_table
        metadata = metadata.loc[metadata["asin"].isin(set(item_index))].copy()
        metadata = metadata.drop_duplicates(subset=["asin"], keep="first")
        metadata["metadata_text"] = metadata.apply(amazon2014_utils.build_metadata_text, axis=1)
        merged = item_table.merge(
            metadata.loc[:, ["asin", "metadata_text"]],
            how="left",
            left_on=ITEM_ID,
            right_on="asin",
        ).drop(columns=["asin"])
        merged["metadata_text"] = merged["metadata_text"].fillna("")
        return merged

    def _download_reviews_frame(self) -> pd.DataFrame:
        gz_path = self._ensure_reviews_gz()
        return amazon2014_utils.reviews_gz_to_frame(gz_path)

    def _download_metadata_frame(self) -> pd.DataFrame:
        gz_path = self._ensure_metadata_gz()
        return amazon2014_utils.metadata_gz_to_frame(gz_path)

    def _ensure_reviews_gz(self) -> Path:
        path = self._reviews_gz_path()
        if path.exists():
            return path
        return self._download_gz(
            amazon2014_utils.reviews_url(self.config.category),
            path,
            description="Amazon 2014 reviews gz",
        )

    def _ensure_metadata_gz(self) -> Path:
        path = self._metadata_gz_path()
        if path.exists():
            return path
        return self._download_gz(
            amazon2014_utils.metadata_url(self.config.category),
            path,
            description="Amazon 2014 metadata gz",
        )

    def _download_gz(self, url: str, path: Path, *, description: str) -> Path:
        if self.config.download_source != "snap":
            raise ValueError(f"Unsupported Amazon 2014 download_source '{self.config.download_source}'.")
        try:
            return amazon2014_utils.download_file(url, path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {description} from {url}. "
                f"Place the original gz file at {path} and rerun, or fix network access."
            ) from exc

    def _metadata_enabled(self) -> bool:
        return self.config.metadata_mode == "sentence"

    def _validate_source_config(self) -> None:
        if self.config.category not in amazon2014_utils.AMAZON2014_AVAILABLE_CATEGORIES:
            raise ValueError(
                f"Category '{self.config.category}' is not available. "
                f"Available categories: {', '.join(amazon2014_utils.AMAZON2014_AVAILABLE_CATEGORIES)}"
            )
        if self.config.metadata_mode not in {"none", "sentence"}:
            raise ValueError(f"Unsupported metadata_mode '{self.config.metadata_mode}'.")
        if self.config.download_source not in {"snap"}:
            raise ValueError(f"Unsupported download_source '{self.config.download_source}'.")

    def _load_parsed_data(self) -> ParsedData:
        return self._parsed_cache().read_parsed()

    def _write_parsed_data(self, parsed: ParsedData) -> None:
        self._parsed_cache().write_parsed(parsed)

    def _parsed_cache_exists(self) -> bool:
        return self._parsed_cache().parsed_exists()

    @property
    def data_dir(self) -> Path:
        return self._parsed_root_dir()

    def _raw_root_dir(self) -> Path:
        return Path(self.config.download_dir) / "amazon2014" / self.config.category

    def _parsed_root_dir(self) -> Path:
        return Path(self.config.processed_dir) / self.config.name / self.config.category / self.config.metadata_mode

    def _raw_cache(self) -> DatasetCache:
        return DatasetCache(self._raw_root_dir())

    def _parsed_cache(self) -> DatasetCache:
        return DatasetCache(self._parsed_root_dir())

    def _reviews_gz_path(self) -> Path:
        return self._raw_cache().path(amazon2014_utils.reviews_gz_name(self.config.category))

    def _metadata_gz_path(self) -> Path:
        return self._raw_cache().path(amazon2014_utils.metadata_gz_name(self.config.category))

    def _load_raw_reviews_frame(self) -> pd.DataFrame:
        return self._raw_cache().read_frame(
            "reviews.jsonl",
            required=True,
            description="Amazon 2014 canonical reviews cache",
        )

    def _load_raw_metadata_frame(self) -> pd.DataFrame:
        return self._raw_cache().read_frame(
            "meta.jsonl",
            required=True,
            description="Amazon 2014 canonical metadata cache",
        )


__all__ = [
    "Amazon2014BaseConfig",
    "Amazon2014BaseParser",
]
