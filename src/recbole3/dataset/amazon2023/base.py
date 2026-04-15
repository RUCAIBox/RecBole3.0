from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from recbole3.dataset.cache import DatasetCache
from recbole3.dataset.config import DatasetConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import ITEM_ID, USER_ID

from . import utils as amazon2023_utils


@dataclass(slots=True)
class Amazon2023BaseConfig(DatasetConfig):
    name: str = field(default="", metadata={"help": "Amazon 2023 dataset name."})
    download_dir: str = field(default="data/raw", metadata={"help": "Raw Amazon 2023 cache root."})
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed Amazon 2023 cache root."})
    refresh_cache: bool = field(default=False, metadata={"help": "Whether to rebuild parser-managed cache files."})
    category: str = field(default="Books", metadata={"help": "Amazon 2023 category name."})
    kcore: Literal["full", "5core"] = field(
        default="full",
        metadata={"help": "Amazon 2023 review subset to download."},
    )
    metadata_mode: Literal["none", "sentence", "fields"] = field(
        default="sentence",
        metadata={"help": "How to materialize item metadata in the item table."},
    )
    download_source: Literal["huggingface", "modelscope"] = field(
        default="modelscope",
        metadata={"help": "Remote source used to snapshot Amazon 2023 raw data."},
    )


class Amazon2023BaseParser(BaseDatasetParser):
    """Shared Amazon Reviews 2023 raw, cache, and metadata parser flow."""

    config_cls = Amazon2023BaseConfig
    config: Amazon2023BaseConfig

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
        """Build task-specific parser interactions from raw Amazon reviews."""

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
        user_index = pd.Index(pd.unique(reviews["user_id"]), name=USER_ID)
        item_index = pd.Index(pd.unique(reviews["parent_asin"]), name=ITEM_ID)
        return user_index, item_index

    def _build_user_table(self, user_index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({USER_ID: user_index.astype(object)})

    def _build_item_table(self, item_index: pd.Index) -> pd.DataFrame:
        item_table = pd.DataFrame({ITEM_ID: item_index.astype(object)})
        if not self._metadata_enabled():
            return item_table
        if self.config.metadata_mode == "fields":
            return self._attach_metadata_fields(item_table, item_index)
        return self._attach_metadata_text(item_table, item_index)

    def _metadata_enabled(self) -> bool:
        return self.config.metadata_mode in ("sentence", "fields")

    def _attach_metadata_text(self, item_table: pd.DataFrame, item_index: pd.Index) -> pd.DataFrame:
        metadata = self._load_raw_metadata_frame()
        metadata = metadata.loc[metadata["parent_asin"].isin(set(item_index))].copy()
        metadata = metadata.drop_duplicates(subset=["parent_asin"], keep="first")
        metadata["metadata_text"] = metadata.apply(amazon2023_utils.build_metadata_text, axis=1)
        merged = item_table.merge(
            metadata.loc[:, ["parent_asin", "metadata_text"]],
            how="left",
            left_on=ITEM_ID,
            right_on="parent_asin",
        ).drop(columns=["parent_asin"])
        merged["metadata_text"] = merged["metadata_text"].fillna("")
        return merged

    def _attach_metadata_fields(self, item_table: pd.DataFrame, item_index: pd.Index) -> pd.DataFrame:
        """Attach individual metadata columns (title, description) to the item table."""
        metadata = self._load_raw_metadata_frame()
        metadata = metadata.loc[metadata["parent_asin"].isin(set(item_index))].copy()
        metadata = metadata.drop_duplicates(subset=["parent_asin"], keep="first")
        # Clean title using feature_to_sentence which handles list/str types
        metadata["title"] = metadata["title"].apply(amazon2023_utils.feature_to_sentence)
        # Build combined description from categories, features, and description
        metadata["description"] = metadata.apply(
            lambda row: " ".join(filter(None, [
                amazon2023_utils.feature_to_sentence(row.get("categories", "")),
                amazon2023_utils.feature_to_sentence(row.get("features", "")),
                amazon2023_utils.feature_to_sentence(row.get("description", "")),
            ])).strip(),
            axis=1,
        )
        keep_cols = ["parent_asin", "title", "description"]
        merged = item_table.merge(
            metadata[keep_cols],
            how="left",
            left_on=ITEM_ID,
            right_on="parent_asin",
        ).drop(columns=["parent_asin"])
        merged["title"] = merged["title"].fillna("")
        merged["description"] = merged["description"].fillna("")
        return merged

    def _load_parsed_data(self) -> ParsedData:
        return self._parsed_cache().read_parsed()

    def _write_parsed_data(self, parsed: ParsedData) -> None:
        self._parsed_cache().write_parsed(parsed)

    def _parsed_cache_exists(self) -> bool:
        return self._parsed_cache().parsed_exists()

    def _download_reviews_frame(self) -> pd.DataFrame:
        reviews = self._load_remote_subset(f"{self.config.kcore}_rating_only_{self.config.category}")
        return amazon2023_utils.dataset_to_frame(reviews, amazon2023_utils.AMAZON2023_REVIEW_COLUMNS)

    def _download_metadata_frame(self) -> pd.DataFrame:
        metadata = self._load_remote_subset(f"raw_meta_{self.config.category}")
        return amazon2023_utils.dataset_to_frame(metadata, amazon2023_utils.AMAZON2023_META_COLUMNS)

    def _load_remote_subset(self, subset_name: str) -> Any:
        if self.config.download_source == "huggingface":
            loader = self._resolve_remote_loader("load_huggingface_dataset")
            return loader(
                amazon2023_utils.AMAZON2023_DATASET_ID,
                subset_name,
                split="full",
                cache_dir=str(self._hf_cache_dir()),
                trust_remote_code=True,
            )
        if self.config.download_source == "modelscope":
            loader = self._resolve_remote_loader("load_modelscope_dataset")
            return loader(
                amazon2023_utils.AMAZON2023_DATASET_ID,
                subset_name,
                split="full",
                trust_remote_code=True,
            )
        raise ValueError(f"Unsupported download_source '{self.config.download_source}'.")

    @staticmethod
    def _resolve_remote_loader(name: str) -> Any:
        utils_loader = getattr(amazon2023_utils, name)
        package = sys.modules.get("recbole3.dataset.amazon2023")
        package_loader = getattr(package, "__dict__", {}).get(name) if package is not None else None
        original_loader = amazon2023_utils.ORIGINAL_LOADERS[name]
        if package_loader is not None and package_loader is not original_loader and package_loader is not utils_loader:
            return package_loader
        return utils_loader

    def _validate_source_config(self) -> None:
        if self.config.category not in amazon2023_utils.AMAZON2023_AVAILABLE_CATEGORIES:
            raise ValueError(
                f"Category '{self.config.category}' is not available. "
                f"Available categories: {', '.join(amazon2023_utils.AMAZON2023_AVAILABLE_CATEGORIES)}"
            )
        if self.config.kcore == "5core" and self.config.category in amazon2023_utils.AMAZON2023_UNSUPPORTED_5CORE_CATEGORIES:
            raise ValueError(f"Category '{self.config.category}' does not provide 5-core reviews.")
        if self.config.metadata_mode not in {"none", "sentence", "fields"}:
            raise ValueError(f"Unsupported metadata_mode '{self.config.metadata_mode}'.")
        if self.config.download_source not in {"huggingface", "modelscope"}:
            raise ValueError(f"Unsupported download_source '{self.config.download_source}'.")

    def _raw_root_dir(self) -> Path:
        return Path(self.config.download_dir) / "amazon2023" / self.config.download_source / self.config.category / self.config.kcore

    def _parsed_root_dir(self) -> Path:
        return (
            Path(self.config.processed_dir)
            / self.config.name
            / self.config.download_source
            / self.config.category
            / self.config.kcore
            / self.config.metadata_mode
        )

    @property
    def data_dir(self) -> Path:
        """Return the root directory for processed data files."""
        return self._parsed_root_dir()

    def _hf_cache_dir(self) -> Path:
        return Path(self.config.download_dir) / "_hf_cache" / "amazon2023"

    def _raw_cache(self) -> DatasetCache:
        return DatasetCache(self._raw_root_dir())

    def _parsed_cache(self) -> DatasetCache:
        return DatasetCache(self._parsed_root_dir())

    def _raw_reviews_path(self) -> Path:
        return self._raw_cache().path("reviews.jsonl")

    def _raw_metadata_path(self) -> Path:
        return self._raw_cache().path("meta.jsonl")

    def _interactions_path(self) -> Path:
        return self._parsed_cache().path("interactions.jsonl")

    def _users_path(self) -> Path:
        return self._parsed_cache().path("users.jsonl")

    def _items_path(self) -> Path:
        return self._parsed_cache().path("items.jsonl")

    def _load_raw_reviews_frame(self) -> pd.DataFrame:
        return self._raw_cache().read_frame(
            "reviews.jsonl",
            required=True,
            description="Amazon 2023 reviews snapshot",
        )

    def _load_raw_metadata_frame(self) -> pd.DataFrame:
        return self._raw_cache().read_frame(
            "meta.jsonl",
            required=True,
            description="Amazon 2023 metadata snapshot",
        )

    def _load_interactions(self) -> pd.DataFrame:
        return self._parsed_cache().read_frame("interactions.jsonl")


__all__ = [
    "Amazon2023BaseConfig",
    "Amazon2023BaseParser",
]
