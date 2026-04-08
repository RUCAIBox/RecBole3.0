from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from recbole3.dataset.base import BaseDatasetParser, DatasetConfig, Interaction, ParsedData, RetrievalDataset
from recbole3.dataset.utils import load_huggingface_dataset, load_modelscope_dataset


AMAZON2023_DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON2023_AVAILABLE_CATEGORIES: tuple[str, ...] = (
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Art_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Unknown",
    "Video_Games",
)

AMAZON2023_UNSUPPORTED_5CORE_CATEGORIES = {
    "Amazon_Fashion",
    "Appliances",
    "Digital_Music",
    "Handmade_Products",
    "Health_and_Personal_Care",
    "Subscription_Boxes",
}

AMAZON2023_REVIEW_COLUMNS: tuple[str, ...] = (
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
)

AMAZON2023_META_COLUMNS: tuple[str, ...] = (
    "parent_asin",
    "title",
    "features",
    "categories",
    "description",
)


@dataclass(slots=True)
class Amazon2023RetrievalConfig(DatasetConfig):
    name: str = field(default="amazon2023_retrieval", metadata={"help": "Amazon 2023 retrieval dataset name."})
    download_dir: str = field(default="data/raw", metadata={"help": "Raw Amazon 2023 cache root."})
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed Amazon 2023 cache root."})
    refresh_cache: bool = field(default=False, metadata={"help": "Whether to rebuild parser-managed cache files."})
    category: str = field(default="Books", metadata={"help": "Amazon 2023 category name."})
    kcore: Literal["full", "5core"] = field(
        default="full",
        metadata={"help": "Amazon 2023 review subset to download."},
    )
    metadata_mode: Literal["none", "sentence"] = field(
        default="sentence",
        metadata={"help": "How to materialize item metadata in the item table."},
    )
    download_source: Literal["huggingface", "modelscope"] = field(
        default="huggingface",
        metadata={"help": "Remote source used to snapshot Amazon 2023 raw data."},
    )


class Amazon2023Parser(BaseDatasetParser):
    """Amazon Reviews 2023 parser with dataset-specific raw and parsed caches."""

    config_cls = Amazon2023RetrievalConfig
    config: Amazon2023RetrievalConfig

    def parse(self) -> ParsedData:
        self._validate_source_config()
        if not self.config.refresh_cache and self._parsed_cache_exists():
            return self._load_parsed_data()

        self._ensure_raw_snapshot(force=self.config.refresh_cache)
        parsed = self._build_parsed_data()
        self._write_parsed_data(parsed)
        return parsed

    def _ensure_raw_snapshot(self, *, force: bool) -> None:
        if force or not self._raw_reviews_path().exists():
            self._write_jsonl_frame(self._raw_reviews_path(), self._download_reviews_frame())
        if self._metadata_enabled() and (force or not self._raw_metadata_path().exists()):
            self._write_jsonl_frame(self._raw_metadata_path(), self._download_metadata_frame())

    def _build_parsed_data(self) -> ParsedData:
        reviews = self._load_raw_reviews_frame()
        user_index, item_index = self._build_entity_indexes(reviews)
        user_id_map = self._build_id_map(user_index)
        item_id_map = self._build_id_map(item_index)
        interactions = self._build_interactions(reviews, user_id_map=user_id_map, item_id_map=item_id_map)
        user_table = self._build_user_table(user_index)
        item_table = self._build_item_table(item_index)
        return ParsedData(interactions=interactions, user_table=user_table, item_table=item_table)

    def _build_entity_indexes(self, reviews: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
        user_index = pd.Index(pd.unique(reviews["user_id"]), name="raw_user_id")
        item_index = pd.Index(pd.unique(reviews["parent_asin"]), name="raw_item_id")
        return user_index, item_index

    @staticmethod
    def _build_id_map(index: pd.Index) -> pd.Series:
        return pd.Series(range(len(index)), index=index)

    def _build_interactions(
        self,
        reviews: pd.DataFrame,
        *,
        user_id_map: pd.Series,
        item_id_map: pd.Series,
    ) -> list[Interaction]:
        mapped_user_ids = reviews["user_id"].map(user_id_map).to_numpy(dtype="int64", copy=False)
        mapped_item_ids = reviews["parent_asin"].map(item_id_map).to_numpy(dtype="int64", copy=False)
        timestamps = tuple(_numeric_or_none(value) for value in reviews["timestamp"].tolist())
        return [
            Interaction(
                user_id=int(user_id),
                item_id=int(item_id),
                timestamp=timestamp,
                label=None,
            )
            for user_id, item_id, timestamp in zip(mapped_user_ids, mapped_item_ids, timestamps, strict=True)
        ]

    def _build_user_table(self, user_index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({"user_id": range(len(user_index)), "raw_user_id": user_index.astype(object)})

    def _build_item_table(self, item_index: pd.Index) -> pd.DataFrame:
        item_table = pd.DataFrame({"item_id": range(len(item_index)), "raw_item_id": item_index.astype(object)})
        if not self._metadata_enabled():
            return item_table
        return self._attach_metadata_text(item_table, item_index)

    def _metadata_enabled(self) -> bool:
        return self.config.metadata_mode == "sentence"

    def _attach_metadata_text(self, item_table: pd.DataFrame, item_index: pd.Index) -> pd.DataFrame:
        metadata = self._load_raw_metadata_frame()
        metadata = metadata.loc[metadata["parent_asin"].isin(set(item_index))].copy()
        metadata = metadata.drop_duplicates(subset=["parent_asin"], keep="first")
        metadata["metadata_text"] = metadata.apply(self._build_metadata_text, axis=1)
        merged = item_table.merge(
            metadata.loc[:, ["parent_asin", "metadata_text"]],
            how="left",
            left_on="raw_item_id",
            right_on="parent_asin",
        ).drop(columns=["parent_asin"])
        merged["metadata_text"] = merged["metadata_text"].fillna("")
        return merged

    def _load_parsed_data(self) -> ParsedData:
        return ParsedData(
            interactions=self._load_interactions(),
            user_table=self._load_jsonl_table(self._users_path()),
            item_table=self._load_jsonl_table(self._items_path()),
        )

    def _write_parsed_data(self, parsed: ParsedData) -> None:
        self._write_jsonl_records(self._interactions_path(), parsed.interactions)
        self._write_jsonl_frame(self._users_path(), parsed.user_table)
        self._write_jsonl_frame(self._items_path(), parsed.item_table)

    def _parsed_cache_exists(self) -> bool:
        return self._interactions_path().exists() and self._users_path().exists() and self._items_path().exists()

    def _download_reviews_frame(self) -> pd.DataFrame:
        reviews = self._load_remote_subset(f"{self.config.kcore}_rating_only_{self.config.category}")
        return self._dataset_to_frame(reviews, AMAZON2023_REVIEW_COLUMNS)

    def _download_metadata_frame(self) -> pd.DataFrame:
        metadata = self._load_remote_subset(f"raw_meta_{self.config.category}")
        return self._dataset_to_frame(metadata, AMAZON2023_META_COLUMNS)

    def _load_remote_subset(self, subset_name: str) -> Any:
        if self.config.download_source == "huggingface":
            return load_huggingface_dataset(
                AMAZON2023_DATASET_ID,
                subset_name,
                split="full",
                cache_dir=str(self._hf_cache_dir()),
                trust_remote_code=True,
            )
        if self.config.download_source == "modelscope":
            return load_modelscope_dataset(
                AMAZON2023_DATASET_ID,
                subset_name,
                split="full",
                trust_remote_code=True,
            )
        raise ValueError(f"Unsupported download_source '{self.config.download_source}'.")

    def _validate_source_config(self) -> None:
        if self.config.category not in AMAZON2023_AVAILABLE_CATEGORIES:
            raise ValueError(
                f"Category '{self.config.category}' is not available. "
                f"Available categories: {', '.join(AMAZON2023_AVAILABLE_CATEGORIES)}"
            )
        if self.config.kcore == "5core" and self.config.category in AMAZON2023_UNSUPPORTED_5CORE_CATEGORIES:
            raise ValueError(f"Category '{self.config.category}' does not provide 5-core reviews.")
        if self.config.metadata_mode not in {"none", "sentence"}:
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

    def _hf_cache_dir(self) -> Path:
        return Path(self.config.download_dir) / "_hf_cache" / "amazon2023"

    def _raw_reviews_path(self) -> Path:
        return self._raw_root_dir() / "reviews.jsonl"

    def _raw_metadata_path(self) -> Path:
        return self._raw_root_dir() / "meta.jsonl"

    def _interactions_path(self) -> Path:
        return self._parsed_root_dir() / "interactions.jsonl"

    def _users_path(self) -> Path:
        return self._parsed_root_dir() / "users.jsonl"

    def _items_path(self) -> Path:
        return self._parsed_root_dir() / "items.jsonl"

    def _load_raw_reviews_frame(self) -> pd.DataFrame:
        return self._read_required_jsonl(self._raw_reviews_path(), description="Amazon 2023 reviews snapshot")

    def _load_raw_metadata_frame(self) -> pd.DataFrame:
        return self._read_required_jsonl(self._raw_metadata_path(), description="Amazon 2023 metadata snapshot")

    @staticmethod
    def _read_required_jsonl(path: Path, *, description: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}.")
        return pd.read_json(path, lines=True, convert_dates=False)

    def _load_interactions(self) -> list[Interaction]:
        interactions: list[Interaction] = []
        with self._interactions_path().open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                interactions.append(
                    Interaction(
                        user_id=int(record["user_id"]),
                        item_id=int(record["item_id"]),
                        timestamp=record.get("timestamp"),
                        label=float(record["label"]) if record.get("label") is not None else None,
                    )
                )
        return interactions

    @staticmethod
    def _load_jsonl_table(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_json(path, lines=True)

    @staticmethod
    def _write_jsonl_frame(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if frame.empty:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8") as handle:
            frame.to_json(handle, orient="records", lines=True, force_ascii=False)

    @staticmethod
    def _write_jsonl_records(path: Path, interactions: list[Interaction]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for interaction in interactions:
                handle.write(
                    json.dumps(
                        {
                            "user_id": interaction.user_id,
                            "item_id": interaction.item_id,
                            "timestamp": interaction.timestamp,
                            "label": interaction.label,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @staticmethod
    def _dataset_to_frame(dataset: Any, columns: Sequence[str]) -> pd.DataFrame:
        if isinstance(dataset, pd.DataFrame):
            frame = dataset.copy()
        elif hasattr(dataset, "to_pandas"):
            frame = dataset.to_pandas()
        elif hasattr(dataset, "to_hf_dataset"):
            frame = dataset.to_hf_dataset().to_pandas()
        else:
            frame = pd.DataFrame(dataset)
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Amazon 2023 source data is missing columns: {missing_columns}")
        return frame.loc[:, list(columns)].reset_index(drop=True)

    @classmethod
    def _clean_text(cls, raw_text: Any) -> str:
        text = cls._stringify_feature(raw_text)
        text = html.unescape(text).strip()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\n\t]", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"[^\x00-\x7F]", " ", text)
        return text.strip()

    @classmethod
    def _stringify_feature(cls, value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, list):
            cleaned_values = [cls._clean_text(item) for item in value]
            return ", ".join(item for item in cleaned_values if item)
        return str(value)

    @classmethod
    def _feature_to_sentence(cls, value: Any) -> str:
        cleaned_value = cls._clean_text(value)
        if not cleaned_value:
            return ""
        return f"{cleaned_value}."

    @classmethod
    def _build_metadata_text(cls, row: pd.Series) -> str:
        sentences = [
            cls._feature_to_sentence(row.get("title")),
            cls._feature_to_sentence(row.get("features")),
            cls._feature_to_sentence(row.get("categories")),
            cls._feature_to_sentence(row.get("description")),
        ]
        return " ".join(sentence for sentence in sentences if sentence).strip()


class Amazon2023RetrievalDataset(RetrievalDataset):
    """Amazon Reviews 2023 retrieval dataset implementation."""

    config_cls = Amazon2023RetrievalConfig
    parser_cls = Amazon2023Parser


def _numeric_or_none(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric_value = float(stripped)
        except ValueError:
            return None
        return int(numeric_value) if numeric_value.is_integer() else numeric_value
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


__all__ = [
    "AMAZON2023_AVAILABLE_CATEGORIES",
    "AMAZON2023_DATASET_ID",
    "Amazon2023Parser",
    "Amazon2023RetrievalConfig",
    "Amazon2023RetrievalDataset",
]
