from __future__ import annotations

import html
import re
from typing import Any, Sequence

import pandas as pd

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

ORIGINAL_LOADERS = {
    "load_huggingface_dataset": load_huggingface_dataset,
    "load_modelscope_dataset": load_modelscope_dataset,
}


def dataset_to_frame(dataset: Any, columns: Sequence[str]) -> pd.DataFrame:
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


def clean_text(raw_text: Any) -> str:
    text = stringify_feature(raw_text)
    text = html.unescape(text).strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text.strip()


def stringify_feature(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        cleaned_values = [clean_text(item) for item in value]
        return ", ".join(item for item in cleaned_values if item)
    return str(value)


def feature_to_sentence(value: Any) -> str:
    cleaned_value = clean_text(value)
    if not cleaned_value:
        return ""
    return f"{cleaned_value}."


def build_metadata_text(row: pd.Series) -> str:
    sentences = [
        feature_to_sentence(row.get("title")),
        feature_to_sentence(row.get("features")),
        feature_to_sentence(row.get("categories")),
        feature_to_sentence(row.get("description")),
    ]
    return " ".join(sentence for sentence in sentences if sentence).strip()


def numeric_or_none(value: Any) -> int | float | None:
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
    "AMAZON2023_META_COLUMNS",
    "AMAZON2023_REVIEW_COLUMNS",
    "AMAZON2023_UNSUPPORTED_5CORE_CATEGORIES",
    "build_metadata_text",
    "clean_text",
    "dataset_to_frame",
    "feature_to_sentence",
    "load_huggingface_dataset",
    "load_modelscope_dataset",
    "numeric_or_none",
    "stringify_feature",
]
