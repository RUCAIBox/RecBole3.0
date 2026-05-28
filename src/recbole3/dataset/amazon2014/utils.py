from __future__ import annotations

import ast
import gzip
import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from recbole3.dataset.utils import OVERALL


AMAZON2014_AVAILABLE_CATEGORIES: tuple[str, ...] = (
    "Books",
    "Electronics",
    "Movies_and_TV",
    "CDs_and_Vinyl",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Sports_and_Outdoors",
    "Cell_Phones_and_Accessories",
    "Health_and_Personal_Care",
    "Toys_and_Games",
    "Video_Games",
    "Tools_and_Home_Improvement",
    "Beauty",
    "Apps_for_Android",
    "Office_Products",
    "Pet_Supplies",
    "Automotive",
    "Grocery_and_Gourmet_Food",
    "Patio_Lawn_and_Garden",
    "Baby",
    "Digital_Music",
    "Musical_Instruments",
    "Amazon_Instant_Video",
)

AMAZON2014_BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
AMAZON2014_META_FIELDS: tuple[str, ...] = (
    "title",
    "price",
    "brand",
    "feature",
    "categories",
    "description",
)


def reviews_gz_name(category: str) -> str:
    return f"reviews_{category}_5.json.gz"


def metadata_gz_name(category: str) -> str:
    return f"meta_{category}.json.gz"


def reviews_url(category: str) -> str:
    return f"{AMAZON2014_BASE_URL}/{reviews_gz_name(category)}"


def metadata_url(category: str) -> str:
    return f"{AMAZON2014_BASE_URL}/{metadata_gz_name(category)}"


def download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def iter_gzip_records(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    record = ast.literal_eval(stripped)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Could not parse {path} line {line_number} as JSON or Python literal.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object record in {path} line {line_number}, got {type(record).__name__}.")
            yield record


def reviews_gz_to_frame(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in iter_gzip_records(path):
        user_id = record.get("reviewerID")
        item_id = record.get("asin")
        timestamp = numeric_or_none(record.get("unixReviewTime"))
        overall = numeric_or_none(record.get(OVERALL))
        if user_id is None or item_id is None:
            raise ValueError("Amazon 2014 review records require non-null reviewerID and asin.")
        if timestamp is None:
            raise ValueError(f"Amazon 2014 review for user={user_id!r}, item={item_id!r} has invalid unixReviewTime.")
        rows.append(
            {
                "reviewerID": str(user_id),
                "asin": str(item_id),
                "unixReviewTime": timestamp,
                OVERALL: overall,
            }
        )
    return pd.DataFrame(rows, columns=["reviewerID", "asin", "unixReviewTime", OVERALL])


def metadata_gz_to_frame(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in iter_gzip_records(path):
        asin = record.get("asin")
        if asin is None:
            continue
        row = {"asin": str(asin)}
        for field in AMAZON2014_META_FIELDS:
            row[field] = record.get(field)
        rows.append(row)
    return pd.DataFrame(rows, columns=("asin",) + AMAZON2014_META_FIELDS)


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
        parts: list[str] = []
        for item in value:
            item_text = stringify_feature(item) if isinstance(item, list) else clean_scalar_text(item)
            if item_text:
                parts.append(item_text)
        return ", ".join(parts)
    return clean_scalar_text(value)


def clean_scalar_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def feature_to_sentence(value: Any) -> str:
    cleaned_value = clean_text(value)
    if not cleaned_value:
        return ""
    return f"{cleaned_value}."


def build_metadata_text(row: pd.Series) -> str:
    sentences = [feature_to_sentence(row.get(field)) for field in AMAZON2014_META_FIELDS]
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
    "AMAZON2014_AVAILABLE_CATEGORIES",
    "AMAZON2014_BASE_URL",
    "AMAZON2014_META_FIELDS",
    "build_metadata_text",
    "clean_text",
    "download_file",
    "iter_gzip_records",
    "metadata_gz_name",
    "metadata_gz_to_frame",
    "metadata_url",
    "numeric_or_none",
    "reviews_gz_name",
    "reviews_gz_to_frame",
    "reviews_url",
]
