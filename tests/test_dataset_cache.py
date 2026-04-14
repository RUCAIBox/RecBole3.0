from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recbole3.dataset.cache import DatasetCache
from recbole3.dataset.parser import ParsedData


def test_frame_cache_round_trips_jsonl_dataframe(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    frame = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "timestamp": 1, "text": "Alpha Pi"},
            {"user_id": "u2", "item_id": "B", "timestamp": 2, "text": "Bravo"},
        ]
    )

    cache.write_frame("nested/interactions.jsonl", frame)

    loaded = cache.read_frame("nested/interactions.jsonl", required=True)
    pd.testing.assert_frame_equal(loaded, frame)


def test_empty_and_none_frames_are_cached_as_empty_tables(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)

    cache.write_frame("empty.jsonl", pd.DataFrame(columns=["user_id"]))
    cache.write_frame("none.jsonl", None)

    assert cache.read_frame("empty.jsonl").empty
    assert cache.read_frame("none.jsonl").empty


def test_missing_required_frame_raises_with_description(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)

    assert cache.read_frame("optional.jsonl").empty
    with pytest.raises(FileNotFoundError, match="Required cache not found"):
        cache.read_frame("required.jsonl", required=True, description="Required cache")


def test_get_or_create_frame_reuses_cache_unless_forced(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    calls = 0

    def build_frame() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame([{"value": calls}])

    first = cache.get_or_create_frame("frame.jsonl", build_frame)
    second = cache.get_or_create_frame("frame.jsonl", build_frame)
    third = cache.get_or_create_frame("frame.jsonl", build_frame, force=True)

    assert first["value"].tolist() == [1]
    assert second["value"].tolist() == [1]
    assert third["value"].tolist() == [2]
    assert calls == 2


def test_parsed_data_cache_round_trips_standard_tables(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    parsed = ParsedData(
        interactions=pd.DataFrame([{"user_id": "u1", "item_id": "A"}]),
        user_table=pd.DataFrame([{"user_id": "u1"}]),
        item_table=pd.DataFrame([{"item_id": "A"}]),
    )

    assert not cache.parsed_exists()
    cache.write_parsed(parsed)

    assert cache.parsed_exists()
    loaded = cache.read_parsed()
    pd.testing.assert_frame_equal(loaded.interactions, parsed.interactions)
    pd.testing.assert_frame_equal(loaded.user_table, parsed.user_table)
    pd.testing.assert_frame_equal(loaded.item_table, parsed.item_table)
