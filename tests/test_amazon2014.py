from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from recbole3.dataset import (
    ITEM_ID,
    LABEL,
    SEEN_ITEM_IDS,
    TIMESTAMP,
    USER_ID,
    Amazon2014RetrievalConfig,
    Amazon2014RetrievalDataset,
    Amazon2014RetrievalParser,
    SplitConfig,
    get_dataset_spec,
)
from recbole3.evaluation import EvalConfig
from recbole3.run import compose_config, run_experiment
from tests.test_helpers import ensure_stub_tables


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _build_config(root: Path, *, metadata_mode: str = "sentence", **overrides: Any) -> Amazon2014RetrievalConfig:
    return Amazon2014RetrievalConfig(
        download_dir=str(root / "raw"),
        processed_dir=str(root / "processed"),
        category="Beauty",
        metadata_mode=metadata_mode,  # type: ignore[arg-type]
        download_source="snap",
        split=SplitConfig(
            strategy="leave_one_out",
            order="chronological",
            per_user=True,
            valid_holdout_num=1,
            test_holdout_num=1,
        ),
        **overrides,
    )


def _write_json_gz(path: Path, rows: list[dict[str, Any]], *, python_literal: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write((repr(row) if python_literal else json.dumps(row)) + "\n")


def _write_amazon2014_source(root: Path) -> None:
    raw_dir = root / "raw" / "amazon2014" / "Beauty"
    _write_json_gz(
        raw_dir / "reviews_Beauty_5.json.gz",
        [
            {"reviewerID": "u1", "asin": "A", "unixReviewTime": 1, "overall": 5.0},
            {"reviewerID": "u1", "asin": "B", "unixReviewTime": 2, "overall": 2.0},
            {"reviewerID": "u1", "asin": "C", "unixReviewTime": 3, "overall": 4.0},
            {"reviewerID": "u2", "asin": "B", "unixReviewTime": 1, "overall": 3.0},
            {"reviewerID": "u2", "asin": "D", "unixReviewTime": 2, "overall": 5.0},
            {"reviewerID": "u2", "asin": "E", "unixReviewTime": 3, "overall": 1.0},
        ],
        python_literal=True,
    )
    _write_json_gz(
        raw_dir / "meta_Beauty.json.gz",
        [
            {
                "asin": "A",
                "title": "<b>Alpha</b>",
                "price": "$1.00",
                "brand": "BrandA",
                "feature": ["Fast", "Light"],
                "categories": [["Beauty", "Skin"]],
                "description": ["Line1\nLine2"],
            },
            {"asin": "B", "title": "Bravo", "feature": ["Durable"], "categories": ["Beauty"], "description": ["Tabbed\ttext"]},
            {"asin": "C", "title": "Charlie", "feature": [], "categories": ["Beauty"], "description": ["Desc"]},
            {"asin": "D", "title": "Delta", "feature": ["Heavy"], "categories": ["Beauty"], "description": ["Desc"]},
        ],
    )


def test_amazon2014_dataset_registry_exposes_retrieval_components() -> None:
    spec = get_dataset_spec("amazon2014_retrieval")

    assert spec.dataset_cls is Amazon2014RetrievalDataset
    assert spec.config_cls is Amazon2014RetrievalConfig
    assert spec.dataset_cls.parser_cls is Amazon2014RetrievalParser


def test_amazon2014_config_loads_expected_defaults() -> None:
    cfg = compose_config(overrides=["dataset=amazon2014_retrieval"])

    assert cfg.dataset.name == "amazon2014_retrieval"
    assert cfg.dataset.category == "Beauty"
    assert cfg.dataset.metadata_mode == "sentence"
    assert cfg.dataset.download_source == "snap"
    assert cfg.dataset.refresh_cache is False
    assert cfg.dataset.split.strategy == "leave_one_out"
    assert cfg.dataset.split.order == "chronological"
    assert cfg.dataset.split.per_user is True
    assert cfg.dataset.split.valid_holdout_num == 1
    assert cfg.dataset.split.test_holdout_num == 1


def test_amazon2014_parser_reads_local_gz_and_materializes_caches(tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    parser = Amazon2014RetrievalParser(_build_config(tmp_path))

    parsed = parser.parse()

    assert len(parsed.interactions) == 6
    assert (tmp_path / "raw" / "amazon2014" / "Beauty" / "reviews.jsonl").exists()
    assert (tmp_path / "raw" / "amazon2014" / "Beauty" / "meta.jsonl").exists()
    assert (tmp_path / "processed" / "amazon2014_retrieval" / "Beauty" / "sentence" / "interactions.jsonl").exists()


def test_amazon2014_parser_reuses_parsed_cache(monkeypatch, tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    parser = Amazon2014RetrievalParser(_build_config(tmp_path))
    first = parser.parse()

    monkeypatch.setattr(
        Amazon2014RetrievalParser,
        "_build_parsed_data",
        lambda self: (_ for _ in ()).throw(AssertionError("should reuse parsed cache")),
    )
    second = parser.parse()

    first_interactions = first.interactions.copy()
    second_interactions = second.interactions.copy()
    first_interactions[LABEL] = first_interactions[LABEL].astype(object).where(first_interactions[LABEL].notna(), None)
    second_interactions[LABEL] = second_interactions[LABEL].astype(object).where(second_interactions[LABEL].notna(), None)
    pd.testing.assert_frame_equal(second_interactions, first_interactions, check_dtype=False)


def test_amazon2014_parser_rebuilds_stale_parsed_cache_without_overall(tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    stale_cache = tmp_path / "processed" / "amazon2014_retrieval" / "Beauty" / "sentence"
    stale_cache.mkdir(parents=True)
    pd.DataFrame(
        [
            {USER_ID: "old_user", ITEM_ID: "old_item", TIMESTAMP: 1, LABEL: None},
        ]
    ).to_json(stale_cache / "interactions.jsonl", orient="records", lines=True)
    pd.DataFrame([{USER_ID: "old_user"}]).to_json(stale_cache / "users.jsonl", orient="records", lines=True)
    pd.DataFrame([{ITEM_ID: "old_item", "metadata_text": "old metadata"}]).to_json(
        stale_cache / "items.jsonl",
        orient="records",
        lines=True,
    )

    parsed = Amazon2014RetrievalParser(_build_config(tmp_path)).parse()

    assert parsed.interactions["overall"].tolist() == [5, 2, 4, 3, 5, 1]
    assert parsed.user_table[USER_ID].tolist() == ["u1", "u2"]


def test_amazon2014_parser_returns_raw_ids_and_metadata_text(tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    parsed = Amazon2014RetrievalParser(_build_config(tmp_path)).parse()

    assert parsed.interactions[[USER_ID, ITEM_ID, TIMESTAMP, LABEL]].to_dict("records") == [
        {USER_ID: "u1", ITEM_ID: "A", TIMESTAMP: 1, LABEL: None},
        {USER_ID: "u1", ITEM_ID: "B", TIMESTAMP: 2, LABEL: None},
        {USER_ID: "u1", ITEM_ID: "C", TIMESTAMP: 3, LABEL: None},
        {USER_ID: "u2", ITEM_ID: "B", TIMESTAMP: 1, LABEL: None},
        {USER_ID: "u2", ITEM_ID: "D", TIMESTAMP: 2, LABEL: None},
        {USER_ID: "u2", ITEM_ID: "E", TIMESTAMP: 3, LABEL: None},
    ]
    assert parsed.interactions["overall"].tolist() == [5, 2, 4, 3, 5, 1]
    assert parsed.user_table[USER_ID].tolist() == ["u1", "u2"]
    assert parsed.item_table[ITEM_ID].tolist() == ["A", "B", "C", "D", "E"]
    first_metadata_text = parsed.item_table.loc[parsed.item_table[ITEM_ID] == "A", "metadata_text"].item()
    missing_metadata_text = parsed.item_table.loc[parsed.item_table[ITEM_ID] == "E", "metadata_text"].item()
    assert "<b>" not in first_metadata_text
    assert "Alpha" in first_metadata_text
    assert "Fast, Light" in first_metadata_text
    assert "Beauty, Skin" in first_metadata_text
    assert "Line1 Line2" in first_metadata_text
    assert missing_metadata_text == ""


def test_amazon2014_parser_fields_mode_preserves_structured_metadata(tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    parsed = Amazon2014RetrievalParser(_build_config(tmp_path, metadata_mode="fields")).parse()

    first_item = parsed.item_table.loc[parsed.item_table[ITEM_ID] == "A"].iloc[0].to_dict()
    missing_item = parsed.item_table.loc[parsed.item_table[ITEM_ID] == "E"].iloc[0].to_dict()

    assert first_item["title"] == "Alpha"
    assert first_item["brand"] == "BrandA"
    assert first_item["feature"] == "Fast, Light"
    assert first_item["categories"] == "Beauty, Skin"
    assert first_item["description"] == "Line1 Line2"
    assert "Alpha" in first_item["metadata_text"]
    assert missing_item["title"] == ""
    assert missing_item["brand"] == ""
    assert missing_item["metadata_text"] == ""


def test_amazon2014_prepare_builds_retrieval_records_in_chronological_order(tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    dataset = Amazon2014RetrievalDataset(_build_config(tmp_path, metadata_mode="none"))
    prepared = dataset.prepare(eval_config=_full_eval_config())

    assert prepared.get_train_dataset().frame[[USER_ID, ITEM_ID, TIMESTAMP]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 0, TIMESTAMP: 1},
        {USER_ID: 1, ITEM_ID: 1, TIMESTAMP: 1},
    ]
    assert prepared.get_eval_dataset("valid").frame[[USER_ID, ITEM_ID, TIMESTAMP, SEEN_ITEM_IDS]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, SEEN_ITEM_IDS: (0,)},
        {USER_ID: 1, ITEM_ID: 3, TIMESTAMP: 2, SEEN_ITEM_IDS: (1,)},
    ]
    assert prepared.get_eval_dataset("test").frame[[USER_ID, ITEM_ID, TIMESTAMP, SEEN_ITEM_IDS]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, SEEN_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 4, TIMESTAMP: 3, SEEN_ITEM_IDS: (1, 3)},
    ]


def test_amazon2014_run_experiment_smoke_with_stub_model(tmp_path: Path) -> None:
    _write_amazon2014_source(tmp_path)
    ensure_stub_tables()

    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: amazon2014_retrieval",
                "  - model: stub_model",
                "  - _self_",
                "runtime:",
                "  device: cpu",
                f"  output_dir: {(tmp_path / 'outputs').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "amazon2014_retrieval.yaml").write_text(
        "\n".join(
            [
                "name: amazon2014_retrieval",
                "category: Beauty",
                "metadata_mode: none",
                "download_source: snap",
                f"download_dir: {(tmp_path / 'raw').as_posix()}",
                f"processed_dir: {(tmp_path / 'processed').as_posix()}",
                "refresh_cache: false",
                "split:",
                "  strategy: leave_one_out",
                "  order: chronological",
                "  per_user: true",
                "  valid_holdout_num: 1",
                "  test_holdout_num: 1",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "model" / "stub_model.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: stub_model",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  optimizer:",
                "    name: SGD",
                "    kwargs:",
                "      lr: 0.001",
                "  eval:",
                "    protocol: sampled",
                "    neg_sampling_num: 2",
                "    candidate_seed: 7",
                "    metrics:",
                "      - name: recall",
                "        ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))
    assert result["prepared_data"].get_num_users() == 2
    assert result["prepared_data"].get_num_items() == 5
    assert len(result["prepared_data"].get_train_dataset()) == 2
    assert len(result["prepared_data"].get_eval_dataset("valid")) == 2
    assert len(result["prepared_data"].get_eval_dataset("test")) == 2
    assert result["test"]["protocol"] == "sampled"
