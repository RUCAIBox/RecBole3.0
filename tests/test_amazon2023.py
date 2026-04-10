from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import recbole3.dataset.amazon2023 as amazon2023_module
from recbole3.dataset import (
    Amazon2023Parser,
    Amazon2023RetrievalConfig,
    Amazon2023RetrievalDataset,
    Interaction,
    RetrievalEvalRequest,
    SplitConfig,
    get_dataset_spec,
)
from recbole3.evaluation import EvalConfig
from recbole3.run import compose_config, run_experiment
from tests.test_helpers import ensure_stub_tables


def _reviews_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 5.0, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "B", "rating": 4.0, "timestamp": 2},
            {"user_id": "u1", "parent_asin": "C", "rating": 3.0, "timestamp": 3},
            {"user_id": "u2", "parent_asin": "B", "rating": 5.0, "timestamp": 1},
            {"user_id": "u2", "parent_asin": "D", "rating": 4.0, "timestamp": 2},
            {"user_id": "u2", "parent_asin": "E", "rating": 3.0, "timestamp": 3},
        ]
    )


def _metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parent_asin": "A",
                "title": "<b>Alpha Pi</b>",
                "features": ["Fast", "Light"],
                "categories": ["Books", "Fiction"],
                "description": ["Line1\nLine2"],
            },
            {
                "parent_asin": "B",
                "title": "Bravo",
                "features": ["Durable"],
                "categories": ["Books"],
                "description": ["Tabbed\ttext"],
            },
            {
                "parent_asin": "C",
                "title": "Charlie",
                "features": [],
                "categories": ["Books"],
                "description": ["Desc"],
            },
            {
                "parent_asin": "D",
                "title": "Delta",
                "features": ["Heavy"],
                "categories": ["Books"],
                "description": ["Desc"],
            },
            {
                "parent_asin": "E",
                "title": "Echo",
                "features": ["Wide"],
                "categories": ["Books"],
                "description": ["Desc"],
            },
        ]
    )


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _install_fake_remote_loaders(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def resolve_subset(subset_name: str) -> pd.DataFrame:
        if subset_name.startswith("raw_meta_"):
            return _metadata_frame()
        if subset_name.endswith("_rating_only_Books"):
            return _reviews_frame()
        raise AssertionError(f"Unexpected Amazon 2023 subset name: {subset_name}")

    def fake_huggingface(dataset_id: str, subset_name: str, *, split: str, cache_dir: str, trust_remote_code: bool) -> Any:
        assert dataset_id == amazon2023_module.AMAZON2023_DATASET_ID
        assert split == "full"
        assert trust_remote_code is True
        assert cache_dir.endswith("amazon2023")
        calls.append(("huggingface", subset_name))
        return resolve_subset(subset_name)

    def fake_modelscope(dataset_id: str, subset_name: str, *, split: str, trust_remote_code: bool) -> Any:
        assert dataset_id == amazon2023_module.AMAZON2023_DATASET_ID
        assert split == "full"
        assert trust_remote_code is True
        calls.append(("modelscope", subset_name))
        return resolve_subset(subset_name)

    monkeypatch.setattr(amazon2023_module, "load_huggingface_dataset", fake_huggingface)
    monkeypatch.setattr(amazon2023_module, "load_modelscope_dataset", fake_modelscope)
    return calls


def _build_config(root: Path, *, download_source: str, metadata_mode: str = "sentence", **overrides: Any) -> Amazon2023RetrievalConfig:
    return Amazon2023RetrievalConfig(
        download_dir=str(root / "raw"),
        processed_dir=str(root / "processed"),
        category="Books",
        kcore="full",
        metadata_mode=metadata_mode,
        download_source=download_source,  # type: ignore[arg-type]
        split=SplitConfig(
            strategy="leave_one_out",
            order="chronological",
            per_user=True,
            valid_holdout_num=1,
            test_holdout_num=1,
        ),
        **overrides,
    )


@pytest.mark.parametrize("download_source", ["huggingface", "modelscope"])
def test_parser_writes_raw_and_parsed_caches(monkeypatch: pytest.MonkeyPatch, download_source: str, tmp_path: Path) -> None:
    calls = _install_fake_remote_loaders(monkeypatch)
    parser = Amazon2023Parser(_build_config(tmp_path, download_source=download_source))

    parsed = parser.parse()

    assert calls == [(download_source, "full_rating_only_Books"), (download_source, "raw_meta_Books")]
    assert len(parsed.interactions) == 6
    assert parser._raw_reviews_path().exists()
    assert parser._raw_metadata_path().exists()
    assert parser._interactions_path().exists()


def test_parser_reuses_parsed_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_remote_loaders(monkeypatch)
    parser = Amazon2023Parser(_build_config(tmp_path, download_source="huggingface"))
    first = parser.parse()
    monkeypatch.setattr(
        Amazon2023Parser,
        "_build_parsed_data",
        lambda self: (_ for _ in ()).throw(AssertionError("should reuse parsed cache")),
    )
    second = parser.parse()
    assert second.interactions == first.interactions


def test_parser_remaps_ids_and_materializes_metadata_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_remote_loaders(monkeypatch)
    parsed = Amazon2023Parser(_build_config(tmp_path, download_source="huggingface")).parse()

    assert [(record.user_id, record.item_id) for record in parsed.interactions] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 3),
        (1, 4),
    ]
    assert parsed.user_table["raw_user_id"].tolist() == ["u1", "u2"]
    assert parsed.item_table["raw_item_id"].tolist() == ["A", "B", "C", "D", "E"]
    first_metadata_text = parsed.item_table.loc[parsed.item_table["item_id"] == 0, "metadata_text"].item()
    assert "<b>" not in first_metadata_text
    assert "Alpha Pi" in first_metadata_text
    assert "Line1 Line2" in first_metadata_text


def test_prepare_builds_retrieval_records_in_chronological_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_remote_loaders(monkeypatch)
    dataset = Amazon2023RetrievalDataset(_build_config(tmp_path, download_source="huggingface", metadata_mode="none"))
    prepared = dataset.prepare(eval_config=_full_eval_config())

    assert list(prepared.get_train_dataset()) == [
        Interaction(user_id=0, item_id=0, timestamp=1, label=None),
        Interaction(user_id=1, item_id=1, timestamp=1, label=None),
    ]
    assert list(prepared.get_eval_dataset("valid")) == [
        RetrievalEvalRequest(user_id=0, item_id=1, timestamp=2, label=None, seen_item_ids=(0,)),
        RetrievalEvalRequest(user_id=1, item_id=3, timestamp=2, label=None, seen_item_ids=(1,)),
    ]
    assert list(prepared.get_eval_dataset("test")) == [
        RetrievalEvalRequest(user_id=0, item_id=2, timestamp=3, label=None, seen_item_ids=(0, 1)),
        RetrievalEvalRequest(user_id=1, item_id=4, timestamp=3, label=None, seen_item_ids=(1, 3)),
    ]


def test_metadata_mode_none_skips_metadata_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_remote_loaders(monkeypatch)
    parsed = Amazon2023Parser(_build_config(tmp_path, download_source="huggingface", metadata_mode="none")).parse()
    assert calls == [("huggingface", "full_rating_only_Books")]
    assert "metadata_text" not in parsed.item_table.columns


@pytest.mark.parametrize(
    ("category", "kcore", "match"),
    [("NotARealCategory", "full", "not available"), ("Appliances", "5core", "does not provide 5-core")],
)
def test_invalid_source_config_raises(category: str, kcore: str, match: str, tmp_path: Path) -> None:
    parser = Amazon2023Parser(
        Amazon2023RetrievalConfig(
            category=category,
            kcore=kcore,
            metadata_mode="none",
            download_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        )
    )
    with pytest.raises(ValueError, match=match):
        parser.parse()


@pytest.mark.parametrize(
    ("loader", "missing_prefix", "match"),
    [
        ("load_huggingface_dataset", "datasets", r"recbole3\[huggingface\]"),
        ("load_modelscope_dataset", "modelscope", r"recbole3\[modelscope\]"),
    ],
)
def test_missing_optional_dependency_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    loader: str,
    missing_prefix: str,
    match: str,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == missing_prefix or name.startswith(f"{missing_prefix}."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match=match):
        if loader == "load_huggingface_dataset":
            amazon2023_module.load_huggingface_dataset(
                amazon2023_module.AMAZON2023_DATASET_ID,
                "full_rating_only_Books",
                split="full",
                cache_dir="cache",
                trust_remote_code=True,
            )
        else:
            amazon2023_module.load_modelscope_dataset(
                amazon2023_module.AMAZON2023_DATASET_ID,
                "full_rating_only_Books",
                split="full",
                trust_remote_code=True,
            )


def test_table_and_run_experiment_support_amazon2023(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_remote_loaders(monkeypatch)
    ensure_stub_tables()
    assert get_dataset_spec("amazon2023_retrieval").dataset_cls is Amazon2023RetrievalDataset

    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: amazon2023_retrieval",
                "  - model: stub_model",
                "  - _self_",
                "runtime:",
                "  seed: 7",
                "  device: cpu",
                f"  output_dir: {(tmp_path / 'outputs').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "amazon2023_retrieval.yaml").write_text(
        "\n".join(
            [
                "name: amazon2023_retrieval",
                "category: Books",
                "kcore: full",
                "metadata_mode: none",
                "download_source: huggingface",
                f"download_dir: {(tmp_path / 'raw').as_posix()}",
                f"processed_dir: {(tmp_path / 'processed').as_posix()}",
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
