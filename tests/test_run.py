from __future__ import annotations

import os
from pathlib import Path

import pytest

from recbole3.run import compose_config, run_experiment
from tests.test_helpers import StubModelDataset, ensure_stub_tables


def test_full_stub_flow_runs(tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: stub_model",
                "  - _self_",
                "runtime:",
                "  device: cpu",
                f"  output_dir: {(tmp_path / 'outputs').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "stub_dataset.yaml").write_text(
        "\n".join(
            [
                "name: stub_dataset",
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
                "  checkpoint:",
                "    save_last: true",
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

    assert len(result["prepared_data"].get_train_dataset()) == 4
    assert len(result["prepared_data"].get_eval_dataset("valid")) == 2
    assert len(result["prepared_data"].get_eval_dataset("test")) == 2
    assert result["fit"]["train_history"][0]["num_batches"] == 2
    assert Path(result["fit"]["checkpoint_paths"]["last"]).exists()
    assert result["test"]["protocol"] == "sampled"




def test_run_wraps_task_dataset_with_model_data_class(tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = tmp_path / "configs_model_data"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: stub_model_with_data",
                "  - _self_",
                "runtime:",
                "  device: cpu",
                f"  output_dir: {(tmp_path / 'outputs_model_data').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "stub_dataset.yaml").write_text(
        "\n".join(
            [
                "name: stub_dataset",
                f"processed_dir: {(tmp_path / 'processed_model_data').as_posix()}",
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
    (config_dir / "model" / "stub_model_with_data.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: stub_model_with_data",
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

    assert isinstance(result["prepared_data"], StubModelDataset)
    assert result["prepared_data"].model_name == "stub_model_with_data"
    assert len(result["prepared_data"].get_train_dataset()) == 4

