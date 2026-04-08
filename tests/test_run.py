from __future__ import annotations

import os
from pathlib import Path

import pytest

from recbole3.run import _accelerate_runtime_device, compose_config, run_experiment
from tests.test_helpers import StubModelDataset, ensure_stub_tables


def test_full_stub_flow_runs(tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)
    (config_dir / "trainer").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: stub_model",
                "  - trainer: stub_trainer",
                "  - _self_",
                "runtime:",
                "  seed: 7",
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
    (config_dir / "model" / "stub_model.yaml").write_text("name: stub_model\n", encoding="utf-8")
    (config_dir / "trainer" / "stub_trainer.yaml").write_text(
        "\n".join(
            [
                "name: stub_trainer",
                "batch_size: 2",
                "shuffle: false",
                "optimizer:",
                "  name: SGD",
                "  kwargs:",
                "    lr: 0.001",
                "checkpoint:",
                "  save_last: true",
                "eval:",
                "  protocol: sampled",
                "  neg_sampling_num: 2",
                "  candidate_seed: 7",
                "  metrics:",
                "    - name: recall",
                "      ks: [3]",
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
    (config_dir / "trainer").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: stub_model_with_data",
                "  - trainer: stub_trainer",
                "  - _self_",
                "runtime:",
                "  seed: 7",
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
    (config_dir / "model" / "stub_model_with_data.yaml").write_text("name: stub_model_with_data\n", encoding="utf-8")
    (config_dir / "trainer" / "stub_trainer.yaml").write_text(
        "\n".join(
            [
                "name: stub_trainer",
                "batch_size: 2",
                "shuffle: false",
                "optimizer:",
                "  name: SGD",
                "  kwargs:",
                "    lr: 0.001",
                "eval:",
                "  protocol: sampled",
                "  neg_sampling_num: 2",
                "  candidate_seed: 7",
                "  metrics:",
                "    - name: recall",
                "      ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))

    assert isinstance(result["prepared_data"], StubModelDataset)
    assert result["prepared_data"].model_name == "stub_model_with_data"
    assert len(result["prepared_data"].get_train_dataset()) == 4


def test_runtime_device_sets_accelerate_env_temporarily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)

    with _accelerate_runtime_device("cuda:0"):
        assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cuda:0"

    assert "ACCELERATE_TORCH_DEVICE" not in os.environ



def test_runtime_device_rejects_distributed_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="runtime.device"):
        with _accelerate_runtime_device("cuda:0"):
            pass
