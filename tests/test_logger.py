from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from recbole3.logger import TrainingLogger, _format_scalar, _is_main_process, _is_nested_container


# ── helpers ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _MockConfig:
    name: str = "test_model"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    use_bias: bool = True


class _MockModel:
    def __init__(self):
        self.config = _MockConfig()

    def parameters(self):
        import torch

        return iter([torch.ones(1000, requires_grad=True), torch.ones(2000, requires_grad=False)])


class _MockPreparedData:
    def __init__(self):
        self.config = _MockConfig(name="test_dataset")

    @property
    def task(self):
        return "retrieval"

    def get_num_users(self):
        return 10000

    def get_num_items(self):
        return 50000

    def get_train_dataset(self):
        return list(range(400000))

    def get_eval_dataset(self, _split: str):
        return list(range(10000))


def _read_log(log_dir: Path) -> str:
    files = list(log_dir.glob("*.log"))
    assert len(files) == 1
    return files[0].read_text()


# ── unit: helpers ────────────────────────────────────────────────────────────


class TestFormatScalar:
    def test_bool(self):
        assert _format_scalar(True) == "true"
        assert _format_scalar(False) == "false"

    def test_float(self):
        assert "0.1" in _format_scalar(0.1)

    def test_regular(self):
        assert _format_scalar(42) == "42"
        assert _format_scalar("hello") == "hello"


class TestIsNestedContainer:
    def test_dict(self):
        assert _is_nested_container({"a": 1}) is True

    def test_list(self):
        assert _is_nested_container([1, 2]) is True

    def test_tuple(self):
        assert _is_nested_container((1, 2)) is True

    def test_string_is_not_nested(self):
        assert _is_nested_container("abc") is False

    def test_scalar_is_not_nested(self):
        assert _is_nested_container(42) is False


class TestIsMainProcess:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert _is_main_process() is True

    def test_rank_zero(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert _is_main_process() is True

    def test_rank_minus_one(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "-1")
        assert _is_main_process() is True

    def test_rank_one_skips(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert _is_main_process() is False


# ── integration: file creation ───────────────────────────────────────────────


class TestLoggerFileCreation:
    def test_creates_log_directory_and_file(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "model_a", "dataset_a", "")
        log_dir = tmp_path / "logs"
        assert log_dir.exists()
        assert log_dir.is_dir()
        files = list(log_dir.glob("*.log"))
        assert len(files) == 1
        assert "model_a" in files[0].name
        assert "dataset_a" in files[0].name
        logger.close()

    def test_filename_contains_model_dataset_timestamp(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "hstu", "ml-1m", "Music")
        log_dir = tmp_path / "logs"
        files = list(log_dir.glob("*.log"))
        name = files[0].name
        assert name.startswith("hstu_ml-1m_Music_")
        assert name.endswith(".log")
        # timestamp portion: YYYYMMDD_HHMMSS
        ts = name[len("hstu_ml-1m_Music_"): -len(".log")]
        assert len(ts) == 15
        assert "_" in ts
        logger.close()

    def test_writes_header(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "RecBole3.0 Training Log" in text
        assert "Model: m" in text
        assert "Dataset: d" in text
        assert "Started:" in text

    def test_output_dir_none_uses_dot(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        logger = TrainingLogger(".", "m", "d", "")
        logger.close()
        assert (tmp_path / "logs").exists()


# ── integration: config logging ──────────────────────────────────────────────


class TestLoggerConfig:
    def test_logs_dataclass_config(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_config("Trainer", _MockConfig())
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Config: Trainer" in text
        assert "hidden_dim: 128" in text
        assert "num_layers: 3" in text
        assert "dropout: 0.1" in text
        assert "use_bias: true" in text

    def test_logs_dict_config(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_config("Custom", {"lr": 0.01, "batch_size": 64})
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "lr: 0.01" in text
        assert "batch_size: 64" in text

    def test_logs_nested_dataclass(self, tmp_path: Path):
        @dataclasses.dataclass
        class Inner:
            value: float = 1.0

        @dataclasses.dataclass
        class Outer:
            inner: Inner = dataclasses.field(default_factory=Inner)
            outer_name: str = "test"

        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_config("Nested", Outer())
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "inner:" in text
        assert "value: 1" in text
        assert "outer_name: test" in text


# ── integration: model info ──────────────────────────────────────────────────


class TestEnsureInitialized:
    """Tests for the BaseModel.ensure_initialized hook."""

    def test_base_model_noop(self):
        from recbole3.model.base import BaseModel, ModelConfig

        class _NoOpModel(BaseModel):
            def build_train_collator(self, prepared_data):
                raise NotImplementedError

            def build_eval_collator(self, prepared_data):
                raise NotImplementedError

            def forward(self, batch):
                raise NotImplementedError

            def compute_loss(self, batch, outputs):
                raise NotImplementedError

        model = _NoOpModel(ModelConfig(name="test"))
        model.ensure_initialized(None)  # should not raise

    def test_hstu_overrides_and_is_idempotent(self, tmp_path: Path):
        pytest.importorskip("fbgemm_gpu")
        from recbole3.model.hstu.config import HSTUConfig
        from recbole3.model.hstu.model import HSTUModel

        model = HSTUModel(HSTUConfig(name="hstu", history_max_length=10))
        assert model._num_items is None

        # First call triggers init
        model.ensure_initialized(_MockPreparedData())
        assert model._num_items == 50000
        assert model._item_embeddings is not None
        params_before = sum(p.numel() for p in model.parameters())

        # Second call is idempotent, keeps same params
        model.ensure_initialized(_MockPreparedData())
        params_after = sum(p.numel() for p in model.parameters())
        assert params_after == params_before

        # Verify logger now sees parameters
        logger = TrainingLogger(str(tmp_path), "hstu", "test_dataset", "")
        logger.log_model_info(model)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Name:         hstu" in text
        assert f"Params:       {params_before:,}" in text


class TestLoggerModelInfo:
    def test_logs_model_info(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_model_info(_MockModel())
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Name:         test_model" in text
        assert "Class:        _MockModel" in text
        assert "Params:       3,000" in text
        assert "Trainable:" in text


# ── integration: dataset info ────────────────────────────────────────────────


class TestLoggerDatasetInfo:
    def test_logs_dataset_info(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_dataset_info(_MockPreparedData())
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Name:         test_dataset" in text
        assert "Task:         retrieval" in text
        assert "Users:        10,000" in text
        assert "Items:        50,000" in text
        assert "Train:        400,000 records" in text
        assert "Valid:        10,000 records" in text
        assert "Test:         10,000 records" in text


# ── integration: epoch logging ───────────────────────────────────────────────


class TestLoggerEpoch:
    def test_logs_epoch_row(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_epoch(1, 10, 0.123456, 1000, 12.34, lr=0.001)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "1/10" in text
        assert "0.123456" in text
        assert "12.34" in text
        assert "1000" in text
        assert "0.001000" in text

    def test_epoch_with_none_loss(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_epoch(1, 5, None, 500, 3.0)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "-" in text  # placeholder for None

    def test_extra_fields_from_first_epoch_determine_columns(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_epoch(1, 10, 1.0, 100, 1.0, lr=0.001, recon_loss=0.5, quant_loss=0.1)
        logger.log_epoch(2, 10, 0.9, 100, 0.9, lr=0.0009, recon_loss=0.4, quant_loss=0.08)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "recon_loss" in text
        assert "quant_loss" in text

    def test_extra_fields_not_in_first_epoch_are_dropped(self, tmp_path: Path):
        """Extra kwargs in later epochs are ignored if not present in epoch 1."""
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_epoch(1, 10, 1.0, 100, 1.0, lr=0.001)
        logger.log_epoch(2, 10, 0.9, 100, 0.9, lr=0.0009, recon_loss=0.4)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "recon_loss" not in text


# ── integration: validation logging ──────────────────────────────────────────


class TestLoggerValidation:
    def test_logs_validation_metrics(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_epoch(1, 10, 0.5, 100, 1.0, lr=0.001)
        logger.log_validation(1, {"ndcg@10": 0.1234, "recall@10": 0.2345})
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Valid epoch 1:" in text
        assert "ndcg@10=0.123400" in text
        assert "recall@10=0.234500" in text

    def test_empty_metrics(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_epoch(1, 10, 0.5, 100, 1.0, lr=0.001)
        logger.log_validation(1, {})
        logger.close()
        text = _read_log(tmp_path / "logs")
        # should not crash; "{}" expected
        assert "Valid epoch 1: {}" in text


# ── integration: best / early-stop / test / summary ──────────────────────────


class TestLoggerBest:
    def test_logs_best_epoch(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_best(8, "ndcg@10", 0.2012)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Best: epoch 8" in text
        assert "ndcg@10=0.201200" in text


class TestLoggerEarlyStopping:
    def test_stopped_true(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_early_stopping(stopped=True, epoch=5, patience=3)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Early stopping triggered" in text
        assert "epoch 5" in text

    def test_stopped_false(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_early_stopping(stopped=False, epoch=10, patience=5)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Training completed" in text


class TestLoggerTestResults:
    def test_logs_test_results(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_test({
            "split": "test",
            "protocol": "retrieval",
            "metrics": {"ndcg@10": 0.1987, "recall@10": 0.3456},
        })
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "TEST RESULTS" in text
        assert "Protocol:     retrieval" in text
        assert "ndcg@10:      0.198700" in text
        assert "recall@10:    0.345600" in text


class TestLoggerSummary:
    def test_logs_summary(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_summary(stopped_early=False, total_epochs=10, best_epoch=8, total_time=95.43)
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Total epochs:   10" in text
        assert "Best epoch:     8" in text
        assert "Stopped early:  False" in text


class TestLoggerFooter:
    def test_close_writes_footer(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.close()
        text = _read_log(tmp_path / "logs")
        assert "Run completed:" in text
        assert "total:" in text


# ── integration: DDP guard ───────────────────────────────────────────────────


class TestLoggerDDPGuard:
    def test_non_main_process_writes_nothing(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("LOCAL_RANK", "1")
        logger = TrainingLogger(str(tmp_path), "m", "d", "")
        logger.log_config("Trainer", _MockConfig())
        logger.log_model_info(_MockModel())
        logger.log_dataset_info(_MockPreparedData())
        logger.log_epoch(1, 10, 0.5, 100, 1.0, lr=0.001)
        logger.log_validation(1, {"ndcg@10": 0.1})
        logger.log_best(1, "ndcg@10", 0.1)
        logger.log_early_stopping(stopped=False, epoch=10, patience=5)
        logger.log_test({"split": "test", "protocol": "full", "metrics": {}})
        logger.log_summary(stopped_early=False, total_epochs=10, best_epoch=1, total_time=1.0)
        logger.close()

        log_dir = tmp_path / "logs"
        assert not log_dir.exists() or len(list(log_dir.glob("*.log"))) == 0


# ── integration: full log file ───────────────────────────────────────────────


class TestFullLogFile:
    def test_full_log_output(self, tmp_path: Path):
        logger = TrainingLogger(str(tmp_path), "hstu", "amazon2023_retrieval", "Musical_Instruments")
        logger.log_config("Trainer", _MockConfig(name="trainer_cfg"))
        logger.log_config("Model", _MockConfig())
        logger.log_config("Dataset", _MockConfig(name="amazon2023_retrieval"))
        logger.log_model_info(_MockModel())
        logger.log_dataset_info(_MockPreparedData())

        logger.log_epoch(1, 3, 2.345678, 1000, 12.34, lr=0.001)
        logger.log_validation(1, {"ndcg@10": 0.1234, "recall@10": 0.2345})
        logger.log_epoch(2, 3, 1.987654, 1000, 11.87, lr=0.00099)
        logger.log_validation(2, {"ndcg@10": 0.1567, "recall@10": 0.2891})
        logger.log_epoch(3, 3, 1.654321, 1000, 11.90, lr=0.00098)
        logger.log_validation(3, {"ndcg@10": 0.1890, "recall@10": 0.3120})

        logger.log_best(3, "ndcg@10", 0.1890)
        logger.log_early_stopping(stopped=False, epoch=3, patience=5)
        logger.log_test({
            "split": "test",
            "protocol": "retrieval",
            "metrics": {"ndcg@10": 0.1850, "recall@10": 0.3080},
        })
        logger.log_summary(stopped_early=False, total_epochs=3, best_epoch=3, total_time=36.11)
        logger.close()

        text = _read_log(tmp_path / "logs")

        # Structural integrity
        assert text.count("====") >= 6  # header, sections, footer rules
        assert "RecBole3.0 Training Log" in text
        assert "Config: Trainer" in text
        assert "Config: Model" in text
        assert "Config: Dataset" in text
        assert "--- Model " in text
        assert "--- Dataset " in text
        assert "--- Epochs " in text
        assert "TEST RESULTS" in text

        # Ordering: header before configs, epochs before test, test before footer
        header_pos = text.index("RecBole3.0")
        config_pos = text.index("Config: Trainer")
        epochs_pos = text.index("--- Epochs")
        test_pos = text.index("TEST RESULTS")
        footer_pos = text.index("Run completed:")

        assert header_pos < config_pos < epochs_pos < test_pos < footer_pos
