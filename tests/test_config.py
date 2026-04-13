from __future__ import annotations

from recbole3.config import instantiate_dataclass
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.trainer import CheckpointConfig, EarlyStoppingConfig, OptimizerConfig, TrainerConfig
from tests.test_helpers import StubTrainerConfig


def test_trainer_config_defaults_to_adam_optimizer() -> None:
    trainer_config = TrainerConfig(eval=EvalConfig(protocol="labeled"))

    assert trainer_config.optimizer == OptimizerConfig(name="Adam", kwargs={"lr": 1e-3})
    assert trainer_config.scheduler is None


def test_eval_config_requires_explicit_protocol() -> None:
    try:
        EvalConfig()  # type: ignore[call-arg]
    except TypeError as exc:
        assert "protocol" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("EvalConfig() should require protocol.")


def test_trainer_config_requires_explicit_eval() -> None:
    try:
        TrainerConfig()  # type: ignore[call-arg]
    except TypeError as exc:
        assert "eval" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("TrainerConfig() should require eval.")


def test_instantiate_dataclass_recursively_builds_nested_trainer_config() -> None:
    """Nested trainer config values should materialize into trainer and eval dataclasses."""

    trainer_config = instantiate_dataclass(
        StubTrainerConfig,
        {
            "batch_size": 2,
            "optimizer": {
                "name": "AdamW",
                "kwargs": {
                    "lr": 0.01,
                    "weight_decay": 0.1,
                },
            },
            "monitor": "recall@5",
            "early_stopping": {
                "enabled": True,
                "patience": 2,
                "min_delta": 0.1,
            },
            "checkpoint": {
                "save_best": True,
                "save_last": True,
            },
            "eval": {
                "protocol": "sampled",
                "metrics": [
                    {"name": "recall", "ks": [5, 10]},
                ],
                "neg_sampling_num": 20,
            },
        },
    )

    assert trainer_config.monitor == "recall@5"
    assert isinstance(trainer_config.optimizer, OptimizerConfig)
    assert trainer_config.optimizer == OptimizerConfig(name="AdamW", kwargs={"lr": 0.01, "weight_decay": 0.1})
    assert isinstance(trainer_config.early_stopping, EarlyStoppingConfig)
    assert trainer_config.early_stopping.enabled is True
    assert trainer_config.early_stopping.patience == 2
    assert trainer_config.early_stopping.min_delta == 0.1
    assert isinstance(trainer_config.checkpoint, CheckpointConfig)
    assert trainer_config.checkpoint.save_best is True
    assert trainer_config.checkpoint.save_last is True
    assert isinstance(trainer_config.eval, EvalConfig)
    assert trainer_config.eval.protocol == "sampled"
    assert trainer_config.eval.neg_sampling_num == 20
    assert trainer_config.eval.metrics == (MetricSpec(name="recall", ks=(5, 10)),)

