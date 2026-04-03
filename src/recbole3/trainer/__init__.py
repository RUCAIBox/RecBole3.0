from __future__ import annotations

from dataclasses import dataclass

from recbole3.trainer.base import (
    CheckpointConfig,
    EarlyStoppingConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainerConfig,
)


@dataclass(frozen=True, slots=True)
class TrainerSpec:
    """Static trainer table entry."""

    trainer_cls: type[Trainer]
    config_cls: type[TrainerConfig]


TRAINER_TABLE: dict[str, TrainerSpec] = {
    "base": TrainerSpec(
        trainer_cls=Trainer,
        config_cls=TrainerConfig,
    ),
}


def get_trainer_spec(name: str) -> TrainerSpec:
    try:
        return TRAINER_TABLE[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRAINER_TABLE)) or "<empty>"
        raise KeyError(f"Unknown trainer '{name}'. Available trainers: {available}") from exc


__all__ = [
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TRAINER_TABLE",
    "Trainer",
    "TrainerConfig",
    "TrainerSpec",
    "get_trainer_spec",
]
