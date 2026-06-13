from __future__ import annotations

from typing import Any

from recbole3.model.minionerec.config import (
    MiniOneRecConfig,
    MiniOneRecRewardType,
    MiniOneRecSIDFileItemIDSpace,
    MiniOneRecStage,
)


__all__ = [
    "MiniOneRecConfig",
    "MiniOneRecRewardType",
    "MiniOneRecSIDFileItemIDSpace",
    "MiniOneRecStage",
    "MiniOneRecSIDCodec",
    "MiniOneRecTrainer",
    "MiniOneRecPipeline",
    "MiniOneRecGRPOTrainer",
]


def __getattr__(name: str) -> Any:
    if name == "MiniOneRecSIDCodec":
        from recbole3.model.minionerec.data import MiniOneRecSIDCodec

        globals()[name] = MiniOneRecSIDCodec
        return MiniOneRecSIDCodec
    if name == "MiniOneRecTrainer":
        from recbole3.model.minionerec.trainer import MiniOneRecTrainer

        globals()[name] = MiniOneRecTrainer
        return MiniOneRecTrainer
    if name == "MiniOneRecPipeline":
        from recbole3.model.minionerec.pipeline import MiniOneRecPipeline

        globals()[name] = MiniOneRecPipeline
        return MiniOneRecPipeline
    if name == "MiniOneRecGRPOTrainer":
        from recbole3.model.minionerec.rl import MiniOneRecGRPOTrainer

        globals()[name] = MiniOneRecGRPOTrainer
        return MiniOneRecGRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
