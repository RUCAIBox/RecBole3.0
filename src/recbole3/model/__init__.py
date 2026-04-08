from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recbole3.dataset import DatasetTask
from recbole3.model.base import (
    BaseCollator,
    BaseModel,
    BaseModelDataset,
    BaseRankingModel,
    BaseRankingModelDataset,
    BaseRetrievalModel,
    BaseRetrievalModelDataset,
    ModelConfig,
)
from recbole3.model.hstu import (
    HSTUConfig,
    HSTUInteraction,
    HSTUModel,
    HSTUModelDataset,
    HSTURetrievalEvalRequest,
)
from recbole3.model.sequential import (
    BaseSequentialRankingModelDataset,
    BaseSequentialRetrievalModelDataset,
    SequentialModelConfig,
    SequentialInteraction,
    SequentialRetrievalEvalRequest,
    build_history_item_ids,
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static model table entry."""

    model_cls: type[BaseModel]
    config_cls: type[ModelConfig]
    model_data_cls: type[BaseModelDataset[Any, Any]] | None = None


MODEL_TABLE: dict[str, ModelSpec] = {
    "hstu": ModelSpec(
        model_cls=HSTUModel,
        config_cls=HSTUConfig,
        model_data_cls=HSTUModelDataset,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_TABLE[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_TABLE)) or "<empty>"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}") from exc


__all__ = [
    "BaseCollator",
    "BaseModel",
    "BaseModelDataset",
    "BaseRankingModel",
    "BaseRankingModelDataset",
    "BaseRetrievalModel",
    "BaseRetrievalModelDataset",
    "BaseSequentialRankingModelDataset",
    "BaseSequentialRetrievalModelDataset",
    "HSTUConfig",
    "HSTUInteraction",
    "HSTUModel",
    "HSTUModelDataset",
    "HSTURetrievalEvalRequest",
    "MODEL_TABLE",
    "ModelConfig",
    "ModelSpec",
    "SequentialModelConfig",
    "SequentialInteraction",
    "SequentialRetrievalEvalRequest",
    "build_history_item_ids",
    "get_model_spec",
]
