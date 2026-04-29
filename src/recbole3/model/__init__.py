from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedModel

from recbole3.dataset import DatasetTask
from recbole3.model.base import (
    BaseCollator,
    BaseModel,
    BaseModelDataset,
    BaseRankingModel,
    BaseRetrievalModel,
    ModelConfig,
    ModelDatasets,
)
from recbole3.model.hstu import (
    HISTORY_TIMESTAMPS,
    HSTUConfig,
    HSTUModel,
    HSTUModelDataset,
)
from recbole3.model.lcrec import LCRecConfig
from recbole3.model.lcrec.pipeline import LCRecPipeline
from recbole3.model.rqvae import (
    RQVAEConfig,
    RQVAEModel,
    RQVAEModelDataset,
    RQVAETrainer,
)
from recbole3.model.tiger import (
    TIGERConfig,
    TIGERModel,
    TIGERModelDataset,
)
from recbole3.model.sequential import (
    BaseSequentialModelDataset,
    HISTORY_ITEM_IDS,
    SequentialModelConfig,
    build_history_item_ids,
)
from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig
from recbole3.pipeline import Pipeline


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static model table entry."""

    model_cls: type[BaseModel] | Any
    config_cls: type[ModelConfig]
    model_data_cls: type[BaseModelDataset[Any, Any]] | None = None
    trainer_cls: type[Trainer] = Trainer
    trainer_config_cls: type[TrainerConfig] = TrainerConfig
    pipeline_cls: type[Pipeline] = Pipeline


MODEL_TABLE: dict[str, ModelSpec] = {
    "hstu": ModelSpec(
        model_cls=HSTUModel,
        config_cls=HSTUConfig,
        model_data_cls=HSTUModelDataset,
        trainer_cls=Trainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=Pipeline,
    ),
    "rqvae": ModelSpec(
        model_cls=RQVAEModel,
        config_cls=RQVAEConfig,
        model_data_cls=RQVAEModelDataset,
        trainer_cls=RQVAETrainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=Pipeline,
    ),
    "lcrec": ModelSpec(
        model_cls=PreTrainedModel,
        config_cls=LCRecConfig,
        pipeline_cls=LCRecPipeline,
    ),
    "tiger": ModelSpec(
        model_cls=TIGERModel,
        config_cls=TIGERConfig,
        model_data_cls=TIGERModelDataset,
        trainer_cls=Trainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=Pipeline,
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
    "BaseRetrievalModel",
    "BaseSequentialModelDataset",
    "HISTORY_ITEM_IDS",
    "HISTORY_TIMESTAMPS",
    "HSTUConfig",
    "HSTUModel",
    "HSTUModelDataset",
    "MODEL_TABLE",
    "RQVAEConfig",
    "RQVAEModel",
    "RQVAEModelDataset",
    "RQVAETrainer",
    "ModelConfig",
    "ModelDatasets",
    "ModelSpec",
    "SequentialModelConfig",
    "TIGERConfig",
    "TIGERModel",
    "TIGERModelDataset",
    "build_history_item_ids",
    "get_model_spec",
]
