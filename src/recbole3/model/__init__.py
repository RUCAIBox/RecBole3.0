from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recbole3.model.base import (
    BaseCollator,
    BaseModel,
    BaseModelDataset,
    BaseRankingModel,
    BaseRankingModelDataset,
    BaseRetrievalModel,
    BaseRetrievalModelDataset,
    ModelConfig,
    ModelDatasets,
)
from recbole3.model.hstu import (
    HISTORY_TIMESTAMPS,
    HSTUConfig,
    HSTUModel,
    HSTUModelDataset,
)
from recbole3.model.llmrank import LLMRankConfig, LLMRankModel, LLMRankModelDataset
from recbole3.model.llmrank.pipeline import LLMRankPipeline
from recbole3.model.llmrank.trainer import LLMRankTrainer, LLMRankTrainerConfig
from recbole3.model.sequential import (
    BaseSequentialRankingModelDataset,
    BaseSequentialRetrievalModelDataset,
    HISTORY_ITEM_IDS,
    SequentialModelConfig,
    build_history_item_ids,
)
from recbole3.pipeline import Pipeline
from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig

try:
    from transformers import PreTrainedModel
    from recbole3.model.lcrec import LCRecConfig
    from recbole3.model.lcrec.pipeline import LCRecPipeline
except ModuleNotFoundError:
    PreTrainedModel = None
    LCRecConfig = None
    LCRecPipeline = None

try:
    from recbole3.model.rqvae import (
        RQVAEConfig,
        RQVAEModel,
        RQVAEModelDataset,
        RQVAETrainer,
    )
except ModuleNotFoundError:
    RQVAEConfig = None
    RQVAEModel = None
    RQVAEModelDataset = None
    RQVAETrainer = None


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
    "llmrank": ModelSpec(
        model_cls=LLMRankModel,
        config_cls=LLMRankConfig,
        model_data_cls=LLMRankModelDataset,
        trainer_cls=LLMRankTrainer,
        trainer_config_cls=LLMRankTrainerConfig,
        pipeline_cls=LLMRankPipeline,
    ),
}

if RQVAEConfig is not None and RQVAEModel is not None and RQVAEModelDataset is not None and RQVAETrainer is not None:
    MODEL_TABLE["rqvae"] = ModelSpec(
        model_cls=RQVAEModel,
        config_cls=RQVAEConfig,
        model_data_cls=RQVAEModelDataset,
        trainer_cls=RQVAETrainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=Pipeline,
    )

if LCRecConfig is not None and LCRecPipeline is not None and PreTrainedModel is not None:
    MODEL_TABLE["lcrec"] = ModelSpec(
        model_cls=PreTrainedModel,
        config_cls=LCRecConfig,
        pipeline_cls=LCRecPipeline,
    )


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
    "HISTORY_ITEM_IDS",
    "HISTORY_TIMESTAMPS",
    "HSTUConfig",
    "HSTUModel",
    "HSTUModelDataset",
    "LLMRankConfig",
    "LLMRankModel",
    "LLMRankModelDataset",
    "MODEL_TABLE",
    "ModelConfig",
    "ModelDatasets",
    "ModelSpec",
    "SequentialModelConfig",
    "build_history_item_ids",
    "get_model_spec",
]
