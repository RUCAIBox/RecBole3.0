from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from recbole3.model.lares import (
    LARESConfig,
    LARESModel,
    LARESModelDataset,
    LARESTrainer,
)
from recbole3.model.letter import (
    LETTERConfig,
    LETTERModel,
    LETTERModelDataset,
    LETTERTrainer,
)
from recbole3.model.lcrec.config import LCRecConfig
from recbole3.model.llmrank import (
    LLMRankConfig,
    LLMRankModel,
    LLMRankModelDataset,
)
from recbole3.model.llmrank.trainer import LLMRankTrainer, LLMRankTrainerConfig
from recbole3.model.rankmixer import (
    RANKMIXER_FEATURES,
    RankMixerConfig,
    RankMixerEvalCollator,
    RankMixerModel,
    RankMixerPipeline,
    RankMixerTrainCollator,
)
from recbole3.model.rpg import (
    RPGConfig,
    RPGModel,
    RPGModelDataset,
    RPGTrainer,
    RPGTrainerConfig,
)
from recbole3.model.rqvae import (
    RQVAEConfig,
    RQVAEModel,
    RQVAEModelDataset,
    RQVAETrainer,
)
from recbole3.model.starec import (
    STARecConfig,
    STARecModel,
    STARecModelDataset,
    STARecTrainer,
    STARecTrainerConfig,
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
from recbole3.utils import LazyImport


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static model table entry."""

    model_cls: type[BaseModel] | LazyImport | Any
    config_cls: type[ModelConfig]
    model_data_cls: type[BaseModelDataset[Any, Any]] | None = None
    trainer_cls: type[Trainer] = Trainer
    trainer_config_cls: type[TrainerConfig] = TrainerConfig
    pipeline_cls: type[Pipeline] | LazyImport = Pipeline


MODEL_TABLE: dict[str, ModelSpec] = {
    "hstu": ModelSpec(
        model_cls=HSTUModel,
        config_cls=HSTUConfig,
        model_data_cls=HSTUModelDataset,
        trainer_cls=Trainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=Pipeline,
    ),
    "lares": ModelSpec(
        model_cls=LARESModel,
        config_cls=LARESConfig,
        model_data_cls=LARESModelDataset,
        trainer_cls=LARESTrainer,
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
    "letter": ModelSpec(
        model_cls=LETTERModel,
        config_cls=LETTERConfig,
        model_data_cls=LETTERModelDataset,
        trainer_cls=LETTERTrainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=Pipeline,
    ),
    "lcrec": ModelSpec(
        model_cls=LazyImport("transformers", "PreTrainedModel"),
        config_cls=LCRecConfig,
        pipeline_cls=LazyImport("recbole3.model.lcrec.pipeline", "LCRecPipeline"),
    ),
    "llmrank": ModelSpec(
        model_cls=LLMRankModel,
        config_cls=LLMRankConfig,
        model_data_cls=LLMRankModelDataset,
        trainer_cls=LLMRankTrainer,
        trainer_config_cls=LLMRankTrainerConfig,
        pipeline_cls=LazyImport("recbole3.model.llmrank.pipeline", "LLMRankPipeline"),
    ),
    "rankmixer": ModelSpec(
        model_cls=RankMixerModel,
        config_cls=RankMixerConfig,
        trainer_cls=Trainer,
        trainer_config_cls=TrainerConfig,
        pipeline_cls=RankMixerPipeline,
    ),
    "rpg": ModelSpec(
        model_cls=RPGModel,
        config_cls=RPGConfig,
        model_data_cls=RPGModelDataset,
        trainer_cls=RPGTrainer,
        trainer_config_cls=RPGTrainerConfig,
        pipeline_cls=Pipeline,
    ),
    "starec": ModelSpec(
        model_cls=STARecModel,
        config_cls=STARecConfig,
        model_data_cls=STARecModelDataset,
        trainer_cls=STARecTrainer,
        trainer_config_cls=STARecTrainerConfig,
        pipeline_cls=LazyImport("recbole3.model.starec.pipeline", "STARecPipeline"),
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
    "LETTERConfig",
    "LETTERModel",
    "LETTERModelDataset",
    "LETTERTrainer",
    "LLMRankConfig",
    "LLMRankModel",
    "LLMRankModelDataset",
    "MODEL_TABLE",
    "RANKMIXER_FEATURES",
    "RQVAEConfig",
    "RQVAEModel",
    "RQVAEModelDataset",
    "RQVAETrainer",
    "ModelConfig",
    "ModelDatasets",
    "ModelSpec",
    "RankMixerConfig",
    "RankMixerEvalCollator",
    "RankMixerModel",
    "RankMixerPipeline",
    "RankMixerTrainCollator",
    "SequentialModelConfig",
    "STARecConfig",
    "STARecModel",
    "STARecModelDataset",
    "STARecTrainer",
    "STARecTrainerConfig",
    "TIGERConfig",
    "TIGERModel",
    "TIGERModelDataset",
    "build_history_item_ids",
    "get_model_spec",
]
