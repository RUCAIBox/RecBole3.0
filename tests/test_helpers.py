from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd
import torch
from torch import nn

from recbole3.dataset import (
    DATASET_TABLE,
    BaseDatasetParser,
    DatasetSpec,
    ITEM_ID,
    LABEL,
    ParsedData,
    SplitConfig,
    BaseTaskDataset,
    USER_ID,
)
from recbole3.dataset.base import DatasetConfig
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import (
    MODEL_TABLE,
    BaseCollator,
    BaseRankingModel,
    BaseRetrievalModel,
    BaseModelDataset,
    ModelConfig,
    ModelDatasets,
    ModelSpec,
)
from recbole3.trainer import OptimizerConfig, Trainer, TrainerConfig


DEFAULT_STUB_SPLIT = SplitConfig(
    strategy="leave_one_out",
    order="chronological",
    per_user=True,
    valid_holdout_num=1,
    test_holdout_num=1,
)


@dataclass(slots=True)
class StubDatasetConfig(DatasetConfig):
    name: str = field(default="stub_dataset", metadata={"help": "Stub dataset name."})
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed data root."})
    split: SplitConfig = field(default_factory=lambda: SplitConfig(strategy="leave_one_out", order="chronological", per_user=True, valid_holdout_num=1, test_holdout_num=1))


@dataclass(slots=True)
class StubRankingDatasetConfig(DatasetConfig):
    name: str = field(default="stub_ranking_dataset", metadata={"help": "Stub ranking dataset name."})
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed data root."})
    split: SplitConfig = field(default_factory=lambda: SplitConfig(strategy="leave_one_out", order="chronological", per_user=True, valid_holdout_num=1, test_holdout_num=1))


class StubParser(BaseDatasetParser):
    def parse(self) -> ParsedData:
        interactions = pd.DataFrame(
            [
                {USER_ID: 0, ITEM_ID: 0, "timestamp": 1, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 1, "timestamp": 2, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 2, "timestamp": 3, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 3, "timestamp": 4, LABEL: 1.0},
                {USER_ID: 1, ITEM_ID: 4, "timestamp": 1, LABEL: 1.0},
                {USER_ID: 1, ITEM_ID: 5, "timestamp": 2, LABEL: 1.0},
                {USER_ID: 1, ITEM_ID: 6, "timestamp": 3, LABEL: 1.0},
                {USER_ID: 1, ITEM_ID: 7, "timestamp": 4, LABEL: 1.0},
            ]
        )
        users = pd.DataFrame([{USER_ID: 0}, {USER_ID: 1}])
        item_titles = (
            "Alpha Quest",
            "Bravo Tales",
            "Charlie Harbor",
            "Delta Echo",
            "Forest Signal",
            "Golden River",
            "Harbor Night",
            "Ivory Path",
        )
        items = pd.DataFrame(
            [{ITEM_ID: item_id, "metadata_text": item_titles[item_id], "title": item_titles[item_id]} for item_id in range(8)]
        )
        return ParsedData(interactions=interactions, user_table=users, item_table=items)


class StubDataset(BaseTaskDataset):
    config_cls = StubDatasetConfig
    parser_cls = StubParser


class StubRankingDataset(BaseTaskDataset):
    config_cls = StubRankingDatasetConfig
    parser_cls = StubParser


@dataclass(slots=True)
class StubModelConfig(ModelConfig):
    name: str = field(default="stub_model", metadata={"help": "Stub model name."})


class StubTrainCollator(BaseCollator):
    def __call__(self, records: pd.DataFrame) -> Mapping[str, Any]:
        item_id = torch.as_tensor(records[ITEM_ID].to_numpy(), dtype=torch.long)
        batch = {
            USER_ID: torch.as_tensor(records[USER_ID].to_numpy(), dtype=torch.long),
            ITEM_ID: item_id,
            LABEL: torch.as_tensor(pd.to_numeric(records[LABEL], errors="coerce").fillna(1.0).to_numpy(), dtype=torch.float32),
        }
        num_items = max(int(self.prepared_data.get_num_items()), 1)
        batch["neg_item_id"] = (item_id + 1) % num_items
        return batch


class StubRetrievalEvalCollator(BaseCollator):
    def __call__(self, records: pd.DataFrame) -> Mapping[str, Any]:
        return {USER_ID: torch.as_tensor(records[USER_ID].to_numpy(), dtype=torch.long)}


class StubModel(BaseRetrievalModel):
    def __init__(self, config: StubModelConfig):
        super().__init__(config)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.num_items = 0

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self.num_items = int(prepared_data.get_num_items())
        return StubTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self.num_items = int(prepared_data.get_num_items())
        return StubRetrievalEvalCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"scores": batch["item_id"].float() * self.scale}

    def compute_loss(self, batch: Mapping[str, Any], outputs: Mapping[str, Any]) -> Any:
        return torch.mean((outputs["scores"] - batch["label"]) ** 2)

    def predict(self, model_inputs: Mapping[str, Any], *, k: int, candidate_item_ids=None, exclude_item_ids=None, exclude_mask=None) -> torch.Tensor:
        if candidate_item_ids is not None:
            candidate_scores = candidate_item_ids.float() * self.scale
            top_indices = torch.topk(candidate_scores, k=k, dim=1).indices
            return torch.gather(candidate_item_ids, 1, top_indices)
        user_ids = model_inputs["user_id"]
        item_ids = torch.arange(self.num_items, device=user_ids.device, dtype=torch.long).unsqueeze(0).expand(user_ids.shape[0], -1)
        scores = item_ids.float() * self.scale
        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
            history_mask = torch.zeros_like(scores, dtype=torch.bool)
            history_mask.scatter_(1, exclude_item_ids, exclude_mask)
            scores = scores.masked_fill(history_mask, float("-inf"))
        return torch.topk(scores, k=k, dim=1).indices.to(dtype=torch.long)


@dataclass(slots=True)
class StubRankingModelConfig(ModelConfig):
    name: str = field(default="stub_ranking_model", metadata={"help": "Stub ranking model name."})


class StubRankingEvalCollator(BaseCollator):
    def __call__(self, records: pd.DataFrame) -> Mapping[str, Any]:
        return {
            USER_ID: torch.as_tensor(records[USER_ID].to_numpy(), dtype=torch.long),
            ITEM_ID: torch.as_tensor(records[ITEM_ID].to_numpy(), dtype=torch.long),
        }


class StubModelDataset(BaseModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        self.model_name = model_config.name
        return ModelDatasets()


class StubRankingModel(BaseRankingModel):
    def __init__(self, config: StubRankingModelConfig):
        super().__init__(config)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def build_train_collator(self, prepared_data) -> BaseCollator:
        return StubTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        return StubRankingEvalCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"scores": batch["item_id"].float() * self.scale}

    def compute_loss(self, batch: Mapping[str, Any], outputs: Mapping[str, Any]) -> Any:
        return torch.mean((outputs["scores"] - batch["label"]) ** 2)

    def predict(self, model_inputs: Mapping[str, Any]) -> torch.Tensor:
        return model_inputs["item_id"].float() * self.scale


@dataclass(slots=True)
class StubTrainerConfig(TrainerConfig):
    batch_size: int = field(default=2, metadata={"help": "Batch size used by the stub trainer."})
    max_epochs: int = field(default=1, metadata={"help": "Epoch count used by the stub trainer."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(protocol="sampled", metrics=(MetricSpec(name="recall", ks=(3,)),)),
        metadata={"help": "Evaluation configuration used by the stub trainer."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(name="SGD", kwargs={"lr": 0.001}),
        metadata={"help": "Optimizer configuration used by the stub trainer."},
    )


class StubTrainer(Trainer):
    pass


@dataclass(slots=True)
class StubRankingTrainerConfig(TrainerConfig):
    batch_size: int = field(default=2, metadata={"help": "Batch size used by the stub trainer."})
    max_epochs: int = field(default=1, metadata={"help": "Epoch count used by the stub trainer."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(protocol="labeled", metrics=(MetricSpec(name="logloss"),)),
        metadata={"help": "Evaluation configuration used by the stub trainer."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(name="SGD", kwargs={"lr": 0.001}),
        metadata={"help": "Optimizer configuration used by the stub trainer."},
    )


class StubRankingTrainer(Trainer):
    pass


def ensure_stub_tables() -> None:
    DATASET_TABLE["stub_dataset"] = DatasetSpec(
        dataset_cls=StubDataset,
        config_cls=StubDatasetConfig,
    )
    DATASET_TABLE["stub_ranking_dataset"] = DatasetSpec(
        dataset_cls=StubRankingDataset,
        config_cls=StubRankingDatasetConfig,
    )
    MODEL_TABLE["stub_model"] = ModelSpec(
        model_cls=StubModel,
        config_cls=StubModelConfig,
        trainer_cls=StubTrainer,
        trainer_config_cls=StubTrainerConfig,
    )
    MODEL_TABLE["stub_model_with_data"] = ModelSpec(
        model_cls=StubModel,
        config_cls=StubModelConfig,
        model_data_cls=StubModelDataset,
        trainer_cls=StubTrainer,
        trainer_config_cls=StubTrainerConfig,
    )
    MODEL_TABLE["stub_ranking_model"] = ModelSpec(
        model_cls=StubRankingModel,
        config_cls=StubRankingModelConfig,
        trainer_cls=StubRankingTrainer,
        trainer_config_cls=StubRankingTrainerConfig,
    )
