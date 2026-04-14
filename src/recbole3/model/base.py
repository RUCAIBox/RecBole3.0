from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Self, TypeVar

import torch
from torch import nn
from torch.utils.data import Dataset

from recbole3.dataset.base import BaseTaskDataset, RankingDataset, RetrievalDataset


TModelTrain = TypeVar("TModelTrain")
TModelEval = TypeVar("TModelEval")


@dataclass(slots=True)
class ModelConfig:
    """Convenience model config template with the framework's standard fields."""

    name: str = field(default="", metadata={"help": "Registered model name."})


@dataclass(slots=True)
class ModelDatasets(Generic[TModelTrain, TModelEval]):
    """Optional model-side split replacements applied by BaseModelDataset."""

    train_dataset: Dataset[TModelTrain] | None = None
    valid_dataset: Dataset[TModelEval] | None = None
    test_dataset: Dataset[TModelEval] | None = None


class BaseCollator(ABC):
    """Turn model-produced feature records into model-ready batches via DataLoader.collate_fn."""

    def __init__(self, config: ModelConfig, prepared_data: BaseTaskDataset):
        self.config = config
        self.prepared_data = prepared_data

    @abstractmethod
    def __call__(self, feature_records: Any) -> Any:
        """Build a model-ready batch from model feature records."""


class BaseModel(nn.Module, ABC):
    """Base interface for all recommendation models."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def build_train_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        """Return the collator used for training batches."""

    @abstractmethod
    def build_eval_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        """Return the collator used to pack evaluation model inputs."""

    @abstractmethod
    def forward(self, batch: Any) -> dict[str, Any]:
        """Run the forward pass on a prepared batch."""

    @abstractmethod
    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        """Compute training loss from a batch and model outputs."""


class BaseRankingModel(BaseModel):
    """Model that scores one provided candidate set per evaluation batch."""

    @abstractmethod
    def predict(self, model_inputs: Any) -> torch.Tensor:
        """Return candidate-aligned scores for one labeled ranking batch."""


class BaseRetrievalModel(BaseModel):
    """Model that returns ordered top-k item ids for one retrieval batch."""

    @abstractmethod
    def predict(
        self,
        model_inputs: Any,
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-k item ids for one retrieval batch."""


class BaseModelDataset(ABC, Generic[TModelTrain, TModelEval]):
    """Model-side prepared-data extension built from one prepared task dataset."""

    @classmethod
    def from_task_dataset(
        cls,
        dataset: BaseTaskDataset,
        *,
        model_config: ModelConfig,
    ) -> Self:
        model_dataset = cls._clone_task_dataset(dataset)
        model_datasets = model_dataset._build_model_datasets(model_config=model_config)
        model_dataset._apply_model_datasets(model_datasets)
        return model_dataset

    @classmethod
    @abstractmethod
    def _clone_task_dataset(cls, dataset: BaseTaskDataset) -> Self:
        """Clone one prepared task dataset into the model-side dataset type."""

    @abstractmethod
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[TModelTrain, TModelEval]:
        """Return model-side prepared split replacements for one model."""

    @staticmethod
    def _copy_task_dataset_state(target: BaseTaskDataset, source: BaseTaskDataset) -> None:
        source._require_prepared()
        target.config = source.config
        target._parser = source._parser
        target._eval_config = source._eval_config
        target._is_prepared = source._is_prepared
        target._interactions = source.get_interactions().copy()
        target._user_table = source.get_user_table().copy()
        target._item_table = source.get_item_table().copy()
        target._num_users = source.get_num_users()
        target._num_items = source.get_num_items()
        target._train_dataset = source.get_train_dataset()
        target._valid_dataset = source.get_eval_dataset("valid")
        target._test_dataset = source.get_eval_dataset("test")

    def _apply_model_datasets(self, model_datasets: ModelDatasets[TModelTrain, TModelEval]) -> None:
        if not isinstance(model_datasets, ModelDatasets):
            if model_datasets is None:
                raise TypeError(
                    f"{type(self).__name__}._build_model_datasets(...) must return ModelDatasets. "
                    "The old manual _set_* protocol has been removed."
                )
            raise TypeError(
                f"{type(self).__name__}._build_model_datasets(...) must return ModelDatasets, "
                f"got {type(model_datasets).__name__}."
            )
        if model_datasets.train_dataset is not None:
            self._train_dataset = self._require_dataset("train_dataset", model_datasets.train_dataset)
        if model_datasets.valid_dataset is not None:
            self._valid_dataset = self._require_dataset("valid_dataset", model_datasets.valid_dataset)
        if model_datasets.test_dataset is not None:
            self._test_dataset = self._require_dataset("test_dataset", model_datasets.test_dataset)

    @staticmethod
    def _require_dataset(name: str, dataset: Dataset[Any]) -> Dataset[Any]:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"ModelDatasets.{name} must be a torch.utils.data.Dataset, got {type(dataset).__name__}.")
        return dataset


class BaseRankingModelDataset(BaseModelDataset[TModelTrain, TModelEval], RankingDataset, ABC):
    """Model-side dataset extension for ranking tasks."""

    @classmethod
    def _clone_task_dataset(cls, dataset: BaseTaskDataset) -> Self:
        if not isinstance(dataset, RankingDataset):
            raise TypeError(f"{cls.__name__} requires a prepared RankingDataset.")
        model_dataset = cls.__new__(cls)
        cls._copy_task_dataset_state(model_dataset, dataset)
        return model_dataset


class BaseRetrievalModelDataset(BaseModelDataset[TModelTrain, TModelEval], RetrievalDataset, ABC):
    """Model-side dataset extension for retrieval tasks."""

    @classmethod
    def _clone_task_dataset(cls, dataset: BaseTaskDataset) -> Self:
        if not isinstance(dataset, RetrievalDataset):
            raise TypeError(f"{cls.__name__} requires a prepared RetrievalDataset.")
        model_dataset = cls.__new__(cls)
        cls._copy_task_dataset_state(model_dataset, dataset)
        return model_dataset
