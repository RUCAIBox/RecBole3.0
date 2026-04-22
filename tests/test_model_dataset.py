from __future__ import annotations

from typing import Any

import pytest

from recbole3.dataset import FrameDataset
from recbole3.evaluation import EvalConfig
from recbole3.model import BaseTaskModelDataset, ModelConfig, ModelDatasets
from tests.test_helpers import StubDataset, StubDatasetConfig


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


class PartialUpdateModelDataset(BaseTaskModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        train_frame = self.get_train_dataset().frame.iloc[:-1].copy()
        return ModelDatasets(train_dataset=FrameDataset(train_frame))


class MetadataOnlyModelDataset(BaseTaskModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        self.model_name = model_config.name
        return ModelDatasets()


class LegacyStyleModelDataset(BaseTaskModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        return None  # type: ignore[return-value]


class InvalidReturnModelDataset(BaseTaskModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        return "invalid"  # type: ignore[return-value]


class InvalidTrainDatasetModelDataset(BaseTaskModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        return ModelDatasets(train_dataset=[1, 2, 3])  # type: ignore[arg-type]


def test_model_dataset_partial_update_preserves_unmodified_eval_splits() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    model_data = PartialUpdateModelDataset.from_task_dataset(prepared, model_config=ModelConfig(name="stub"))

    assert len(model_data.get_train_dataset()) == len(prepared.get_train_dataset()) - 1
    assert model_data.get_eval_dataset("valid") is prepared.get_eval_dataset("valid")
    assert model_data.get_eval_dataset("test") is prepared.get_eval_dataset("test")


def test_model_dataset_empty_update_preserves_all_splits_and_allows_metadata() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    model_data = MetadataOnlyModelDataset.from_task_dataset(prepared, model_config=ModelConfig(name="metadata_only"))

    assert model_data.model_name == "metadata_only"
    assert model_data.get_train_dataset() is prepared.get_train_dataset()
    assert model_data.get_eval_dataset("valid") is prepared.get_eval_dataset("valid")
    assert model_data.get_eval_dataset("test") is prepared.get_eval_dataset("test")


def test_model_dataset_rejects_legacy_none_return_value() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(TypeError, match="must return ModelDatasets"):
        LegacyStyleModelDataset.from_task_dataset(prepared, model_config=ModelConfig(name="legacy"))


def test_model_dataset_rejects_invalid_return_type() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(TypeError, match="got str"):
        InvalidReturnModelDataset.from_task_dataset(prepared, model_config=ModelConfig(name="invalid"))


def test_model_dataset_rejects_non_dataset_split_replacement() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(TypeError, match="ModelDatasets.train_dataset"):
        InvalidTrainDatasetModelDataset.from_task_dataset(prepared, model_config=ModelConfig(name="invalid_split"))
