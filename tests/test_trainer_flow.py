from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from recbole3.evaluation import EvalConfig, MetricSpec, RankingEvalData, RetrievalEvalData
from recbole3.trainer import CheckpointConfig, EarlyStoppingConfig, OptimizerConfig, SchedulerConfig, Trainer, TrainerConfig
from tests.test_helpers import (
    StubDataset,
    StubDatasetConfig,
    StubModel,
    StubModelConfig,
    StubRankingDataset,
    StubRankingDatasetConfig,
    StubRankingModel,
    StubRankingModelConfig,
    StubRankingTrainer,
    StubRankingTrainerConfig,
    StubTrainer,
    StubTrainerConfig,
)


class MissingRetrievalPredictionModel(StubModel):
    def predict(self, model_inputs, *, k: int, candidate_item_ids=None, exclude_item_ids=None, exclude_mask=None):
        raise NotImplementedError("Missing retrieval predict implementation.")


class TrackingLoadStateDictRankingModel(StubRankingModel):
    def __init__(self, config: StubRankingModelConfig):
        super().__init__(config)
        self.load_state_dict_calls = 0

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):  # type: ignore[override]
        self.load_state_dict_calls += 1
        return super().load_state_dict(state_dict, strict=strict)


def _mock_eval_result(split: str, metric_name: str, value: float) -> dict[str, Any]:
    return {
        "split": split,
        "protocol": "labeled",
        "loss": None,
        "metrics": {metric_name: value},
        "num_batches": 1,
        "data_stats": {"num_users": 2, "num_items": 8},
    }


def _labeled_eval_config() -> EvalConfig:
    return EvalConfig(protocol="labeled")


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _sampled_eval_config(*, neg_sampling_num: int = 2, candidate_seed: int = 7) -> EvalConfig:
    return EvalConfig(protocol="sampled", neg_sampling_num=neg_sampling_num, candidate_seed=candidate_seed)


def test_train_dataloader_uses_prepared_split_dataset() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())
    trainer = StubRankingTrainer(StubRankingTrainerConfig(batch_size=2, shuffle=False))
    model = StubRankingModel(StubRankingModelConfig())

    train_dataset = prepared.get_train_dataset()
    train_loader = trainer.build_dataloader(train_dataset, model.build_train_collator(prepared), shuffle=False)
    train_batch = next(iter(train_loader))

    assert train_dataset[0] == list(prepared.get_train_dataset())[0]
    assert train_batch["item_id"].tolist() == [0, 1]
    assert "neg_item_id" in train_batch

    valid_method = trainer.create_evaluation_method(prepared)
    valid_loader = trainer.build_dataloader(
        prepared.get_eval_dataset("valid"),
        valid_method.build_eval_collate_fn(model, prepared),
        shuffle=False,
    )
    model_inputs, records = next(iter(valid_loader))
    assert model_inputs["item_id"].tolist() == [2, 6]
    assert [record.item_id for record in records] == [2, 6]
    assert "neg_item_id" not in model_inputs



def test_ranking_method_owns_metric_aggregation() -> None:
    trainer = StubRankingTrainer(StubRankingTrainerConfig())
    result = trainer.create_evaluation_method().compute_metrics(
        [
            RankingEvalData(scores=np.array([0.5]), labels=np.array([1.0]), group_ids=np.array([0])),
            RankingEvalData(scores=np.array([0.5]), labels=np.array([0.0]), group_ids=np.array([1])),
        ]
    )
    assert result["logloss"] == pytest.approx(float(np.log(2.0)))



def test_retrieval_method_owns_metric_aggregation() -> None:
    trainer = StubTrainer(
        StubTrainerConfig(
            eval=EvalConfig(
                protocol="sampled",
                metrics=(MetricSpec(name="recall", ks=(3,)), MetricSpec(name="ndcg", ks=(3,))),
                neg_sampling_num=2,
                candidate_seed=7,
            )
        )
    )
    result = trainer.create_evaluation_method().compute_metrics(
        [
            RetrievalEvalData(pred_item_ids=np.array([[3, 2, -1]]), target_item_ids=np.array([[2]]), target_mask=np.array([[True]])),
            RetrievalEvalData(pred_item_ids=np.array([[5, 4, 1]]), target_item_ids=np.array([[1]]), target_mask=np.array([[True]])),
        ]
    )
    assert result["recall@3"] == pytest.approx(1.0)
    assert result["ndcg@3"] > 0.0



def test_sampled_evaluation_uses_dataset_prepared_candidates() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    trainer = StubTrainer(
        StubTrainerConfig(
            batch_size=2,
            shuffle=False,
            eval=EvalConfig(
                protocol="sampled",
                metrics=(MetricSpec(name="recall", ks=(3,)), MetricSpec(name="ndcg", ks=(3,))),
                neg_sampling_num=2,
                candidate_seed=7,
            ),
        )
    )
    result = trainer.evaluate(StubModel(StubModelConfig()), prepared, split="test")

    assert result["protocol"] == "sampled"
    assert result["metrics"]["recall@3"] == 1.0
    sampled_records = list(prepared.get_eval_dataset("test"))
    assert [len(record.candidate_item_ids or ()) for record in sampled_records] == [3, 3]



def test_full_evaluation_uses_seen_history_filtering() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    trainer = StubTrainer(
        StubTrainerConfig(
            batch_size=2,
            shuffle=False,
            eval=EvalConfig(protocol="full", metrics=(MetricSpec(name="recall", ks=(5,)),), exclude_history=True),
        )
    )
    result = trainer.evaluate(StubModel(StubModelConfig()), prepared, split="test")
    assert result["protocol"] == "full"
    assert result["metrics"]["recall@5"] == 1.0



def test_fit_runs_validation_every_epoch() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())
    trainer = StubRankingTrainer(StubRankingTrainerConfig(batch_size=2, shuffle=False, max_epochs=3))
    model = StubRankingModel(StubRankingModelConfig())
    call_count = {"value": 0}
    original_create_method = trainer.create_evaluation_method

    def counting_create_method(prepared_data):
        call_count["value"] += 1
        return original_create_method(prepared_data)

    trainer.create_evaluation_method = counting_create_method  # type: ignore[method-assign]
    result = trainer.fit(model, prepared)
    assert call_count["value"] == 3
    assert len(result["valid_history"]) == 3



def test_fit_stops_early_when_monitor_stalls() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())
    trainer = StubRankingTrainer(
        StubRankingTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=5,
            monitor="logloss",
            early_stopping=EarlyStoppingConfig(enabled=True, patience=2),
        )
    )
    values = iter([1.0, 1.1, 1.2, 1.3, 1.4])

    def fake_run_evaluation(model, prepared_data, *, split, accelerator, model_is_prepared):
        metric_value = 0.5 if split == "test" else next(values)
        return _mock_eval_result(split, "logloss", metric_value)

    trainer._run_evaluation = fake_run_evaluation  # type: ignore[method-assign]
    result = trainer.fit(StubRankingModel(StubRankingModelConfig()), prepared)
    assert result["stopped_early"] is True
    assert result["best_epoch"] == 1



def test_run_loads_best_checkpoint_before_test(tmp_path: Path) -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())
    trainer = StubRankingTrainer(
        StubRankingTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=2,
            monitor="logloss",
            checkpoint=CheckpointConfig(save_best=True),
        )
    )
    model = TrackingLoadStateDictRankingModel(StubRankingModelConfig())
    values = iter([1.0, 1.1])

    def fake_run_evaluation(model, prepared_data, *, split, accelerator, model_is_prepared):
        metric_value = 0.25 if split == "test" else next(values)
        return _mock_eval_result(split, "logloss", metric_value)

    trainer._run_evaluation = fake_run_evaluation  # type: ignore[method-assign]
    result = trainer.run(model, prepared, output_dir=tmp_path)
    assert model.load_state_dict_calls == 1
    assert result["test"]["split"] == "test"



def test_retrieval_protocol_rejects_ranking_dataset_prepare() -> None:
    dataset = StubRankingDataset(StubRankingDatasetConfig())
    with pytest.raises(ValueError, match="only support eval protocol 'labeled'"):
        dataset.prepare(eval_config=EvalConfig(protocol="sampled"))



def test_sampled_evaluation_requires_retrieval_predict_method() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    trainer = StubTrainer(
        StubTrainerConfig(
            batch_size=2,
            shuffle=False,
            eval=EvalConfig(protocol="sampled", metrics=(MetricSpec(name="recall", ks=(3,)),), neg_sampling_num=2, candidate_seed=7),
        )
    )
    with pytest.raises(NotImplementedError, match="retrieval predict"):
        trainer.evaluate(MissingRetrievalPredictionModel(StubModelConfig()), prepared, split="test")



def test_base_trainer_builds_configured_optimizer_and_scheduler() -> None:
    trainer = Trainer(
        TrainerConfig(
            eval=EvalConfig(protocol="labeled"),
            optimizer=OptimizerConfig(name="AdamW", kwargs={"lr": 0.01, "weight_decay": 0.1}),
            scheduler=SchedulerConfig(name="OneCycleLR", kwargs={"max_lr": 0.1}),
            max_epochs=2,
        )
    )
    model = StubRankingModel(StubRankingModelConfig())
    optimizer = trainer.build_optimizer(model)
    scheduler = trainer.build_scheduler(optimizer, num_training_steps=10, steps_per_epoch=5)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    assert scheduler.total_steps == 10



def test_base_trainer_rejects_invalid_optimizer_and_scheduler() -> None:
    model = StubRankingModel(StubRankingModelConfig())
    with pytest.raises(ValueError, match="Unknown torch optimizer"):
        Trainer(
            TrainerConfig(
                eval=EvalConfig(protocol="labeled"),
                optimizer=OptimizerConfig(name="NotAnOptimizer", kwargs={"lr": 0.01}),
            )
        ).build_optimizer(model)
    trainer = Trainer(
        TrainerConfig(
            eval=EvalConfig(protocol="labeled"),
            optimizer=OptimizerConfig(name="SGD", kwargs={"lr": 0.01}),
            scheduler=SchedulerConfig(name="NotAScheduler", kwargs={}),
        )
    )
    optimizer = Trainer(
        TrainerConfig(
            eval=EvalConfig(protocol="labeled"),
            optimizer=OptimizerConfig(name="SGD", kwargs={"lr": 0.01}),
        )
    ).build_optimizer(model)
    with pytest.raises(ValueError, match="Unknown torch scheduler"):
        trainer.build_scheduler(optimizer, num_training_steps=4, steps_per_epoch=2)
