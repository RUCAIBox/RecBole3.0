from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from recbole3.dataset import BaseDatasetParser, Interaction, ParsedData, RetrievalDataset, SplitConfig
from recbole3.dataset.base import DatasetConfig
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import HSTUConfig, HSTUInteraction, HSTUModel, HSTUModelDataset, HSTURetrievalEvalRequest, get_model_spec
from recbole3.model.hstu.data import HSTUEvalCollator, HSTUTrainCollator
from recbole3.run import compose_config, run_experiment
from recbole3.trainer import Trainer, TrainerConfig
from tests.test_helpers import StubDataset, StubDatasetConfig, ensure_stub_tables


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


@dataclass(slots=True)
class MissingTimestampDatasetConfig(DatasetConfig):
    name: str = field(default="missing_timestamp_dataset", metadata={"help": "Dataset with one missing timestamp."})
    split: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            strategy="leave_one_out",
            order="chronological",
            per_user=True,
            valid_holdout_num=1,
            test_holdout_num=1,
        )
    )


class MissingTimestampParser(BaseDatasetParser):
    def parse(self) -> ParsedData:
        interactions = [
            Interaction(user_id=0, item_id=0, timestamp=None, label=1.0),
            Interaction(user_id=0, item_id=1, timestamp=2, label=1.0),
            Interaction(user_id=0, item_id=2, timestamp=3, label=1.0),
        ]
        users = pd.DataFrame([{"user_id": 0}])
        items = pd.DataFrame([{"item_id": 0}, {"item_id": 1}, {"item_id": 2}])
        return ParsedData(interactions=interactions, user_table=users, item_table=items)


class MissingTimestampDataset(RetrievalDataset):
    config_cls = MissingTimestampDatasetConfig
    parser_cls = MissingTimestampParser


def test_hstu_model_dataset_builds_histories_with_timestamps_across_splits() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    hstu_data = HSTUModelDataset.from_task_dataset(
        prepared,
        model_config=HSTUConfig(history_max_length=2),
    )

    assert list(hstu_data.get_train_dataset()) == [
        HSTUInteraction(user_id=0, item_id=0, timestamp=1, label=1.0, history_item_ids=(), history_timestamps=()),
        HSTUInteraction(user_id=0, item_id=1, timestamp=2, label=1.0, history_item_ids=(0,), history_timestamps=(1.0,)),
        HSTUInteraction(user_id=1, item_id=4, timestamp=1, label=1.0, history_item_ids=(), history_timestamps=()),
        HSTUInteraction(user_id=1, item_id=5, timestamp=2, label=1.0, history_item_ids=(4,), history_timestamps=(1.0,)),
    ]
    assert list(hstu_data.get_eval_dataset("valid")) == [
        HSTURetrievalEvalRequest(
            user_id=0,
            item_id=2,
            timestamp=3,
            label=1.0,
            seen_item_ids=(0, 1),
            candidate_item_ids=None,
            history_item_ids=(0, 1),
            history_timestamps=(1.0, 2.0),
        ),
        HSTURetrievalEvalRequest(
            user_id=1,
            item_id=6,
            timestamp=3,
            label=1.0,
            seen_item_ids=(4, 5),
            candidate_item_ids=None,
            history_item_ids=(4, 5),
            history_timestamps=(1.0, 2.0),
        ),
    ]
    assert list(hstu_data.get_eval_dataset("test")) == [
        HSTURetrievalEvalRequest(
            user_id=0,
            item_id=3,
            timestamp=4,
            label=1.0,
            seen_item_ids=(0, 1, 2),
            candidate_item_ids=None,
            history_item_ids=(1, 2),
            history_timestamps=(2.0, 3.0),
        ),
        HSTURetrievalEvalRequest(
            user_id=1,
            item_id=7,
            timestamp=4,
            label=1.0,
            seen_item_ids=(4, 5, 6),
            candidate_item_ids=None,
            history_item_ids=(5, 6),
            history_timestamps=(2.0, 3.0),
        ),
    ]


def test_hstu_model_dataset_rejects_missing_timestamps() -> None:
    prepared = MissingTimestampDataset(MissingTimestampDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(ValueError, match="timestamp"):
        HSTUModelDataset.from_task_dataset(prepared, model_config=HSTUConfig(history_max_length=2))


def test_hstu_collators_pad_history_sequences() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    hstu_data = HSTUModelDataset.from_task_dataset(prepared, model_config=HSTUConfig(history_max_length=2))
    train_records = list(hstu_data.get_train_dataset())
    eval_records = list(hstu_data.get_eval_dataset("test"))

    train_batch = HSTUTrainCollator(HSTUConfig(history_max_length=2), prepared_data=hstu_data)(train_records[:2])
    eval_batch = HSTUEvalCollator(HSTUConfig(history_max_length=2), prepared_data=hstu_data)(eval_records)

    assert train_batch["history_lengths"].tolist() == [0, 1]
    assert train_batch["history_item_ids"].tolist() == [[0], [0]]
    assert train_batch["history_timestamps"].tolist() == [[0.0], [1.0]]
    assert train_batch["item_id"].tolist() == [0, 1]

    assert eval_batch["history_lengths"].tolist() == [2, 2]
    assert eval_batch["history_item_ids"].tolist() == [[1, 2], [5, 6]]
    assert eval_batch["history_timestamps"].tolist() == [[2.0, 3.0], [2.0, 3.0]]


def test_hstu_model_registration_and_retrieval_trainer_registration_exist() -> None:
    model_spec = get_model_spec("hstu")
    assert model_spec.config_cls is HSTUConfig
    assert model_spec.trainer_cls is Trainer
    assert model_spec.trainer_config_cls is TrainerConfig


def test_hstu_model_requires_fbgemm_gpu_dependency() -> None:
    with pytest.raises(RuntimeError, match="fbgemm-gpu"):
        HSTUModel(HSTUConfig())


def test_hstu_predict_supports_sampled_and_full_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    model = HSTUModel(HSTUConfig(history_max_length=2, normalize_embeddings=False, temperature=1.0))
    model._num_items = 4
    model._item_embeddings = nn.Embedding(5, 2, padding_idx=0)
    with torch.no_grad():
        model._item_embeddings.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [4.0, 1.0],
                    [3.0, 0.0],
                    [1.0, 5.0],
                    [0.0, 2.0],
                ]
            )
        )
    model._empty_history_embedding = nn.Parameter(torch.zeros(2))
    monkeypatch.setattr(
        HSTUModel,
        "_encode_user_embeddings",
        lambda self, batch: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )
    batch = {
        "history_item_ids": torch.zeros((2, 0), dtype=torch.long),
        "history_timestamps": torch.zeros((2, 0), dtype=torch.float32),
        "history_lengths": torch.zeros(2, dtype=torch.long),
    }

    sampled_pred = model.predict(batch, k=2, candidate_item_ids=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
    full_pred = model.predict(
        batch,
        k=2,
        exclude_item_ids=torch.tensor([[0], [2]], dtype=torch.long),
        exclude_mask=torch.tensor([[True], [True]], dtype=torch.bool),
    )

    assert sampled_pred.tolist() == [[0, 1], [2, 3]]
    assert full_pred.tolist() == [[1, 2], [3, 0]]


def test_run_experiment_with_hstu_fails_fast_without_fbgemm_gpu(tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: hstu",
                "  - _self_",
                "runtime:",
                "  seed: 7",
                "  device: cpu",
                f"  output_dir: {(tmp_path / 'outputs').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "stub_dataset.yaml").write_text(
        "\n".join(
            [
                "name: stub_dataset",
                f"processed_dir: {(tmp_path / 'processed').as_posix()}",
                "split:",
                "  strategy: leave_one_out",
                "  order: chronological",
                "  per_user: true",
                "  valid_holdout_num: 1",
                "  test_holdout_num: 1",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "model" / "hstu.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: hstu",
                "  history_max_length: 2",
                "  embedding_dim: 8",
                "  num_layers: 1",
                "  num_heads: 1",
                "  attention_dim: 4",
                "  linear_hidden_dim: 4",
                "  linear_dropout_rate: 0.0",
                "  attn_dropout_rate: 0.0",
                "  temperature: 1.0",
                "  normalize_embeddings: false",
                "  num_time_buckets: 16",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  optimizer:",
                "    name: SGD",
                "    kwargs:",
                "      lr: 0.001",
                "  eval:",
                "    protocol: full",
                "    metrics:",
                "      - name: recall",
                "        ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fbgemm-gpu"):
        run_experiment(compose_config(config_dir=config_dir))
