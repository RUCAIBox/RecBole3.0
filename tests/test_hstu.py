from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from recbole3.dataset import ITEM_ID, LABEL, SEEN_ITEM_IDS, TIMESTAMP, USER_ID, BaseDatasetParser, ParsedData, SplitConfig, BaseTaskDataset
from recbole3.dataset.base import DatasetConfig
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import HISTORY_ITEM_IDS, HISTORY_TIMESTAMPS, HSTUConfig, HSTUModel, HSTUModelDataset, get_model_spec
from recbole3.model.hstu.config import HSTU_PADDING_ITEM_ID
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
        interactions = pd.DataFrame(
            [
                {USER_ID: 0, ITEM_ID: 0, TIMESTAMP: None, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0},
            ]
        )
        users = pd.DataFrame([{USER_ID: 0}])
        items = pd.DataFrame([{ITEM_ID: 0}, {ITEM_ID: 1}, {ITEM_ID: 2}])
        return ParsedData(interactions=interactions, user_table=users, item_table=items)


class MissingTimestampDataset(BaseTaskDataset):
    config_cls = MissingTimestampDatasetConfig
    parser_cls = MissingTimestampParser


def test_hstu_model_dataset_builds_histories_with_timestamps_across_splits() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    hstu_data = HSTUModelDataset.from_task_dataset(
        prepared,
        model_config=HSTUConfig(history_max_length=2),
    )

    assert hstu_data.get_train_dataset().frame[
        [USER_ID, ITEM_ID, TIMESTAMP, LABEL, HISTORY_ITEM_IDS, HISTORY_TIMESTAMPS]
    ].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (0,), HISTORY_TIMESTAMPS: (1.0,)},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (4,), HISTORY_TIMESTAMPS: (1.0,)},
    ]
    eval_columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS, HISTORY_ITEM_IDS, HISTORY_TIMESTAMPS]
    assert hstu_data.get_eval_dataset("valid").frame[eval_columns].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1), HISTORY_ITEM_IDS: (0, 1), HISTORY_TIMESTAMPS: (1.0, 2.0)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5), HISTORY_ITEM_IDS: (4, 5), HISTORY_TIMESTAMPS: (1.0, 2.0)},
    ]
    assert hstu_data.get_eval_dataset("test").frame[eval_columns].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1, 2), HISTORY_ITEM_IDS: (1, 2), HISTORY_TIMESTAMPS: (2.0, 3.0)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5, 6), HISTORY_ITEM_IDS: (5, 6), HISTORY_TIMESTAMPS: (2.0, 3.0)},
    ]


def test_hstu_model_dataset_rejects_missing_timestamps() -> None:
    prepared = MissingTimestampDataset(MissingTimestampDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(ValueError, match="timestamp"):
        HSTUModelDataset.from_task_dataset(prepared, model_config=HSTUConfig(history_max_length=2))


def test_hstu_collators_pad_history_sequences() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    hstu_data = HSTUModelDataset.from_task_dataset(prepared, model_config=HSTUConfig(history_max_length=2))
    train_records = hstu_data.get_train_dataset().frame.reset_index(drop=True)
    eval_records = hstu_data.get_eval_dataset("test").frame

    train_batch = HSTUTrainCollator(HSTUConfig(history_max_length=2), prepared_data=hstu_data)(train_records)
    eval_batch = HSTUEvalCollator(HSTUConfig(history_max_length=2), prepared_data=hstu_data)(eval_records)

    assert train_batch["history_lengths"].tolist() == [1, 1]
    assert train_batch["history_item_ids"].tolist() == [[0, 1], [4, 5]]
    assert train_batch["history_timestamps"].tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert train_batch["item_id"].tolist() == [1, 5]

    assert eval_batch["history_lengths"].tolist() == [2, 2]
    assert eval_batch["history_item_ids"].tolist() == [[1, 2, HSTU_PADDING_ITEM_ID], [5, 6, HSTU_PADDING_ITEM_ID]]
    assert eval_batch["history_timestamps"].tolist() == [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]


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
    model._num_items = 5
    model._item_embeddings = nn.Embedding(6, 2, padding_idx=HSTU_PADDING_ITEM_ID)
    with torch.no_grad():
        model._item_embeddings.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
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

    sampled_pred = model.predict(batch, k=2, candidate_item_ids=torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long))
    full_pred = model.predict(
        batch,
        k=2,
        exclude_item_ids=torch.tensor([[1], [3]], dtype=torch.long),
        exclude_mask=torch.tensor([[True], [True]], dtype=torch.bool),
    )

    assert sampled_pred.tolist() == [[1, 2], [3, 4]]
    assert full_pred.tolist() == [[2, 3], [4, 1]]


def test_hstu_compute_loss_uses_ar_sampled_softmax_with_item_zero_and_collision_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    model = HSTUModel(HSTUConfig(history_max_length=2, normalize_embeddings=False, temperature=1.0, num_negatives=3))
    model._num_items = 3
    model._item_embeddings = nn.Embedding(4, 2, padding_idx=HSTU_PADDING_ITEM_ID)
    with torch.no_grad():
        model._item_embeddings.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )
        )

    negative_item_ids = torch.tensor(
        [
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1],
        ],
        dtype=torch.long,
    )

    def fake_randint(low, high, size, *, device=None, dtype=None, **kwargs):
        del kwargs
        assert (low, high, tuple(size)) == (0, 3, (3, 3))
        return negative_item_ids.to(device=device, dtype=dtype or torch.long)

    monkeypatch.setattr(torch, "randint", fake_randint)
    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[0, 1, 2], [1, 0, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([2, 1], dtype=torch.long),
    }
    sequence_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[1.0, 1.0], [8.0, 8.0], [7.0, 7.0]],
        ],
        dtype=torch.float32,
    )

    loss = model.compute_loss(batch, {"sequence_embeddings": sequence_embeddings})

    flat_prediction_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    positive_item_ids = torch.tensor([1, 2, 0], dtype=torch.long)
    real_item_embeddings = model._item_embeddings.weight[1:]
    positive_logits = torch.sum(flat_prediction_embeddings * real_item_embeddings[positive_item_ids], dim=1)
    negative_logits = torch.einsum("bd,bkd->bk", flat_prediction_embeddings, real_item_embeddings[negative_item_ids])
    negative_logits = negative_logits.masked_fill(negative_item_ids == positive_item_ids.unsqueeze(1), -5e4)
    expected = F.cross_entropy(
        torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1),
        torch.zeros(3, dtype=torch.long),
    )
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item())


def test_hstu_eval_encode_uses_query_timestamp_without_target_item(monkeypatch: pytest.MonkeyPatch) -> None:
    records = pd.DataFrame(
        [
            {
                USER_ID: 0,
                ITEM_ID: 0,
                TIMESTAMP: 10,
                LABEL: 1.0,
                HISTORY_ITEM_IDS: (0,),
                HISTORY_TIMESTAMPS: (5.0,),
            }
        ]
    )
    batch = HSTUEvalCollator(HSTUConfig(history_max_length=2), prepared_data=object())(records)
    assert batch[HISTORY_ITEM_IDS].tolist() == [[0, HSTU_PADDING_ITEM_ID]]
    assert batch[HISTORY_TIMESTAMPS].tolist() == [[5.0, 10.0]]

    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    monkeypatch.setattr(
        HSTUModel,
        "_complete_cumsum",
        lambda self, lengths: torch.cat([lengths.new_zeros(1), torch.cumsum(lengths.to(torch.int32), dim=0)]),
    )
    model = HSTUModel(HSTUConfig(history_max_length=2, normalize_embeddings=False, temperature=1.0))
    model._num_items = 2
    model._item_embeddings = nn.Embedding(3, 2, padding_idx=HSTU_PADDING_ITEM_ID)
    with torch.no_grad():
        model._item_embeddings.weight.copy_(torch.tensor([[0.0, 0.0], [3.0, 4.0], [9.0, 9.0]]))
    model._empty_history_embedding = nn.Parameter(torch.zeros(2))

    captured: dict[str, torch.Tensor] = {}

    class CapturePreprocessor(nn.Module):
        def forward(self, *, past_lengths, past_ids, past_embeddings, past_payloads):
            del past_payloads
            captured["past_lengths"] = past_lengths.detach().cpu()
            captured["past_ids"] = past_ids.detach().cpu()
            return past_lengths, past_embeddings, {}

    class CaptureEncoder(nn.Module):
        def forward(self, *, x, x_offsets, all_timestamps, invalid_attn_mask):
            del x_offsets, invalid_attn_mask
            captured["all_timestamps"] = all_timestamps.detach().cpu()
            return x

    model._input_preprocessor = CapturePreprocessor()
    model._encoder = CaptureEncoder()

    user_embeddings = model._encode_user_embeddings(batch)

    assert captured["past_lengths"].tolist() == [2]
    assert captured["past_ids"].tolist() == [[1, HSTU_PADDING_ITEM_ID]]
    assert captured["all_timestamps"].tolist() == [[5.0, 10.0]]
    assert user_embeddings.tolist() == [[3.0, 4.0]]


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
