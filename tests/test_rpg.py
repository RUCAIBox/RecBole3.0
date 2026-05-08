from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from recbole3.dataset import ITEM_ID, SEEN_ITEM_IDS, USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.model import get_model_spec
from recbole3.model.rpg import (
    RPG_INPUT_IDS,
    RPG_LABELS,
    RPG_SEQ_LENS,
    RPGConfig,
    RPGModel,
    RPGModelDataset,
    RPGSemanticTokenizer,
    RPGTrainCollator,
    RPGTrainer,
    RPGTrainerConfig,
)
from tests.test_helpers import StubDataset, StubDatasetConfig


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _write_sid_file(tmp_path: Path, *, num_items: int = 8) -> Path:
    sid_file = tmp_path / "rpg_sids.json"
    item2sids = {str(item_id): [item_id % 4, item_id // 4] for item_id in range(num_items)}
    sid_file.write_text(json.dumps(item2sids), encoding="utf-8")
    return sid_file


def _rpg_config(sid_file: Path) -> RPGConfig:
    return RPGConfig(
        semantic_id_file=str(sid_file),
        n_codebook=2,
        codebook_size=4,
        max_item_seq_len=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_inner=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        temperature=1.0,
    )


def test_rpg_model_dataset_tokenizes_train_and_eval_sequences(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _rpg_config(_write_sid_file(tmp_path))

    rpg_data = RPGModelDataset.from_task_dataset(prepared, model_config=config)

    train_records = rpg_data.get_train_dataset().frame
    assert train_records[[USER_ID, ITEM_ID, RPG_INPUT_IDS, "attention_mask", RPG_LABELS, RPG_SEQ_LENS]].to_dict("records") == [
        {
            USER_ID: 0,
            ITEM_ID: 1,
            RPG_INPUT_IDS: [1, 0],
            "attention_mask": [1, 0],
            RPG_LABELS: [2, -100],
            RPG_SEQ_LENS: 1,
        },
        {
            USER_ID: 1,
            ITEM_ID: 5,
            RPG_INPUT_IDS: [5, 0],
            "attention_mask": [1, 0],
            RPG_LABELS: [6, -100],
            RPG_SEQ_LENS: 1,
        },
    ]

    valid_records = rpg_data.get_eval_dataset("valid").frame
    assert valid_records[[USER_ID, ITEM_ID, SEEN_ITEM_IDS, RPG_INPUT_IDS, RPG_LABELS, RPG_SEQ_LENS]].to_dict("records") == [
        {
            USER_ID: 0,
            ITEM_ID: 2,
            SEEN_ITEM_IDS: (0, 1),
            RPG_INPUT_IDS: [1, 2],
            RPG_LABELS: [3],
            RPG_SEQ_LENS: 2,
        },
        {
            USER_ID: 1,
            ITEM_ID: 6,
            SEEN_ITEM_IDS: (4, 5),
            RPG_INPUT_IDS: [5, 6],
            RPG_LABELS: [7],
            RPG_SEQ_LENS: 2,
        },
    ]

    batch = RPGTrainCollator(config, prepared_data=rpg_data)(train_records)
    assert batch[RPG_INPUT_IDS].tolist() == [[1, 0], [5, 0]]
    assert batch[RPG_LABELS].tolist() == [[2, -100], [6, -100]]


def test_rpg_model_registration_exists() -> None:
    model_spec = get_model_spec("rpg")
    assert model_spec.config_cls is RPGConfig
    assert model_spec.model_cls is RPGModel
    assert model_spec.model_data_cls is RPGModelDataset
    assert model_spec.trainer_cls is RPGTrainer
    assert model_spec.trainer_config_cls is RPGTrainerConfig


def test_rpg_predict_supports_sampled_and_full_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RPGModel(RPGConfig(n_codebook=2, codebook_size=4, temperature=1.0))
    model._num_items = 3
    model._n_pred_head = 2
    model._codebook_size = 4
    model.register_buffer(
        "item_id2tokens",
        torch.tensor(
            [
                [0, 0],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=torch.long,
        ),
        persistent=False,
    )

    def fake_token_logits(self, batch):
        return batch["token_logits"]

    monkeypatch.setattr(RPGModel, "_token_logits", fake_token_logits)
    batch = {
        RPG_INPUT_IDS: torch.zeros((2, 1), dtype=torch.long),
        "token_logits": torch.tensor(
            [
                [10.0, 8.0, 1.0, 0.0, 10.0, 8.0, 1.0, 0.0],
                [0.0, 1.0, 9.0, 0.0, 0.0, 1.0, 9.0, 0.0],
            ]
        ),
    }

    sampled_pred = model.predict(batch, k=1, candidate_item_ids=torch.tensor([[1, 2], [0, 2]], dtype=torch.long))
    full_pred = model.predict(
        batch,
        k=2,
        exclude_item_ids=torch.tensor([[0], [2]], dtype=torch.long),
        exclude_mask=torch.tensor([[True], [True]], dtype=torch.bool),
    )

    assert sampled_pred.tolist() == [[1], [2]]
    assert full_pred.tolist() == [[1, 2], [1, 0]]


def test_rpg_predict_rejects_out_of_range_candidate_items(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RPGModel(RPGConfig(n_codebook=2, codebook_size=4, temperature=1.0))
    model._num_items = 3
    model._n_pred_head = 2
    model._codebook_size = 4
    model.register_buffer(
        "item_id2tokens",
        torch.tensor([[0, 0], [1, 5], [2, 6], [3, 7]], dtype=torch.long),
        persistent=False,
    )

    def fake_token_logits(self, batch):
        return batch["token_logits"]

    monkeypatch.setattr(RPGModel, "_token_logits", fake_token_logits)
    batch = {
        RPG_INPUT_IDS: torch.zeros((1, 1), dtype=torch.long),
        "token_logits": torch.zeros((1, 8), dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="RPG item ids must be in"):
        model.predict(batch, k=1, candidate_item_ids=torch.tensor([[0, 3]], dtype=torch.long))


def test_rpg_semantic_id_file_validation_rejects_bad_codes(tmp_path: Path) -> None:
    sid_file = _write_sid_file(tmp_path)
    item2sids = json.loads(sid_file.read_text(encoding="utf-8"))
    item2sids["0"] = [0, 4]
    sid_file.write_text(json.dumps(item2sids), encoding="utf-8")

    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    with pytest.raises(ValueError, match="must be in"):
        RPGModelDataset.from_task_dataset(prepared, model_config=_rpg_config(sid_file))


def test_rpg_missing_explicit_semantic_id_file_fails(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _rpg_config(tmp_path / "missing_sids.json")

    with pytest.raises(FileNotFoundError, match="semantic_id_file"):
        RPGModelDataset.from_task_dataset(prepared, model_config=config)


def test_rpg_metadata_text_falls_back_to_title_and_description(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    tokenizer = RPGSemanticTokenizer(_rpg_config(_write_sid_file(tmp_path)), prepared)

    assert tokenizer._metadata_text({"title": "Alpha", "description": "Quest"}) == "Alpha Quest"
    assert tokenizer._metadata_text({"metadata_text": "Configured text", "title": "Alpha"}) == "Configured text"


def test_rpg_codebook_size_must_be_power_of_two() -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        RPGSemanticTokenizer._get_codebook_bits(3)


def test_rpg_trainer_steps_count_optimizer_steps_with_accumulation() -> None:
    trainer = RPGTrainer(
        RPGTrainerConfig(
            max_epochs=3,
            steps=None,
            gradient_accumulation_steps=4,
        )
    )

    assert trainer._optimizer_steps_per_epoch(10) == 3
    assert trainer._resolve_total_steps(10) == 9

    trainer.config.steps = 5
    assert trainer._resolve_epoch_count(10) == 2


def test_rpg_forward_computes_finite_loss(tmp_path: Path) -> None:
    pytest.importorskip("transformers")

    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _rpg_config(_write_sid_file(tmp_path))
    rpg_data = RPGModelDataset.from_task_dataset(prepared, model_config=config)
    model = RPGModel(config)

    records = rpg_data.get_train_dataset().frame
    batch = model.build_train_collator(rpg_data)(records)
    outputs = model.forward(batch)
    loss = model.compute_loss(batch, outputs)

    assert torch.isfinite(loss)
