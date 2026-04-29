from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from recbole3.dataset import ITEM_ID, SEEN_ITEM_IDS, USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.model import HISTORY_ITEM_IDS, TIGERConfig, TIGERModel, TIGERModelDataset, get_model_spec
from recbole3.model.tiger import TIGEREvalCollator, TIGERSIDCodec, TIGERTrainCollator
from recbole3.run import compose_config, run_experiment
from recbole3.trainer import Trainer, TrainerConfig
from tests.test_helpers import StubDataset, StubDatasetConfig, ensure_stub_tables


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _write_sid_file(path: Path, *, num_items: int = 8, width: int = 2) -> Path:
    path.write_text(
        json.dumps({str(item_id): [item_id, item_id + 10][:width] for item_id in range(num_items)}),
        encoding="utf-8",
    )
    return path


def _prepared_tiger_data(tmp_path: Path, *, history_max_length: int = 2) -> TIGERModelDataset:
    sid_file = _write_sid_file(tmp_path / "item_sids.json")
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    return TIGERModelDataset.from_task_dataset(
        prepared,
        model_config=TIGERConfig(sid_file=str(sid_file), history_max_length=history_max_length, num_beams=3, eval_topk=(2, 3)),
    )


def _tiny_tiger_config(sid_file: Path, *, num_beams: int = 3) -> TIGERConfig:
    return TIGERConfig(
        sid_file=str(sid_file),
        history_max_length=2,
        n_user_tokens=1,
        num_beams=num_beams,
        eval_topk=(2, 3),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )


def test_tiger_model_registration_exposes_expected_components() -> None:
    spec = get_model_spec("tiger")

    assert spec.config_cls is TIGERConfig
    assert spec.model_cls is TIGERModel
    assert spec.model_data_cls is TIGERModelDataset
    assert spec.trainer_cls is Trainer
    assert spec.trainer_config_cls is TrainerConfig

    from recbole3.model import TIGERConfig as ExportedTIGERConfig
    from recbole3.model import TIGERModel as ExportedTIGERModel
    from recbole3.model import TIGERModelDataset as ExportedTIGERModelDataset

    assert ExportedTIGERConfig is TIGERConfig
    assert ExportedTIGERModel is TIGERModel
    assert ExportedTIGERModelDataset is TIGERModelDataset


def test_tiger_config_loads_expected_defaults() -> None:
    cfg = compose_config(overrides=["model=tiger"])

    assert cfg.model.name == "tiger"
    assert cfg.model.sid_file == ""
    assert cfg.model.history_max_length == 20
    assert cfg.model.num_beams == 50
    assert list(cfg.model.eval_topk) == [5, 10]
    assert cfg.trainer.batch_size == 256
    assert cfg.trainer.max_epochs == 150
    assert cfg.trainer.monitor == "ndcg@10"
    assert cfg.trainer.optimizer.name == "AdamW"
    assert cfg.trainer.optimizer.kwargs.lr == 0.003
    assert cfg.trainer.optimizer.kwargs.weight_decay == 0.05
    assert cfg.trainer.eval.protocol == "full"
    assert cfg.trainer.eval.exclude_history is True


def test_tiger_sid_codec_reads_zero_based_item_sids_and_offsets_tokens(tmp_path: Path) -> None:
    sid_file = _write_sid_file(tmp_path / "item_sids.json", num_items=4)

    codec = TIGERSIDCodec.from_file(str(sid_file), num_items=4)

    assert codec.item_to_sid[0] == (0, 10)
    assert codec.item_to_tokens[0] == (1, 11)
    assert codec.token_tuple_to_item((1, 11)) == 0
    assert codec.n_digit == 2
    assert codec.fallback_item_ids == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, "missing 4 item ids"),
        ({"0": [0, 1], "1": [1, 2], "2": [2], "3": [3, 4]}, "same length"),
        ({"0": [0, 1], "1": [1, 2], "2": [2, -1], "3": [3, 4]}, "negative"),
        ({"0": [0, 1], "1": [1, 2], "2": [2, "x"], "3": [3, 4]}, "non-negative integers"),
        ({"0": [0, 1], "bad": [1, 2], "2": [2, 3], "3": [3, 4]}, "remapped item_id string"),
        ({"0": [0, 1], "1": [1, 2], "2": [2, 3], "4": [3, 4]}, "outside dataset range"),
        ({"0": [0, 1], "1": [0, 1], "2": [2, 3], "3": [3, 4]}, "duplicate SID"),
    ],
)
def test_tiger_sid_codec_rejects_invalid_sid_files(tmp_path: Path, payload: dict, match: str) -> None:
    sid_file = tmp_path / "item_sids.json"
    sid_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        TIGERSIDCodec.from_file(str(sid_file), num_items=4)


def test_tiger_sid_codec_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sid_file does not exist"):
        TIGERSIDCodec.from_file(str(tmp_path / "missing.json"), num_items=4)


def test_tiger_model_dataset_builds_histories_across_splits(tmp_path: Path) -> None:
    tiger_data = _prepared_tiger_data(tmp_path, history_max_length=2)

    assert tiger_data.get_train_dataset().frame[[USER_ID, ITEM_ID, HISTORY_ITEM_IDS]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 0, ITEM_ID: 1, HISTORY_ITEM_IDS: (0,)},
        {USER_ID: 1, ITEM_ID: 4, HISTORY_ITEM_IDS: ()},
        {USER_ID: 1, ITEM_ID: 5, HISTORY_ITEM_IDS: (4,)},
    ]
    assert tiger_data.get_eval_dataset("valid").frame[[USER_ID, ITEM_ID, SEEN_ITEM_IDS, HISTORY_ITEM_IDS]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 2, SEEN_ITEM_IDS: (0, 1), HISTORY_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 6, SEEN_ITEM_IDS: (4, 5), HISTORY_ITEM_IDS: (4, 5)},
    ]
    assert tiger_data.get_eval_dataset("test").frame[[USER_ID, ITEM_ID, SEEN_ITEM_IDS, HISTORY_ITEM_IDS]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 3, SEEN_ITEM_IDS: (0, 1, 2), HISTORY_ITEM_IDS: (1, 2)},
        {USER_ID: 1, ITEM_ID: 7, SEEN_ITEM_IDS: (4, 5, 6), HISTORY_ITEM_IDS: (5, 6)},
    ]


def test_tiger_model_dataset_validates_num_beams_against_eval_topk(tmp_path: Path) -> None:
    sid_file = _write_sid_file(tmp_path / "item_sids.json")
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(ValueError, match="num_beams"):
        TIGERModelDataset.from_task_dataset(
            prepared,
            model_config=TIGERConfig(sid_file=str(sid_file), num_beams=2, eval_topk=(3,)),
        )


def test_tiger_collators_build_teacher_forcing_and_generation_batches(tmp_path: Path) -> None:
    tiger_data = _prepared_tiger_data(tmp_path, history_max_length=2)
    config = TIGERConfig(sid_file=str(tmp_path / "item_sids.json"), history_max_length=2, num_beams=3, eval_topk=(2, 3))
    train_records = tiger_data.get_train_dataset().frame.iloc[:2].reset_index(drop=True)
    eval_records = tiger_data.get_eval_dataset("test").frame

    train_batch = TIGERTrainCollator(config, prepared_data=tiger_data)(train_records)
    eval_batch = TIGEREvalCollator(config, prepared_data=tiger_data)(eval_records)

    user_token = tiger_data.tiger_codec.semantic_vocab_size + 1
    eos_token = user_token + 1
    assert set(train_batch) == {"input_ids", "attention_mask", "labels"}
    assert train_batch["input_ids"].dtype == torch.long
    assert train_batch["attention_mask"].dtype == torch.long
    assert train_batch["labels"].dtype == torch.long
    assert train_batch["input_ids"].shape == (2, 6)
    assert train_batch["attention_mask"].shape == (2, 6)
    assert train_batch["labels"].shape == (2, 3)
    assert train_batch["input_ids"].tolist() == [
        [user_token, eos_token, 0, 0, 0, 0],
        [user_token, 1, 11, eos_token, 0, 0],
    ]
    assert train_batch["attention_mask"].tolist() == [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
    ]
    assert train_batch["labels"].tolist() == [
        [1, 11, eos_token],
        [2, 12, eos_token],
    ]
    assert set(eval_batch) == {"input_ids", "attention_mask"}
    assert eval_batch["input_ids"].shape == (2, 6)


def test_tiger_lightweight_forward_and_predict_on_cpu(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    sid_file = _write_sid_file(tmp_path / "item_sids.json")
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_tiger_config(sid_file)
    tiger_data = TIGERModelDataset.from_task_dataset(prepared, model_config=config)
    model = TIGERModel(config)
    train_batch = model.build_train_collator(tiger_data)(tiger_data.get_train_dataset().frame.iloc[:2].reset_index(drop=True))
    eval_batch = model.build_eval_collator(tiger_data)(tiger_data.get_eval_dataset("test").frame)

    outputs = model.forward(train_batch)
    loss = model.compute_loss(train_batch, outputs)
    predictions = model.predict(
        eval_batch,
        k=3,
        exclude_item_ids=torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.long),
        exclude_mask=torch.ones((2, 3), dtype=torch.bool),
    )

    assert torch.isfinite(loss)
    assert predictions.shape == (2, 3)
    assert predictions.dtype == torch.long
    assert predictions.min() >= 0
    assert predictions.max() < tiger_data.get_num_items()


def test_tiger_predict_rejects_sampled_evaluation_candidates(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    sid_file = _write_sid_file(tmp_path / "item_sids.json")
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_tiger_config(sid_file)
    tiger_data = TIGERModelDataset.from_task_dataset(prepared, model_config=config)
    model = TIGERModel(config)
    eval_batch = model.build_eval_collator(tiger_data)(tiger_data.get_eval_dataset("test").frame)

    with pytest.raises(NotImplementedError, match="sampled evaluation"):
        model.predict(eval_batch, k=2, candidate_item_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.long))


def test_run_experiment_with_tiny_tiger_smoke(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    ensure_stub_tables()
    sid_file = _write_sid_file(tmp_path / "item_sids.json")
    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: tiger",
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
    (config_dir / "model" / "tiger.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: tiger",
                f"  sid_file: {(sid_file).as_posix()}",
                "  history_max_length: 2",
                "  n_user_tokens: 1",
                "  num_beams: 2",
                "  eval_topk: [2]",
                "  num_layers: 1",
                "  num_decoder_layers: 1",
                "  d_model: 8",
                "  d_ff: 16",
                "  num_heads: 2",
                "  d_kv: 4",
                "  dropout_rate: 0.0",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  max_epochs: 1",
                "  optimizer:",
                "    name: AdamW",
                "    kwargs:",
                "      lr: 0.001",
                "  eval:",
                "    protocol: full",
                "    exclude_history: true",
                "    metrics:",
                "      - name: recall",
                "        ks: [2]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))

    assert result["prepared_data"].get_num_users() == 2
    assert result["prepared_data"].get_num_items() == 8
    assert len(result["fit"]["train_history"]) == 1
    assert result["test"]["protocol"] == "full"
    assert "recall@2" in result["test"]["metrics"]
