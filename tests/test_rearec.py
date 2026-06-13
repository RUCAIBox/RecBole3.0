from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

from recbole3.dataset import ITEM_ID, LABEL, SEEN_ITEM_IDS, TIMESTAMP, USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.model import (
    MODEL_TABLE,
    HISTORY_ITEM_IDS,
    HISTORY_TIMESTAMPS,
    ReaRecConfig,
    ReaRecModel,
    ReaRecModelDataset,
    ReaRecTrainer,
    get_model_spec,
)
from recbole3.model.hstu.config import HSTU_PADDING_ITEM_ID, ITEM_ID_OFFSET
from recbole3.model.hstu.model import HSTUModel
from recbole3.model.rearec.data import (
    ReaRecEvalCollator,
    ReaRecHSTUEvalCollator,
    ReaRecHSTUTrainCollator,
    ReaRecTrainCollator,
)
from recbole3.model.rearec.layers import (
    HSTUBackbone,
    ReaRecAutoRegressiveWrapper,
    SASRecBackbone,
    TransformerEncoder,
    build_causal_attention_mask,
)
from recbole3.trainer import Trainer, TrainerConfig
from tests.test_helpers import StubDataset, StubDatasetConfig

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_NUM_ITEMS = 50
_B = 4
_L = 8
_D = 16
_K = 2


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _sasrec_config(**kwargs: Any) -> ReaRecConfig:
    defaults: dict[str, Any] = dict(
        name="rearec",
        backbone="sasrec",
        history_max_length=_L,
        embedding_dim=_D,
        num_layers=1,
        num_heads=2,
        inner_size=32,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        dropout=0.0,
        initializer_range=0.02,
        learning_strategy="prl",
        reason_step=_K,
        temperature=0.07,
        kl_weight=0.05,
        pl_weight=1.0,
        temp_scale=5.0,
        noise_factor=0.01,
        cl_weight=1.0,
        warmup_epochs=0,
    )
    defaults.update(kwargs)
    return ReaRecConfig(**defaults)


def _make_sasrec_batch() -> dict[str, torch.Tensor]:
    lengths = torch.randint(1, _L + 1, (_B,))
    history = torch.full((_B, _L), _NUM_ITEMS, dtype=torch.long)
    for i, ln in enumerate(lengths.tolist()):
        history[i, _L - int(ln) :] = torch.randint(0, _NUM_ITEMS, (int(ln),))
    return {
        HISTORY_ITEM_IDS: history,
        "history_lengths": lengths,
        ITEM_ID: torch.randint(0, _NUM_ITEMS, (_B,)),
    }


def _make_encoder() -> TransformerEncoder:
    return TransformerEncoder(
        n_layers=1,
        n_heads=2,
        hidden_size=_D,
        inner_size=32,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_rearec_config_has_correct_defaults() -> None:
    cfg = ReaRecConfig()
    assert cfg.name == "rearec"
    assert cfg.backbone == "sasrec"
    assert cfg.learning_strategy == "prl"
    assert cfg.reason_step == 2
    assert cfg.embedding_dim == 256
    assert cfg.warmup_epochs == 0


def test_rearec_config_inherits_sequential_model_config() -> None:
    from recbole3.model.sequential import SequentialModelConfig

    cfg = ReaRecConfig()
    assert isinstance(cfg, SequentialModelConfig)


def test_rearec_config_hstu_hyperparameters_present() -> None:
    cfg = ReaRecConfig()
    assert hasattr(cfg, "attention_dim")
    assert hasattr(cfg, "linear_hidden_dim")
    assert hasattr(cfg, "num_time_buckets")
    assert cfg.attention_dim == 32
    assert cfg.num_time_buckets == 128


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------


def test_rearec_model_is_registered_in_model_table() -> None:
    assert "rearec" in MODEL_TABLE


def test_rearec_model_registration_uses_correct_classes() -> None:
    spec = get_model_spec("rearec")
    assert spec.model_cls is ReaRecModel
    assert spec.config_cls is ReaRecConfig
    assert spec.model_data_cls is ReaRecModelDataset
    assert spec.trainer_cls is ReaRecTrainer
    assert spec.trainer_config_cls is TrainerConfig


def test_rearec_model_device_returns_parameter_device() -> None:
    """_model_device() must reflect the device of the model parameters."""
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)

    expected = next(model.parameters()).device
    assert model._model_device() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA to test cross-device handling")
def test_rearec_predict_handles_cpu_batch_with_cuda_model() -> None:
    """Regression: framework eval may deliver CPU batches while the model is on CUDA
    (e.g. via _ExistingAcceleratorEvalContext.prepare being a no-op). predict() must
    migrate the batch internally so it does not crash with a device mismatch.
    """
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)
    model.cuda()
    model.eval()

    batch = _make_sasrec_batch()  # CPU tensors
    assert batch[HISTORY_ITEM_IDS].device.type == "cpu"

    with torch.no_grad():
        top_k = model.predict(batch, k=5)

    assert top_k.shape == (_B, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA to test cross-device handling")
def test_rearec_forward_handles_cpu_batch_with_cuda_model() -> None:
    """Regression: forward() must migrate CPU batch tensors to model device."""
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)
    model.cuda()
    model.train()

    batch = _make_sasrec_batch()  # CPU tensors
    assert batch[HISTORY_ITEM_IDS].device.type == "cpu"

    outputs = model.forward(batch)
    loss = model.compute_loss(batch, outputs)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_trainer_backfills_is_main_process_on_existing_accelerator_context() -> None:
    """Regression guard: ReaRecTrainer.create_accelerator must inject is_main_process
    when the base returns the _ExistingAcceleratorEvalContext stand-in (which lacks
    that attribute upstream). See framework regression in commit 0960bc4.
    """
    from recbole3.trainer import _ExistingAcceleratorEvalContext

    stand_in = _ExistingAcceleratorEvalContext()
    assert not hasattr(stand_in, "is_main_process")  # pre-condition: upstream still broken

    trainer = ReaRecTrainer.__new__(ReaRecTrainer)  # skip __init__ (it expects configs)
    # Stub super().create_accelerator() to return the bare stand-in
    trainer_super = Trainer.create_accelerator
    Trainer.create_accelerator = lambda self: stand_in  # type: ignore[assignment]
    try:
        result = trainer.create_accelerator()
    finally:
        Trainer.create_accelerator = trainer_super  # type: ignore[assignment]

    assert result is stand_in
    assert result.is_main_process is True


# ---------------------------------------------------------------------------
# ReaRecModelDataset
# ---------------------------------------------------------------------------


def test_rearec_model_dataset_sasrec_builds_history_item_ids_only() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    rearec_data = ReaRecModelDataset.from_task_dataset(
        prepared,
        model_config=ReaRecConfig(backbone="sasrec", history_max_length=2),
    )
    train_frame = rearec_data.get_train_dataset().frame
    assert HISTORY_ITEM_IDS in train_frame.columns
    assert HISTORY_TIMESTAMPS not in train_frame.columns


def test_rearec_model_dataset_sasrec_history_values_are_correct() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    rearec_data = ReaRecModelDataset.from_task_dataset(
        prepared,
        model_config=ReaRecConfig(backbone="sasrec", history_max_length=2),
    )
    rows = (
        rearec_data.get_train_dataset()
        .frame[[USER_ID, ITEM_ID, HISTORY_ITEM_IDS]]
        .to_dict("records")
    )
    assert rows[0][HISTORY_ITEM_IDS] == ()
    assert rows[1][HISTORY_ITEM_IDS] == (0,)


def test_rearec_model_dataset_hstu_builds_both_history_item_ids_and_timestamps() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    rearec_data = ReaRecModelDataset.from_task_dataset(
        prepared,
        model_config=ReaRecConfig(backbone="hstu", history_max_length=2),
    )
    train_frame = rearec_data.get_train_dataset().frame
    assert HISTORY_ITEM_IDS in train_frame.columns
    assert HISTORY_TIMESTAMPS in train_frame.columns


def test_rearec_model_dataset_hstu_train_uses_prefix_splitting() -> None:
    """HSTU training must produce one record per interaction, not one per user."""
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    rearec_data = ReaRecModelDataset.from_task_dataset(
        prepared,
        model_config=ReaRecConfig(backbone="hstu", history_max_length=10),
    )
    train_frame = rearec_data.get_train_dataset().frame
    # StubDataset leave-one-out: 2 users × 2 training interactions = 4 records total.
    assert len(train_frame) == 4
    # First interaction of each user must have an empty prefix history.
    for uid in [0, 1]:
        user_rows = train_frame[train_frame[USER_ID] == uid].sort_values(TIMESTAMP)
        assert user_rows.iloc[0][HISTORY_ITEM_IDS] == ()
        assert user_rows.iloc[0][HISTORY_TIMESTAMPS] == ()
        # Second interaction carries the first item as its history.
        assert len(user_rows.iloc[1][HISTORY_ITEM_IDS]) == 1
        assert len(user_rows.iloc[1][HISTORY_TIMESTAMPS]) == 1


# ---------------------------------------------------------------------------
# SASRec collators
# ---------------------------------------------------------------------------


def test_rearec_train_collator_left_pads_and_includes_target() -> None:
    num_items = 10
    max_len = 4
    records = pd.DataFrame(
        [
            {HISTORY_ITEM_IDS: (0, 1, 2), ITEM_ID: 3},
            {HISTORY_ITEM_IDS: (4,), ITEM_ID: 5},
        ]
    )
    collator = ReaRecTrainCollator(
        ReaRecConfig(), object(), num_items=num_items, history_max_length=max_len
    )
    batch = collator(records)

    assert batch[HISTORY_ITEM_IDS].shape == (2, max_len)
    assert batch[HISTORY_ITEM_IDS][0].tolist() == [num_items, 0, 1, 2]
    assert batch[HISTORY_ITEM_IDS][1].tolist() == [num_items, num_items, num_items, 4]
    assert batch["history_lengths"].tolist() == [3, 1]
    assert batch[ITEM_ID].tolist() == [3, 5]


def test_rearec_eval_collator_left_pads_without_target() -> None:
    num_items = 10
    max_len = 3
    records = pd.DataFrame([{HISTORY_ITEM_IDS: (1, 2), ITEM_ID: 9}])
    collator = ReaRecEvalCollator(
        ReaRecConfig(), object(), num_items=num_items, history_max_length=max_len
    )
    batch = collator(records)

    assert HISTORY_ITEM_IDS in batch
    assert ITEM_ID not in batch
    assert batch[HISTORY_ITEM_IDS][0].tolist() == [num_items, 1, 2]


def test_rearec_train_collator_truncates_long_histories() -> None:
    num_items = 20
    max_len = 2
    records = pd.DataFrame([{HISTORY_ITEM_IDS: (0, 1, 2, 3, 4), ITEM_ID: 5}])
    collator = ReaRecTrainCollator(
        ReaRecConfig(), object(), num_items=num_items, history_max_length=max_len
    )
    batch = collator(records)

    assert batch[HISTORY_ITEM_IDS][0].tolist() == [3, 4]
    assert batch["history_lengths"].tolist() == [2]


# ---------------------------------------------------------------------------
# HSTU collators
# ---------------------------------------------------------------------------


def test_rearec_hstu_train_collator_right_pads_with_timestamps_and_target() -> None:
    max_len = 4
    records = pd.DataFrame(
        [
            {
                HISTORY_ITEM_IDS: (0, 1, 2),
                HISTORY_TIMESTAMPS: (1.0, 2.0, 3.0),
                ITEM_ID: 3,
                TIMESTAMP: 10.0,
            },
            {
                HISTORY_ITEM_IDS: (4,),
                HISTORY_TIMESTAMPS: (5.0,),
                ITEM_ID: 6,
                TIMESTAMP: 20.0,
            },
        ]
    )
    collator = ReaRecHSTUTrainCollator(
        ReaRecConfig(), object(), history_max_length=max_len
    )
    batch = collator(records)

    # Shape is (B, max_len + 1): L real slots + 1 virtual query-timestamp slot.
    assert batch[HISTORY_ITEM_IDS].shape == (2, max_len + 1)
    assert batch[HISTORY_TIMESTAMPS].shape == (2, max_len + 1)
    assert ITEM_ID in batch
    # Item IDs: history items then HSTU_PADDING_ITEM_ID; target NOT in sequence.
    assert batch[HISTORY_ITEM_IDS][0].tolist() == [0, 1, 2, HSTU_PADDING_ITEM_ID, HSTU_PADDING_ITEM_ID]
    assert batch[HISTORY_ITEM_IDS][1].tolist() == [
        4,
        HSTU_PADDING_ITEM_ID,
        HSTU_PADDING_ITEM_ID,
        HSTU_PADDING_ITEM_ID,
        HSTU_PADDING_ITEM_ID,
    ]
    # history_lengths is the real (non-padded) history length, NOT including the query slot.
    assert batch["history_lengths"].tolist() == [3, 1]
    assert batch[ITEM_ID].tolist() == [3, 6]
    # Query timestamp written at position == history_lengths[b].
    assert batch[HISTORY_TIMESTAMPS][0][3].item() == pytest.approx(10.0)
    assert batch[HISTORY_TIMESTAMPS][1][1].item() == pytest.approx(20.0)


def test_rearec_hstu_eval_collator_right_pads_without_target() -> None:
    max_len = 3
    records = pd.DataFrame(
        [
            {
                HISTORY_ITEM_IDS: (1, 2),
                HISTORY_TIMESTAMPS: (2.0, 3.0),
                ITEM_ID: 9,
                TIMESTAMP: 99.0,
            }
        ]
    )
    collator = ReaRecHSTUEvalCollator(
        ReaRecConfig(), object(), history_max_length=max_len
    )
    batch = collator(records)

    assert ITEM_ID not in batch
    # Shape is (B, max_len + 1).
    assert batch[HISTORY_ITEM_IDS].shape == (1, max_len + 1)
    assert batch[HISTORY_ITEM_IDS][0].tolist() == [1, 2, HSTU_PADDING_ITEM_ID, HSTU_PADDING_ITEM_ID]
    # Query timestamp (99.0) written at position 2 (== history_lengths[0]).
    assert batch[HISTORY_TIMESTAMPS][0].tolist() == pytest.approx([2.0, 3.0, 99.0, 0.0])


def test_rearec_hstu_train_collator_does_not_append_target_to_sequence() -> None:
    max_len = 3
    target_item = 99
    records = pd.DataFrame(
        [
            {
                HISTORY_ITEM_IDS: (1, 2),
                HISTORY_TIMESTAMPS: (1.0, 2.0),
                ITEM_ID: target_item,
                TIMESTAMP: 50.0,
            }
        ]
    )
    collator = ReaRecHSTUTrainCollator(
        ReaRecConfig(), object(), history_max_length=max_len
    )
    batch = collator(records)

    history_ids = batch[HISTORY_ITEM_IDS][0].tolist()
    assert target_item not in history_ids
    # Shape is (B, max_len + 1): history slots + 1 virtual query slot.
    assert history_ids == [1, 2, HSTU_PADDING_ITEM_ID, HSTU_PADDING_ITEM_ID]
    assert batch[ITEM_ID].tolist() == [target_item]


def test_rearec_hstu_collator_timestamps_query_slot_and_zero_padding() -> None:
    max_len = 4
    records = pd.DataFrame(
        [
            {
                HISTORY_ITEM_IDS: (0,),
                HISTORY_TIMESTAMPS: (10.0,),
                ITEM_ID: 1,
                TIMESTAMP: 20.0,
            }
        ]
    )
    collator = ReaRecHSTUTrainCollator(
        ReaRecConfig(), object(), history_max_length=max_len
    )
    batch = collator(records)

    # Width = max_len + 1 = 5.
    # Position 0 = item history timestamp; position 1 = query timestamp; 2..4 = zeros.
    assert batch[HISTORY_TIMESTAMPS][0].tolist() == pytest.approx([10.0, 20.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# build_causal_attention_mask
# ---------------------------------------------------------------------------


def test_build_causal_attention_mask_shape() -> None:
    lengths = torch.tensor([3, 2])
    mask = build_causal_attention_mask(4, lengths, original_seq_len=4)
    assert mask.shape == (2, 1, 4, 4)


def test_build_causal_attention_mask_blocks_left_padding_columns() -> None:
    lengths = torch.tensor([3])
    mask = build_causal_attention_mask(4, lengths, original_seq_len=4)
    assert (mask[0, 0, :, 0] < -1e9).all()
    assert mask[0, 0, 1, 1] == 0.0


def test_build_causal_attention_mask_is_causal() -> None:
    lengths = torch.tensor([4])
    mask = build_causal_attention_mask(4, lengths, original_seq_len=4)
    assert mask[0, 0, 0, 1] < -1e9
    assert mask[0, 0, 0, 2] < -1e9
    assert mask[0, 0, 2, 0] == 0.0
    assert mask[0, 0, 3, 0] == 0.0


def test_build_causal_attention_mask_reasoning_extension_not_masked() -> None:
    lengths = torch.tensor([2])
    mask = build_causal_attention_mask(5, lengths, original_seq_len=4)
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 4, 4] == 0.0


# ---------------------------------------------------------------------------
# SASRecBackbone
# ---------------------------------------------------------------------------


def test_sasrec_backbone_initial_encode_returns_correct_shapes() -> None:
    enc = _make_encoder()
    backbone = SASRecBackbone(enc)
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    last_hidden, state = backbone.initial_encode(seq, lengths)

    assert last_hidden.shape == (_B, 1, _D)
    assert "kv_caches" in state
    assert len(state["kv_caches"]) == 1


def test_sasrec_backbone_step_encode_returns_correct_shape_and_extends_kv_cache() -> None:
    enc = _make_encoder()
    backbone = SASRecBackbone(enc)
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    _, state = backbone.initial_encode(seq, lengths)
    new_token = torch.randn(_B, 1, _D)
    last_hidden2, state2 = backbone.step_encode(
        new_token, lengths, step=1, original_seq_len=_L, state=state
    )

    assert last_hidden2.shape == (_B, 1, _D)
    assert state2["kv_caches"][0]["k"].shape[2] == _L + 1


def test_sasrec_backbone_ignores_raw_context() -> None:
    enc = _make_encoder()
    backbone = SASRecBackbone(enc)
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    last_hidden, _ = backbone.initial_encode(
        seq, lengths, raw_context={"item_ids": torch.zeros(_B, _L, dtype=torch.long)}
    )
    assert last_hidden.shape == (_B, 1, _D)


def test_sasrec_backbone_kv_cache_grows_with_each_step() -> None:
    enc = _make_encoder()
    backbone = SASRecBackbone(enc)
    seq = torch.randn(_B, _L, _D)
    lengths = torch.full((_B,), _L, dtype=torch.long)

    _, state = backbone.initial_encode(seq, lengths)
    for step in range(1, _K + 1):
        new_token = torch.randn(_B, 1, _D)
        _, state = backbone.step_encode(
            new_token, lengths, step=step, original_seq_len=_L, state=state
        )
        assert state["kv_caches"][0]["k"].shape[2] == _L + step


# ---------------------------------------------------------------------------
# ReaRecAutoRegressiveWrapper
# ---------------------------------------------------------------------------


def test_ar_wrapper_reason_step_zero_returns_single_hidden_state() -> None:
    backbone = SASRecBackbone(_make_encoder())
    wrapper = ReaRecAutoRegressiveWrapper(backbone=backbone, hidden_size=_D, reason_step=0)
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    out = wrapper(seq, lengths)

    assert out.shape == (_B, 1, _D)


def test_ar_wrapper_reason_step_k_returns_k_plus_one_hidden_states() -> None:
    backbone = SASRecBackbone(_make_encoder())
    wrapper = ReaRecAutoRegressiveWrapper(backbone=backbone, hidden_size=_D, reason_step=_K)
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    out = wrapper(seq, lengths)

    assert out.shape == (_B, _K + 1, _D)


def test_ar_wrapper_prl_noise_doubles_batch_in_training() -> None:
    backbone = SASRecBackbone(_make_encoder())
    wrapper = ReaRecAutoRegressiveWrapper(backbone=backbone, hidden_size=_D, reason_step=_K)
    wrapper.train()
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    out = wrapper(seq, lengths, noise_factor=0.1)

    assert out.shape == (2 * _B, _K + 1, _D)


def test_ar_wrapper_no_batch_doubling_in_eval_mode() -> None:
    backbone = SASRecBackbone(_make_encoder())
    wrapper = ReaRecAutoRegressiveWrapper(backbone=backbone, hidden_size=_D, reason_step=_K)
    wrapper.eval()
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    out = wrapper(seq, lengths, noise_factor=0.1)

    assert out.shape == (_B, _K + 1, _D)


def test_ar_wrapper_no_batch_doubling_when_reason_step_zero() -> None:
    backbone = SASRecBackbone(_make_encoder())
    wrapper = ReaRecAutoRegressiveWrapper(backbone=backbone, hidden_size=_D, reason_step=0)
    wrapper.train()
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    out = wrapper(seq, lengths, noise_factor=0.1)

    assert out.shape == (_B, 1, _D)


def test_ar_wrapper_dropout_p_defaults_match_cfg() -> None:
    """Wrapper dropout must come from cfg.dropout, not the previous hard-coded 0.2."""
    cfg = _sasrec_config(dropout=0.5)
    model = ReaRecModel(cfg)
    model._init_params(_NUM_ITEMS)

    assert model._ar_wrapper.dropout.p == pytest.approx(0.5)


def test_ar_wrapper_dropout_p_respects_custom_value() -> None:
    cfg = _sasrec_config(dropout=0.3)
    model = ReaRecModel(cfg)
    model._init_params(_NUM_ITEMS)

    assert model._ar_wrapper.dropout.p == pytest.approx(0.3)


def test_ar_wrapper_raw_context_doubled_with_batch() -> None:
    backbone = SASRecBackbone(_make_encoder())
    wrapper = ReaRecAutoRegressiveWrapper(backbone=backbone, hidden_size=_D, reason_step=_K)
    wrapper.train()
    seq = torch.randn(_B, _L, _D)
    lengths = torch.randint(1, _L + 1, (_B,))

    captured: dict[str, Any] = {}
    original_initial = backbone.initial_encode

    def patched_initial(s, h, raw_context=None):  # type: ignore[return]
        if raw_context is not None:
            captured["raw_context_batch_size"] = raw_context["item_ids"].shape[0]
        return original_initial(s, h, raw_context=raw_context)

    backbone.initial_encode = patched_initial  # type: ignore[method-assign]

    raw_context = {"item_ids": torch.zeros(_B, _L, dtype=torch.long)}
    wrapper(seq, lengths, noise_factor=0.1, raw_context=raw_context)

    assert captured.get("raw_context_batch_size") == 2 * _B


# ---------------------------------------------------------------------------
# HSTUBackbone interface
# ---------------------------------------------------------------------------


def test_hstu_backbone_requires_hstu_model_argument() -> None:
    with pytest.raises(TypeError):
        HSTUBackbone()  # type: ignore[call-arg]


def test_hstu_backbone_get_item_embs_skips_padding_slot() -> None:
    class _FakeEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.randn(_NUM_ITEMS + 1, _D)

    class _FakeHSTU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._emb = _FakeEmbedding()

        def _item_embedding_module(self) -> _FakeEmbedding:
            return self._emb

    backbone = HSTUBackbone(_FakeHSTU())
    scoring = backbone.get_item_embs()

    assert scoring is not None
    assert scoring.shape == (_NUM_ITEMS, _D)


def test_hstu_backbone_get_item_embs_starts_at_item_id_offset() -> None:
    n_rows = _NUM_ITEMS + 1
    data = torch.arange(n_rows * _D, dtype=torch.float32).view(n_rows, _D)

    class _FakeEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = data

    class _FakeHSTU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._emb = _FakeEmbedding()

        def _item_embedding_module(self) -> _FakeEmbedding:
            return self._emb

    backbone = HSTUBackbone(_FakeHSTU())
    scoring = backbone.get_item_embs()

    assert torch.equal(scoring[0], data[ITEM_ID_OFFSET])


# ---------------------------------------------------------------------------
# ReaRecModel: init
# ---------------------------------------------------------------------------


def test_rearec_model_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="backbone"):
        ReaRecModel(ReaRecConfig(backbone="unknown"))


def test_rearec_model_sasrec_init_creates_item_pos_embeddings() -> None:
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)

    assert model._item_emb is not None
    assert model._pos_emb is not None
    assert model._ar_wrapper is not None
    assert model._item_emb.weight.shape == (_NUM_ITEMS + 1, _D)
    assert model._pos_emb.weight.shape == (_L + 1, _D)


def test_rearec_model_init_is_idempotent() -> None:
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)
    model._init_params(_NUM_ITEMS)

    assert model._num_items == _NUM_ITEMS


def test_rearec_model_hstu_init_creates_no_item_pos_emb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    cfg = ReaRecConfig(
        name="rearec",
        backbone="hstu",
        history_max_length=_L,
        embedding_dim=_D,
        num_layers=1,
        num_heads=2,
        reason_step=_K,
        temperature=0.07,
        attention_dim=8,
        linear_hidden_dim=8,
        num_time_buckets=16,
    )
    model = ReaRecModel(cfg)
    model._init_params(_NUM_ITEMS)

    assert model._item_emb is None
    assert model._pos_emb is None
    assert isinstance(model._ar_wrapper.backbone, HSTUBackbone)


def test_rearec_model_hstu_config_expands_max_encoder_length_for_k_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    cfg = ReaRecConfig(
        name="rearec",
        backbone="hstu",
        history_max_length=_L,
        embedding_dim=_D,
        num_layers=1,
        num_heads=2,
        reason_step=_K,
        temperature=0.07,
        attention_dim=8,
        linear_hidden_dim=8,
        num_time_buckets=16,
    )
    model = ReaRecModel(cfg)
    model._init_params(_NUM_ITEMS)

    hstu = model._ar_wrapper.backbone._hstu
    expected_hstu_max = _L + max(_K, 1) - 1
    assert hstu.config.history_max_length == expected_hstu_max


# ---------------------------------------------------------------------------
# ReaRecModel: ERL
# ---------------------------------------------------------------------------


def test_rearec_erl_forward_output_shape() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="erl"))
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    outputs = model.forward(batch)

    assert "model_output" in outputs
    assert "item_emb_weight" in outputs
    assert outputs["model_output"].shape == (_B, _K + 1, _D)
    assert outputs["item_emb_weight"].shape[0] >= _NUM_ITEMS


def test_rearec_erl_loss_is_finite_scalar() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="erl", kl_weight=0.05))
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_erl_loss_kl_weight_zero_produces_valid_loss() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="erl", kl_weight=0.0))
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ReaRecModel: PRL
# ---------------------------------------------------------------------------


def test_rearec_prl_forward_doubles_batch_in_training() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="prl", noise_factor=0.1))
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    outputs = model.forward(batch)

    assert outputs["model_output"].shape == (2 * _B, _K + 1, _D)


def test_rearec_prl_forward_no_doubling_in_eval() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="prl", noise_factor=0.1))
    model._init_params(_NUM_ITEMS)
    model.eval()

    batch = _make_sasrec_batch()
    with torch.no_grad():
        outputs = model.forward(batch)

    assert outputs["model_output"].shape == (_B, _K + 1, _D)


def test_rearec_prl_loss_is_finite_scalar() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="prl", noise_factor=0.1))
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_prl_warmup_suppresses_noise() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="prl", noise_factor=0.1, warmup_epochs=3)
    )
    model._init_params(_NUM_ITEMS)
    model.train()
    model._steps_per_epoch = 10

    model._train_steps = 0
    assert model._effective_noise_factor() == 0.0

    model._train_steps = 25
    assert model._effective_noise_factor() == 0.0

    model._train_steps = 31
    assert model._effective_noise_factor() > 0.0


def test_rearec_prl_warmup_zero_always_active() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="prl", noise_factor=0.1, warmup_epochs=0)
    )
    model._init_params(_NUM_ITEMS)
    model.train()
    model._train_steps = 0

    assert model._effective_noise_factor() == pytest.approx(0.1)


def test_rearec_effective_noise_zero_for_erl() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="erl", noise_factor=0.1))
    model._init_params(_NUM_ITEMS)
    model.train()

    assert model._effective_noise_factor() == 0.0


def test_rearec_effective_noise_zero_in_eval_mode() -> None:
    model = ReaRecModel(_sasrec_config(learning_strategy="prl", noise_factor=0.1))
    model._init_params(_NUM_ITEMS)
    model.eval()

    assert model._effective_noise_factor() == 0.0


def test_rearec_effective_loss_type_auto_sasrec() -> None:
    model = ReaRecModel(_sasrec_config(backbone="sasrec", loss_type="auto"))
    assert model._effective_loss_type() == "ce"


def test_rearec_effective_loss_type_auto_hstu() -> None:
    model = ReaRecModel(ReaRecConfig(backbone="hstu", loss_type="auto"))
    assert model._effective_loss_type() == "sampled_softmax"


def test_rearec_effective_loss_type_explicit_override() -> None:
    model = ReaRecModel(_sasrec_config(backbone="sasrec", loss_type="sampled_softmax"))
    assert model._effective_loss_type() == "sampled_softmax"

    model2 = ReaRecModel(ReaRecConfig(backbone="hstu", loss_type="ce"))
    assert model2._effective_loss_type() == "ce"


def test_rearec_effective_loss_type_invalid_raises() -> None:
    model = ReaRecModel(_sasrec_config(loss_type="unknown"))
    with pytest.raises(ValueError, match="loss_type"):
        model._effective_loss_type()


def test_rearec_sampled_softmax_erl_loss_is_finite() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="erl", loss_type="sampled_softmax", num_negatives=16)
    )
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_sampled_softmax_prl_loss_is_finite() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="prl", loss_type="sampled_softmax", num_negatives=16, noise_factor=0.1)
    )
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_sampled_softmax_erl_kl_term_is_finite() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="erl", loss_type="sampled_softmax", num_negatives=16, kl_weight=0.1)
    )
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_prl_reason_step_zero_no_progressive_loss() -> None:
    model = ReaRecModel(_sasrec_config(reason_step=0, learning_strategy="prl"))
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ReaRecModel: predict
# ---------------------------------------------------------------------------


def test_rearec_predict_returns_topk_shape() -> None:
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)
    model.eval()

    batch = _make_sasrec_batch()
    with torch.no_grad():
        top_k = model.predict(batch, k=10)

    assert top_k.shape == (_B, 10)
    assert top_k.dtype == torch.long


def test_rearec_predict_excludes_history_items() -> None:
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)
    model.eval()

    batch = _make_sasrec_batch()
    exclude_ids = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long
    )
    exclude_mask = torch.ones_like(exclude_ids, dtype=torch.bool)

    with torch.no_grad():
        top_k = model.predict(
            batch, k=5, exclude_item_ids=exclude_ids, exclude_mask=exclude_mask
        )

    assert top_k.shape == (_B, 5)
    for b in range(_B):
        for excl in exclude_ids[b].tolist():
            assert excl not in top_k[b].tolist()


def test_rearec_predict_candidate_mode_returns_subset() -> None:
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)
    model.eval()

    batch = _make_sasrec_batch()
    candidates = torch.randint(0, _NUM_ITEMS, (_B, 20), dtype=torch.long)

    with torch.no_grad():
        top_k = model.predict(batch, k=5, candidate_item_ids=candidates)

    assert top_k.shape == (_B, 5)
    for b in range(_B):
        for pred in top_k[b].tolist():
            assert pred in candidates[b].tolist()


# ---------------------------------------------------------------------------
# ReaRecModel: _scoring_embs
# ---------------------------------------------------------------------------


def test_rearec_sasrec_scoring_embs_excludes_padding_row() -> None:
    model = ReaRecModel(_sasrec_config())
    model._init_params(_NUM_ITEMS)

    scoring = model._scoring_embs()

    assert scoring.shape == (_NUM_ITEMS, _D)


def test_rearec_hstu_scoring_embs_delegates_to_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    cfg = ReaRecConfig(
        name="rearec",
        backbone="hstu",
        history_max_length=_L,
        embedding_dim=_D,
        num_layers=1,
        num_heads=2,
        reason_step=_K,
        temperature=0.07,
        attention_dim=8,
        linear_hidden_dim=8,
        num_time_buckets=16,
    )
    model = ReaRecModel(cfg)
    model._init_params(_NUM_ITEMS)

    scoring = model._scoring_embs()

    assert scoring.shape == (_NUM_ITEMS, _D)
    assert isinstance(model._ar_wrapper.backbone, HSTUBackbone)


# ---------------------------------------------------------------------------
# normalize_embeddings
# ---------------------------------------------------------------------------


def test_rearec_config_normalize_embeddings_defaults_to_true() -> None:
    cfg = ReaRecConfig()
    assert cfg.normalize_embeddings is True


def test_rearec_normalize_embeddings_true_erl_loss_is_finite() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="erl", normalize_embeddings=True)
    )
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_normalize_embeddings_true_prl_loss_is_finite() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="prl", normalize_embeddings=True, noise_factor=0.1)
    )
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_normalize_embeddings_false_erl_loss_is_finite() -> None:
    model = ReaRecModel(
        _sasrec_config(learning_strategy="erl", normalize_embeddings=False)
    )
    model._init_params(_NUM_ITEMS)
    model.train()

    batch = _make_sasrec_batch()
    loss = model.compute_loss(batch, model.forward(batch))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rearec_should_normalize_is_false_for_sasrec() -> None:
    """normalize_embeddings has no effect on SASRec backbone (_should_normalize always False)."""
    model_true = ReaRecModel(_sasrec_config(normalize_embeddings=True))
    model_false = ReaRecModel(_sasrec_config(normalize_embeddings=False))

    assert model_true._should_normalize() is False
    assert model_false._should_normalize() is False


def test_rearec_should_normalize_respects_config_for_hstu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """normalize_embeddings is respected for HSTU backbone."""
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)

    base_kwargs = dict(
        backbone="hstu", history_max_length=_L, embedding_dim=_D,
        num_layers=1, num_heads=2, attention_dim=8, linear_hidden_dim=8, num_time_buckets=16,
    )
    model_true = ReaRecModel(ReaRecConfig(**base_kwargs, normalize_embeddings=True))
    model_false = ReaRecModel(ReaRecConfig(**base_kwargs, normalize_embeddings=False))

    assert model_true._should_normalize() is True
    assert model_false._should_normalize() is False


def test_rearec_normalize_embeddings_sasrec_loss_unchanged() -> None:
    """SASRec loss must be identical regardless of normalize_embeddings value."""
    torch.manual_seed(42)
    model_norm = ReaRecModel(_sasrec_config(normalize_embeddings=True))
    model_norm._init_params(_NUM_ITEMS)
    model_norm.train()

    torch.manual_seed(42)
    model_raw = ReaRecModel(_sasrec_config(normalize_embeddings=False))
    model_raw._init_params(_NUM_ITEMS)
    model_raw.train()

    batch = _make_sasrec_batch()
    torch.manual_seed(0)
    loss_norm = model_norm.compute_loss(batch, model_norm.forward(batch))
    torch.manual_seed(0)
    loss_raw = model_raw.compute_loss(batch, model_raw.forward(batch))

    # normalize_embeddings is ignored for SASRec: losses must be equal
    assert torch.isclose(loss_norm, loss_raw)


def test_rearec_normalize_embeddings_predict_returns_valid_shape() -> None:
    model = ReaRecModel(_sasrec_config(normalize_embeddings=True))
    model._init_params(_NUM_ITEMS)
    model.eval()

    batch = _make_sasrec_batch()
    with torch.no_grad():
        top_k = model.predict(batch, k=5)

    assert top_k.shape == (_B, 5)
    assert top_k.dtype == torch.long


def test_rearec_hstu_config_inherits_normalize_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HSTUConfig built inside _init_params should reflect ReaRec's normalize_embeddings."""
    monkeypatch.setattr(HSTUModel, "_require_runtime_support", lambda self: None)
    cfg = ReaRecConfig(
        backbone="hstu",
        history_max_length=_L,
        embedding_dim=_D,
        num_layers=1,
        num_heads=2,
        attention_dim=8,
        linear_hidden_dim=8,
        num_time_buckets=16,
        normalize_embeddings=False,
    )
    model = ReaRecModel(cfg)
    model._init_params(_NUM_ITEMS)

    backbone = model._ar_wrapper.backbone
    assert isinstance(backbone, HSTUBackbone)
    assert backbone._hstu.config.normalize_embeddings is False
