from __future__ import annotations

import pytest
import torch

from recbole3.dataset import ITEM_ID, SEEN_ITEM_IDS
from recbole3.evaluation import EvalConfig
from recbole3.model import HISTORY_ITEM_IDS, get_model_spec
from recbole3.model.lares import (
    LARESConfig,
    LARESEvalCollator,
    LARESModel,
    LARESModelDataset,
    LARESTrainCollator,
    LARESTrainer,
)
from recbole3.model.lares.model import (
    ContrastiveLoss,
    MultiHeadAttention,
    TransformerEncoder,
)
from recbole3.trainer import TrainerConfig
from tests.test_helpers import StubDataset, StubDatasetConfig


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _prepared_lares_data(
    history_max_length: int | None = 50,
) -> LARESModelDataset:
    prepared = StubDataset(StubDatasetConfig()).prepare(
        eval_config=_full_eval_config()
    )
    return LARESModelDataset.from_task_dataset(
        prepared,
        model_config=LARESConfig(history_max_length=history_max_length),
    )


# ---- Registration ----

def test_lares_model_registration() -> None:
    spec = get_model_spec("lares")

    assert spec.config_cls is LARESConfig
    assert spec.model_cls is LARESModel
    assert spec.model_data_cls is LARESModelDataset
    assert spec.trainer_cls is LARESTrainer
    assert spec.trainer_config_cls is TrainerConfig


# ---- Config ----

def test_lares_config_defaults() -> None:
    cfg = LARESConfig()
    assert cfg.name == "lares"
    assert cfg.history_max_length == 50
    assert cfg.hidden_size == 64
    assert cfg.n_pre_layers == 1
    assert cfg.n_core_layers == 1
    assert cfg.n_heads == 2
    assert cfg.inner_size == 256
    assert cfg.mean_recurrence == 4.0
    assert cfg.state_init_method == "normal"
    assert cfg.adapter_type == "add"
    assert cfg.tau == 0.07
    assert cfg.alpha == 0.1
    assert cfg.gamma == 0.1
    assert cfg.sem_func == "cos"
    assert cfg.same_step is True


# ---- Model Dataset ----

def test_lares_model_dataset_builds_histories() -> None:
    lares_data = _prepared_lares_data(history_max_length=50)

    train_frame = lares_data.get_train_dataset().frame
    assert HISTORY_ITEM_IDS in train_frame.columns
    assert train_frame[HISTORY_ITEM_IDS].tolist() == [(0,), (4,)]

    eval_frame = lares_data.get_eval_dataset("valid").frame
    assert eval_frame[HISTORY_ITEM_IDS].tolist() == [(0, 1), (4, 5)]


def test_lares_model_dataset_computes_same_target_index() -> None:
    lares_data = _prepared_lares_data()

    idx = lares_data.same_target_index
    assert isinstance(idx, dict)
    # StubDataset has 4 train records with items [0, 1, 4, 5] — each unique,
    # so no items have >= 2 occurrences and same_target_index should be empty.
    assert idx == {}


def test_lares_model_dataset_stores_full_train_frame() -> None:
    lares_data = _prepared_lares_data()
    frame = lares_data.full_train_frame
    assert frame is not None
    assert HISTORY_ITEM_IDS in frame.columns
    assert len(frame) == 2


# ---- Collators ----

def test_lares_train_collator_pads_histories_and_augmentations() -> None:
    lares_data = _prepared_lares_data()
    config = LARESConfig(history_max_length=50)
    train_records = (
        lares_data.get_train_dataset().frame
        .reset_index(drop=True)
    )

    batch = LARESTrainCollator(config, prepared_data=lares_data)(train_records)

    assert HISTORY_ITEM_IDS in batch
    assert "history_lengths" in batch
    assert ITEM_ID in batch
    assert "aug_history_item_ids" in batch
    assert "aug_history_lengths" in batch

    # history_lengths should match actual histories (empty histories filtered out)
    assert batch["history_lengths"].tolist() == [1, 1]

    # padding check: max length should be >= the longest history
    assert batch[HISTORY_ITEM_IDS].shape[0] == 2
    # Augmented sequences fallback to original when no same-target pairs exist
    assert batch["aug_history_item_ids"].shape[0] == 2
    assert batch["aug_history_lengths"].shape[0] == 2


def test_lares_eval_collator_pads_histories_only() -> None:
    lares_data = _prepared_lares_data()
    config = LARESConfig(history_max_length=50)
    eval_records = lares_data.get_eval_dataset("test").frame

    batch = LARESEvalCollator(config, prepared_data=lares_data)(eval_records)

    assert HISTORY_ITEM_IDS in batch
    assert "history_lengths" in batch
    # No augmentation for eval
    assert "aug_history_item_ids" not in batch
    assert ITEM_ID not in batch

    # StubDataset test split: user 0 history=[0,1,2], user 1 history=[4,5,6]
    assert batch["history_lengths"].tolist() == [3, 3]
    assert batch[HISTORY_ITEM_IDS].shape[0] == 2


# ---- SASRec Transformer Encoder ----

def test_transformer_encoder_shapes() -> None:
    B, L, H = 2, 5, 64
    encoder = TransformerEncoder(
        n_layers=2,
        n_heads=2,
        hidden_size=H,
        inner_size=256,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    )
    x = torch.randn(B, L, H)
    attn_mask = torch.zeros(B, L, L, dtype=torch.float32)

    out = encoder(x, attn_mask)
    assert out.shape == (B, L, H)


def test_multi_head_attention_causal_mask() -> None:
    B, L, H = 2, 4, 64
    mha = MultiHeadAttention(n_heads=2, hidden_size=H, attn_dropout_prob=0.0)
    x = torch.randn(B, L, H)
    tril_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
    causal = torch.zeros(B, L, L).masked_fill(~tril_mask.unsqueeze(0), float("-inf"))

    out = mha(x, x, x, causal)
    assert out.shape == (B, L, H)
    assert not torch.isnan(out).any()


# ---- ContrastiveLoss ----

def test_contrastive_loss_cosine() -> None:
    cl = ContrastiveLoss(tau=0.07, sem_func="cos")
    x = torch.randn(8, 64)
    y = torch.randn(8, 64)

    loss = cl(x, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_contrastive_loss_dot() -> None:
    cl = ContrastiveLoss(tau=0.07, sem_func="dot")
    x = torch.randn(8, 64)
    y = torch.randn(8, 64)

    loss = cl(x, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---- LARESModel ----

def _initialized_model(
    num_items: int = 8,
    hidden_size: int = 16,
    n_pre_layers: int = 1,
    n_core_layers: int = 1,
    n_heads: int = 2,
) -> LARESModel:
    config = LARESConfig(
        history_max_length=50,
        hidden_size=hidden_size,
        n_pre_layers=n_pre_layers,
        n_core_layers=n_core_layers,
        n_heads=n_heads,
        inner_size=64,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
    )
    model = LARESModel(config)
    model._init_modules(num_items)
    return model


def test_lares_model_lazy_initialization() -> None:
    model = LARESModel(LARESConfig())
    assert model._num_items is None
    assert model._item_emb is None

    model._init_modules(8)
    assert model._num_items == 8
    assert model._item_emb is not None
    assert model._pre_encoder is not None
    assert model._core_encoder is not None
    assert model._pos_emb is not None

    # Re-initialization with same num_items should be a no-op
    model._init_modules(8)

    # Re-initialization with different num_items should raise
    with pytest.raises(ValueError, match="initialized for num_items"):
        model._init_modules(9)


def test_lares_model_item_embedding_offset() -> None:
    """Item ID 0 is padding, real items start at index 1."""
    model = _initialized_model(num_items=5, hidden_size=8)
    emb = model._item_emb
    assert emb.weight.shape[0] == 6  # num_items + 1
    assert emb.padding_idx == 0


def test_lares_model_forward_basic() -> None:
    """Forward pass returns user_embeddings."""
    model = _initialized_model(num_items=8, hidden_size=16)
    model.eval()

    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([3, 2], dtype=torch.long),
    }

    with torch.no_grad():
        outputs = model.forward(batch)

    assert "user_embeddings" in outputs
    assert outputs["user_embeddings"].shape == (2, 16)


def test_lares_model_forward_with_augmentation() -> None:
    """Forward pass with aug sequences returns both embeddings."""
    model = _initialized_model(num_items=8, hidden_size=16)
    model.eval()

    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[1, 2, 0], [4, 5, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([2, 2], dtype=torch.long),
        "aug_history_item_ids": torch.tensor([[1, 0, 0], [5, 0, 0]], dtype=torch.long),
        "aug_history_lengths": torch.tensor([1, 1], dtype=torch.long),
    }

    with torch.no_grad():
        outputs = model.forward(batch)

    assert "user_embeddings" in outputs
    assert "aug_user_embeddings" in outputs
    assert outputs["user_embeddings"].shape == (2, 16)
    assert outputs["aug_user_embeddings"].shape == (2, 16)


def test_lares_model_compute_loss_returns_finite() -> None:
    model = _initialized_model(num_items=8, hidden_size=16)
    model.train()

    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([3, 2], dtype=torch.long),
        ITEM_ID: torch.tensor([3, 5], dtype=torch.long),
        "aug_history_item_ids": torch.tensor([[1, 2, 0], [5, 0, 0]], dtype=torch.long),
        "aug_history_lengths": torch.tensor([2, 1], dtype=torch.long),
    }

    # forward first
    outputs = model.forward(batch)
    loss = model.compute_loss(batch, outputs)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_lares_model_predict_full() -> None:
    """Full evaluation: score all items, return top-k."""
    model = _initialized_model(num_items=8, hidden_size=16)
    model.eval()

    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([3, 2], dtype=torch.long),
    }

    with torch.no_grad():
        pred = model.predict(batch, k=3)

    assert pred.shape == (2, 3)
    assert pred.dtype == torch.long
    assert pred.min() >= 0
    assert pred.max() < 8


def test_lares_model_predict_with_exclusion() -> None:
    """Predict with seen-item exclusion."""
    model = _initialized_model(num_items=8, hidden_size=16)
    model.eval()

    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([3, 2], dtype=torch.long),
    }

    with torch.no_grad():
        pred = model.predict(
            batch,
            k=3,
            exclude_item_ids=torch.tensor([[1, 2], [4, 5]], dtype=torch.long),
            exclude_mask=torch.tensor([[True, True], [True, True]], dtype=torch.bool),
        )

    assert pred.shape == (2, 3)
    # Excluded items should not appear
    for i in range(2):
        excluded = {1, 2} if i == 0 else {4, 5}
        for p in pred[i].tolist():
            assert p not in excluded, f"item {p} should have been excluded"


def test_lares_model_predict_sampled() -> None:
    """Sampled evaluation: score only candidate items."""
    model = _initialized_model(num_items=8, hidden_size=16)
    model.eval()

    batch = {
        HISTORY_ITEM_IDS: torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "history_lengths": torch.tensor([3, 2], dtype=torch.long),
    }

    candidates = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)

    with torch.no_grad():
        pred = model.predict(batch, k=2, candidate_item_ids=candidates)

    assert pred.shape == (2, 2)
    # Predictions come from candidates
    for i in range(2):
        for p in pred[i].tolist():
            assert p in candidates[i].tolist()


# ---- Recurrence Step Sampling ----

def test_lares_recurrence_sampling_deterministic_at_eval() -> None:
    model = _initialized_model(num_items=8, hidden_size=16, n_pre_layers=1, n_core_layers=1)
    model.eval()
    steps = model._sample_T()
    assert isinstance(steps, int)
    assert steps == int(model.config.mean_recurrence)


def test_lares_recurrence_sampling_stochastic_at_train() -> None:
    model = _initialized_model(num_items=8, hidden_size=16)
    model.train()
    steps = model._sample_T()
    assert isinstance(steps, int)
    assert steps >= 1


def test_lares_recurrence_override() -> None:
    model = _initialized_model(num_items=8, hidden_size=16)
    model.eval()

    model._eval_recurrence_override = 10
    steps = model._sample_T()
    assert steps == 10

    model._eval_recurrence_override = None
    steps = model._sample_T()
    assert steps == int(model.config.mean_recurrence)


# ---- Build Collators via Model ----

def test_lares_model_build_train_collator() -> None:
    model = LARESModel(LARESConfig(history_max_length=50))
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    lares_data = LARESModelDataset.from_task_dataset(
        prepared, model_config=LARESConfig(history_max_length=50)
    )

    collator = model.build_train_collator(lares_data)
    assert isinstance(collator, LARESTrainCollator)
    assert model._num_items is not None


def test_lares_model_build_eval_collator() -> None:
    model = LARESModel(LARESConfig(history_max_length=50))
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    lares_data = LARESModelDataset.from_task_dataset(
        prepared, model_config=LARESConfig(history_max_length=50)
    )

    collator = model.build_eval_collator(lares_data)
    assert isinstance(collator, LARESEvalCollator)
    assert model._num_items is not None


# ---- End-to-end train/eval flow ----

def test_lares_full_train_eval_flow() -> None:
    """Full train (1 step) and eval flow with LARES on CPU."""
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = LARESConfig(
        history_max_length=50,
        hidden_size=16,
        n_pre_layers=1,
        n_core_layers=1,
        n_heads=2,
        inner_size=64,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
    )
    lares_data = LARESModelDataset.from_task_dataset(prepared, model_config=config)

    model = LARESModel(config)
    train_collator = model.build_train_collator(lares_data)
    eval_collator = model.build_eval_collator(lares_data)

    train_dataset = lares_data.get_train_dataset()
    eval_dataset = lares_data.get_eval_dataset("test")

    # Train forward + loss + backward
    model.train()
    train_records = train_dataset.frame.reset_index(drop=True)
    train_batch = train_collator(train_records)
    outputs = model.forward(train_batch)
    loss = model.compute_loss(train_batch, outputs)

    assert torch.isfinite(loss)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Eval predict
    model.eval()
    eval_records = eval_dataset.frame
    eval_batch = eval_collator(eval_records)

    with torch.no_grad():
        pred = model.predict(
            eval_batch,
            k=3,
            exclude_item_ids=torch.tensor(
                [list(r[SEEN_ITEM_IDS]) for _, r in eval_records.iterrows()],
                dtype=torch.long,
            ),
            exclude_mask=torch.ones((2, 3), dtype=torch.bool),
        )

    assert pred.shape == (2, 3)
    assert pred.dtype == torch.long
