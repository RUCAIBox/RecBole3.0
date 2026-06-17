"""Unit tests for the E4SRec model migration.

Run with:
    pytest tests/test_e4srec.py -v

Requires: torch, transformers, peft, pandas
For GPU tests: bitsandbytes (for 8-bit loading)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_ITEMS = 50
EMBED_DIM = 64
BATCH_SIZE = 3


def _make_pretrained_embeds(num_items: int = NUM_ITEMS, embed_dim: int = EMBED_DIM) -> str:
    """Create a temporary .npy file with random item embeddings."""
    embeds = torch.randn(num_items, embed_dim)
    tmpdir = tempfile.mkdtemp()
    path = str(Path(tmpdir) / "item_embeds.npy")
    np.save(path, embeds.numpy())
    return path


def _mock_records() -> pd.DataFrame:
    """Build realistic mock feature records (matching BaseSequentialModelDataset output)."""
    from recbole3.dataset import ITEM_ID
    from recbole3.model.sequential import HISTORY_ITEM_IDS

    return pd.DataFrame(
        [
            {ITEM_ID: 5, HISTORY_ITEM_IDS: (1, 2, 3)},
            {ITEM_ID: 10, HISTORY_ITEM_IDS: (4, 5, 6, 7)},
            {ITEM_ID: 3, HISTORY_ITEM_IDS: (8,)},
        ]
    )


class _MockPreparedData:
    """Minimal prepared-data stub for collator & model initialization."""

    def get_num_users(self) -> int:
        return 10

    def get_num_items(self) -> int:
        return NUM_ITEMS


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestE4SRecConfig:
    """Verify E4SRecConfig dataclass."""

    def test_defaults(self):
        from recbole3.model.e4srec.config import E4SRecConfig

        cfg = E4SRecConfig()
        assert cfg.name == "e4srec"
        assert cfg.history_max_length == 50
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.05
        assert cfg.load_in_8bit is True
        assert cfg.load_in_4bit is False
        assert cfg.item_embed_dim == 64
        assert "{instruction}" in cfg.prompt_template

    def test_custom_values(self):
        from recbole3.model.e4srec.config import E4SRecConfig

        cfg = E4SRecConfig(
            base_model="meta-llama/Llama-2-7b-hf",
            history_max_length=20,
            lora_r=8,
            lora_alpha=32,
            lora_target_modules=("q_proj", "v_proj"),
        )
        assert cfg.base_model == "meta-llama/Llama-2-7b-hf"
        assert cfg.history_max_length == 20
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 32
        assert cfg.lora_target_modules == ("q_proj", "v_proj")

    def test_inherits_sequential_model_config(self):
        from recbole3.model.e4srec.config import E4SRecConfig
        from recbole3.model.sequential import SequentialModelConfig

        cfg = E4SRecConfig()
        assert isinstance(cfg, SequentialModelConfig)


# ---------------------------------------------------------------------------
# Collator tests (no model needed)
# ---------------------------------------------------------------------------

class TestCollators:
    """Verify collator padding, offset, and shapes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from recbole3.model.e4srec.config import E4SRecConfig

        self.config = E4SRecConfig(history_max_length=50)
        self.mock_data = _MockPreparedData()

    def test_train_collator_shapes(self):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(self.config, prepared_data=self.mock_data)
        batch = collator(_mock_records())

        assert batch["input_ids"].shape == (BATCH_SIZE, 4)  # max history len = 4
        assert batch["attention_mask"].shape == (BATCH_SIZE, 4)
        assert batch["labels"].shape == (BATCH_SIZE,)
        assert batch["attention_mask"].sum(dim=1).tolist() == [3, 4, 1]  # seq lengths

    def test_train_collator_item_id_offset(self):
        from recbole3.model.e4srec.data import E4SRecCollator, ITEM_ID_OFFSET
        from recbole3.dataset import ITEM_ID
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        collator = E4SRecCollator(self.config, prepared_data=self.mock_data)

        # Two records: shorter one gets right-padded to max_len
        record = pd.DataFrame([
            {ITEM_ID: 0, HISTORY_ITEM_IDS: (0,)},       # 1 item → padded
            {ITEM_ID: 1, HISTORY_ITEM_IDS: (0, 1)},     # 2 items → max_len
        ])
        batch = collator(record)

        # Framework ID 0 → internal ID 1
        assert batch["input_ids"][0, 0].item() == 0 + ITEM_ID_OFFSET
        assert batch["labels"][0].item() == 0 + ITEM_ID_OFFSET

        # First record is right-padded at the last position
        assert batch["input_ids"][0, -1].item() == 0  # PAD_TOKEN

    def test_train_collator_attention_mask(self):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(self.config, prepared_data=self.mock_data)
        batch = collator(_mock_records())

        # Row 0: 3 items → mask [1,1,1,0]
        assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0]
        # Row 1: 4 items → mask [1,1,1,1]
        assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1]
        # Row 2: 1 item → mask [1,0,0,0]
        assert batch["attention_mask"][2].tolist() == [1, 0, 0, 0]

    def test_eval_collator_no_labels(self):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(self.config, prepared_data=self.mock_data, include_labels=False)
        batch = collator(_mock_records())

        assert "labels" not in batch
        assert "input_ids" in batch
        assert "attention_mask" in batch

    def test_collator_history_truncation(self):
        from recbole3.model.e4srec.config import E4SRecConfig
        from recbole3.model.e4srec.data import E4SRecCollator
        from recbole3.dataset import ITEM_ID
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        short_cfg = E4SRecConfig(history_max_length=2)
        collator = E4SRecCollator(short_cfg, prepared_data=self.mock_data)

        # History longer than max → truncated to last 2
        record = pd.DataFrame([{ITEM_ID: 99, HISTORY_ITEM_IDS: (10, 20, 30, 40)}])
        batch = collator(record)

        # Should keep last 2 items only
        assert batch["attention_mask"][0].sum().item() == 2
        # Original items (40, 30) after truncation → (last 2: 30, 40) + 1 offset
        assert batch["input_ids"][0, :2].tolist() == [31, 41]  # 30+1, 40+1

    def test_collator_empty_history(self):
        from recbole3.model.e4srec.data import E4SRecCollator
        from recbole3.dataset import ITEM_ID
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        collator = E4SRecCollator(self.config, prepared_data=self.mock_data)

        record = pd.DataFrame([{ITEM_ID: 7, HISTORY_ITEM_IDS: ()}])
        batch = collator(record)

        # All padding, length 0
        assert batch["attention_mask"][0].sum().item() == 0
        # With max_len=0, input_ids is shape (1, 0) — empty sequence
        assert batch["input_ids"].shape == (1, 0)

    def test_collator_from_list_of_dicts(self):
        """Collator should accept list[dict] as well as DataFrame."""
        from recbole3.model.e4srec.data import E4SRecCollator
        from recbole3.dataset import ITEM_ID
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        collator = E4SRecCollator(self.config, prepared_data=self.mock_data)
        records = [
            {ITEM_ID: 1, HISTORY_ITEM_IDS: (0,)},
            {ITEM_ID: 2, HISTORY_ITEM_IDS: (0, 1)},
        ]
        batch = collator(records)
        assert batch["input_ids"].shape == (2, 2)
        assert batch["labels"].tolist() == [2, 3]  # 1+1, 2+1


# ---------------------------------------------------------------------------
# ModelDataset tests
# ---------------------------------------------------------------------------

class TestModelDataset:
    """Verify E4SRecModelDataset extends BaseSequentialModelDataset correctly."""

    def test_subclass(self):
        from recbole3.model.e4srec.data import E4SRecModelDataset
        from recbole3.model.sequential import BaseSequentialModelDataset

        assert issubclass(E4SRecModelDataset, BaseSequentialModelDataset)


# ---------------------------------------------------------------------------
# Model tests (with tiny backbone – CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_e4srec_model():
    """Create an E4SRecModel with a tiny GPT-2 backbone (CPU-safe, no 8-bit).

    This fixture is module-scoped so the model download happens once.
    """
    from recbole3.model.e4srec.config import E4SRecConfig
    from recbole3.model.e4srec.model import E4SRecModel

    embed_path = _make_pretrained_embeds()

    config = E4SRecConfig(
        name="e4srec",
        history_max_length=10,
        base_model="sshleifer/tiny-gpt2",
        load_in_8bit=False,
        load_in_4bit=False,
        torch_dtype="float32",
        use_gradient_checkpointing=False,
        lora_r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=("c_attn",),  # GPT-2 attention projection
        item_embed_path=embed_path,
        item_embed_dim=EMBED_DIM,
        instruction_text="Predict next item.",
        prompt_template="{instruction}",
        response_split="\nResponse:",
    )

    model = E4SRecModel(config)
    mock_data = _MockPreparedData()
    model.ensure_initialized(mock_data)
    return model


@pytest.fixture
def mock_data():
    return _MockPreparedData()


@pytest.fixture
def model_config():
    from recbole3.model.e4srec.config import E4SRecConfig

    return E4SRecConfig()


class TestE4SRecModelInit:
    """Verify lazy-init mechanics and parameter shapes."""

    def test_not_initialized_after_constructor(self):
        from recbole3.model.e4srec.config import E4SRecConfig
        from recbole3.model.e4srec.model import E4SRecModel

        model = E4SRecModel(E4SRecConfig())
        assert model._num_items == 0
        assert not hasattr(model, "llm_model") or model.llm_model is None

    def test_ensure_initialized_sets_num_items(self, tiny_e4srec_model):
        assert tiny_e4srec_model._num_items == NUM_ITEMS

    def test_ensure_initialized_idempotent(self, tiny_e4srec_model, mock_data):
        """Calling ensure_initialized with same num_items should be a no-op."""
        model = tiny_e4srec_model
        # Should not raise
        model.ensure_initialized(mock_data)

    def test_score_layer_dimensions(self, tiny_e4srec_model):
        """score output dim = num_items + 1 (position 0 is padding)."""
        assert tiny_e4srec_model.score.out_features == NUM_ITEMS + 1

    def test_embedding_table_dimensions(self, tiny_e4srec_model):
        """Embedding table should have (num_items + 1, embed_dim)."""
        assert tiny_e4srec_model.input_embeds.weight.shape == (NUM_ITEMS + 1, EMBED_DIM)

    def test_embedding_frozen(self, tiny_e4srec_model):
        """Pre-trained CF embeddings must be frozen."""
        assert not tiny_e4srec_model.input_embeds.weight.requires_grad

    def test_prompt_tokenized(self, tiny_e4srec_model):
        """Instruction and response should be tokenized to non-empty tensors."""
        assert tiny_e4srec_model._instruct_ids.numel() > 0
        assert tiny_e4srec_model._response_ids.numel() > 0
        assert tiny_e4srec_model._instruct_ids.shape[0] == 1  # batch dim
        assert tiny_e4srec_model._response_ids.shape[0] == 1


class TestE4SRecModelForward:
    """Verify forward() shapes & gradients."""

    def test_forward_output_shape(self, tiny_e4srec_model, model_config, mock_data):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(model_config, prepared_data=mock_data)
        batch = collator(_mock_records())

        tiny_e4srec_model.eval()
        with torch.no_grad():
            outputs = tiny_e4srec_model.forward(**batch)

        assert "logits" in outputs
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_ITEMS + 1)

    def test_forward_with_empty_history(self, tiny_e4srec_model, model_config, mock_data):
        """Model should handle entirely empty item sequences gracefully."""
        from recbole3.model.e4srec.data import E4SRecCollator
        from recbole3.dataset import ITEM_ID
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        # Need at least one record with non-empty to set max_len ≥ 1
        # Actually, let's test with empty history in isolation
        collator = E4SRecCollator(model_config, prepared_data=mock_data)
        # Single record with empty history, will pad to at least 1 position
        records = pd.DataFrame([{ITEM_ID: 7, HISTORY_ITEM_IDS: ()}])
        batch = collator(records)

        tiny_e4srec_model.eval()
        with torch.no_grad():
            outputs = tiny_e4srec_model.forward(**batch)

        # Should still produce valid logits (just from prompt + response)
        assert outputs["logits"].shape[1] == NUM_ITEMS + 1

    def test_compute_loss_backward(self, tiny_e4srec_model, model_config, mock_data):
        """Gradients should flow through trainable params (LoRA + projections)."""
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(model_config, prepared_data=mock_data)
        batch = collator(_mock_records())

        tiny_e4srec_model.train()
        outputs = tiny_e4srec_model.forward(**batch)
        loss = outputs.loss
        loss.backward()

        # Check that at least input_proj has gradients (it's always trainable)
        assert tiny_e4srec_model.input_proj.weight.grad is not None
        # Score layer should also have gradients
        assert tiny_e4srec_model.score.weight.grad is not None


class TestE4SRecModelPredict:
    """Verify predict() for full and sampled evaluation protocols."""

    def _make_batch(self, tiny_e4srec_model, model_config, mock_data):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(model_config, prepared_data=mock_data, include_labels=False)
        return collator(_mock_records())

    def test_predict_full_no_exclusion(self, tiny_e4srec_model, model_config, mock_data):
        """Full eval without history exclusion returns 0-indexed item IDs."""
        batch = self._make_batch(tiny_e4srec_model, model_config, mock_data)

        tiny_e4srec_model.eval()
        with torch.no_grad():
            preds = tiny_e4srec_model.predict(
                batch,
                k=5,
                candidate_item_ids=None,
                exclude_item_ids=None,
                exclude_mask=None,
            )

        assert preds.shape == (BATCH_SIZE, 5)
        # All returned IDs should be 0-indexed (0 to num_items-1)
        assert preds.min() >= 0
        assert preds.max() < NUM_ITEMS
        # Padding position (internal 0) should never appear
        assert (preds < 0).sum() == 0

    def test_predict_full_with_exclusion(self, tiny_e4srec_model, model_config, mock_data):
        """Excluded items should NOT appear in top-k results."""
        batch = self._make_batch(tiny_e4srec_model, model_config, mock_data)

        # Exclude item 0 (framework ID 0 → internal ID 1) for all rows
        exclude_item_ids = torch.tensor([
            [0, 0],  # exclusions for row 0
            [0, 0],  # exclusions for row 1
            [0, 0],  # exclusions for row 2
        ])
        exclude_mask = torch.tensor([
            [True, False],  # only item 0 is valid exclusion
            [True, False],
            [True, False],
        ])

        tiny_e4srec_model.eval()
        with torch.no_grad():
            preds = tiny_e4srec_model.predict(
                batch,
                k=10,
                candidate_item_ids=None,
                exclude_item_ids=exclude_item_ids,
                exclude_mask=exclude_mask,
            )

        # Item 0 should not appear in predictions
        for b in range(BATCH_SIZE):
            assert 0 not in preds[b].tolist(), f"Excluded item 0 found in row {b}"

    def test_predict_sampled(self, tiny_e4srec_model, model_config, mock_data):
        """Sampled eval should return from candidate set only."""
        batch = self._make_batch(tiny_e4srec_model, model_config, mock_data)

        candidates = torch.tensor([
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [0, 1, 2, 3, 4],
        ])

        tiny_e4srec_model.eval()
        with torch.no_grad():
            preds = tiny_e4srec_model.predict(
                batch,
                k=3,
                candidate_item_ids=candidates,
            )

        assert preds.shape == (BATCH_SIZE, 3)
        # All predictions must be from the candidate set
        for b in range(BATCH_SIZE):
            for p in preds[b].tolist():
                assert p in candidates[b].tolist(), f"Pred {p} not in candidates for row {b}"

    def test_predict_k_larger_than_items(self, tiny_e4srec_model, model_config, mock_data):
        """When k > num_items, topk should still work (return all items)."""
        batch = self._make_batch(tiny_e4srec_model, model_config, mock_data)

        tiny_e4srec_model.eval()
        with torch.no_grad():
            preds = tiny_e4srec_model.predict(
                batch, k=NUM_ITEMS + 10, exclude_item_ids=None, exclude_mask=None
            )

        # torch.topk limits k to the actual dim size; we'll get fewer than requested
        assert preds.shape[1] <= NUM_ITEMS

    def test_predict_returns_zero_indexed(self, tiny_e4srec_model, model_config, mock_data):
        """Returned item IDs must always be in [0, num_items)."""
        batch = self._make_batch(tiny_e4srec_model, model_config, mock_data)

        tiny_e4srec_model.eval()
        with torch.no_grad():
            preds = tiny_e4srec_model.predict(
                batch, k=5, exclude_item_ids=None, exclude_mask=None
            )

        assert preds.dtype == torch.long
        assert (preds >= 0).all()
        assert (preds < NUM_ITEMS).all()


# ---------------------------------------------------------------------------
# Item ID consistency tests
# ---------------------------------------------------------------------------

class TestItemIDConsistency:
    """End-to-end verification of the ITEM_ID_OFFSET contract."""

    def test_framework_roundtrip(self, tiny_e4srec_model, model_config, mock_data):
        """Framework IDs (0-indexed) → collator (1-indexed) → model → predict → 0-indexed."""
        from recbole3.model.e4srec.data import E4SRecCollator
        from recbole3.dataset import ITEM_ID
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        collator = E4SRecCollator(model_config, prepared_data=mock_data)

        # Single record with known items
        records = pd.DataFrame([{ITEM_ID: 5, HISTORY_ITEM_IDS: (0, 1, 2)}])
        batch = collator(records)

        # Collator adds ITEM_ID_OFFSET
        assert batch["input_ids"][0, :3].tolist() == [1, 2, 3]  # 0+1, 1+1, 2+1

        tiny_e4srec_model.eval()
        with torch.no_grad():
            preds = tiny_e4srec_model.predict(
                batch, k=NUM_ITEMS, exclude_item_ids=None, exclude_mask=None
            )

        # predict() subtracts ITEM_ID_OFFSET
        assert (preds >= 0).all()
        assert (preds < NUM_ITEMS).all()

    def test_padding_embedding_is_mean(self):
        """Padding embedding (index 0) should equal mean of all item embeddings."""
        embed_path = _make_pretrained_embeds()

        pretrained = torch.as_tensor(np.load(embed_path), dtype=torch.float32)

        expected_pad = pretrained[:NUM_ITEMS].mean(dim=0)

        from recbole3.model.e4srec.config import E4SRecConfig
        from recbole3.model.e4srec.model import E4SRecModel

        config = E4SRecConfig(
            name="e4srec",
            history_max_length=10,
            base_model="sshleifer/tiny-gpt2",
            load_in_8bit=False,
            load_in_4bit=False,
            torch_dtype="float32",
            use_gradient_checkpointing=False,
            lora_r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            lora_target_modules=("c_attn",),
            item_embed_path=embed_path,
            item_embed_dim=EMBED_DIM,
            instruction_text="test",
            prompt_template="{instruction}",
            response_split="\nR:",
        )

        model = E4SRecModel(config)
        model.ensure_initialized(_MockPreparedData())

        pad_embed = model.input_embeds.weight[0]
        assert torch.allclose(pad_embed, expected_pad, atol=1e-5)


# ---------------------------------------------------------------------------
# Module export test
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_init_exports(self):
        from recbole3.model.e4srec import (
            E4SRecCollator,
            E4SRecConfig,
            E4SRecModel,
            E4SRecModelDataset,
            E4SRecTrainer,
            ITEM_ID_OFFSET,
            PAD_TOKEN,
        )
        assert E4SRecConfig.__name__ == "E4SRecConfig"
        assert E4SRecModel.__name__ == "E4SRecModel"
        assert E4SRecModelDataset.__name__ == "E4SRecModelDataset"
        assert E4SRecCollator.__name__ == "E4SRecCollator"
        assert E4SRecTrainer.__name__ == "E4SRecTrainer"
        assert E4SRecCollator.__name__ == "E4SRecCollator"
        assert E4SRecTrainer.__name__ == "E4SRecTrainer"
        assert ITEM_ID_OFFSET == 1
        assert PAD_TOKEN == 0

    def test_model_table_entry(self):
        from recbole3.model import get_model_spec

        spec = get_model_spec("e4srec")
        assert spec.config_cls.__name__ == "E4SRecConfig"
        assert spec.model_cls.__name__ == "E4SRecModel"
        assert spec.model_data_cls.__name__ == "E4SRecModelDataset"
        assert spec.trainer_cls.__name__ == "E4SRecTrainer"


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestE4SRecCheckpoint:
    """Verify checkpoint save & load via safetensors (HF Trainer format)."""

    def test_save_state_dict_as_safetensors(self, tiny_e4srec_model, tmp_path):
        """Saving state_dict via safetensors should create model.safetensors."""
        from safetensors.torch import save_file

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        path = checkpoint_dir / "model.safetensors"
        save_file(tiny_e4srec_model.state_dict(), str(path))
        assert path.exists()

    def test_save_and_reload_roundtrip(self, tiny_e4srec_model, model_config, mock_data, tmp_path):
        """Checkpoint save + reload should preserve forward output."""
        from safetensors.torch import save_file, load_file
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(model_config, prepared_data=mock_data)
        batch = collator(_mock_records())

        tiny_e4srec_model.eval()
        with torch.no_grad():
            original_output = tiny_e4srec_model.forward(**batch)

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        path = checkpoint_dir / "model.safetensors"
        save_file(tiny_e4srec_model.state_dict(), str(path))

        # Reload and verify
        state_dict = load_file(str(path))
        tiny_e4srec_model.load_state_dict(state_dict, strict=False)

        tiny_e4srec_model.eval()
        with torch.no_grad():
            reloaded_output = tiny_e4srec_model.forward(**batch)

        assert torch.allclose(
            original_output["logits"], reloaded_output["logits"], atol=1e-4
        ), "Logits changed after checkpoint roundtrip"


class TestForwardHFCompatible:
    """Verify forward() returns loss when labels are present (HF Trainer convention)."""

    def test_forward_returns_loss_with_labels(self, tiny_e4srec_model, model_config, mock_data):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(model_config, prepared_data=mock_data)
        batch = collator(_mock_records())

        tiny_e4srec_model.train()
        result = tiny_e4srec_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        assert "loss" in result
        assert result["loss"].requires_grad
        assert "logits" in result

    def test_forward_no_loss_without_labels(self, tiny_e4srec_model, model_config, mock_data):
        from recbole3.model.e4srec.data import E4SRecCollator

        collator = E4SRecCollator(model_config, prepared_data=mock_data, include_labels=False)
        batch = collator(_mock_records())

        tiny_e4srec_model.eval()
        with torch.no_grad():
            result = tiny_e4srec_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        assert "loss" not in result
        assert "logits" in result


class TestE4SRecTrainerInit:
    """Verify E4SRecTrainer stores config and supports required protocol."""

    def test_takes_trainer_config(self):
        from recbole3.evaluation.config import EvalConfig
        from recbole3.trainer_config import TrainerConfig
        from recbole3.model.e4srec.trainer import E4SRecTrainer

        cfg = TrainerConfig(eval=EvalConfig(protocol="full"))
        trainer = E4SRecTrainer(cfg)
        assert trainer.config is cfg

    def test_has_run_method(self):
        from recbole3.evaluation.config import EvalConfig
        from recbole3.trainer_config import TrainerConfig
        from recbole3.model.e4srec.trainer import E4SRecTrainer
        import inspect

        trainer = E4SRecTrainer(TrainerConfig(eval=EvalConfig(protocol="full")))
        assert hasattr(trainer, "run")
        sig = inspect.signature(trainer.run)
        assert "model" in sig.parameters
        assert "prepared_data" in sig.parameters
        assert "output_dir" in sig.parameters


class TestDeviceMapResolution:
    """Verify DDP vs single-GPU device_map resolution."""

    def test_single_gpu_no_local_rank(self, monkeypatch):
        """Without LOCAL_RANK, should pin to GPU 0 if CUDA is available."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        from recbole3.model.e4srec.model import E4SRecModel

        result = E4SRecModel._resolve_device_map("auto")
        # If CUDA is available: {"": 0}; otherwise: None (CPU)
        import torch
        if torch.cuda.is_available():
            assert result == {"": 0}
        else:
            assert result is None

    def test_ddp_with_local_rank(self, monkeypatch):
        """With LOCAL_RANK=1, device_map should pin to that GPU."""
        monkeypatch.setenv("LOCAL_RANK", "1")

        from recbole3.model.e4srec.model import E4SRecModel

        result = E4SRecModel._resolve_device_map("auto")
        assert result == {"": 1}

    def test_device_map_explicit_value(self, monkeypatch):
        """Explicit non-'auto' device_map should still be overridden in DDP."""
        monkeypatch.setenv("LOCAL_RANK", "2")

        from recbole3.model.e4srec.model import E4SRecModel

        result = E4SRecModel._resolve_device_map("sequential")
        assert result == {"": 2}  # DDP overrides regardless of input


# ---------------------------------------------------------------------------
# Peft helper
# ---------------------------------------------------------------------------


def _peft_base_model(peft_model):
    """Extract the raw HuggingFace backbone from a PeftModel wrapper.

    PeftModel wraps the base transformer as::

        PeftModel → .base_model (LoraModel) → .model (HF backbone)

    This helper unwraps both levels so that callers can access
    ``config``, ``forward()``, and other HF-standard attributes.
    """
    model = peft_model
    # Unwrap PeftModel → LoraModel (or similar adapter wrapper)
    if hasattr(model, "base_model"):
        model = model.base_model
    # Unwrap LoraModel → raw HF backbone
    if hasattr(model, "model"):
        model = model.model
    return model


class TestPeftBaseModel:
    """Verify _peft_base_model extracts the raw backbone."""

    def test_extract_from_peft_model(self, tiny_e4srec_model):
        base = _peft_base_model(tiny_e4srec_model.llm_model)
        # The base model should not be a PeftModel
        assert not hasattr(base, "peft_config") or "Peft" not in type(base).__name__
        # Should have the standard HF model attributes
        assert hasattr(base, "config")
        assert hasattr(base, "forward")

