"""Comprehensive tests for the BIGRec model integration.

Coverage areas
--------------
1.  Config — default values, field types, SequentialModelConfig inheritance.
2.  Registration — MODEL_TABLE entry, LazyImport pipeline.
3.  Prompt utilities — all domains, unknown-domain fallback, prompt structure.
4.  BIGRecModelDataset — history injection, truncation, cross-split accumulation.
5.  build_item_text_lookup — primary field, fallback, missing column handling.
6.  BIGRecSFTDataset — tokenisation, label masking, history truncation.
7.  batchify — various sizes, empty input, exact multiples.
8.  build_eval_prompts — prompt construction from eval frame.
9.  BIGRecTrainer._compute_metrics — Recall@K and NDCG@K correctness.
10. BIGRecTrainer._extract_embeddings — deterministic shape with fake model.
11. BIGRecTrainer._precompute_item_embeddings — cache save / load round-trip.
12. BIGRecTrainer._rank_from_texts — L2 ranking & metric computation (mock model).
13. BIGRecTrainer.fit — smoke test with mocked HF Trainer.
14. BIGRecTrainer.evaluate — smoke test with monkeypatched internals.
15. Utility — _is_main_process / _get_device_map.
16. Grounding weight injection — _apply_grounding_weights Eq.3 math,
    _compute_popularity_weights normalisation, _load_cf_weights validation,
    _build_grounding_weights mode dispatch, end-to-end ranking change.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from recbole3.dataset import ITEM_ID, LABEL, SEEN_ITEM_IDS, USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.evaluation.metric import NDCGMetric, RecallMetric, RetrievalEvalData
from recbole3.model import MODEL_TABLE, get_model_spec
from recbole3.model.bigrec import (
    BIGRecConfig,
    BIGRecModelDataset,
    BIGRecSFTDataset,
    BIGRecTrainer,
    batchify,
    build_eval_prompts,
    build_input_block,
    build_instruction,
    build_item_text_lookup,
    build_prompt,
)
from recbole3.model.bigrec.data import _DOMAIN_VOCAB, _PROMPT_PREAMBLE
from recbole3.model.sequential import HISTORY_ITEM_IDS
from tests.test_helpers import StubDataset, StubDatasetConfig


# ── Shared fixtures ────────────────────────────────────────────────────────────


def _full_eval() -> EvalConfig:
    return EvalConfig(protocol="full")


def _sampled_eval() -> EvalConfig:
    return EvalConfig(protocol="sampled")


def _prepare_bigrec_data(
    *,
    history_max_length: int | None = None,
    eval_protocol: str = "full",
) -> tuple[BIGRecModelDataset, BIGRecConfig]:
    """Build a BIGRecModelDataset from StubDataset.

    StubParser provides 2 users × 4 items each (item_ids 0-7) with title and
    metadata_text columns in the item_table.
    """
    prepared = StubDataset(StubDatasetConfig()).prepare(
        eval_config=EvalConfig(protocol=eval_protocol)
    )
    cfg = BIGRecConfig(
        history_max_length=history_max_length,
        eval_protocol=eval_protocol,
    )
    return BIGRecModelDataset.from_task_dataset(prepared, model_config=cfg), cfg


class _BatchEncodingMock(dict):
    """Dict subclass that supports ``.to(device)`` like HuggingFace BatchEncoding.

    The real tokenizer returns a ``BatchEncoding`` object (a dict subclass) with
    a ``.to()`` method.  Our mock must match this interface so that code doing
    ``tokenizer(...).to(device)`` works without a real tokenizer installed.
    """

    def to(self, device: Any) -> "_BatchEncodingMock":  # noqa: D401
        return _BatchEncodingMock({k: v.to(device) for k, v in self.items()})


class _MockTokenizer:
    """Minimal tokenizer stub — no real LLM dependency.

    Encodes text as character ordinals mod 30 (capped to max_length) so that
    the output length scales predictably with the input text length.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.padding_side = "right"
        self.truncation_side = "right"

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        ids = [ord(c) % 30 + 3 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        # Never return the LLaMA spurious space sentinel "▁".
        return ["<tok>"] * len(ids)

    def save_pretrained(self, path: str) -> None:
        pass

    def batch_decode(
        self,
        sequences: Any,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> list[str]:
        """Return a quoted placeholder title for each sequence."""
        return [f'"MockTitle{i}"' for i, _ in enumerate(sequences)]

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> _BatchEncodingMock:
        if isinstance(texts, str):
            texts = [texts]
        cap = min(max_length or 32, 32)
        B = len(texts)
        return _BatchEncodingMock({
            "input_ids": torch.zeros(B, cap, dtype=torch.long),
            "attention_mask": torch.ones(B, cap, dtype=torch.long),
        })


class _FakeOutput:
    """Fake model forward output with deterministic hidden states."""

    def __init__(self, B: int, seq_len: int, H: int = 4) -> None:
        hidden = torch.randn(B, seq_len, H)
        # Tuple of (num_layers+1) tensors; trainer uses index -1.
        self.hidden_states: tuple[torch.Tensor, ...] = (hidden,) * 3


class _FakeModel(nn.Module):
    """Fake CausalLM that returns reproducible fixed-dimension embeddings."""

    H: int = 4  # hidden size

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def eval(self) -> "_FakeModel":
        return self

    def forward(self, **kwargs: Any) -> _FakeOutput:
        B, seq_len = kwargs["input_ids"].shape
        return _FakeOutput(B, seq_len, self.H)

    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Append 5 zero tokens to the prompt."""
        B = input_ids.shape[0]
        new_tokens = torch.zeros(B, 5, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return iter([self._dummy])


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config
# ══════════════════════════════════════════════════════════════════════════════


class TestBIGRecConfig:
    def test_default_name(self) -> None:
        assert BIGRecConfig().name == "bigrec"

    def test_default_domain_is_product(self) -> None:
        assert BIGRecConfig().domain == "product"

    def test_default_history_max_length(self) -> None:
        assert BIGRecConfig().history_max_length == 10

    def test_default_eval_protocol_is_sampled(self) -> None:
        assert BIGRecConfig().eval_protocol == "sampled"

    def test_default_lora_params(self) -> None:
        cfg = BIGRecConfig()
        assert cfg.use_lora is True
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_target_modules == ("q_proj", "v_proj")

    def test_default_eval_topk(self) -> None:
        assert BIGRecConfig().eval_topk == (1, 5, 10, 20)

    def test_default_eval_metrics(self) -> None:
        assert BIGRecConfig().eval_metrics == ("recall", "ndcg")

    def test_checkpoint_path_defaults_none(self) -> None:
        assert BIGRecConfig().checkpoint_path is None

    def test_pipeline_stage_defaults_training(self) -> None:
        assert BIGRecConfig().pipeline_stage == "training"

    def test_config_override(self) -> None:
        cfg = BIGRecConfig(domain="movie", lora_r=16, history_max_length=5)
        assert cfg.domain == "movie"
        assert cfg.lora_r == 16
        assert cfg.history_max_length == 5

    def test_inherits_sequential_model_config(self) -> None:
        from recbole3.model.sequential import SequentialModelConfig

        assert issubclass(BIGRecConfig, SequentialModelConfig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Registration
# ══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_bigrec_in_model_table(self) -> None:
        assert "bigrec" in MODEL_TABLE

    def test_model_spec_config_cls(self) -> None:
        assert get_model_spec("bigrec").config_cls is BIGRecConfig

    def test_pipeline_is_lazy_import(self) -> None:
        from recbole3.utils import LazyImport

        spec = get_model_spec("bigrec")
        assert isinstance(spec.pipeline_cls, LazyImport)

    def test_lazy_pipeline_resolves_to_bigrec_pipeline(self) -> None:
        from recbole3.model.bigrec.pipeline import BIGRecPipeline

        spec = get_model_spec("bigrec")
        assert spec.pipeline_cls.resolve() is BIGRecPipeline


# ══════════════════════════════════════════════════════════════════════════════
# 3. Prompt utilities
# ══════════════════════════════════════════════════════════════════════════════


class TestPromptUtilities:

    # -- build_instruction -----------------------------------------------------

    @pytest.mark.parametrize("domain,expected_item,expected_action", [
        ("movie",   "movie",   "watched"),
        ("product", "product", "purchased"),
        ("item",    "item",    "interacted with"),
    ])
    def test_build_instruction_known_domains(
        self, domain: str, expected_item: str, expected_action: str
    ) -> None:
        text = build_instruction(domain)
        assert expected_item in text
        assert expected_action in text
        assert "a list of" in text  # official template wording (no count N)

    def test_build_instruction_unknown_domain_falls_back_to_item(self) -> None:
        text = build_instruction("unknown_xyz")
        assert "item" in text

    def test_build_instruction_uses_official_wording(self) -> None:
        """Official BIGRec instruction: 'Given a list of … before, please recommend …'"""
        text = build_instruction("movie")
        assert "a list of" in text
        assert "before" in text
        assert "please recommend" in text

    # -- build_input_block ------------------------------------------------------

    def test_build_input_block_quotes_titles(self) -> None:
        block = build_input_block("movie", ["Alpha", "Beta"])
        assert '"Alpha"' in block
        assert '"Beta"' in block

    def test_build_input_block_includes_all_titles(self) -> None:
        titles = ["A", "B", "C", "D"]
        block = build_input_block("product", titles)
        for t in titles:
            assert t in block

    def test_build_input_block_empty_history(self) -> None:
        # Should not raise; produces a grammatically odd but valid string.
        # Official template has no space between "before:" and the quoted titles.
        block = build_input_block("item", [])
        assert "before:" in block

    # -- build_prompt ----------------------------------------------------------

    def test_build_prompt_contains_all_sections(self) -> None:
        prompt = build_prompt("movie", ["Inception"], include_response_prefix=True)
        assert _PROMPT_PREAMBLE in prompt
        assert "### Instruction:\n" in prompt
        assert "### Input:\n" in prompt
        assert "### Response:\n" in prompt

    def test_build_prompt_without_response_prefix(self) -> None:
        prompt = build_prompt("movie", ["Inception"], include_response_prefix=False)
        assert "### Response:" not in prompt

    def test_build_prompt_history_titles_appear_in_output(self) -> None:
        titles = ["The Matrix", "Inception"]
        prompt = build_prompt("movie", titles)
        for t in titles:
            assert t in prompt

    def test_build_prompt_uses_official_instruction_wording(self) -> None:
        """Official BIGRec instruction uses 'a list of' (no count N in the template)."""
        titles = ["A", "B", "C"]
        prompt = build_prompt("product", titles)
        assert "a list of products" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# 4. BIGRecModelDataset
# ══════════════════════════════════════════════════════════════════════════════


def _frame_rows(dataset: Any, columns: list[str]) -> list[dict]:
    return dataset.frame.loc[:, columns].to_dict("records")


class TestBIGRecModelDataset:

    def test_history_item_ids_injected_into_train_split(self) -> None:
        data, _ = _prepare_bigrec_data()
        train_df = data.get_train_dataset().frame
        assert HISTORY_ITEM_IDS in train_df.columns

    def test_history_item_ids_injected_into_valid_split(self) -> None:
        data, _ = _prepare_bigrec_data()
        valid_df = data.get_eval_dataset("valid").frame
        assert HISTORY_ITEM_IDS in valid_df.columns

    def test_history_item_ids_injected_into_test_split(self) -> None:
        data, _ = _prepare_bigrec_data()
        test_df = data.get_eval_dataset("test").frame
        assert HISTORY_ITEM_IDS in test_df.columns

    def test_train_first_interaction_has_empty_history(self) -> None:
        data, _ = _prepare_bigrec_data()
        train_df = data.get_train_dataset().frame
        # Both users' very first interactions should have empty history.
        first_rows = train_df.groupby(USER_ID).first()
        for _, row in first_rows.iterrows():
            assert row[HISTORY_ITEM_IDS] == ()

    def test_train_second_interaction_includes_first_item(self) -> None:
        """User 0 buys item_0 then item_1; history at item_1 should be (0,)."""
        data, _ = _prepare_bigrec_data()
        train_df = data.get_train_dataset().frame
        user0 = train_df[train_df[USER_ID] == 0].reset_index(drop=True)
        assert user0.loc[1, HISTORY_ITEM_IDS] == (0,)

    def test_history_max_length_truncates_history_in_test(self) -> None:
        """With max_length=2, test split for user 0 (items 0-3) should see at most 2 items."""
        data, _ = _prepare_bigrec_data(history_max_length=2)
        test_df = data.get_eval_dataset("test").frame
        for _, row in test_df.iterrows():
            assert len(row[HISTORY_ITEM_IDS]) <= 2

    def test_history_accumulates_across_train_into_valid(self) -> None:
        """Histories in the valid split should include training items."""
        data, _ = _prepare_bigrec_data()
        train_df = data.get_train_dataset().frame
        valid_df = data.get_eval_dataset("valid").frame

        # User 0: trains on items 0, 1 → valid history should be (0, 1).
        user0_valid = valid_df[valid_df[USER_ID] == 0].iloc[0]
        assert 0 in user0_valid[HISTORY_ITEM_IDS]
        assert 1 in user0_valid[HISTORY_ITEM_IDS]

    def test_seen_item_ids_preserved_in_eval_splits(self) -> None:
        """Full-protocol eval must still carry seen_item_ids for exclusion."""
        data, _ = _prepare_bigrec_data(eval_protocol="full")
        valid_df = data.get_eval_dataset("valid").frame
        assert SEEN_ITEM_IDS in valid_df.columns

    def test_item_table_and_num_items_accessible(self) -> None:
        data, _ = _prepare_bigrec_data()
        assert data.get_num_items() > 0
        assert not data.get_item_table().empty


# ══════════════════════════════════════════════════════════════════════════════
# 5. build_item_text_lookup
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildItemTextLookup:

    def test_lookup_length_equals_num_items(self) -> None:
        data, cfg = _prepare_bigrec_data()
        lookup = build_item_text_lookup(data, cfg)
        assert len(lookup) == data.get_num_items()

    def test_lookup_reads_primary_title_field(self) -> None:
        data, cfg = _prepare_bigrec_data()
        lookup = build_item_text_lookup(data, cfg)
        # StubParser uses "title" column with values like "Alpha Quest".
        assert any("Quest" in s or "Tales" in s for s in lookup)

    def test_lookup_falls_back_to_metadata_text(self) -> None:
        """When title field is absent, fallback_item_text_field is used."""
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(
            item_text_field="nonexistent_column",
            fallback_item_text_field="metadata_text",
        )
        lookup = build_item_text_lookup(data, cfg)
        # metadata_text is same as title in StubParser.
        assert any("Quest" in s or "Tales" in s for s in lookup)

    def test_lookup_uses_placeholder_when_no_column_exists(self) -> None:
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(
            item_text_field="absent_col",
            fallback_item_text_field=None,
        )
        lookup = build_item_text_lookup(data, cfg)
        # Should be all placeholders like "item_0", "item_1", …
        assert all(s.startswith("item_") for s in lookup)

    def test_lookup_item0_is_placeholder_or_title(self) -> None:
        """item_id 0 is either a real title or the 'item_0' placeholder."""
        data, cfg = _prepare_bigrec_data()
        lookup = build_item_text_lookup(data, cfg)
        assert isinstance(lookup[0], str) and len(lookup[0]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. BIGRecSFTDataset
# ══════════════════════════════════════════════════════════════════════════════


def _make_sft_dataset(
    *,
    history_max_length: int | None = 3,
    max_input_length: int = 64,
    max_new_tokens: int = 16,
) -> BIGRecSFTDataset:
    data, _ = _prepare_bigrec_data(history_max_length=history_max_length)
    cfg = BIGRecConfig(
        history_max_length=history_max_length,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        domain="product",
    )
    item_lookup = build_item_text_lookup(data, cfg)
    train_frame = data.get_train_dataset().frame
    return BIGRecSFTDataset(
        records=train_frame,
        tokenizer=_MockTokenizer(),
        item_text_lookup=item_lookup,
        config=cfg,
    )


class TestBIGRecSFTDataset:

    def test_length_matches_training_rows(self) -> None:
        ds = _make_sft_dataset()
        data, _ = _prepare_bigrec_data(history_max_length=3)
        assert len(ds) == len(data.get_train_dataset().frame)

    def test_getitem_returns_input_ids_and_labels(self) -> None:
        ds = _make_sft_dataset()
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample

    def test_input_ids_and_labels_same_length(self) -> None:
        ds = _make_sft_dataset()
        for i in range(len(ds)):
            s = ds[i]
            assert len(s["input_ids"]) == len(s["labels"]), (
                f"Sample {i}: input_ids length {len(s['input_ids'])} != "
                f"labels length {len(s['labels'])}"
            )

    def test_prompt_portion_masked_with_minus_100(self) -> None:
        """Labels for the prompt portion must be -100 when train_on_inputs=False."""
        # Explicitly request response-only supervision.
        data, _ = _prepare_bigrec_data(history_max_length=3)
        cfg = BIGRecConfig(
            history_max_length=3,
            max_input_length=64,
            max_new_tokens=16,
            domain="product",
            train_on_inputs=False,  # explicit: mask prompt tokens
        )
        item_lookup = build_item_text_lookup(data, cfg)
        train_frame = data.get_train_dataset().frame
        ds = BIGRecSFTDataset(
            records=train_frame,
            tokenizer=_MockTokenizer(),
            item_text_lookup=item_lookup,
            config=cfg,
        )
        sample = ds[0]
        labels = sample["labels"]
        assert -100 in labels, "Expected prompt portion to be masked in labels"

    def test_full_sequence_supervised_when_train_on_inputs_true(self) -> None:
        """When train_on_inputs=True, labels equal input_ids (no -100 masking)."""
        data, _ = _prepare_bigrec_data(history_max_length=3)
        cfg = BIGRecConfig(
            history_max_length=3,
            max_input_length=64,
            max_new_tokens=16,
            domain="product",
            train_on_inputs=True,
        )
        item_lookup = build_item_text_lookup(data, cfg)
        train_frame = data.get_train_dataset().frame
        ds = BIGRecSFTDataset(
            records=train_frame,
            tokenizer=_MockTokenizer(),
            item_text_lookup=item_lookup,
            config=cfg,
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert -100 not in sample["labels"], (
                f"Sample {i}: labels should not contain -100 when train_on_inputs=True"
            )
            assert sample["labels"] == sample["input_ids"], (
                f"Sample {i}: labels should equal input_ids when train_on_inputs=True"
            )

    def test_response_portion_has_real_token_ids(self) -> None:
        """At least the last token (EOS) should not be -100."""
        ds = _make_sft_dataset()
        for i in range(len(ds)):
            labels = ds[i]["labels"]
            non_masked = [l for l in labels if l != -100]
            assert len(non_masked) > 0, f"Sample {i} has no supervised tokens"

    def test_total_length_does_not_exceed_max(self) -> None:
        max_input = 32
        max_new = 8
        ds = _make_sft_dataset(max_input_length=max_input, max_new_tokens=max_new)
        cap = max_input + max_new
        for i in range(len(ds)):
            assert len(ds[i]["input_ids"]) <= cap

    def test_history_truncation_via_config(self) -> None:
        """Samples built with max_length=1 should have very short prompts."""
        ds_short = _make_sft_dataset(history_max_length=1)
        ds_long = _make_sft_dataset(history_max_length=10)
        # Shorter history → shorter prompt → fewer input_ids on average.
        avg_short = np.mean([len(ds_short[i]["input_ids"]) for i in range(len(ds_short))])
        avg_long = np.mean([len(ds_long[i]["input_ids"]) for i in range(len(ds_long))])
        assert avg_short <= avg_long


# ══════════════════════════════════════════════════════════════════════════════
# 7. batchify
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchify:

    def test_exact_multiple(self) -> None:
        assert list(batchify([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_remainder_batch(self) -> None:
        assert list(batchify([1, 2, 3, 4, 5], 3)) == [[1, 2, 3], [4, 5]]

    def test_single_batch(self) -> None:
        assert list(batchify([1, 2], 10)) == [[1, 2]]

    def test_empty_input(self) -> None:
        assert list(batchify([], 4)) == []

    def test_batch_size_one(self) -> None:
        result = list(batchify([10, 20, 30], 1))
        assert result == [[10], [20], [30]]

    def test_items_preserved_in_order(self) -> None:
        items = list(range(100))
        flat = [x for batch in batchify(items, 7) for x in batch]
        assert flat == items


# ══════════════════════════════════════════════════════════════════════════════
# 8. build_eval_prompts
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildEvalPrompts:

    def _make_eval_frame(self) -> tuple[pd.DataFrame, list[str]]:
        data, cfg = _prepare_bigrec_data()
        item_lookup = build_item_text_lookup(data, cfg)
        eval_frame = data.get_eval_dataset("test").frame
        return eval_frame, item_lookup

    def test_returns_one_prompt_per_row(self) -> None:
        eval_frame, item_lookup = self._make_eval_frame()
        cfg = BIGRecConfig(domain="product")
        prompts = build_eval_prompts(eval_frame, item_lookup, cfg)
        assert len(prompts) == len(eval_frame)

    def test_each_prompt_has_response_prefix(self) -> None:
        eval_frame, item_lookup = self._make_eval_frame()
        cfg = BIGRecConfig(domain="product")
        for prompt in build_eval_prompts(eval_frame, item_lookup, cfg):
            assert "### Response:\n" in prompt

    def test_prompt_reflects_domain_vocabulary(self) -> None:
        eval_frame, item_lookup = self._make_eval_frame()
        cfg = BIGRecConfig(domain="movie")
        for prompt in build_eval_prompts(eval_frame, item_lookup, cfg):
            assert "movie" in prompt.lower() or "watched" in prompt.lower()

    def test_history_truncation_via_config(self) -> None:
        """Prompts built with history_max_length=1 should be shorter."""
        eval_frame, item_lookup = self._make_eval_frame()
        cfg_short = BIGRecConfig(history_max_length=1)
        cfg_long = BIGRecConfig(history_max_length=100)
        short_prompts = build_eval_prompts(eval_frame, item_lookup, cfg_short)
        long_prompts = build_eval_prompts(eval_frame, item_lookup, cfg_long)
        avg_short = np.mean([len(p) for p in short_prompts])
        avg_long = np.mean([len(p) for p in long_prompts])
        assert avg_short <= avg_long


# ══════════════════════════════════════════════════════════════════════════════
# 9. BIGRecTrainer._compute_metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeMetrics:
    """Unit tests for the metric computation path using hand-crafted eval data."""

    @staticmethod
    def _perfect_hit_eval_data(topk: int = 5) -> RetrievalEvalData:
        """Target item at rank-0 for every row → Recall@K = 1.0."""
        return RetrievalEvalData(
            pred_item_ids=np.arange(topk * 3).reshape(3, topk),
            target_item_ids=np.array([[0], [topk], [2 * topk]]),
            target_mask=np.ones((3, 1), dtype=bool),
        )

    @staticmethod
    def _zero_hit_eval_data(topk: int = 5) -> RetrievalEvalData:
        """Target item never appears in predictions → Recall@K = 0.0."""
        return RetrievalEvalData(
            pred_item_ids=np.arange(topk * 3).reshape(3, topk),
            # Targets are all -1, guaranteed not in preds.
            target_item_ids=np.full((3, 1), -1, dtype=np.int64),
            target_mask=np.ones((3, 1), dtype=bool),
        )

    def _trainer(self, topk=(1, 5), metrics=("recall", "ndcg")) -> BIGRecTrainer:
        return BIGRecTrainer(BIGRecConfig(eval_topk=topk, eval_metrics=metrics))

    def test_perfect_recall_is_one(self) -> None:
        trainer = self._trainer(topk=(1, 5))
        results = trainer._compute_metrics(self._perfect_hit_eval_data(topk=5))
        assert results["recall@1"] == pytest.approx(1.0)
        assert results["recall@5"] == pytest.approx(1.0)

    def test_zero_recall_is_zero(self) -> None:
        trainer = self._trainer(topk=(1, 5))
        results = trainer._compute_metrics(self._zero_hit_eval_data(topk=5))
        assert results["recall@1"] == pytest.approx(0.0)
        assert results["recall@5"] == pytest.approx(0.0)

    def test_perfect_ndcg_at_1_is_one(self) -> None:
        trainer = self._trainer(topk=(1,))
        results = trainer._compute_metrics(self._perfect_hit_eval_data(topk=5))
        assert results["ndcg@1"] == pytest.approx(1.0)

    def test_hit_at_rank_2_gives_correct_ndcg(self) -> None:
        """One user: target item at rank index 1 (0-indexed) → DCG = 1/log2(3)."""
        eval_data = RetrievalEvalData(
            pred_item_ids=np.array([[10, 7, 3]]),
            target_item_ids=np.array([[7]]),
            target_mask=np.array([[True]]),
        )
        trainer = self._trainer(topk=(3,), metrics=("ndcg",))
        results = trainer._compute_metrics(eval_data)
        expected_dcg = 1.0 / np.log2(3.0)  # rank=2 (1-indexed), discount=1/log2(2+1)
        assert results["ndcg@3"] == pytest.approx(expected_dcg)

    def test_unknown_metric_is_skipped(self) -> None:
        trainer = BIGRecTrainer(
            BIGRecConfig(eval_topk=(5,), eval_metrics=("recall", "foobar_metric"))
        )
        results = trainer._compute_metrics(self._perfect_hit_eval_data(topk=5))
        assert "recall@5" in results
        assert "foobar_metric@5" not in results

    def test_both_recall_and_ndcg_returned(self) -> None:
        trainer = self._trainer(topk=(5, 10), metrics=("recall", "ndcg"))
        results = trainer._compute_metrics(self._perfect_hit_eval_data(topk=10))
        assert "recall@5" in results
        assert "ndcg@10" in results


# ══════════════════════════════════════════════════════════════════════════════
# 10. BIGRecTrainer._extract_embeddings
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractEmbeddings:

    def _trainer(self) -> BIGRecTrainer:
        return BIGRecTrainer(BIGRecConfig(max_input_length=32, embedding_batch_size=4))

    def test_output_shape_num_texts(self) -> None:
        trainer = self._trainer()
        texts = ["hello world", "foo bar baz", "rec sys"]
        embs = trainer._extract_embeddings(
            _FakeModel(), _MockTokenizer(), texts, batch_size=2, device=torch.device("cpu")
        )
        assert embs.shape[0] == len(texts)

    def test_output_hidden_dim_is_model_h(self) -> None:
        trainer = self._trainer()
        texts = ["a", "b", "c", "d"]
        embs = trainer._extract_embeddings(
            _FakeModel(), _MockTokenizer(), texts, batch_size=2, device=torch.device("cpu")
        )
        assert embs.shape[1] == _FakeModel.H

    def test_output_is_float32_on_cpu(self) -> None:
        trainer = self._trainer()
        embs = trainer._extract_embeddings(
            _FakeModel(), _MockTokenizer(), ["test"], batch_size=1, device=torch.device("cpu")
        )
        assert embs.device.type == "cpu"
        assert embs.dtype == torch.float32

    def test_single_text(self) -> None:
        trainer = self._trainer()
        embs = trainer._extract_embeddings(
            _FakeModel(), _MockTokenizer(), ["singleton"], batch_size=1, device=torch.device("cpu")
        )
        assert embs.shape == (1, _FakeModel.H)

    def test_tokenizer_padding_side_restored(self) -> None:
        """_extract_embeddings must restore tokenizer.padding_side after the call."""
        trainer = self._trainer()
        tok = _MockTokenizer()
        tok.padding_side = "right"
        trainer._extract_embeddings(
            _FakeModel(), tok, ["x"], batch_size=1, device=torch.device("cpu")
        )
        assert tok.padding_side == "right"


# ══════════════════════════════════════════════════════════════════════════════
# 11. BIGRecTrainer._precompute_item_embeddings — cache round-trip
# ══════════════════════════════════════════════════════════════════════════════


class TestPrecomputeItemEmbeddings:

    def _trainer(self) -> BIGRecTrainer:
        return BIGRecTrainer(
            BIGRecConfig(embedding_batch_size=4, max_input_length=32, refresh_embedding_cache=False)
        )

    def test_cache_is_saved_and_reloaded(self, tmp_path) -> None:
        trainer = self._trainer()
        cache_path = str(tmp_path / "item_embs.pt")
        texts = ["item_a", "item_b", "item_c"]

        embs = trainer._precompute_item_embeddings(
            _FakeModel(), _MockTokenizer(), texts, cache_path, torch.device("cpu")
        )
        assert os.path.isfile(cache_path)

        # Second call should load from cache without calling _extract_embeddings.
        with patch.object(trainer, "_extract_embeddings") as mock_extract:
            embs2 = trainer._precompute_item_embeddings(
                _FakeModel(), _MockTokenizer(), texts, cache_path, torch.device("cpu")
            )
            mock_extract.assert_not_called()

        assert embs.shape == embs2.shape

    def test_refresh_flag_recomputes_embeddings(self, tmp_path) -> None:
        trainer = BIGRecTrainer(
            BIGRecConfig(embedding_batch_size=4, max_input_length=32, refresh_embedding_cache=True)
        )
        cache_path = str(tmp_path / "item_embs.pt")
        texts = ["x", "y"]

        trainer._precompute_item_embeddings(
            _FakeModel(), _MockTokenizer(), texts, cache_path, torch.device("cpu")
        )
        assert os.path.isfile(cache_path)

        with patch.object(trainer, "_extract_embeddings", wraps=trainer._extract_embeddings) as mock:
            trainer._precompute_item_embeddings(
                _FakeModel(), _MockTokenizer(), texts, cache_path, torch.device("cpu")
            )
            mock.assert_called_once()

    def test_output_shape_correct(self, tmp_path) -> None:
        trainer = self._trainer()
        cache_path = str(tmp_path / "embs.pt")
        texts = ["a", "b", "c", "d", "e"]
        embs = trainer._precompute_item_embeddings(
            _FakeModel(), _MockTokenizer(), texts, cache_path, torch.device("cpu")
        )
        assert embs.shape == (len(texts), _FakeModel.H)


# ══════════════════════════════════════════════════════════════════════════════
# 12. BIGRecTrainer._rank_from_texts — L2 ranking & metrics
# ══════════════════════════════════════════════════════════════════════════════


def _make_eval_frame_for_trainer(num_users: int = 2) -> tuple[pd.DataFrame, list[str]]:
    """Build a minimal eval frame and item lookup (used by grounding tests for num_items)."""
    data, cfg = _prepare_bigrec_data()
    item_lookup = build_item_text_lookup(data, cfg)
    eval_frame = data.get_eval_dataset("test").frame.head(num_users).reset_index(drop=True)
    return eval_frame, item_lookup


class TestRankFromTexts:
    """Test _rank_from_texts with deterministic mocks via monkeypatching."""

    def _trainer(self, eval_protocol: str = "full") -> BIGRecTrainer:
        return BIGRecTrainer(
            BIGRecConfig(
                eval_topk=(1, 3),
                eval_metrics=("recall", "ndcg"),
                eval_protocol=eval_protocol,
                eval_batch_size=2,
                max_input_length=32,
                max_new_tokens=8,
                num_beams=1,
                history_max_length=3,
            )
        )

    def test_returns_recall_and_ndcg_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer = self._trainer()
        data, cfg = _prepare_bigrec_data()
        item_lookup = build_item_text_lookup(data, cfg)
        num_items = len(item_lookup)

        def fake_extract(model, tok, texts, batch_size, device):
            return torch.zeros(len(texts), 4)

        monkeypatch.setattr(trainer, "_extract_embeddings", fake_extract)

        eval_frame = data.get_eval_dataset("test").frame.head(2).reset_index(drop=True)
        generated_texts = ["MockTitle0", "MockTitle1"]
        target_ids: list[int] = eval_frame[ITEM_ID].tolist()
        cand_lists: list[list[int] | None] = [None] * len(target_ids)

        results = trainer._rank_from_texts(
            emb_model=_FakeModel(),
            tokenizer=_MockTokenizer(),
            item_emb_device=torch.zeros(num_items, 4),
            generated_texts=generated_texts,
            target_ids=target_ids,
            cand_lists=cand_lists,
            grounding_weights=None,
            device=torch.device("cpu"),
        )

        assert "recall@1" in results
        assert "recall@3" in results
        assert "ndcg@1" in results
        assert "ndcg@3" in results

    def test_l2_ranking_selects_nearest_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Oracle embedding ≈ item_2's embedding → item_2 should rank first."""
        item_emb = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # closest to oracle below
            [0.5, 0.5, 0.0, 0.0],
        ])

        trainer = BIGRecTrainer(
            BIGRecConfig(
                eval_topk=(1, 4),
                eval_metrics=("recall",),
                eval_protocol="full",
                eval_batch_size=1,
                max_input_length=32,
                max_new_tokens=8,
                num_beams=1,
                history_max_length=3,
            )
        )

        oracle_vec = torch.tensor([[0.85, 0.15, 0.0, 0.0]])

        def fake_extract(model, tok, texts, batch_size, device):
            return oracle_vec.expand(len(texts), -1)

        monkeypatch.setattr(trainer, "_extract_embeddings", fake_extract)

        results = trainer._rank_from_texts(
            emb_model=_FakeModel(),
            tokenizer=_MockTokenizer(),
            item_emb_device=item_emb,
            generated_texts=["oracle_title"],
            target_ids=[2],  # item_2 is the target
            cand_lists=[None],
            grounding_weights=None,
            device=torch.device("cpu"),
        )

        # Oracle is closest to item_2 → should appear at rank-1 → recall@1 = 1.0
        assert results["recall@1"] == pytest.approx(1.0)

    def test_sampled_protocol_restricts_to_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With sampled protocol only candidate_item_ids are ranked."""
        data, cfg = _prepare_bigrec_data()
        num_items = len(build_item_text_lookup(data, cfg))

        trainer = self._trainer(eval_protocol="sampled")

        item_emb = torch.zeros(num_items, 4)
        item_emb[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # item_3 is in candidates

        # Oracle closest to item_3 among candidates [3, 4, 5].
        def fake_extract(model, tok, texts, batch_size, device):
            return torch.tensor([[0.0, 0.0, 0.0, 0.9]] * len(texts))

        monkeypatch.setattr(trainer, "_extract_embeddings", fake_extract)

        results = trainer._rank_from_texts(
            emb_model=_FakeModel(),
            tokenizer=_MockTokenizer(),
            item_emb_device=item_emb,
            generated_texts=["oracle_title"],
            target_ids=[3],
            cand_lists=[[3, 4, 5]],
            grounding_weights=None,
            device=torch.device("cpu"),
        )
        # item_3 is closest in the candidate set and is the target → recall@1 = 1.0.
        assert results["recall@1"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 13. BIGRecTrainer.fit — smoke test
# ══════════════════════════════════════════════════════════════════════════════


def _stub_fit_externals(
    trainer: BIGRecTrainer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: "_MockTokenizer | None" = None,
) -> "_MockTokenizer":
    """Monkeypatch all external LLM / HF calls inside fit() for smoke tests.

    Stubs: _load_tokenizer, _load_model.
    fit() no longer loads a base model or pre-computes embeddings during training
    (aligned with official BIGRec: training uses a single model, recommendation
    metrics are computed post-training by evaluate()).

    Returns the tokenizer stub that was installed (useful for asserting
    save_pretrained calls).
    """
    tok = tokenizer or _MockTokenizer()
    fake_model = _FakeModel()
    monkeypatch.setattr(trainer, "_load_tokenizer", lambda *a, **kw: tok)
    monkeypatch.setattr(trainer, "_load_model", lambda *a, **kw: fake_model)
    return tok


class TestTrainerFit:
    """Smoke tests for BIGRecTrainer.fit — mocks out all HF Trainer machinery."""

    def test_fit_returns_checkpoint_path(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(
            train_batch_size=1,
            num_train_epochs=1,
            history_max_length=2,
            max_input_length=32,
            max_new_tokens=8,
        )
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        # Stub HF Trainer to avoid actual training.
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = None
        mock_hf_trainer.save_state.return_value = None
        mock_hf_trainer.save_model.return_value = None

        with patch("recbole3.model.bigrec.trainer.HFTrainer", return_value=mock_hf_trainer):
            result = trainer.fit(data, output_dir=str(tmp_path))

        assert "checkpoint_path" in result
        assert result["checkpoint_path"] == str(tmp_path)

    def test_fit_calls_hf_trainer_train(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(history_max_length=2, max_input_length=32, max_new_tokens=8)
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        mock_hf_trainer = MagicMock()
        with patch("recbole3.model.bigrec.trainer.HFTrainer", return_value=mock_hf_trainer):
            trainer.fit(data, output_dir=str(tmp_path))

        mock_hf_trainer.train.assert_called_once()

    def test_fit_saves_tokenizer(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(history_max_length=2, max_input_length=32, max_new_tokens=8)
        trainer = BIGRecTrainer(cfg)

        tok = _MockTokenizer()
        tok_save_calls: list[str] = []

        def patched_save(path: str) -> None:
            tok_save_calls.append(path)

        tok.save_pretrained = patched_save  # type: ignore[method-assign]
        _stub_fit_externals(trainer, monkeypatch, tokenizer=tok)

        with patch("recbole3.model.bigrec.trainer.HFTrainer", return_value=MagicMock()):
            trainer.fit(data, output_dir=str(tmp_path))

        assert any(str(tmp_path) in p for p in tok_save_calls)

    def test_fit_passes_early_stopping_callback_to_hf_trainer(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """fit() must include an EarlyStoppingCallback in the HFTrainer callbacks list.

        Official BIGRec uses HF's built-in EarlyStoppingCallback monitoring LM
        validation loss (not a custom recommendation-metric callback).
        """
        from transformers import EarlyStoppingCallback

        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(history_max_length=2, max_input_length=32, max_new_tokens=8)
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        captured_kwargs: dict[str, Any] = {}

        def _capture_trainer(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("recbole3.model.bigrec.trainer.HFTrainer", side_effect=_capture_trainer):
            trainer.fit(data, output_dir=str(tmp_path))

        callbacks = captured_kwargs.get("callbacks", [])
        assert any(isinstance(cb, EarlyStoppingCallback) for cb in callbacks), (
            "Expected an EarlyStoppingCallback in the HFTrainer callbacks list"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 14. BIGRecTrainer.evaluate — smoke test
# ══════════════════════════════════════════════════════════════════════════════


class TestTrainerEvaluate:
    """Smoke tests for BIGRecTrainer.evaluate — stubs all heavy operations."""

    def _trainer(self) -> BIGRecTrainer:
        return BIGRecTrainer(
            BIGRecConfig(
                eval_topk=(1, 3),
                eval_metrics=("recall", "ndcg"),
                eval_protocol="full",
                eval_batch_size=2,
                max_input_length=32,
                max_new_tokens=8,
                num_beams=1,
                history_max_length=3,
            )
        )

    def test_evaluate_returns_metric_dict(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, _ = _prepare_bigrec_data()
        trainer = self._trainer()
        num_items = data.get_num_items()

        monkeypatch.setattr(trainer, "_load_trained_model", lambda cp: _FakeModel())
        # Stub base-model loader so no real LLM is touched (embedding_use_base_model=True default).
        monkeypatch.setattr(trainer, "_load_base_model_for_embedding", lambda *a: _FakeModel())
        monkeypatch.setattr(trainer, "_load_tokenizer", lambda **kw: _MockTokenizer())
        monkeypatch.setattr(
            trainer,
            "_precompute_item_embeddings",
            lambda *a, **kw: torch.zeros(num_items, _FakeModel.H),
        )
        monkeypatch.setattr(
            trainer,
            "_extract_embeddings",
            lambda model, tok, texts, batch_size, device: torch.zeros(len(texts), _FakeModel.H),
        )

        results = trainer.evaluate(data, checkpoint_path=str(tmp_path), split="test")
        assert isinstance(results, dict)
        assert "recall@1" in results or "recall@3" in results

    def test_evaluate_valid_split_runs_without_error(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, _ = _prepare_bigrec_data()
        trainer = self._trainer()
        num_items = data.get_num_items()

        monkeypatch.setattr(trainer, "_load_trained_model", lambda cp: _FakeModel())
        monkeypatch.setattr(trainer, "_load_base_model_for_embedding", lambda *a: _FakeModel())
        monkeypatch.setattr(trainer, "_load_tokenizer", lambda **kw: _MockTokenizer())
        monkeypatch.setattr(
            trainer,
            "_precompute_item_embeddings",
            lambda *a, **kw: torch.zeros(num_items, _FakeModel.H),
        )
        monkeypatch.setattr(
            trainer,
            "_extract_embeddings",
            lambda model, tok, texts, batch_size, device: torch.zeros(len(texts), _FakeModel.H),
        )

        # Should not raise.
        results = trainer.evaluate(data, checkpoint_path=str(tmp_path), split="valid")
        assert isinstance(results, dict)


# ══════════════════════════════════════════════════════════════════════════════
# 15. Utility: _is_main_process / _get_device_map
# ══════════════════════════════════════════════════════════════════════════════


class TestTrainerUtility:

    def test_is_main_process_true_when_rank_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RANK", raising=False)
        assert BIGRecTrainer(BIGRecConfig())._is_main_process() is True

    def test_is_main_process_true_when_rank_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "0")
        assert BIGRecTrainer(BIGRecConfig())._is_main_process() is True

    def test_is_main_process_false_when_rank_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "1")
        assert BIGRecTrainer(BIGRecConfig())._is_main_process() is False

    def test_get_device_map_returns_device_id_dict_when_no_local_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-process mode must use {"": device_id}, never "auto".

        device_map="auto" shards the model across all visible GPUs which causes
        CUDA peer-mapping errors during HF Trainer training loops.
        """
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        device_map = BIGRecTrainer(BIGRecConfig(device_id=0))._get_device_map()
        assert device_map == {"": 0}

    def test_get_device_map_always_returns_logical_zero_in_single_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_get_device_map always returns {"": 0} in single-process mode.

        fit()/evaluate() set CUDA_VISIBLE_DEVICES=device_id before CUDA init,
        so the target physical GPU always appears as logical GPU 0.
        """
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        device_map = BIGRecTrainer(BIGRecConfig(device_id=2))._get_device_map()
        assert device_map == {"": 0}


# ══════════════════════════════════════════════════════════════════════════════
# 16. Grounding weight injection (Eq. 3)
# ══════════════════════════════════════════════════════════════════════════════


class TestApplyGroundingWeights:
    """Unit tests for the static Eq. 3 formula in _apply_grounding_weights."""

    @staticmethod
    def _apply(dist: torch.Tensor, weights: torch.Tensor, gamma: float) -> torch.Tensor:
        return BIGRecTrainer._apply_grounding_weights(dist, weights, gamma)

    def test_gamma_zero_returns_uniform_dist_hat(self) -> None:
        """When gamma=0, D̃ = D̂ × (1+W)^0 = D̂ — weights have no effect."""
        dist = torch.tensor([[0.0, 1.0, 4.0]])           # [1, 3]
        weights = torch.tensor([0.0, 0.9, 0.5])          # [3]
        result = self._apply(dist, weights, gamma=0.0)
        # D̂ = min-max of [0, 1, 4] = [0, 0.25, 1.0]
        expected = torch.tensor([[0.0, 0.25, 1.0]])
        assert torch.allclose(result, expected, atol=1e-4)

    def test_high_weight_lowers_effective_distance(self) -> None:
        """Item with higher weight should get a lower effective distance than an
        equally-ranked item without weight.

        Key insight of Eq. 3: only the item with D̂=0 (the absolute L2 minimum)
        has a guaranteed D̃=0 regardless of weights.  Among items that are NOT the
        L2 minimum, a higher weight genuinely reduces D̃ relative to zero weight.

        Setup:  3 items so the range is well-defined.
            dist = [1.0, 2.0, 3.0]  →  D̂ = [0.0, 0.5, 1.0]
            weights = [0.0, 1.0, 0.0],  gamma = 1
            D̃_1 = 0.5 × (1+1)^(−1) = 0.25
            D̃_2 = 1.0 × (1+0)^(−1) = 1.0
        item_1 (high weight) should rank above item_2 (no weight).
        """
        dist = torch.tensor([[1.0, 2.0, 3.0]])           # [1, 3]
        weights = torch.tensor([0.0, 1.0, 0.0])          # item_1 has max weight
        result = self._apply(dist, weights, gamma=1.0)
        # item_1 D̃ should be lower than item_2 D̃.
        assert result[0, 1] < result[0, 2]

    def test_output_shape_preserved(self) -> None:
        B, N = 4, 10
        dist = torch.rand(B, N)
        weights = torch.rand(N)
        result = self._apply(dist, weights, gamma=2.0)
        assert result.shape == (B, N)

    def test_per_row_normalisation_each_row_min_is_zero(self) -> None:
        """After min-max normalisation, each row's min D̂ should be 0."""
        dist = torch.rand(5, 8) * 10 + 1
        weights = torch.zeros(8)                         # gamma=1 → D̃ = D̂/2
        result = self._apply(dist, weights, gamma=0.0)   # gamma=0 → D̃ = D̂
        row_mins = result.min(dim=1)[0]
        assert torch.allclose(row_mins, torch.zeros(5), atol=1e-5)

    def test_all_zero_weights_identical_to_pure_l2_norm(self) -> None:
        """Zero weights with any gamma should match simple per-row normalisation."""
        dist = torch.tensor([[1.0, 2.0, 3.0, 6.0]])     # [1, 4]
        weights = torch.zeros(4)
        result = self._apply(dist, weights, gamma=99.0)
        # (1+0)^(-99) = 1, so result = D̂
        expected = (dist - dist.min(1, keepdim=True)[0]) / (
            dist.max(1, keepdim=True)[0] - dist.min(1, keepdim=True)[0] + 1e-8
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_large_gamma_strongly_promotes_high_weight_item(self) -> None:
        """With very large γ, a high-weight item leaps above lower-weight items
        at similar (but non-minimum) raw distances.

        Eq. 3 cannot beat the item with the absolute L2 minimum (D̂=0), but it
        CAN reorder all other items.  Here item_2 (farthest raw distance, weight=1)
        should rank above item_1 (middle distance, weight=0) with γ=100.

            dist = [0.3, 0.5, 0.8]   (item_0 is closest)
            D̂   = [0.0, 0.4, 1.0]   (range = 0.8−0.3 = 0.5)
            weights = [0.0, 0.0, 1.0]
            γ = 100
            D̃_1 = 0.4 × 1^(−100) = 0.4
            D̃_2 = 1.0 × 2^(−100) ≈ 7.9e-31  ← nearly zero!
        So item_2 ranks above item_1 despite its larger raw distance.
        """
        dist = torch.tensor([[0.3, 0.5, 0.8]])           # [1, 3]; item_0 is closest
        weights = torch.tensor([0.0, 0.0, 1.0])          # item_2 has highest weight
        result = self._apply(dist, weights, gamma=100.0)
        # item_2 D̃ ≈ 0 should be lower than item_1 D̃ = 0.4.
        assert result[0, 2] < result[0, 1]


class TestComputePopularityWeights:
    """Tests for _compute_popularity_weights normalisation and correctness."""

    def _make_trainer_and_data(self) -> tuple[BIGRecTrainer, Any]:
        data, cfg = _prepare_bigrec_data()
        return BIGRecTrainer(cfg), data

    def test_output_shape_matches_num_items(self) -> None:
        trainer, data = self._make_trainer_and_data()
        num_items = data.get_num_items()
        weights = trainer._compute_popularity_weights(data, num_items)
        assert weights.shape == (num_items,)

    def test_output_values_in_zero_one(self) -> None:
        trainer, data = self._make_trainer_and_data()
        num_items = data.get_num_items()
        weights = trainer._compute_popularity_weights(data, num_items)
        assert weights.min().item() >= 0.0 - 1e-6
        assert weights.max().item() <= 1.0 + 1e-6

    def test_output_min_is_zero(self) -> None:
        """After min-max normalisation, the global minimum must be 0."""
        trainer, data = self._make_trainer_and_data()
        num_items = data.get_num_items()
        weights = trainer._compute_popularity_weights(data, num_items)
        # Allow floating-point noise.
        assert weights.min().item() == pytest.approx(0.0, abs=1e-5)

    def test_output_max_is_one(self) -> None:
        """After min-max normalisation, the global maximum must be 1."""
        trainer, data = self._make_trainer_and_data()
        num_items = data.get_num_items()
        weights = trainer._compute_popularity_weights(data, num_items)
        assert weights.max().item() == pytest.approx(1.0, abs=1e-5)

    def test_more_popular_item_gets_higher_weight(self) -> None:
        """An item with more training interactions should receive a higher weight."""
        trainer, data = self._make_trainer_and_data()
        num_items = data.get_num_items()
        weights = trainer._compute_popularity_weights(data, num_items)

        train_df = data.get_train_dataset().frame
        counts = train_df[ITEM_ID].value_counts()
        if len(counts) < 2:
            pytest.skip("Need at least two distinct items in training data.")

        most_popular = int(counts.index[0])
        least_popular = int(counts.index[-1])

        # Only compare if they are genuinely different popularity.
        if counts.iloc[0] > counts.iloc[-1]:
            assert weights[most_popular].item() >= weights[least_popular].item()

    def test_item_not_in_train_has_zero_weight(self) -> None:
        """An item with no training interactions should have zero normalised weight.

        Because zero counts → zero frequency → min after normalisation is also 0
        (the least popular item always gets 0 after min-max).
        """
        trainer, data = self._make_trainer_and_data()
        num_items = data.get_num_items()
        weights = trainer._compute_popularity_weights(data, num_items)

        train_df = data.get_train_dataset().frame
        all_item_ids = set(range(num_items))
        seen_ids = set(int(x) for x in train_df[ITEM_ID].unique())
        unseen_ids = all_item_ids - seen_ids

        if unseen_ids:
            for uid in unseen_ids:
                # Unseen items get 0 raw frequency; after normalisation they should
                # have the minimum weight (≤ any seen-item weight).
                assert weights[uid].item() <= weights.max().item()


class TestLoadCFWeights:
    """Tests for _load_cf_weights file validation and normalisation."""

    def _trainer(self) -> BIGRecTrainer:
        return BIGRecTrainer(BIGRecConfig())

    def test_loads_and_normalises_to_zero_one(self, tmp_path) -> None:
        num_items = 6
        raw = torch.tensor([2.0, 5.0, 1.0, 8.0, 3.0, 6.0])
        cf_path = str(tmp_path / "cf_scores.pt")
        torch.save(raw, cf_path)

        trainer = BIGRecTrainer(BIGRecConfig(cf_score_path=cf_path))
        weights = trainer._load_cf_weights(num_items)

        assert weights.min().item() == pytest.approx(0.0, abs=1e-5)
        assert weights.max().item() == pytest.approx(1.0, abs=1e-5)

    def test_shape_mismatch_raises_value_error(self, tmp_path) -> None:
        raw = torch.rand(10)
        cf_path = str(tmp_path / "cf_scores.pt")
        torch.save(raw, cf_path)

        trainer = BIGRecTrainer(BIGRecConfig(cf_score_path=cf_path))
        with pytest.raises(ValueError, match="does not match num_items"):
            trainer._load_cf_weights(5)

    def test_missing_file_raises_file_not_found(self) -> None:
        trainer = BIGRecTrainer(BIGRecConfig(cf_score_path="/nonexistent/path.pt"))
        with pytest.raises(FileNotFoundError):
            trainer._load_cf_weights(10)

    def test_none_cf_score_path_raises_file_not_found(self) -> None:
        trainer = BIGRecTrainer(BIGRecConfig(cf_score_path=None))
        with pytest.raises(FileNotFoundError):
            trainer._load_cf_weights(10)

    def test_uniform_scores_normalise_to_zeros(self, tmp_path) -> None:
        """All-equal scores → zero range → result should be all-zero."""
        raw = torch.ones(5) * 3.7
        cf_path = str(tmp_path / "uniform_cf.pt")
        torch.save(raw, cf_path)

        trainer = BIGRecTrainer(BIGRecConfig(cf_score_path=cf_path))
        weights = trainer._load_cf_weights(5)
        assert torch.all(weights == 0.0)

    def test_relative_order_preserved(self, tmp_path) -> None:
        """After normalisation, the ranking order of scores must be unchanged."""
        raw = torch.tensor([1.0, 5.0, 3.0])
        cf_path = str(tmp_path / "cf.pt")
        torch.save(raw, cf_path)

        trainer = BIGRecTrainer(BIGRecConfig(cf_score_path=cf_path))
        weights = trainer._load_cf_weights(3)
        assert weights[1] > weights[2] > weights[0]


class TestBuildGroundingWeights:
    """Tests for _build_grounding_weights mode dispatch."""

    def _make_trainer_data(
        self, mode: str = "none", cf_path: str | None = None
    ) -> tuple[BIGRecTrainer, Any]:
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(grounding_mode=mode, cf_score_path=cf_path)
        return BIGRecTrainer(cfg), data

    def test_none_mode_returns_none(self) -> None:
        trainer, data = self._make_trainer_data(mode="none")
        result = trainer._build_grounding_weights(data, data.get_num_items())
        assert result is None

    def test_popularity_mode_returns_tensor(self) -> None:
        trainer, data = self._make_trainer_data(mode="popularity")
        num_items = data.get_num_items()
        result = trainer._build_grounding_weights(data, num_items)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (num_items,)

    def test_popularity_weights_in_zero_one(self) -> None:
        trainer, data = self._make_trainer_data(mode="popularity")
        num_items = data.get_num_items()
        result = trainer._build_grounding_weights(data, num_items)
        assert result is not None
        assert result.min().item() >= 0.0 - 1e-6
        assert result.max().item() <= 1.0 + 1e-6

    def test_cf_mode_returns_tensor(self, tmp_path) -> None:
        num_items: int
        data, _ = _prepare_bigrec_data()
        num_items = data.get_num_items()

        raw = torch.rand(num_items)
        cf_path = str(tmp_path / "cf.pt")
        torch.save(raw, cf_path)

        cfg = BIGRecConfig(grounding_mode="cf", cf_score_path=cf_path)
        trainer = BIGRecTrainer(cfg)
        result = trainer._build_grounding_weights(data, num_items)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (num_items,)

    def test_popularity_plus_cf_renormalises(self, tmp_path) -> None:
        """Combined mode: sum is re-normalised so max is 1."""
        data, _ = _prepare_bigrec_data()
        num_items = data.get_num_items()

        raw = torch.rand(num_items)
        cf_path = str(tmp_path / "cf.pt")
        torch.save(raw, cf_path)

        cfg = BIGRecConfig(grounding_mode="popularity+cf", cf_score_path=cf_path)
        trainer = BIGRecTrainer(cfg)
        result = trainer._build_grounding_weights(data, num_items)
        assert result is not None
        assert result.max().item() == pytest.approx(1.0, abs=1e-5)
        assert result.min().item() == pytest.approx(0.0, abs=1e-5)

    def test_case_insensitive_mode(self) -> None:
        """Mode string should be normalised; 'POPULARITY' == 'popularity'."""
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(grounding_mode="POPULARITY")
        trainer = BIGRecTrainer(cfg)
        result = trainer._build_grounding_weights(data, data.get_num_items())
        assert result is not None


class TestGroundingWeightsEndToEnd:
    """Integration test: grounding weights change evaluation ranking order."""

    def test_popularity_weight_promotes_target_item(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A target item with very high popularity weight ranks above equal-distance
        competitors, demonstrating that Eq. 3 reorders non-minimum-distance items.

        Setup:
          - 4 items total.
          - item_0 is uniquely closest to the oracle (D̂=0; always ranks first).
          - item_1 and item_2 are equidistant from the oracle (same raw L2).
          - item_3 is farthest.
          - TARGET = item_2 gets weight=1.0; item_1 gets weight=0.
          - With γ=50, D̃_2 = D̂_2 × 2^(−50) ≈ 0, so item_2 leaps to rank #2.
          - recall@2 should be 1.0.
        """
        num_items = 4
        # Oracle = zero vector; items have varying raw L2 distances.
        # L2(oracle, item_0)=0.1, L2(oracle,item_1)=L2(oracle,item_2)=0.5, item_3=1.0
        item_emb = torch.tensor([
            [0.1, 0.0, 0.0, 0.0],   # item_0: closest (D̂ = 0)
            [0.5, 0.0, 0.0, 0.0],   # item_1: mid-distance (D̂ = 0.444…)
            [0.0, 0.5, 0.0, 0.0],   # item_2: mid-distance (D̂ = 0.444…), TARGET
            [1.0, 0.0, 0.0, 0.0],   # item_3: farthest   (D̂ = 1.0)
        ])

        TARGET = 2

        cfg = BIGRecConfig(
            eval_topk=(2,),          # recall@2 — item_2 should be in top-2
            eval_metrics=("recall",),
            eval_protocol="full",
            eval_batch_size=1,
            max_input_length=32,
            max_new_tokens=8,
            num_beams=1,
            history_max_length=3,
            grounding_mode="popularity",
            grounding_gamma=50.0,
        )
        trainer = BIGRecTrainer(cfg)

        # Oracle = zero vector → item_0 is L2 closest (distance 0.1).
        def fake_extract(model, tok, texts, batch_size, device):
            return torch.zeros(len(texts), 4)

        monkeypatch.setattr(trainer, "_extract_embeddings", fake_extract)

        # Weights: only TARGET (item_2) gets max weight; others are 0.
        fake_weights = torch.zeros(num_items)
        fake_weights[TARGET] = 1.0

        results = trainer._rank_from_texts(
            emb_model=_FakeModel(),
            tokenizer=_MockTokenizer(),
            item_emb_device=item_emb,
            generated_texts=["oracle_title"],
            target_ids=[TARGET],
            cand_lists=[None],
            grounding_weights=fake_weights,
            device=torch.device("cpu"),
        )

        # item_2 (equal raw dist with item_1, but max weight) → rank #2 → recall@2=1.0
        assert results["recall@2"] == pytest.approx(1.0)

    def test_no_weights_produces_pure_l2_ranking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With grounding_weights=None the ranking is identical to pure L2."""
        _, item_lookup = _make_eval_frame_for_trainer(num_users=1)
        num_items = len(item_lookup)

        TARGET = 0

        cfg = BIGRecConfig(
            eval_topk=(1,),
            eval_metrics=("recall",),
            eval_protocol="full",
            eval_batch_size=1,
            max_input_length=32,
            max_new_tokens=8,
            num_beams=1,
            grounding_mode="none",
        )
        trainer = BIGRecTrainer(cfg)

        # Oracle = item_0 direction → pure L2 → item_0 ranks first.
        item_emb = torch.eye(num_items, 4)

        def fake_extract(model, tok, texts, batch_size, device):
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(texts))

        monkeypatch.setattr(trainer, "_extract_embeddings", fake_extract)

        results = trainer._rank_from_texts(
            emb_model=_FakeModel(),
            tokenizer=_MockTokenizer(),
            item_emb_device=item_emb,
            generated_texts=["oracle_title"],
            target_ids=[TARGET],
            cand_lists=[None],
            grounding_weights=None,   # explicit None → pure L2
            device=torch.device("cpu"),
        )

        assert results["recall@1"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 17. New config fields (train_on_inputs, embedding_use_base_model,
#     grounding_gamma_search, grounding_gamma_search_values)
# ══════════════════════════════════════════════════════════════════════════════


class TestNewConfigFields:
    """Verify defaults and overrides for the four new BIGRecConfig fields."""

    def test_default_train_on_inputs_is_true(self) -> None:
        assert BIGRecConfig().train_on_inputs is True

    def test_default_embedding_use_base_model_is_true(self) -> None:
        assert BIGRecConfig().embedding_use_base_model is True

    def test_default_grounding_gamma_search_is_false(self) -> None:
        assert BIGRecConfig().grounding_gamma_search is False

    def test_default_grounding_gamma_search_values_is_empty(self) -> None:
        assert BIGRecConfig().grounding_gamma_search_values == ()

    def test_train_on_inputs_can_be_overridden(self) -> None:
        cfg = BIGRecConfig(train_on_inputs=False)
        assert cfg.train_on_inputs is False

    def test_embedding_use_base_model_can_be_overridden(self) -> None:
        cfg = BIGRecConfig(embedding_use_base_model=False)
        assert cfg.embedding_use_base_model is False

    def test_grounding_gamma_search_can_be_enabled(self) -> None:
        cfg = BIGRecConfig(grounding_gamma_search=True)
        assert cfg.grounding_gamma_search is True

    def test_grounding_gamma_search_values_custom(self) -> None:
        cfg = BIGRecConfig(grounding_gamma_search_values=(0.5, 1.0, 2.0))
        assert cfg.grounding_gamma_search_values == (0.5, 1.0, 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# 18. _default_gamma_search_values
# ══════════════════════════════════════════════════════════════════════════════


class TestDefaultGammaSearchValues:
    """Unit tests for the static helper that generates the official gamma grid."""

    @staticmethod
    def _values() -> tuple[float, ...]:
        return BIGRecTrainer._default_gamma_search_values()

    def test_total_count_is_199(self) -> None:
        assert len(self._values()) == 199

    def test_starts_with_zero(self) -> None:
        assert self._values()[0] == pytest.approx(0.0)

    def test_fine_grained_section_ends_at_099(self) -> None:
        vals = self._values()
        assert vals[99] == pytest.approx(0.99, abs=1e-6)

    def test_coarse_section_starts_at_1(self) -> None:
        vals = self._values()
        assert vals[100] == pytest.approx(1.0)

    def test_ends_at_99(self) -> None:
        vals = self._values()
        assert vals[-1] == pytest.approx(99.0)

    def test_all_values_are_non_negative(self) -> None:
        assert all(v >= 0.0 for v in self._values())


# ══════════════════════════════════════════════════════════════════════════════
# 19. _run_gamma_search and _evaluate_from_dist_per_k_gammas
# ══════════════════════════════════════════════════════════════════════════════


class TestGammaSearch:
    """Tests for the gamma grid-search helpers with synthetic distances."""

    @staticmethod
    def _make_trainer(topk: tuple[int, ...] = (1, 3)) -> BIGRecTrainer:
        return BIGRecTrainer(
            BIGRecConfig(
                eval_topk=topk,
                eval_metrics=("recall",),
                eval_protocol="full",
                grounding_mode="popularity",
                grounding_gamma=1.0,
            )
        )

    def test_run_gamma_search_returns_keys_per_metric_k(self) -> None:
        """_run_gamma_search must return a dict with keys 'recall@1' and 'recall@3'."""
        trainer = self._make_trainer()
        device = torch.device("cpu")
        num_items = 5
        n = 4

        dist = torch.rand(n, num_items)
        weights = torch.rand(num_items)
        target_ids = list(range(n))
        cand_lists: list[list[int] | None] = [None] * n

        gamma_values: tuple[float, ...] = (0.0, 1.0, 10.0)
        best_gammas = trainer._run_gamma_search(
            dist, weights, target_ids, cand_lists, device, gamma_values
        )
        assert "recall@1" in best_gammas
        assert "recall@3" in best_gammas

    def test_run_gamma_search_best_gamma_in_candidate_set(self) -> None:
        """Every returned gamma must come from the provided gamma_values."""
        trainer = self._make_trainer()
        device = torch.device("cpu")
        num_items = 6
        n = 3

        dist = torch.rand(n, num_items)
        weights = torch.rand(num_items)
        target_ids = [0, 1, 2]
        cand_lists: list[list[int] | None] = [None] * n

        gamma_values: tuple[float, ...] = (0.1, 0.5, 2.0)
        best_gammas = trainer._run_gamma_search(
            dist, weights, target_ids, cand_lists, device, gamma_values
        )
        for key, gamma in best_gammas.items():
            assert gamma in gamma_values, (
                f"Best gamma for {key} ({gamma}) is not in candidate set {gamma_values}"
            )

    def test_evaluate_from_dist_per_k_gammas_returns_all_keys(self) -> None:
        """_evaluate_from_dist_per_k_gammas must produce 'recall@K' for every K."""
        trainer = self._make_trainer(topk=(1, 3))
        device = torch.device("cpu")
        num_items = 5
        n = 4

        dist = torch.rand(n, num_items)
        weights = torch.rand(num_items)
        target_ids = list(range(n))
        cand_lists: list[list[int] | None] = [None] * n

        best_gammas = {"recall@1": 0.5, "recall@3": 2.0}
        results = trainer._evaluate_from_dist_per_k_gammas(
            dist, weights, target_ids, cand_lists, best_gammas, device
        )
        assert "recall@1" in results
        assert "recall@3" in results

    def test_evaluate_from_dist_per_k_gammas_perfect_hit(self) -> None:
        """When the oracle embedding exactly matches item_0 and target=0, recall@1=1."""
        trainer = self._make_trainer(topk=(1,))
        device = torch.device("cpu")

        # distance[0, 0] = 0.0 (perfect match), others > 0
        dist = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # [1, 4]
        weights: torch.Tensor | None = None
        target_ids = [0]
        cand_lists: list[list[int] | None] = [None]

        best_gammas = {"recall@1": 0.0}
        results = trainer._evaluate_from_dist_per_k_gammas(
            dist, weights, target_ids, cand_lists, best_gammas, device
        )
        assert results["recall@1"] == pytest.approx(1.0)

    def test_gamma_search_selects_optimal_gamma(self) -> None:
        """The search must find the gamma that maximises recall on the validation data.

        Setup: 1 user, 4 items, item_2 is the TARGET (topk=2).
          dist    = [0.3, 0.5, 0.7, 2.0]
          D̂       = [0.0, ~0.12, ~0.24, 1.0]   (range = 1.7)
          weights = [0.0, 0.0,  1.0,  0.0]     (item_2 = TARGET has weight=1.0)

          gamma=0:   D̃ = D̂ → ranking: item_0, item_1, item_2, item_3 → recall@2 = 0
          gamma=100: D̃_2 = 0.24×2^(−100) ≈ 0 → leaps above item_1 → recall@2 = 1.0

        Key: item_0 always holds rank-1 (D̂=0 → D̃=0 always).  Eq. 3 promotes
        item_2 to rank-2, pushing item_1 down.  The search should select gamma=100.
        """
        trainer = self._make_trainer(topk=(2,))
        device = torch.device("cpu")

        dist = torch.tensor([[0.3, 0.5, 0.7, 2.0]])  # [1, 4]; item_0 is L2 closest
        weights = torch.tensor([0.0, 0.0, 1.0, 0.0])  # item_2 = TARGET has max weight
        target_ids = [2]
        cand_lists: list[list[int] | None] = [None]

        # Search only gamma=0 and gamma=100 to keep the test fast.
        best_gammas = trainer._run_gamma_search(
            dist, weights, target_ids, cand_lists, device, (0.0, 100.0)
        )
        # gamma=100 gives recall@2=1.0; gamma=0 gives recall@2=0 → best is 100.
        assert best_gammas["recall@2"] == pytest.approx(100.0)


# ══════════════════════════════════════════════════════════════════════════════
# 20. New training hyper-parameter defaults (matching official BIGRec train.py)
# ══════════════════════════════════════════════════════════════════════════════


class TestOfficialDefaultHyperparams:
    """Verify that BIGRecConfig defaults match the official BIGRec train.py."""

    def test_learning_rate_default_is_3e4(self) -> None:
        assert BIGRecConfig().learning_rate == pytest.approx(3e-4)

    def test_warmup_steps_default_is_20(self) -> None:
        assert BIGRecConfig().warmup_steps == 20

    def test_warmup_ratio_default_is_zero(self) -> None:
        """warmup_ratio should be 0.0; warmup_steps is the primary control."""
        assert BIGRecConfig().warmup_ratio == pytest.approx(0.0)

    def test_fp16_default_is_true(self) -> None:
        assert BIGRecConfig().fp16 is True

    def test_bf16_default_is_false(self) -> None:
        assert BIGRecConfig().bf16 is False

    def test_optim_default_is_adamw_torch(self) -> None:
        assert BIGRecConfig().optim == "adamw_torch"

    def test_save_total_limit_default_is_1(self) -> None:
        assert BIGRecConfig().save_total_limit == 1

    def test_load_best_model_at_end_default_is_true(self) -> None:
        assert BIGRecConfig().load_best_model_at_end is True

    def test_save_strategy_default_is_epoch(self) -> None:
        """Official BIGRec uses save_strategy='epoch' (matches evaluation_strategy)."""
        assert BIGRecConfig().save_strategy == "epoch"

    def test_early_stopping_patience_default_is_5(self) -> None:
        """Official BIGRec EarlyStoppingCallback uses patience=5."""
        assert BIGRecConfig().early_stopping_patience == 5

    def test_warmup_steps_overridable(self) -> None:
        cfg = BIGRecConfig(warmup_steps=100)
        assert cfg.warmup_steps == 100

    def test_warmup_steps_can_be_none(self) -> None:
        """Setting warmup_steps=None falls back to warmup_ratio."""
        cfg = BIGRecConfig(warmup_steps=None, warmup_ratio=0.05)
        assert cfg.warmup_steps is None
        assert cfg.warmup_ratio == pytest.approx(0.05)

    def test_fp16_overridable(self) -> None:
        cfg = BIGRecConfig(fp16=False)
        assert cfg.fp16 is False

    def test_early_stopping_patience_overridable(self) -> None:
        cfg = BIGRecConfig(early_stopping_patience=3)
        assert cfg.early_stopping_patience == 3

    def test_max_steps_default_is_500(self) -> None:
        """Default max_steps=500 caps training on large datasets."""
        assert BIGRecConfig().max_steps == 500

    def test_max_steps_disabled_with_minus_one(self) -> None:
        cfg = BIGRecConfig(max_steps=-1)
        assert cfg.max_steps == -1

    def test_max_steps_overridable(self) -> None:
        cfg = BIGRecConfig(max_steps=1000)
        assert cfg.max_steps == 1000

    def test_fit_passes_max_steps_to_training_arguments(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When max_steps < epoch_steps, effective_max_steps == max_steps.

        StubDataset has 4 training rows; effective_batch=32; num_epochs=3
        → epoch_steps = ceil(4/32)*3 = 3.  max_steps=1 < 3, so
        effective_max_steps=1 is forwarded to HF TrainingArguments.
        """
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(
            history_max_length=2, max_input_length=32, max_new_tokens=8,
            max_steps=1,  # 1 < epoch_steps(3) → effective_max_steps=1
        )
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        captured_args: dict[str, Any] = {}

        def _capture(**kwargs: Any) -> MagicMock:
            captured_args.update(kwargs)
            return MagicMock()

        with patch("recbole3.model.bigrec.trainer.HFTrainer", side_effect=_capture):
            trainer.fit(data, output_dir=str(tmp_path))

        training_args = captured_args.get("args")
        assert training_args is not None
        assert training_args.max_steps == 1

    def test_fit_disables_max_steps_when_epochs_finish_first(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When max_steps > epoch_steps, effective_max_steps is -1 (epochs control).

        StubDataset has 4 training rows; epoch_steps = ceil(4/32)*3 = 3.
        max_steps=500 > 3 → effective_max_steps=-1 so that HF Trainer stops
        after num_train_epochs instead of running 500÷3≈166 epochs.
        """
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(
            history_max_length=2, max_input_length=32, max_new_tokens=8,
            max_steps=500,  # 500 > epoch_steps(3) → effective_max_steps=-1
        )
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        captured_args: dict[str, Any] = {}

        def _capture(**kwargs: Any) -> MagicMock:
            captured_args.update(kwargs)
            return MagicMock()

        with patch("recbole3.model.bigrec.trainer.HFTrainer", side_effect=_capture):
            trainer.fit(data, output_dir=str(tmp_path))

        training_args = captured_args.get("args")
        assert training_args is not None
        assert training_args.max_steps == -1


# ══════════════════════════════════════════════════════════════════════════════
# 21. fit() uses EarlyStoppingCallback with correct patience, and passes
#     eval_dataset (val SFT) + evaluation_strategy="epoch" to HFTrainer.
# ══════════════════════════════════════════════════════════════════════════════


class TestFitEarlyStoppingAlignment:
    """Verify that fit() passes the correct arguments to HFTrainer for the
    official BIGRec training setup (LM val-loss early stopping)."""

    def test_fit_passes_eval_dataset_to_hf_trainer(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HFTrainer must receive a non-None eval_dataset (the val SFT dataset)."""
        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(history_max_length=2, max_input_length=32, max_new_tokens=8)
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        captured_kwargs: dict[str, Any] = {}

        def _capture_trainer(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("recbole3.model.bigrec.trainer.HFTrainer", side_effect=_capture_trainer):
            trainer.fit(data, output_dir=str(tmp_path))

        assert captured_kwargs.get("eval_dataset") is not None, (
            "fit() must pass eval_dataset (val SFT dataset) to HFTrainer"
        )

    def test_early_stopping_patience_from_config(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EarlyStoppingCallback patience must come from config.early_stopping_patience."""
        from transformers import EarlyStoppingCallback

        data, _ = _prepare_bigrec_data()
        cfg = BIGRecConfig(
            history_max_length=2,
            max_input_length=32,
            max_new_tokens=8,
            early_stopping_patience=7,
        )
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        captured_kwargs: dict[str, Any] = {}

        def _capture_trainer(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("recbole3.model.bigrec.trainer.HFTrainer", side_effect=_capture_trainer):
            trainer.fit(data, output_dir=str(tmp_path))

        callbacks = captured_kwargs.get("callbacks", [])
        early_cb = next(
            (cb for cb in callbacks if isinstance(cb, EarlyStoppingCallback)), None
        )
        assert early_cb is not None, "EarlyStoppingCallback must be in HFTrainer callbacks"
        # HF EarlyStoppingCallback stores patience as early_stopping_patience attribute.
        assert early_cb.early_stopping_patience == 7


class TestFitValidationCapping:
    """Verify that fit() caps the validation set proportionally when max_steps > 0."""

    def _capture_eval_dataset(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
        cfg: BIGRecConfig,
    ) -> Any:
        data, _ = _prepare_bigrec_data()
        trainer = BIGRecTrainer(cfg)
        _stub_fit_externals(trainer, monkeypatch)

        captured: dict[str, Any] = {}

        def _capture(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch("recbole3.model.bigrec.trainer.HFTrainer", side_effect=_capture):
            trainer.fit(data, output_dir=str(tmp_path))

        return captured.get("eval_dataset")

    def test_eval_dataset_capped_when_max_steps_positive(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When max_steps > 0, eval_dataset must be capped to max_steps × eval_batch_size."""
        # StubDataset: 2 users × 4 items → 2 valid samples (1 per user, leave-one-out).
        # max_steps=1, eval_batch_size=2 → cap = 1 × 2 = 2 (≤ 2, so no actual truncation
        # in stub data, but the cap path is exercised).
        cfg = BIGRecConfig(
            history_max_length=2,
            max_input_length=32,
            max_new_tokens=8,
            max_steps=1,
            eval_batch_size=1,  # cap = 1 × 1 = 1 → truncates the 2-row valid split
        )
        eval_ds = self._capture_eval_dataset(tmp_path, monkeypatch, cfg)
        assert eval_ds is not None
        # Cap = max_steps(1) × eval_batch_size(1) = 1 → only 1 validation sample.
        assert len(eval_ds) == 1

    def test_eval_dataset_not_capped_when_max_steps_minus_one(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When max_steps=-1, the full validation set is used (no cap)."""
        data, _ = _prepare_bigrec_data()
        full_valid_len = len(data.get_eval_dataset("valid").frame)

        cfg = BIGRecConfig(
            history_max_length=2,
            max_input_length=32,
            max_new_tokens=8,
            max_steps=-1,
        )
        eval_ds = self._capture_eval_dataset(tmp_path, monkeypatch, cfg)
        assert eval_ds is not None
        assert len(eval_ds) == full_valid_len
