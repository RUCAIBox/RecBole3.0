from __future__ import annotations

from pathlib import Path

import pytest
import torch

from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import LLMRankConfig, LLMRankModel, LLMRankModelDataset, get_model_spec
from recbole3.trainer import LLMRankTrainer, LLMRankTrainerConfig, get_trainer_spec
from tests.test_helpers import StubDataset, StubDatasetConfig, ensure_stub_tables


def _sampled_eval_config() -> EvalConfig:
    return EvalConfig(
        protocol="sampled",
        metrics=(MetricSpec(name="recall", ks=(3,)), MetricSpec(name="ndcg", ks=(3,))),
        neg_sampling_num=2,
        candidate_seed=7,
    )


class _FakeAccelerator:
    def prepare(self, *args):
        if len(args) == 1:
            return args[0]
        return args

    def unwrap_model(self, model):
        return model


def test_llmrank_prompt_uses_paper_style_recency_prompting() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig(domain="product"))
    model = LLMRankModel(LLMRankConfig(domain="product", prompt_strategy="recency_focused"))
    model.build_eval_collator(model_data)

    prompt = model.build_prompt(
        ["Alpha Quest", "Bravo Tales"],
        [2, 3, 4],
    )

    assert "I've purchased the following products in the past in order" in prompt
    assert "Note that my most recently purchased product is Bravo Tales" in prompt
    assert "candidate products" in prompt
    assert "Please think step by step." in prompt
    assert "You must rank the given candidate products only." in prompt
    assert "['0. Alpha Quest', '1. Bravo Tales']" in prompt


def test_llmrank_predict_parses_mock_title_rankings() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            backend="mock",
            parsing_strategy="title",
            mock_responses=("0. Harbor Night\n1. Forest Signal\n2. Charlie Harbor",),
        )
    )
    collator = model.build_eval_collator(model_data)
    records = list(model_data.get_eval_dataset("valid"))[:1]
    model_inputs = collator(records)
    candidate_item_ids = torch.tensor([[2, 4, 6]], dtype=torch.long)

    pred_item_ids = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert pred_item_ids.tolist() == [[6, 4, 2]]


def test_llmrank_predict_parses_fuzzy_movie_titles() -> None:
    model = LLMRankModel(LLMRankConfig(domain="movie", parsing_strategy="title"))
    model._item_text_lookup = ("The Matrix", "Toy Story", "A Bug's Life")

    parsed_item_ids = model.parse_response(
        "1. Matrix\n2. Toy Story (1995)\n3. Bugs Life",
        [0, 1, 2],
    )

    assert parsed_item_ids == [0, 1, 2]


def test_llmrank_openai_backend_uses_batched_dispatch() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            backend="openai",
            domain="item",
            parsing_strategy="title",
        )
    )
    collator = model.build_eval_collator(model_data)
    records = list(model_data.get_eval_dataset("valid"))[:2]
    model_inputs = collator(records)
    candidate_item_ids = torch.tensor([[2, 4, 6], [1, 3, 5]], dtype=torch.long)
    recorded_prompts: list[str] = []

    def _fake_request_openai_responses(prompts: list[str], round_indices: list[int]) -> list[str]:
        recorded_prompts.extend(prompts)
        assert round_indices == [0, 0]
        return [
            "1. Harbor Night\n2. Forest Signal\n3. Charlie Harbor",
            "1. Golden River\n2. Delta Echo\n3. Bravo Tales",
        ]

    model._request_openai_responses = _fake_request_openai_responses  # type: ignore[method-assign]

    pred_item_ids = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert len(recorded_prompts) == 2
    assert pred_item_ids.tolist() == [[6, 4, 2], [5, 3, 1]]


def test_llmrank_registry_and_inference_only_run() -> None:
    ensure_stub_tables()
    assert get_model_spec("llmrank").config_cls is LLMRankConfig
    assert get_trainer_spec("llmrank").config_cls.__name__ == "LLMRankTrainerConfig"
    output_dir = Path("test_outputs") / "llmrank_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            item_text_field="metadata_text",
            prompt_strategy="recency_focused",
            domain="item",
            backend="heuristic_overlap",
            parsing_strategy="title",
            bootstrap_rounds=1,
            candidate_shuffle=False,
        )
    )
    trainer = LLMRankTrainer(
        LLMRankTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=0,
            eval=_sampled_eval_config(),
        )
    )
    trainer.create_accelerator = lambda: _FakeAccelerator()  # type: ignore[method-assign]

    result = trainer.run(model, model_data, output_dir=output_dir)

    assert result["fit"]["train_history"] == []
    assert len(result["fit"]["valid_history"]) == 1
    assert result["fit"]["valid_history"][0]["protocol"] == "sampled"
    assert result["test"]["split"] == "test"
    assert "recall@3" in result["test"]["metrics"]
    assert "ndcg@3" in result["test"]["metrics"]


def test_llmrank_model_rejects_full_retrieval_predict() -> None:
    model = LLMRankModel(LLMRankConfig())
    with pytest.raises(NotImplementedError, match="sampled candidate reranking"):
        model.predict({"history_texts": [[]]}, k=3, candidate_item_ids=None)
