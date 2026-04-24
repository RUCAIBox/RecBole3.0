from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.run import compose_config, run_experiment
from recbole3.model import LLMRankConfig, LLMRankModel, LLMRankModelDataset, get_model_spec
from recbole3.model.llmrank.candidates import BM25CandidateGenerator, HSTUCandidateGenerator, RandomCandidateGenerator
from recbole3.model.llmrank.pipeline import LLMRankPipeline
from recbole3.model.llmrank.trainer import LLMRankTrainer, LLMRankTrainerConfig
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


def test_llmrank_local_hf_backend_uses_batched_dispatch() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            backend="local_hf",
            parsing_strategy="title",
            local_model_path="/mnt/data/model/Qwen2.5-1.5B-Instruct",
            local_batch_size=2,
        )
    )
    collator = model.build_eval_collator(model_data)
    records = list(model_data.get_eval_dataset("valid"))[:2]
    model_inputs = collator(records)
    candidate_item_ids = torch.tensor([[2, 4, 6], [1, 3, 5]], dtype=torch.long)
    recorded_prompts: list[str] = []

    def _fake_request_local_hf_responses(prompts: list[str]) -> list[str]:
        recorded_prompts.extend(prompts)
        return [
            "1. Harbor Night\n2. Forest Signal\n3. Charlie Harbor",
            "1. Golden River\n2. Delta Echo\n3. Bravo Tales",
        ]

    model._request_local_hf_responses = _fake_request_local_hf_responses  # type: ignore[method-assign]

    pred_item_ids = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert len(recorded_prompts) == 2
    assert pred_item_ids.tolist() == [[6, 4, 2], [5, 3, 1]]


def test_llmrank_openai_backend_reuses_response_cache(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            backend="openai",
            parsing_strategy="title",
            api_response_cache_path=str(tmp_path / "api_cache.jsonl"),
        )
    )
    collator = model.build_eval_collator(model_data)
    records = list(model_data.get_eval_dataset("valid"))[:1]
    model_inputs = collator(records)
    candidate_item_ids = torch.tensor([[2, 4, 6]], dtype=torch.long)
    call_count = 0

    def _fake_request_openai_response(prompt: str, *, round_index: int) -> str:
        del prompt
        del round_index
        nonlocal call_count
        call_count += 1
        return "1. Harbor Night\n2. Forest Signal\n3. Charlie Harbor"

    model._request_openai_response_uncached = _fake_request_openai_response  # type: ignore[method-assign]

    first_pred = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)
    second_pred = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert call_count == 1
    assert first_pred.tolist() == [[6, 4, 2]]
    assert second_pred.tolist() == [[6, 4, 2]]


def test_llmrank_prompt_and_overlap_caches_reduce_repeat_work() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            backend="heuristic_overlap",
            candidate_shuffle=False,
        )
    )
    model.build_eval_collator(model_data)
    build_prompt_call_count = 0
    overlap_call_count = 0
    original_build_prompt_uncached = model._build_prompt_uncached
    original_rank_with_overlap = model._rank_with_overlap

    def _counted_build_prompt(history_texts, candidate_item_ids):
        nonlocal build_prompt_call_count
        build_prompt_call_count += 1
        return original_build_prompt_uncached(history_texts, candidate_item_ids)

    def _counted_rank_with_overlap(history_texts, candidate_item_ids):
        nonlocal overlap_call_count
        overlap_call_count += 1
        return original_rank_with_overlap(history_texts, candidate_item_ids)

    model._build_prompt_uncached = _counted_build_prompt  # type: ignore[method-assign]
    model._rank_with_overlap = _counted_rank_with_overlap  # type: ignore[method-assign]

    history_texts = ["Alpha Quest", "Bravo Tales"]
    candidate_item_ids = [2, 4, 6]

    prompt_one = model.build_prompt(history_texts, candidate_item_ids)
    prompt_two = model.build_prompt(history_texts, candidate_item_ids)
    ranked_one = model.rank_candidates(history_texts, candidate_item_ids)
    ranked_two = model.rank_candidates(history_texts, candidate_item_ids)

    assert prompt_one == prompt_two
    assert ranked_one == ranked_two
    assert build_prompt_call_count == 1
    assert overlap_call_count == 1


def test_llmrank_registry_and_inference_only_run() -> None:
    ensure_stub_tables()
    model_spec = get_model_spec("llmrank")
    assert model_spec.config_cls is LLMRankConfig
    assert model_spec.trainer_config_cls is LLMRankTrainerConfig
    assert model_spec.pipeline_cls is LLMRankPipeline
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


def test_random_candidate_generator_filters_history_and_prepends_target() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    generator = RandomCandidateGenerator(
        prepared,
        model_config=LLMRankConfig(candidate_source="random", candidate_topk=3, candidate_cache_dir="test_outputs/candidate_cache"),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )

    valid_frame = generator.build_split_frame("valid")

    assert all(len(candidate_item_ids) == 3 for candidate_item_ids in valid_frame["candidate_item_ids"].tolist())
    for record in valid_frame.to_dict("records"):
        candidate_item_ids = tuple(record["candidate_item_ids"])
        assert candidate_item_ids[0] == int(record["item_id"])
        assert set(candidate_item_ids[1:]).isdisjoint(set(record["seen_item_ids"]))


def test_random_candidate_generator_reuses_cache(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    generator = RandomCandidateGenerator(
        prepared,
        model_config=LLMRankConfig(candidate_source="random", candidate_topk=3, candidate_cache_dir=str(tmp_path / "candidate_cache")),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )

    valid_frame_first = generator.build_split_frame("valid")
    cache_files = list((tmp_path / "candidate_cache").rglob("valid.jsonl"))
    assert cache_files

    generator_reloaded = RandomCandidateGenerator(
        prepared,
        model_config=LLMRankConfig(
            candidate_source="random",
            candidate_topk=3,
            candidate_seed=999,
            candidate_cache_dir=str(tmp_path / "candidate_cache"),
        ),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )
    valid_frame_second = generator_reloaded.build_split_frame("valid")

    assert valid_frame_first["candidate_item_ids"].tolist() == valid_frame_second["candidate_item_ids"].tolist()


def test_bm25_candidate_generator_uses_history_text_and_filters_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())

    monkeypatch.setattr(
        BM25CandidateGenerator,
        "_load_segment_text",
        staticmethod(lambda texts: [str(text).lower().split() for text in texts]),
    )

    generator = BM25CandidateGenerator(
        prepared,
        model_config=LLMRankConfig(
            candidate_source="bm25",
            candidate_topk=3,
            candidate_cache_dir=str(tmp_path / "candidate_cache"),
            bm25_item_text_field="metadata_text",
        ),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )

    valid_frame = generator.build_split_frame("valid")

    for record in valid_frame.to_dict("records"):
        candidate_item_ids = tuple(record["candidate_item_ids"])
        assert len(candidate_item_ids) == 3
        assert candidate_item_ids[0] == int(record["item_id"])
        assert set(candidate_item_ids[1:]).isdisjoint(set(record["seen_item_ids"]))


def test_hstu_candidate_generator_can_run_with_mock_backbone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_sampled_eval_config())
    generator = HSTUCandidateGenerator(
        prepared,
        model_config=LLMRankConfig(candidate_source="hstu", candidate_topk=3, candidate_cache_dir=str(tmp_path / "candidate_cache")),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )

    class _FakeHSTUModel:
        def build_eval_collator(self, prepared_data):
            def _collator(records):
                return {"batch_size": len(records)}

            return _collator

        def predict(self, model_inputs, *, k, candidate_item_ids=None, exclude_item_ids=None, exclude_mask=None):
            del model_inputs
            del candidate_item_ids
            assert exclude_item_ids is not None
            assert exclude_mask is not None
            batch_size = int(exclude_item_ids.shape[0])
            rows = []
            for row_index in range(batch_size):
                history = set(exclude_item_ids[row_index][exclude_mask[row_index]].tolist())
                ranked = [item_id for item_id in range(prepared.get_num_items()) if item_id not in history]
                rows.append(ranked[:k])
            return torch.tensor(rows, dtype=torch.long)

    monkeypatch.setattr(generator, "_load_trained_hstu_model", lambda: _FakeHSTUModel())

    valid_frame = generator.build_split_frame("valid")

    for record in valid_frame.to_dict("records"):
        candidate_item_ids = tuple(record["candidate_item_ids"])
        assert len(candidate_item_ids) == 3
        assert candidate_item_ids[0] == int(record["item_id"])
        assert set(candidate_item_ids[1:]).isdisjoint(set(record["seen_item_ids"]))


def test_llmrank_pipeline_end_to_end_with_random_candidates(tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: llmrank_test",
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
    (config_dir / "model" / "llmrank_test.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: llmrank",
                "  candidate_source: random",
                "  candidate_topk: 3",
                f"  candidate_cache_dir: {(tmp_path / 'candidate_cache').as_posix()}",
                "  item_text_field: metadata_text",
                "  backend: heuristic_overlap",
                "  domain: item",
                "  candidate_shuffle: true",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  max_epochs: 0",
                "  eval:",
                "    protocol: sampled",
                "    neg_sampling_num: 2",
                "    candidate_seed: 7",
                "    metrics:",
                "      - name: recall",
                "        ks: [3]",
                "      - name: ndcg",
                "        ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))

    valid_dataset = result["prepared_data"].get_eval_dataset("valid")
    test_dataset = result["prepared_data"].get_eval_dataset("test")
    assert "candidate_item_ids" in valid_dataset.frame.columns
    assert "candidate_item_ids" in test_dataset.frame.columns
    assert all(len(values) == 3 for values in valid_dataset.frame["candidate_item_ids"].tolist())
    assert all(len(values) == 3 for values in test_dataset.frame["candidate_item_ids"].tolist())
    assert "recall@3" in result["test"]["metrics"]
    assert "ndcg@3" in result["test"]["metrics"]
