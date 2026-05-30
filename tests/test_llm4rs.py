from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from random import Random
from typing import Any

import pytest
import torch

from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import LLM4RSConfig, LLM4RSModel, LLM4RSModelDataset, get_model_spec
from recbole3.model.llm4rs.model import LLM4RSOutcome, LLM4RS_RECORD_INDEX
from recbole3.model.llm4rs.pipeline import LLM4RSPipeline
from recbole3.model.llm4rs.trainer import LLM4RSTrainer, LLM4RSTrainerConfig
from recbole3.run import compose_config, run_experiment
from tests.test_helpers import StubDataset, StubDatasetConfig, ensure_stub_tables


@pytest.fixture
def local_tmp_path() -> Path:
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="llm4rs-", dir=root))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _prepared_data(config: LLM4RSConfig) -> LLM4RSModelDataset:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=EvalConfig(protocol="full"))
    valid_frame = prepared.get_eval_dataset("valid").frame.copy()
    test_frame = prepared.get_eval_dataset("test").frame.copy()
    valid_frame[CANDIDATE_ITEM_IDS] = [(2, 3, 4), (6, 5, 7)]
    test_frame[CANDIDATE_ITEM_IDS] = [(3, 2, 4), (7, 5, 6)]
    prepared._valid_dataset = FrameDataset(valid_frame)
    prepared._test_dataset = FrameDataset(test_frame)
    return LLM4RSModelDataset.from_task_dataset(prepared, model_config=config)


def test_llm4rs_registration() -> None:
    spec = get_model_spec("llm4rs")

    assert spec.config_cls is LLM4RSConfig
    assert spec.model_cls is LLM4RSModel
    assert spec.model_data_cls is LLM4RSModelDataset
    assert spec.trainer_cls is LLM4RSTrainer


def test_llm4rs_defaults_use_native_dataset_metadata_mode() -> None:
    cfg = compose_config(overrides=["dataset=amazon2014_retrieval", "model=llm4rs"])

    assert LLM4RSConfig().domain == "agnostic"
    assert cfg.model.domain == "agnostic"
    assert cfg.dataset.metadata_mode == "sentence"


def test_llm4rs_random_fallback_samples_non_target_items_without_masking_history() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=EvalConfig(protocol="full"))
    frame = LLM4RSPipeline._random_candidate_frame(
        prepared,
        split="test",
        config=LLM4RSConfig(candidate_num=8, example_num=0, shuffle_candidates=False),
        selected_user_ids=(0, 1),
    )

    assert set(frame.iloc[0][CANDIDATE_ITEM_IDS]) == set(range(8))
    assert set(frame.iloc[0]["seen_item_ids"]).issubset(set(frame.iloc[0][CANDIDATE_ITEM_IDS]))


def test_llm4rs_list_prompt_retains_official_format_and_examples() -> None:
    config = LLM4RSConfig(ranking_policy="list", domain="Movie", example_num=1, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)
    model.configure_examples(prepared.get_eval_dataset("valid").frame.iloc[:1])

    prompt = model.build_prompt(["Alpha Quest", "Bravo Tales"], [2, 3, 4])

    assert prompt.startswith("You are a movie recommender system now.\n")
    assert "Input: Here is the watching history of a user: Alpha Quest, Bravo Tales." in prompt
    assert "please rank the following candidate movies: (A) Charlie Harbor (B) Delta Echo (C) Forest Signal" in prompt
    assert "Output: The answer index is" in prompt


def test_llm4rs_agnostic_prompt_is_suitable_for_native_product_datasets() -> None:
    config = LLM4RSConfig(ranking_policy="list", example_num=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)

    prompt = model.build_prompt(["Alpha Quest"], [2, 3, 4])

    assert prompt.startswith("You are a general-purpose recommender system now.\n")
    assert "\n\nInput:" not in prompt
    assert "interaction history" in prompt
    assert "candidate items" in prompt


def test_llm4rs_prompt_without_instruction_starts_at_query() -> None:
    config = LLM4RSConfig(ranking_policy="list", no_instruction=True, example_num=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)

    prompt = model.build_prompt(["Alpha Quest"], [2, 3, 4])

    assert prompt.startswith("Input:")


def test_llm4rs_eval_collator_retains_candidate_target_and_record_alignment() -> None:
    config = LLM4RSConfig(ranking_policy="pair", candidate_num=3, example_num=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    collator = model.build_eval_collator(prepared)
    records = prepared.get_eval_dataset("test").frame.iloc[:1].copy()
    records[LLM4RS_RECORD_INDEX] = [7]

    batch = collator(records)

    assert batch["candidate_item_ids"] == [(3, 2, 4)]
    assert batch["target_item_ids"] == [3]
    assert batch["record_indices"] == [7]


def test_llm4rs_point_policy_retains_target_tie_positions() -> None:
    config = LLM4RSConfig(ranking_policy="point", example_num=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)

    outcome = model.rank_candidate_batches([["Alpha Quest"]], [[2, 3, 4]])[0]

    assert outcome.scores == (3, 3, 3)
    assert outcome.target_ranks(2) == (0, 1, 2)


def test_llm4rs_point_policy_keeps_rows_with_partial_subrequest_failures() -> None:
    model = LLM4RSModel(LLM4RSConfig(ranking_policy="point", example_num=0, backend="identity"))

    partial = model._parse_row_outcome((2, 3, 4), [((0,), "5"), ((1,), "invalid"), ((2,), "1")])
    failed = model._parse_row_outcome((2, 3, 4), [((0,), None), ((1,), "invalid"), ((2,), None)])

    assert partial.scores == (5, 0, 1)
    assert partial.failed_subrequests == 1
    assert partial.error is None
    assert partial.target_ranks(2) == (0,)
    assert failed.failed_subrequests == 3
    assert failed.target_ranks(2) == ()


def test_llm4rs_point_examples_follow_official_scores_and_shuffled_negative_choice() -> None:
    config = LLM4RSConfig(ranking_policy="point", candidate_num=5, example_num=3, candidate_seed=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)
    candidate_item_ids = (2, 3, 4, 5, 6)
    target_position = 0
    defined_point_scores = (1, 3, 5)

    examples: list[str] = []
    for example_index, rating in enumerate(defined_point_scores):
        negative_positions = [position for position in range(len(candidate_item_ids)) if position != target_position]
        Random(int(config.candidate_seed) + example_index).shuffle(negative_positions)
        expected_position = negative_positions[0] if rating <= 3 else target_position
        example = model._build_example(
            history_texts=["Alpha Quest"],
            candidate_item_ids=candidate_item_ids,
            target_item_id=candidate_item_ids[target_position],
            example_index=example_index,
            example_count=len(defined_point_scores),
        )
        examples.append(example)

        assert model._item_text(candidate_item_ids[expected_position]) in example
        assert f"Output: {rating}." in example

    assert "Golden River" in examples[0]
    assert "Delta Echo" not in examples[0]


def test_llm4rs_pair_policy_votes_over_all_candidate_pairs() -> None:
    config = LLM4RSConfig(ranking_policy="pair", example_num=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)

    outcome = model.rank_candidate_batches([["Alpha Quest"]], [[2, 3, 4]])[0]

    assert outcome.scores == (2, 1, 0)
    assert outcome.target_ranks(3) == (1,)
    assert len(outcome.responses) == 3


def test_llm4rs_pair_policy_keeps_rows_with_partial_subrequest_failures() -> None:
    model = LLM4RSModel(LLM4RSConfig(ranking_policy="pair", example_num=0, backend="identity"))

    partial = model._parse_row_outcome((2, 3, 4), [((0, 1), "A"), ((0, 2), "invalid"), ((1, 2), "B")])
    failed = model._parse_row_outcome((2, 3, 4), [((0, 1), None), ((0, 2), "invalid"), ((1, 2), None)])

    assert partial.scores == (1, 0, 1)
    assert partial.failed_subrequests == 1
    assert partial.error is None
    assert partial.target_ranks(2) == (0, 1)
    assert failed.failed_subrequests == 3
    assert failed.target_ranks(2) == ()


def test_llm4rs_parsers_accept_short_answer_sentences() -> None:
    candidates = (2, 3, 4)

    assert LLM4RSModel._parse_list_response("The answer index is A B C.", candidates) == candidates
    assert LLM4RSModel._parse_pair_response("The answer index is B.") == 1
    assert LLM4RSModel._parse_pair_response("I choose B.") == 1


def test_llm4rs_pair_evaluation_randomizes_target_choice_position() -> None:
    config = LLM4RSConfig(ranking_policy="pair", example_num=0, backend="identity", candidate_seed=2023)
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)

    outcome = model.rank_candidate_batches([["Alpha Quest"]], [[2, 3, 4]], target_item_ids=[2])[0]

    assert outcome.scores == (1, 1, 1)
    assert outcome.target_ranks(2) == (0, 1, 2)


def test_llm4rs_pair_prompts_are_stable_across_batch_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLM4RSConfig(ranking_policy="pair", example_num=0, backend="identity", candidate_seed=2023)
    prepared = _prepared_data(config)
    histories = [["Alpha Quest"], ["Bravo Tales"], ["Alpha Quest"]]
    candidates = [[2, 3, 4], [2, 3, 4], [2, 3, 4]]
    targets = [2, 2, 2]
    record_indices = [0, 1, 2]

    full_model = LLM4RSModel(config)
    full_model.build_eval_collator(prepared)
    full_prompts: list[str] = []

    def capture_full(prompts: list[str], defaults: list[str]) -> list[str]:
        full_prompts.extend(prompts)
        return defaults

    monkeypatch.setattr(full_model, "_generate_responses", capture_full)
    full_outcomes = full_model.rank_candidate_batches(
        histories,
        candidates,
        target_item_ids=targets,
        record_indices=record_indices,
    )

    split_model = LLM4RSModel(config)
    split_model.build_eval_collator(prepared)
    split_prompts: list[str] = []

    def capture_split(prompts: list[str], defaults: list[str]) -> list[str]:
        split_prompts.extend(prompts)
        return defaults

    monkeypatch.setattr(split_model, "_generate_responses", capture_split)
    split_outcomes = []
    for row_index in record_indices:
        split_outcomes.extend(
            split_model.rank_candidate_batches(
                histories[row_index : row_index + 1],
                candidates[row_index : row_index + 1],
                target_item_ids=targets[row_index : row_index + 1],
                record_indices=[row_index],
            )
        )

    assert split_prompts == full_prompts
    assert [outcome.scores for outcome in split_outcomes] == [outcome.scores for outcome in full_outcomes]


def test_llm4rs_pair_predict_requires_and_passes_collated_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLM4RSConfig(ranking_policy="pair", candidate_num=3, example_num=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    collator = model.build_eval_collator(prepared)
    candidate_item_ids = torch.tensor([[3, 2, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="requires collated target_item_ids"):
        model.predict({"history_texts": [["Alpha Quest"]]}, k=3, candidate_item_ids=candidate_item_ids)

    records = prepared.get_eval_dataset("test").frame.iloc[:1].copy()
    records[LLM4RS_RECORD_INDEX] = [11]
    model_inputs = collator(records)
    captured: dict[str, object] = {}

    def capture_rank(
        history_text_batches: list[list[str]],
        candidate_batches: list[list[int]],
        *,
        target_item_ids: list[int] | None = None,
        record_indices: list[int] | None = None,
    ) -> list[LLM4RSOutcome]:
        captured["targets"] = target_item_ids
        captured["indices"] = record_indices
        return [LLM4RSOutcome(tuple(candidate_batches[0]))]

    monkeypatch.setattr(model, "rank_candidate_batches", capture_rank)

    model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert captured == {"targets": [3], "indices": [11]}


def test_llm4rs_trainer_uses_collated_metadata_instead_of_batch_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLM4RSConfig(ranking_policy="pair", candidate_num=3, example_num=0, begin_index=0, backend="identity")
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    trainer = LLM4RSTrainer(
        LLM4RSTrainerConfig(
            batch_size=1,
            eval=EvalConfig(protocol="full", metrics=(MetricSpec(name="recall", ks=(1,)),)),
        )
    )
    calls: list[tuple[list[tuple[int, ...]], list[int], list[int]]] = []

    def reverse_batches(dataset: FrameDataset, collate_fn: Any, *, shuffle: bool) -> list[dict[str, object]]:
        assert shuffle is False
        return [
            collate_fn(dataset.frame.iloc[[1]]),
            collate_fn(dataset.frame.iloc[[0]]),
        ]

    def capture_rank(
        history_text_batches: list[list[str]],
        candidate_batches: list[tuple[int, ...]],
        *,
        target_item_ids: list[int] | None = None,
        record_indices: list[int] | None = None,
    ) -> list[LLM4RSOutcome]:
        assert target_item_ids is not None
        assert record_indices is not None
        calls.append((candidate_batches, target_item_ids, record_indices))
        return [
            LLM4RSOutcome(
                tuple(candidates),
                ranked_item_ids=(target, *tuple(item_id for item_id in candidates if item_id != target)),
            )
            for candidates, target in zip(candidate_batches, target_item_ids, strict=True)
        ]

    monkeypatch.setattr(trainer, "build_dataloader", reverse_batches)
    monkeypatch.setattr(model, "rank_candidate_batches", capture_rank)

    result = trainer.evaluate(model, prepared, split="test")

    assert calls == [([(7, 5, 6)], [7], [1]), ([(3, 2, 4)], [3], [0])]
    assert result["metrics"]["recall@1"] == pytest.approx(1.0)


def test_llm4rs_tie_aware_metrics_match_official_average() -> None:
    trainer = LLM4RSTrainer(
        LLM4RSTrainerConfig(
            eval=EvalConfig(
                protocol="full",
                metrics=(
                    MetricSpec(name="ndcg", ks=(3,)),
                    MetricSpec(name="recall", ks=(1,)),
                ),
            )
        )
    )

    metrics = trainer._compute_metrics([(0, 1, 2)])

    expected_ndcg = (1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)) / 3.0
    assert metrics["ndcg@3"] == pytest.approx(expected_ndcg)
    assert metrics["recall@1"] == pytest.approx(1.0 / 3.0)


def test_llm4rs_failed_responses_count_as_metric_misses() -> None:
    trainer = LLM4RSTrainer(
        LLM4RSTrainerConfig(
            eval=EvalConfig(protocol="full", metrics=(MetricSpec(name="recall", ks=(1,)),))
        )
    )

    metrics = trainer._compute_metrics([(0,), ()])

    assert metrics["recall@1"] == pytest.approx(0.5)


def test_llm4rs_openai_backend_calls_local_compatible_endpoint(local_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLM4RSConfig(
        ranking_policy="list",
        example_num=0,
        backend="openai",
        async_dispatch=False,
        api_response_cache_path=str(local_tmp_path / "responses.jsonl"),
    )
    prepared = _prepared_data(config)
    model = LLM4RSModel(config)
    model.build_eval_collator(prepared)

    def _fake_request(prompt: str) -> str:
        del prompt
        return "A B C"

    monkeypatch.setattr(model, "_request_openai_response", _fake_request)

    outcome = model.rank_candidate_batches([["Alpha Quest"]], [[2, 3, 4]])[0]

    assert outcome.ordered_item_ids() == (2, 3, 4)


def test_llm4rs_pipeline_runs_end_to_end_with_official_candidate_injection(local_tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = local_tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: llm4rs_test",
                "  - _self_",
                "runtime:",
                "  device: cpu",
                f"  output_dir: {(local_tmp_path / 'outputs').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "stub_dataset.yaml").write_text(
        "\n".join(
            [
                "name: stub_dataset",
                f"processed_dir: {(local_tmp_path / 'processed').as_posix()}",
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
    (config_dir / "model" / "llm4rs_test.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "model:",
                "  name: llm4rs",
                "  ranking_policy: list",
                "  domain: Movie",
                "  history_max_length: 5",
                "  candidate_num: 3",
                "  candidate_source: random",
                "  backbone_topk: 4",
                "  selected_user_count: -1",
                "  example_num: 0",
                "  begin_index: 0",
                "  shuffle_candidates: false",
                f"  candidate_cache_dir: {(local_tmp_path / 'candidate_cache').as_posix()}",
                f"  candidate_file_dir: {(local_tmp_path / 'candidate_files').as_posix()}",
                "  use_candidate_file: false",
                "  backend: identity",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  max_epochs: 0",
                "  eval:",
                "    protocol: full",
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

    test_frame = result["prepared_data"].get_eval_dataset("test").frame
    assert all(len(candidates) == 3 for candidates in test_frame[CANDIDATE_ITEM_IDS].tolist())
    assert all(int(target) in candidates for target, candidates in zip(test_frame["item_id"], test_frame[CANDIDATE_ITEM_IDS]))
    assert result["test"]["evaluated_records"] == len(test_frame)
    assert "recall@3" in result["test"]["metrics"]
