from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import pytest
import torch

from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import LLMRankConfig, LLMRankModel, LLMRankModelDataset
from recbole3.dataset.candidates import BM25CandidateGenerator, HSTUCandidateGenerator, RandomCandidateGenerator
from recbole3.run import compose_config, run_experiment
from tests.test_helpers import StubDataset, StubDatasetConfig, ensure_stub_tables


@pytest.fixture
def local_tmp_path() -> Path:
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="llmrank-", dir=root))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _full_eval_config() -> EvalConfig:
    return EvalConfig(
        protocol="full",
        metrics=(MetricSpec(name="recall", ks=(3,)), MetricSpec(name="ndcg", ks=(3,))),
        neg_sampling_num=0,
        candidate_seed=7,
    )


def _sampled_eval_config() -> EvalConfig:
    return EvalConfig(
        protocol="sampled",
        metrics=(MetricSpec(name="recall", ks=(3,)),),
        neg_sampling_num=3,
        candidate_seed=7,
    )


def test_llmrank_product_title_prompt_matches_official_style() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig(domain="product"))
    model = LLMRankModel(LLMRankConfig(domain="product", parsing_strategy="title"))
    model.build_eval_collator(model_data)

    prompt = model.build_prompt(["Alpha Quest", "Bravo Tales"], [2, 3, 4])

    assert "I've purchased the following products in the past in order:" in prompt
    assert "Now there are 3 candidate products that I can consider to purchase next:" in prompt
    assert "Please think step by step." in prompt
    assert "Please show me your ranking results with order numbers." in prompt
    assert "You MUST rank the given candidate products." in prompt
    assert "['0. Alpha Quest', '1. Bravo Tales']" in prompt
    assert "['0. Charlie Harbor', '1. Delta Echo', '2. Forest Signal']" in prompt


def test_llmrank_index_prompt_matches_official_comment_style() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig(domain="product"))
    model = LLMRankModel(LLMRankConfig(domain="product", parsing_strategy="index"))
    model.build_eval_collator(model_data)

    prompt = model.build_prompt(["Alpha Quest", "Bravo Tales"], [2, 3, 4])

    assert "Please only output the order numbers after ranking." in prompt
    assert "Split these order numbers with line break." in prompt
    assert "Please think step by step." not in prompt


def test_llmrank_prompt_supports_recency_focused_strategy() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig(domain="product"))
    model = LLMRankModel(LLMRankConfig(domain="product", prompt_strategy="recency_focused"))
    model.build_eval_collator(model_data)

    prompt = model.build_prompt(["Alpha Quest", "Bravo Tales"], [2, 3, 4])

    assert "Note that my most recently purchased product is Bravo Tales." in prompt


def test_llmrank_prompt_supports_in_context_learning_strategy() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig(domain="product"))
    model = LLMRankModel(LLMRankConfig(domain="product", prompt_strategy="in_context_learning"))
    model.build_eval_collator(model_data)

    prompt = model.build_prompt(["Alpha Quest", "Bravo Tales"], [2, 3, 4])

    assert "Then if I ask you to recommend a new product to me according to the given purchasing history" in prompt
    assert "you should recommend Bravo Tales" in prompt


def test_llmrank_parse_response_supports_title_strategy() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(LLMRankConfig(parsing_strategy="title"))
    model.build_eval_collator(model_data)

    parsed = model.parse_response(
        "0. Harbor Night\n1. Forest Signal\n2. Charlie Harbor",
        [2, 4, 6],
    )

    assert parsed == [6, 4, 2]


def test_llmrank_parse_response_supports_index_strategy() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(LLMRankConfig(parsing_strategy="index"))
    model.build_eval_collator(model_data)

    parsed = model.parse_response("2\n0\n1", [2, 4, 6])

    assert parsed == [6, 2, 4]


def test_llmrank_identity_backend_keeps_candidate_order() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(LLMRankConfig(backend="identity"))
    collator = model.build_eval_collator(model_data)
    records = list(model_data.get_eval_dataset("valid"))[:1]
    model_inputs = collator(records)
    candidate_item_ids = torch.tensor([[2, 4, 6]], dtype=torch.long)

    pred_item_ids = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert pred_item_ids.tolist() == [[2, 4, 6]]


def test_llmrank_openai_backend_batches_requests_like_official() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    model_data = LLMRankModelDataset.from_task_dataset(prepared, model_config=LLMRankConfig())
    model = LLMRankModel(
        LLMRankConfig(
            backend="openai",
            parsing_strategy="title",
            api_batch=2,
            async_dispatch=True,
        )
    )
    collator = model.build_eval_collator(model_data)
    records = list(model_data.get_eval_dataset("valid"))[:2]
    model_inputs = collator(records)
    candidate_item_ids = torch.tensor([[2, 4, 6], [1, 3, 5]], dtype=torch.long)
    requested: list[tuple[str, int]] = []

    def _fake_request(prompt: str, *, round_index: int) -> str:
        requested.append((prompt, round_index))
        if len(requested) == 1:
            return "1. Harbor Night\n2. Forest Signal\n3. Charlie Harbor"
        return "1. Golden River\n2. Delta Echo\n3. Bravo Tales"

    model._request_openai_response = _fake_request  # type: ignore[method-assign]

    pred_item_ids = model.predict(model_inputs, k=3, candidate_item_ids=candidate_item_ids)

    assert len(requested) == 2
    assert [round_index for _, round_index in requested] == [0, 0]
    assert pred_item_ids.tolist() == [[6, 4, 2], [5, 3, 1]]


def test_random_candidate_generator_masks_seen_items(local_tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    generator = RandomCandidateGenerator(
        prepared,
        model_config=LLMRankConfig(
            candidate_source="random",
            backbone_topk=4,
            recall_budget=3,
            candidate_cache_dir=str(local_tmp_path / "candidate_cache"),
        ),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )

    valid_frame = generator.build_split_frame("valid")

    for record in valid_frame.to_dict("records"):
        candidate_item_ids = tuple(record["candidate_item_ids"])
        assert len(candidate_item_ids) == 4
        assert set(candidate_item_ids).isdisjoint(set(record["seen_item_ids"]))


def test_bm25_candidate_generator_masks_seen_items(local_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    monkeypatch.setattr(
        BM25CandidateGenerator,
        "_load_segment_text",
        staticmethod(lambda texts: [str(text).lower().split() for text in texts]),
    )
    generator = BM25CandidateGenerator(
        prepared,
        model_config=LLMRankConfig(
            candidate_source="bm25",
            backbone_topk=4,
            recall_budget=3,
            candidate_cache_dir=str(local_tmp_path / "candidate_cache"),
            bm25_item_text_field="metadata_text",
        ),
        runtime_cfg=None,
        dataset_cfg={},
        trainer_cfg={},
    )

    valid_frame = generator.build_split_frame("valid")

    for record in valid_frame.to_dict("records"):
        candidate_item_ids = tuple(record["candidate_item_ids"])
        assert len(candidate_item_ids) == 4
        assert set(candidate_item_ids).isdisjoint(set(record["seen_item_ids"]))


# def test_hstu_candidate_generator_uses_model_backbone_for_raw_candidates(local_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
#     prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
#     generator = HSTUCandidateGenerator(
#         prepared,
#         model_config=LLMRankConfig(
#             candidate_source="hstu",
#             backbone_topk=4,
#             recall_budget=3,
#             candidate_cache_dir=str(local_tmp_path / "candidate_cache"),
#         ),
#         runtime_cfg=None,
#         dataset_cfg={},
#         trainer_cfg={},
#     )

#     class _FakeHSTUModel:
#         def build_eval_collator(self, prepared_data):
#             def _collator(records):
#                 return {"history_item_ids": torch.zeros((len(records), 1), dtype=torch.long)}

#             return _collator

#         def predict(self, model_inputs, *, k, candidate_item_ids=None, exclude_item_ids=None, exclude_mask=None):
#             del candidate_item_ids, exclude_item_ids, exclude_mask
#             batch_size = int(model_inputs["history_item_ids"].shape[0])
#             base = torch.tensor([[7, 6, 5, 4], [3, 2, 1, 0]], dtype=torch.long)
#             return base[:batch_size, :k]

#     monkeypatch.setattr(generator, "_load_trained_backbone_model", lambda: _FakeHSTUModel())

#     valid_frame = generator.build_split_frame("valid")

#     assert len(valid_frame) == len(prepared.get_eval_dataset("valid"))
#     for record in valid_frame.to_dict("records"):
#         candidate_item_ids = tuple(record["candidate_item_ids"])
#         assert len(candidate_item_ids) == 4


def test_llmrank_pipeline_end_to_end_with_full_protocol_and_identity_backend(local_tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = local_tmp_path / "configs"
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
    (config_dir / "model" / "llmrank_test.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: llmrank",
                "  candidate_source: random",
                "  backbone_topk: 4",
                "  recall_budget: 3",
                f"  candidate_cache_dir: {(local_tmp_path / 'candidate_cache').as_posix()}",
                "  item_text_field: metadata_text",
                "  backend: identity",
                "  parsing_strategy: index",
                "  domain: item",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  max_epochs: 0",
                "  eval:",
                "    protocol: full",
                "    neg_sampling_num: 0",
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
    assert all(len(values) == 4 for values in valid_dataset.frame["candidate_item_ids"].tolist())
    assert all(len(values) == 4 for values in test_dataset.frame["candidate_item_ids"].tolist())
    assert "recall@3" in result["test"]["metrics"]
    assert "ndcg@3" in result["test"]["metrics"]


def test_llmrank_pipeline_with_sampled_protocol_keeps_candidates_in_eval_frame(local_tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = local_tmp_path / "configs"
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
    (config_dir / "model" / "llmrank_test.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: llmrank",
                "  candidate_source: random",
                "  backbone_topk: 4",
                "  recall_budget: 3",
                f"  candidate_cache_dir: {(local_tmp_path / 'candidate_cache').as_posix()}",
                "  item_text_field: metadata_text",
                "  backend: identity",
                "trainer:",
                "  batch_size: 2",
                "  shuffle: false",
                "  max_epochs: 0",
                "  eval:",
                "    protocol: sampled",
                "    neg_sampling_num: 3",
                "    candidate_seed: 7",
                "    metrics:",
                "      - name: recall",
                "        ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))

    valid_dataset = result["prepared_data"].get_eval_dataset("valid")
    assert "candidate_item_ids" in valid_dataset.frame.columns
    assert all(len(values) == 4 for values in valid_dataset.frame["candidate_item_ids"].tolist())
