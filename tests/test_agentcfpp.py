from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recbole3.dataset import DATASET_TABLE, get_dataset_spec
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MetricSpec
from recbole3.model import MODEL_TABLE
from recbole3.model.agentcfpp.config import AgentCFPPConfig
from recbole3.model.agentcfpp.model import AgentCFPPModel
from recbole3.model.agentcfpp.trainer import AgentCFPPTrainerConfig, _compute_mrr
from recbole3.evaluation.metric import RetrievalEvalData


DOMAINS = ["Books", "Video_Games", "Movies_and_TV"]
MAIN_CATS = {"Books": "Books", "Video_Games": "Video Games", "Movies_and_TV": "Movies & TV"}


class FakeLLMClient:
    """Deterministic LLM stub that returns well-formed AgentCF++ outputs."""

    def __init__(self, item_titles: dict[str, str] | None = None):
        self._item_titles = item_titles or {}

    def chat_completion_batch(self, messages_list, *, temperature=None):
        results = []
        for messages in messages_list:
            content = messages[-1]["content"]
            results.append(self._respond(content))
        return results

    def chat_completion(self, messages, *, temperature=None):
        return self._respond(messages[-1]["content"])

    def embedding_batch(self, texts):
        # Deterministic pseudo-embeddings derived from text hash.
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rng.normal(size=16).tolist())
        return out

    def _respond(self, content: str) -> str:
        if "interest_tags" in content:
            return json.dumps({"interest_tags": ["action", "classic"]})
        if "summary phrase" in content or "single phrase" in content:
            return "action lovers"
        if "Choice:" in content:
            # Pick the first title-looking token after "title:".
            return "Choice: Item A\nExplanation: fits my taste."
        if "My updated self-introduction:" in content:
            return "My updated self-introduction: I like action items."
        if "My deduced preference:" in content:
            return "My deduced preference: cross-domain action fan."
        if "first item" in content:
            return (
                "The updated description of the first item is: a dull item. "
                "The updated description of the second item is: an exciting item."
            )
        if "Rank:" in content:
            # Rank candidate titles in arbitrary deterministic order.
            return "Rank: {1. Item A\n2. Item B\n3. Item C}"
        return "ok"


def _write_cross_domain_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "random").mkdir(parents=True, exist_ok=True)

    # 4 items across 3 domains.
    items = [
        ("b1", "Item A", "Books"),
        ("b2", "Item B", "Books"),
        ("v1", "Item C", "Video_Games"),
        ("m1", "Item D", "Movies_and_TV"),
    ]
    meta = pd.DataFrame(
        {
            "parent_asin": [i[0] for i in items],
            "title": [i[1] for i in items],
            "main_category": [MAIN_CATS[i[2]] for i in items],
            "categories": ["cat"] * len(items),
            "price": ["10"] * len(items),
            "subtitle": [""] * len(items),
        }
    )
    meta.to_csv(root / "meta_crossdomain.csv", index=False)

    # Interactions: two users, train + test.
    train = pd.DataFrame({"user_id": ["u1", "u2", "u1"], "parent_asin": ["b1", "v1", "m1"]})
    test = pd.DataFrame({"user_id": ["u1", "u2"], "parent_asin": ["b2", "v1"]})
    train.to_csv(root / "inter_crossdomain_timesequence_train.csv", index=False)
    test.to_csv(root / "inter_crossdomain_timesequence_test.csv", index=False)

    # Per-domain candidate pools (column "Unnamed: 0" = user id).
    books_pool = pd.DataFrame({"Unnamed: 0": ["u1", "u2"], "item_0": ["b1", "b2"], "item_1": ["b2", "b1"]})
    vg_pool = pd.DataFrame({"Unnamed: 0": ["u1", "u2"], "item_0": ["v1", "v1"], "item_1": ["v1", "v1"]})
    mv_pool = pd.DataFrame({"Unnamed: 0": ["u1", "u2"], "item_0": ["m1", "m1"], "item_1": ["m1", "m1"]})
    books_pool.to_csv(root / "random" / "random_Books.csv", index=False)
    vg_pool.to_csv(root / "random" / "random_Video_Games.csv", index=False)
    mv_pool.to_csv(root / "random" / "random_Movies_and_TV.csv", index=False)


def _build_prepared(root: Path):
    spec = get_dataset_spec("agentcfpp_cross")
    config = spec.config_cls(
        name="agentcfpp_cross",
        source="local_csv",
        data_dir=str(root),
        random_files=("random/random_Books.csv", "random/random_Video_Games.csv", "random/random_Movies_and_TV.csv"),
        domain_list=tuple(DOMAINS),
    )
    dataset = spec.dataset_cls(config)
    eval_config = EvalConfig(
        protocol="full",
        metrics=(MetricSpec(name="ndcg", ks=(1, 3)), MetricSpec(name="recall", ks=(1, 3))),
        neg_sampling_num=0,
        candidate_seed=42,
    )
    return dataset.prepare(eval_config=eval_config)


def test_registration() -> None:
    assert "agentcfpp" in MODEL_TABLE
    assert "agentcfpp_cross" in DATASET_TABLE


def test_dataset_parses_domains_and_pools(tmp_path) -> None:
    _write_cross_domain_data(tmp_path)
    prepared = _build_prepared(tmp_path)

    item_domains = prepared.get_item_domains()
    assert set(item_domains.values()) <= set(DOMAINS)
    assert len(item_domains) == 4

    pools = prepared.get_domain_candidate_pools()
    assert "Books" in pools
    # Framework ids are ints.
    for domain_pool in pools.values():
        for user_id, items in domain_pool.items():
            assert isinstance(user_id, int)
            assert all(isinstance(i, int) for i in items)


def test_train_predict_smoke(tmp_path) -> None:
    _write_cross_domain_data(tmp_path)
    prepared = _build_prepared(tmp_path)

    model_config = AgentCFPPConfig(name="agentcfpp", domain_list=tuple(DOMAINS), candidate_num=3, use_group_memory=True)
    model = AgentCFPPModel(model_config)
    model.set_llm_client(FakeLLMClient())
    model.set_cross_domain_context(
        item_domains=prepared.get_item_domains(),
        domain_candidate_pools=prepared.get_domain_candidate_pools(),
    )
    model.ensure_initialized(prepared)

    # Train step over the train split.
    train_collator = model.build_train_collator(prepared)
    train_frame = prepared.get_train_dataset().frame
    batch = train_collator(train_frame)
    result = model.train_step(batch)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0

    # User memory should be populated for the trained domains.
    agent = model._user_agents[batch["user_ids"][0].item()]
    assert agent.active_domains

    # Group memory build.
    from recbole3.model.agentcfpp.group_memory import build_group_state

    group_state = build_group_state(model, prepared, model_config)
    model.set_group_state(group_state)

    # Predict over an injected candidate set.
    import torch

    user_ids = torch.tensor([0], dtype=torch.long)
    item_domains = prepared.get_item_domains()
    # Build a candidate row from the Books pool for user 0.
    candidate_item_ids = torch.tensor([[0, 1]], dtype=torch.long)
    preds = model.predict({"user_ids": user_ids, "history_item_ids": None}, k=2, candidate_item_ids=candidate_item_ids)
    assert preds.shape == (1, 2)
    assert set(preds[0].tolist()) <= {0, 1}


def test_mrr_metric() -> None:
    eval_data = RetrievalEvalData(
        pred_item_ids=np.array([[5, 3, 1], [2, 9, 8]]),
        target_item_ids=np.array([[3], [2]]),
        target_mask=np.array([[True], [True]]),
    )
    out = _compute_mrr(eval_data, (1, 3))
    # Row 0: target 3 at rank 2 -> 0.5; row 1: target 2 at rank 1 -> 1.0.
    assert out["mrr@1"] == pytest.approx(0.5)  # only row1 hits at rank 1
    assert out["mrr@3"] == pytest.approx((0.5 + 1.0) / 2)


def test_amazon2023_cross_build(monkeypatch, tmp_path) -> None:
    """source=amazon2023 builds a cross-domain dataset from HF-downloaded files (mocked)."""
    from recbole3.dataset.agentcfpp_cross import AgentCFPPCrossParser
    from recbole3.dataset.utils import ITEM_ID as IID, TIMESTAMP as TS, USER_ID as UID

    # Two users active in two domains each (so they survive min_domains_per_user=2).
    fake_ratings = {
        "Books": pd.DataFrame(
            {UID: ["u1", "u2", "u1"], IID: ["b1", "b2", "b3"], TS: [1, 2, 3]}
        ),
        "Video_Games": pd.DataFrame({UID: ["u1", "u2"], IID: ["v1", "v2"], TS: [4, 5]}),
    }
    fake_titles = {
        "Books": {"b1": "Book One", "b2": "Book Two", "b3": "Book Three"},
        "Video_Games": {"v1": "Game One", "v2": "Game Two"},
    }

    def fake_download_ratings(self, domain):
        return fake_ratings[domain].copy()

    def fake_download_titles(self, domain, kept_items):
        return {k: v for k, v in fake_titles[domain].items() if k in kept_items}

    monkeypatch.setattr(AgentCFPPCrossParser, "_download_ratings", fake_download_ratings)
    monkeypatch.setattr(AgentCFPPCrossParser, "_download_item_titles", fake_download_titles)

    spec = get_dataset_spec("agentcfpp_cross")
    config = spec.config_cls(
        name="agentcfpp_cross",
        source="amazon2023",
        domain_list=("Books", "Video_Games"),
        max_users=10,
        min_domains_per_user=2,
        min_inter_per_user=2,
        n_random_item=2,
        download_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "proc"),
        dump_dir=str(tmp_path / "dump"),
    )
    dataset = spec.dataset_cls(config)
    eval_config = EvalConfig(
        protocol="full",
        metrics=(MetricSpec(name="ndcg", ks=(1,)),),
        neg_sampling_num=0,
        candidate_seed=42,
    )
    prepared = dataset.prepare(eval_config=eval_config)

    item_domains = prepared.get_item_domains()
    assert set(item_domains.values()) == {"Books", "Video_Games"}
    pools = prepared.get_domain_candidate_pools()
    assert "Books" in pools and "Video_Games" in pools
    # Both cross-domain users are kept.
    assert prepared.get_num_users() == 2

    # dump_dir wrote local_csv-format files that read back to the same shape.
    dump = tmp_path / "dump"
    assert (dump / "meta_crossdomain.csv").exists()
    assert (dump / "inter_crossdomain_timesequence_train.csv").exists()
    assert (dump / "inter_crossdomain_timesequence_test.csv").exists()
    assert (dump / "random" / "random_Books.csv").exists()

    config2 = spec.config_cls(
        name="agentcfpp_cross",
        source="local_csv",
        data_dir=str(dump),
        domain_list=("Books", "Video_Games"),
        random_files=("random/random_Books.csv", "random/random_Video_Games.csv"),
    )
    prepared2 = spec.dataset_cls(config2).prepare(eval_config=eval_config)
    assert prepared2.get_num_users() == prepared.get_num_users()
    assert prepared2.get_num_items() == prepared.get_num_items()
    assert set(prepared2.get_item_domains().values()) == {"Books", "Video_Games"}
