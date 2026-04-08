from __future__ import annotations

from recbole3.dataset import Interaction, RetrievalEvalRequest, SplitConfig
from recbole3.evaluation import EvalConfig
from tests.test_helpers import StubDataset, StubDatasetConfig, StubRankingDataset, StubRankingDatasetConfig


def _labeled_eval_config() -> EvalConfig:
    return EvalConfig(protocol="labeled")


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def test_ratio_split_with_chronological_order_and_per_user_groups() -> None:
    prepared = StubRankingDataset(
        StubRankingDatasetConfig(
            split=SplitConfig(
                strategy="ratio",
                order="chronological",
                per_user=True,
                train_ratio=0.5,
                valid_ratio=0.25,
                test_ratio=0.25,
            )
        )
    ).prepare(eval_config=_labeled_eval_config())

    assert list(prepared.get_train_dataset()) == [
        Interaction(user_id=0, item_id=0, timestamp=1, label=1.0),
        Interaction(user_id=0, item_id=1, timestamp=2, label=1.0),
        Interaction(user_id=1, item_id=4, timestamp=1, label=1.0),
        Interaction(user_id=1, item_id=5, timestamp=2, label=1.0),
    ]
    assert list(prepared.get_eval_dataset("valid")) == [
        Interaction(user_id=0, item_id=2, timestamp=3, label=1.0),
        Interaction(user_id=1, item_id=6, timestamp=3, label=1.0),
    ]
    assert list(prepared.get_eval_dataset("test")) == [
        Interaction(user_id=0, item_id=3, timestamp=4, label=1.0),
        Interaction(user_id=1, item_id=7, timestamp=4, label=1.0),
    ]


def test_ratio_split_uses_all_three_ratios() -> None:
    prepared = StubRankingDataset(
        StubRankingDatasetConfig(
            split=SplitConfig(
                strategy="ratio",
                order="chronological",
                per_user=False,
                train_ratio=2,
                valid_ratio=1,
                test_ratio=1,
            )
        )
    ).prepare(eval_config=_labeled_eval_config())

    assert [record.item_id for record in prepared.get_train_dataset()] == [0, 4, 1, 5]
    assert [record.item_id for record in prepared.get_eval_dataset("valid")] == [2, 6]
    assert [record.item_id for record in prepared.get_eval_dataset("test")] == [3, 7]


def test_random_per_user_split_is_deterministic_for_same_seed() -> None:
    split = SplitConfig(
        strategy="ratio",
        order="random",
        per_user=True,
        train_ratio=0.5,
        valid_ratio=0.25,
        test_ratio=0.25,
        seed=17,
    )
    prepared_a = StubRankingDataset(StubRankingDatasetConfig(split=split)).prepare(eval_config=_labeled_eval_config())
    prepared_b = StubRankingDataset(StubRankingDatasetConfig(split=split)).prepare(eval_config=_labeled_eval_config())

    assert list(prepared_a.get_train_dataset()) == list(prepared_b.get_train_dataset())
    assert list(prepared_a.get_eval_dataset("valid")) == list(prepared_b.get_eval_dataset("valid"))
    assert list(prepared_a.get_eval_dataset("test")) == list(prepared_b.get_eval_dataset("test"))


def test_retrieval_prepare_builds_request_level_eval_records() -> None:
    prepared = StubDataset(
        StubDatasetConfig(
            split=SplitConfig(
                strategy="leave_one_out",
                order="chronological",
                per_user=True,
                valid_holdout_num=1,
                test_holdout_num=1,
            )
        )
    ).prepare(eval_config=_full_eval_config())

    assert list(prepared.get_train_dataset()) == [
        Interaction(user_id=0, item_id=0, timestamp=1, label=1.0),
        Interaction(user_id=0, item_id=1, timestamp=2, label=1.0),
        Interaction(user_id=1, item_id=4, timestamp=1, label=1.0),
        Interaction(user_id=1, item_id=5, timestamp=2, label=1.0),
    ]
    assert list(prepared.get_eval_dataset("valid")) == [
        RetrievalEvalRequest(user_id=0, item_id=2, timestamp=3, label=1.0, seen_item_ids=(0, 1)),
        RetrievalEvalRequest(user_id=1, item_id=6, timestamp=3, label=1.0, seen_item_ids=(4, 5)),
    ]
    assert list(prepared.get_eval_dataset("test")) == [
        RetrievalEvalRequest(user_id=0, item_id=3, timestamp=4, label=1.0, seen_item_ids=(0, 1, 2)),
        RetrievalEvalRequest(user_id=1, item_id=7, timestamp=4, label=1.0, seen_item_ids=(4, 5, 6)),
    ]


def test_retrieval_prepare_expands_seen_history_within_split() -> None:
    prepared = StubDataset(
        StubDatasetConfig(
            split=SplitConfig(
                strategy="leave_one_out",
                order="chronological",
                per_user=True,
                valid_holdout_num=2,
                test_holdout_num=1,
            )
        )
    ).prepare(eval_config=_full_eval_config())

    assert list(prepared.get_eval_dataset("valid")) == [
        RetrievalEvalRequest(user_id=0, item_id=1, timestamp=2, label=1.0, seen_item_ids=(0,)),
        RetrievalEvalRequest(user_id=0, item_id=2, timestamp=3, label=1.0, seen_item_ids=(0, 1)),
        RetrievalEvalRequest(user_id=1, item_id=5, timestamp=2, label=1.0, seen_item_ids=(4,)),
        RetrievalEvalRequest(user_id=1, item_id=6, timestamp=3, label=1.0, seen_item_ids=(4, 5)),
    ]


def test_prepared_dataset_exposes_runtime_views() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())

    train_dataset = prepared.get_train_dataset()
    valid_dataset = prepared.get_eval_dataset("valid")
    interactions = prepared.get_interactions()

    assert train_dataset[0] == Interaction(user_id=0, item_id=0, timestamp=1, label=1.0)
    assert list(valid_dataset) == [
        Interaction(user_id=0, item_id=2, timestamp=3, label=1.0),
        Interaction(user_id=1, item_id=6, timestamp=3, label=1.0),
    ]
    assert interactions[0] == Interaction(user_id=0, item_id=0, timestamp=1, label=1.0)
