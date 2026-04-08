from __future__ import annotations

import pytest

from recbole3.evaluation import EvalConfig
from recbole3.model import (
    BaseSequentialRankingModelDataset,
    BaseSequentialRetrievalModelDataset,
    ModelConfig,
    SequentialModelConfig,
    SequentialInteraction,
    SequentialRetrievalEvalRequest,
    build_history_item_ids,
)
from tests.test_helpers import StubDataset, StubDatasetConfig, StubRankingDataset, StubRankingDatasetConfig


class StubSequentialRankingDataset(BaseSequentialRankingModelDataset):
    pass


class StubSequentialRetrievalDataset(BaseSequentialRetrievalModelDataset):
    pass


def _labeled_eval_config() -> EvalConfig:
    return EvalConfig(protocol="labeled")


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def test_build_history_item_ids_skips_non_positive_updates() -> None:
    records = [
        SequentialInteraction(user_id=0, item_id=1, label=1.0),
        SequentialInteraction(user_id=0, item_id=2, label=0.0),
        SequentialInteraction(user_id=0, item_id=3, label=1.0),
    ]

    history_item_ids, history_state = build_history_item_ids(records)

    assert history_item_ids == [(), (1,), (1,)]
    assert history_state == {0: (1, 3)}


def test_build_history_item_ids_truncates_to_most_recent_items() -> None:
    records = [
        SequentialInteraction(user_id=0, item_id=4, label=1.0),
        SequentialInteraction(user_id=0, item_id=5, label=1.0),
    ]

    history_item_ids, history_state = build_history_item_ids(
        records,
        initial_histories={0: (1, 2, 3)},
        history_max_length=2,
    )

    assert history_item_ids == [(2, 3), (3, 4)]
    assert history_state == {0: (4, 5)}


@pytest.mark.parametrize("history_max_length", [0, -1])
def test_build_history_item_ids_rejects_non_positive_history_max_length(history_max_length: int) -> None:
    with pytest.raises(ValueError, match="history_max_length"):
        build_history_item_ids(
            [SequentialInteraction(user_id=0, item_id=1, label=1.0)],
            history_max_length=history_max_length,
        )


def test_sequential_ranking_dataset_builds_prefix_histories_across_splits() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())

    sequential_data = StubSequentialRankingDataset.from_task_dataset(prepared, model_config=ModelConfig(name="stub"))

    assert list(sequential_data.get_train_dataset()) == [
        SequentialInteraction(user_id=0, item_id=0, timestamp=1, label=1.0, history_item_ids=()),
        SequentialInteraction(user_id=0, item_id=1, timestamp=2, label=1.0, history_item_ids=(0,)),
        SequentialInteraction(user_id=1, item_id=4, timestamp=1, label=1.0, history_item_ids=()),
        SequentialInteraction(user_id=1, item_id=5, timestamp=2, label=1.0, history_item_ids=(4,)),
    ]
    assert list(sequential_data.get_eval_dataset("valid")) == [
        SequentialInteraction(user_id=0, item_id=2, timestamp=3, label=1.0, history_item_ids=(0, 1)),
        SequentialInteraction(user_id=1, item_id=6, timestamp=3, label=1.0, history_item_ids=(4, 5)),
    ]
    assert list(sequential_data.get_eval_dataset("test")) == [
        SequentialInteraction(user_id=0, item_id=3, timestamp=4, label=1.0, history_item_ids=(0, 1, 2)),
        SequentialInteraction(user_id=1, item_id=7, timestamp=4, label=1.0, history_item_ids=(4, 5, 6)),
    ]


def test_sequential_ranking_dataset_truncates_prefix_histories_across_splits() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())

    sequential_data = StubSequentialRankingDataset.from_task_dataset(
        prepared,
        model_config=SequentialModelConfig(name="stub", history_max_length=2),
    )

    assert list(sequential_data.get_train_dataset()) == [
        SequentialInteraction(user_id=0, item_id=0, timestamp=1, label=1.0, history_item_ids=()),
        SequentialInteraction(user_id=0, item_id=1, timestamp=2, label=1.0, history_item_ids=(0,)),
        SequentialInteraction(user_id=1, item_id=4, timestamp=1, label=1.0, history_item_ids=()),
        SequentialInteraction(user_id=1, item_id=5, timestamp=2, label=1.0, history_item_ids=(4,)),
    ]
    assert list(sequential_data.get_eval_dataset("valid")) == [
        SequentialInteraction(user_id=0, item_id=2, timestamp=3, label=1.0, history_item_ids=(0, 1)),
        SequentialInteraction(user_id=1, item_id=6, timestamp=3, label=1.0, history_item_ids=(4, 5)),
    ]
    assert list(sequential_data.get_eval_dataset("test")) == [
        SequentialInteraction(user_id=0, item_id=3, timestamp=4, label=1.0, history_item_ids=(1, 2)),
        SequentialInteraction(user_id=1, item_id=7, timestamp=4, label=1.0, history_item_ids=(5, 6)),
    ]


def test_sequential_retrieval_dataset_preserves_eval_contract_and_histories() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    sequential_data = StubSequentialRetrievalDataset.from_task_dataset(prepared, model_config=ModelConfig(name="stub"))

    assert list(sequential_data.get_train_dataset()) == [
        SequentialInteraction(user_id=0, item_id=0, timestamp=1, label=1.0, history_item_ids=()),
        SequentialInteraction(user_id=0, item_id=1, timestamp=2, label=1.0, history_item_ids=(0,)),
        SequentialInteraction(user_id=1, item_id=4, timestamp=1, label=1.0, history_item_ids=()),
        SequentialInteraction(user_id=1, item_id=5, timestamp=2, label=1.0, history_item_ids=(4,)),
    ]
    assert list(sequential_data.get_eval_dataset("valid")) == [
        SequentialRetrievalEvalRequest(
            user_id=0,
            item_id=2,
            timestamp=3,
            label=1.0,
            seen_item_ids=(0, 1),
            candidate_item_ids=None,
            history_item_ids=(0, 1),
        ),
        SequentialRetrievalEvalRequest(
            user_id=1,
            item_id=6,
            timestamp=3,
            label=1.0,
            seen_item_ids=(4, 5),
            candidate_item_ids=None,
            history_item_ids=(4, 5),
        ),
    ]
    assert list(sequential_data.get_eval_dataset("test")) == [
        SequentialRetrievalEvalRequest(
            user_id=0,
            item_id=3,
            timestamp=4,
            label=1.0,
            seen_item_ids=(0, 1, 2),
            candidate_item_ids=None,
            history_item_ids=(0, 1, 2),
        ),
        SequentialRetrievalEvalRequest(
            user_id=1,
            item_id=7,
            timestamp=4,
            label=1.0,
            seen_item_ids=(4, 5, 6),
            candidate_item_ids=None,
            history_item_ids=(4, 5, 6),
        ),
    ]


def test_sequential_retrieval_dataset_truncates_histories_without_changing_eval_contract() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    sequential_data = StubSequentialRetrievalDataset.from_task_dataset(
        prepared,
        model_config=SequentialModelConfig(name="stub", history_max_length=2),
    )

    assert list(sequential_data.get_eval_dataset("valid")) == [
        SequentialRetrievalEvalRequest(
            user_id=0,
            item_id=2,
            timestamp=3,
            label=1.0,
            seen_item_ids=(0, 1),
            candidate_item_ids=None,
            history_item_ids=(0, 1),
        ),
        SequentialRetrievalEvalRequest(
            user_id=1,
            item_id=6,
            timestamp=3,
            label=1.0,
            seen_item_ids=(4, 5),
            candidate_item_ids=None,
            history_item_ids=(4, 5),
        ),
    ]
    assert list(sequential_data.get_eval_dataset("test")) == [
        SequentialRetrievalEvalRequest(
            user_id=0,
            item_id=3,
            timestamp=4,
            label=1.0,
            seen_item_ids=(0, 1, 2),
            candidate_item_ids=None,
            history_item_ids=(1, 2),
        ),
        SequentialRetrievalEvalRequest(
            user_id=1,
            item_id=7,
            timestamp=4,
            label=1.0,
            seen_item_ids=(4, 5, 6),
            candidate_item_ids=None,
            history_item_ids=(5, 6),
        ),
    ]
