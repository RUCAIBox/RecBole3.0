from __future__ import annotations

import pandas as pd
import pytest

from recbole3.dataset import ITEM_ID, LABEL, SEEN_ITEM_IDS, TIMESTAMP, USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.model import (
    HISTORY_ITEM_IDS,
    BaseSequentialModelDataset,
    ModelConfig,
    SequentialModelConfig,
    build_history_item_ids,
)
from tests.test_helpers import StubDataset, StubDatasetConfig, StubRankingDataset, StubRankingDatasetConfig


class StubSequentialRankingDataset(BaseSequentialModelDataset):
    pass


class StubSequentialRetrievalDataset(BaseSequentialModelDataset):
    pass


def _labeled_eval_config() -> EvalConfig:
    return EvalConfig(protocol="labeled")


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _rows(dataset, columns: list[str]) -> list[dict]:
    return dataset.frame.loc[:, columns].to_dict("records")


def test_build_history_item_ids_skips_non_positive_updates() -> None:
    records = pd.DataFrame(
        [
            {USER_ID: 0, ITEM_ID: 1, LABEL: 1.0},
            {USER_ID: 0, ITEM_ID: 2, LABEL: 0.0},
            {USER_ID: 0, ITEM_ID: 3, LABEL: 1.0},
        ]
    )

    history_item_ids, history_state = build_history_item_ids(records)

    assert history_item_ids == [(), (1,), (1,)]
    assert history_state == {0: (1, 3)}


def test_build_history_item_ids_truncates_to_most_recent_items() -> None:
    records = pd.DataFrame(
        [
            {USER_ID: 0, ITEM_ID: 4, LABEL: 1.0},
            {USER_ID: 0, ITEM_ID: 5, LABEL: 1.0},
        ]
    )

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
            pd.DataFrame([{USER_ID: 0, ITEM_ID: 1, LABEL: 1.0}]),
            history_max_length=history_max_length,
        )


def test_sequential_ranking_dataset_builds_prefix_histories_across_splits() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())

    sequential_data = StubSequentialRankingDataset.from_task_dataset(prepared, model_config=ModelConfig(name="stub"))

    columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL, HISTORY_ITEM_IDS]
    assert _rows(sequential_data.get_train_dataset(), columns) == [
        {USER_ID: 0, ITEM_ID: 0, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (0,)},
        {USER_ID: 1, ITEM_ID: 4, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (4,)},
    ]
    assert _rows(sequential_data.get_eval_dataset("valid"), columns) == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0, HISTORY_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0, HISTORY_ITEM_IDS: (4, 5)},
    ]
    assert _rows(sequential_data.get_eval_dataset("test"), columns) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 4, LABEL: 1.0, HISTORY_ITEM_IDS: (0, 1, 2)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 4, LABEL: 1.0, HISTORY_ITEM_IDS: (4, 5, 6)},
    ]


def test_sequential_ranking_dataset_truncates_prefix_histories_across_splits() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())

    sequential_data = StubSequentialRankingDataset.from_task_dataset(
        prepared,
        model_config=SequentialModelConfig(name="stub", history_max_length=2),
    )

    columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL, HISTORY_ITEM_IDS]
    assert _rows(sequential_data.get_train_dataset(), columns) == [
        {USER_ID: 0, ITEM_ID: 0, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (0,)},
        {USER_ID: 1, ITEM_ID: 4, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (4,)},
    ]
    assert _rows(sequential_data.get_eval_dataset("valid"), columns) == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0, HISTORY_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0, HISTORY_ITEM_IDS: (4, 5)},
    ]
    assert _rows(sequential_data.get_eval_dataset("test"), columns) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 4, LABEL: 1.0, HISTORY_ITEM_IDS: (1, 2)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 4, LABEL: 1.0, HISTORY_ITEM_IDS: (5, 6)},
    ]


def test_sequential_retrieval_dataset_preserves_eval_contract_and_histories() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    sequential_data = StubSequentialRetrievalDataset.from_task_dataset(prepared, model_config=ModelConfig(name="stub"))

    assert _rows(sequential_data.get_train_dataset(), [USER_ID, ITEM_ID, TIMESTAMP, LABEL, HISTORY_ITEM_IDS]) == [
        {USER_ID: 0, ITEM_ID: 0, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (0,)},
        {USER_ID: 1, ITEM_ID: 4, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (4,)},
    ]
    eval_columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS, HISTORY_ITEM_IDS]
    assert _rows(sequential_data.get_eval_dataset("valid"), eval_columns) == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1), HISTORY_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5), HISTORY_ITEM_IDS: (4, 5)},
    ]
    assert _rows(sequential_data.get_eval_dataset("test"), eval_columns) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1, 2), HISTORY_ITEM_IDS: (0, 1, 2)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5, 6), HISTORY_ITEM_IDS: (4, 5, 6)},
    ]


def test_sequential_retrieval_dataset_truncates_histories_without_changing_eval_contract() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    sequential_data = StubSequentialRetrievalDataset.from_task_dataset(
        prepared,
        model_config=SequentialModelConfig(name="stub", history_max_length=2),
    )

    eval_columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS, HISTORY_ITEM_IDS]
    assert _rows(sequential_data.get_eval_dataset("valid"), eval_columns) == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1), HISTORY_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5), HISTORY_ITEM_IDS: (4, 5)},
    ]
    assert _rows(sequential_data.get_eval_dataset("test"), eval_columns) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1, 2), HISTORY_ITEM_IDS: (1, 2)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5, 6), HISTORY_ITEM_IDS: (5, 6)},
    ]
