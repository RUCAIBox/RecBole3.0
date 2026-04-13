from __future__ import annotations

import pandas as pd

from recbole3.dataset import CANDIDATE_ITEM_IDS, ITEM_ID, LABEL, SEEN_ITEM_IDS, SplitConfig, TIMESTAMP, USER_ID
from recbole3.evaluation import EvalConfig
from tests.test_helpers import StubDataset, StubDatasetConfig, StubRankingDataset, StubRankingDatasetConfig


def _labeled_eval_config() -> EvalConfig:
    return EvalConfig(protocol="labeled")


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _rows(dataset, columns: list[str] | None = None) -> list[dict]:
    frame = dataset.frame
    if columns is not None:
        frame = frame.loc[:, columns]
    return frame.to_dict("records")


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

    columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL]
    assert _rows(prepared.get_train_dataset(), columns) == [
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 1, LABEL: 1.0},
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 2, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 1, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 2, LABEL: 1.0},
    ]
    assert _rows(prepared.get_eval_dataset("valid"), columns) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 3, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 3, LABEL: 1.0},
    ]
    assert _rows(prepared.get_eval_dataset("test"), columns) == [
        {USER_ID: 0, ITEM_ID: 4, TIMESTAMP: 4, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 8, TIMESTAMP: 4, LABEL: 1.0},
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

    assert prepared.get_train_dataset().frame[ITEM_ID].tolist() == [1, 5, 2, 6]
    assert prepared.get_eval_dataset("valid").frame[ITEM_ID].tolist() == [3, 7]
    assert prepared.get_eval_dataset("test").frame[ITEM_ID].tolist() == [4, 8]


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

    assert prepared_a.get_train_dataset().frame.equals(prepared_b.get_train_dataset().frame)
    assert prepared_a.get_eval_dataset("valid").frame.equals(prepared_b.get_eval_dataset("valid").frame)
    assert prepared_a.get_eval_dataset("test").frame.equals(prepared_b.get_eval_dataset("test").frame)


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

    assert _rows(prepared.get_train_dataset(), [USER_ID, ITEM_ID, TIMESTAMP, LABEL]) == [
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 1, LABEL: 1.0},
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 2, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 1, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 2, LABEL: 1.0},
    ]
    assert _rows(prepared.get_eval_dataset("valid"), [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS]) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (1, 2)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (5, 6)},
    ]
    assert _rows(prepared.get_eval_dataset("test"), [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS]) == [
        {USER_ID: 0, ITEM_ID: 4, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (1, 2, 3)},
        {USER_ID: 1, ITEM_ID: 8, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (5, 6, 7)},
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

    assert _rows(prepared.get_eval_dataset("valid"), [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS]) == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 2, LABEL: 1.0, SEEN_ITEM_IDS: (1,)},
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (1, 2)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 2, LABEL: 1.0, SEEN_ITEM_IDS: (5,)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (5, 6)},
    ]


def test_retrieval_eval_frame_keeps_first_seen_unique_histories() -> None:
    dataset = StubDataset(StubDatasetConfig())
    history = pd.DataFrame(
        [
            {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 1, LABEL: 1.0},
            {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 2, LABEL: 1.0},
            {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 3, LABEL: 1.0},
            {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 1, LABEL: 1.0},
            {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 2, LABEL: 1.0},
        ]
    )
    positives = pd.DataFrame(
        [
            {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 4, LABEL: 1.0},
            {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 5, LABEL: 1.0},
            {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 6, LABEL: 1.0},
            {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0},
        ]
    )

    requests = dataset._build_retrieval_eval_frame(positives, seen_history_interactions=history)

    assert requests[[USER_ID, ITEM_ID, SEEN_ITEM_IDS]].to_dict("records") == [
        {USER_ID: 0, ITEM_ID: 2, SEEN_ITEM_IDS: (1, 2)},
        {USER_ID: 0, ITEM_ID: 3, SEEN_ITEM_IDS: (1, 2)},
        {USER_ID: 0, ITEM_ID: 3, SEEN_ITEM_IDS: (1, 2, 3)},
        {USER_ID: 1, ITEM_ID: 6, SEEN_ITEM_IDS: (5,)},
    ]


def test_prepared_dataset_exposes_runtime_views() -> None:
    prepared = StubRankingDataset(StubRankingDatasetConfig()).prepare(eval_config=_labeled_eval_config())

    train_dataset = prepared.get_train_dataset()
    valid_dataset = prepared.get_eval_dataset("valid")
    interactions = prepared.get_interactions()

    assert train_dataset[0] == {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 1, LABEL: 1.0}
    assert _rows(valid_dataset, [USER_ID, ITEM_ID, TIMESTAMP, LABEL]) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 3, LABEL: 1.0},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 3, LABEL: 1.0},
    ]
    assert interactions.loc[0, ITEM_ID] == 1


def test_sampled_candidates_are_equal_width_and_skip_pad_item() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=EvalConfig(protocol="sampled", neg_sampling_num=2))
    candidates = prepared.get_eval_dataset("test").frame[CANDIDATE_ITEM_IDS].tolist()

    assert all(isinstance(values, tuple) for values in candidates)
    assert [len(values) for values in candidates] == [3, 3]
    assert all(0 not in values for values in candidates)
    assert candidates == [(4, 5, 6), (8, 5, 1)]
