from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from recbole3.config import project_root
from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID, SEEN_ITEM_IDS, USER_ID
from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.starec.config import STARecConfig
from recbole3.model.starec.feedback import (
    is_positive_or_unlabeled_record,
    positive_or_unlabeled,
    positive_or_unlabeled_mask,
)


STARecHistoryFrames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[int, ...]]


def build_history_limited_frames(
    task_data: BaseTaskDataset,
    *,
    model_config: STARecConfig,
) -> STARecHistoryFrames:
    train_frame = _frame_dataset_frame(task_data.get_train_dataset(), split="train")
    valid_frame = _frame_dataset_frame(task_data.get_eval_dataset("valid"), split="valid")
    test_frame = _frame_dataset_frame(task_data.get_eval_dataset("test"), split="test")

    valid_frame = positive_or_unlabeled(valid_frame, model_config=model_config).reset_index(drop=True)
    test_frame = positive_or_unlabeled(test_frame, model_config=model_config).reset_index(drop=True)
    require_train_warmup = _requires_train_warmup(model_config)
    ordered_user_ids, candidate_user_ids, requested_user_ids, selected_user_count = _candidate_history_user_ids(
        test_frame,
        model_config=model_config,
    )
    train_positions_by_user = _group_positions_by_user(train_frame)
    valid_positions_by_user = _group_positions_by_user(valid_frame)
    test_positions_by_user = _group_positions_by_user(test_frame)
    retained_by_user: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for user_id in candidate_user_ids:
        user_train = _user_frame(train_frame, train_positions_by_user, user_id)
        user_valid = _user_frame(valid_frame, valid_positions_by_user, user_id)
        user_test = _user_frame(test_frame, test_positions_by_user, user_id)
        if user_valid.empty or user_test.empty:
            continue
        retained_train, retained_valid, retained_test = _history_limited_user_frames(
            train_frame=user_train,
            valid_frame=user_valid,
            test_frame=user_test,
            history_max_length=model_config.history_max_length,
            model_config=model_config,
        )
        if _is_user_eligible(
            retained_train=retained_train,
            retained_valid=retained_valid,
            retained_test=retained_test,
            train_init_interactions=int(model_config.train_init_interactions),
            history_min_length=int(model_config.history_min_length),
            require_train_warmup=require_train_warmup,
            model_config=model_config,
        ):
            retained_by_user[int(user_id)] = (retained_train, retained_valid, retained_test)
            if selected_user_count != -1 and len(retained_by_user) >= selected_user_count:
                break

    selected_user_ids = _finalize_selected_user_ids(
        ordered_user_ids=ordered_user_ids,
        eligible_user_ids=tuple(retained_by_user),
        requested_user_ids=requested_user_ids,
        selected_user_count=selected_user_count,
        candidate_seed=int(model_config.candidate_seed),
    )

    selected_train_frames: list[pd.DataFrame] = []
    selected_valid_frames: list[pd.DataFrame] = []
    selected_test_frames: list[pd.DataFrame] = []
    for user_id in selected_user_ids:
        user_train, user_valid, user_test = retained_by_user[int(user_id)]
        selected_train_frames.append(user_train)
        selected_valid_frames.append(user_valid)
        selected_test_frames.append(user_test)

    return (
        _concat_like(selected_train_frames, train_frame),
        _concat_like(selected_valid_frames, valid_frame),
        _concat_like(selected_test_frames, test_frame),
        selected_user_ids,
    )


def select_history_eligible_user_ids(
    task_data: BaseTaskDataset,
    *,
    model_config: STARecConfig,
) -> tuple[int, ...]:
    """Return STARec-eligible user ids without materializing per-user frames."""

    train_frame = _frame_dataset_frame(task_data.get_train_dataset(), split="train")
    valid_frame = _frame_dataset_frame(task_data.get_eval_dataset("valid"), split="valid")
    test_frame = _frame_dataset_frame(task_data.get_eval_dataset("test"), split="test")

    valid_frame = positive_or_unlabeled(valid_frame, model_config=model_config).reset_index(drop=True)
    test_frame = positive_or_unlabeled(test_frame, model_config=model_config).reset_index(drop=True)
    require_train_warmup = _requires_train_warmup(model_config)
    ordered_user_ids, candidate_user_ids, requested_user_ids, selected_user_count = _candidate_history_user_ids(
        test_frame,
        model_config=model_config,
    )
    train_positions_by_user = _group_positions_by_user(train_frame)
    valid_positions_by_user = _group_positions_by_user(valid_frame)
    test_positions_by_user = _group_positions_by_user(test_frame)
    train_positive_mask = positive_or_unlabeled_mask(train_frame, model_config=model_config).to_numpy(dtype=bool)

    eligible_by_candidate_order: list[int] = []
    for user_id in candidate_user_ids:
        if _is_user_eligible_by_positions(
            train_positions=train_positions_by_user.get(int(user_id), ()),
            valid_positions=valid_positions_by_user.get(int(user_id), ()),
            test_positions=test_positions_by_user.get(int(user_id), ()),
            train_positive_mask=train_positive_mask,
            history_max_length=model_config.history_max_length,
            train_init_interactions=int(model_config.train_init_interactions),
            history_min_length=int(model_config.history_min_length),
            require_train_warmup=require_train_warmup,
        ):
            eligible_by_candidate_order.append(int(user_id))
            if selected_user_count != -1 and len(eligible_by_candidate_order) >= selected_user_count:
                break

    return _finalize_selected_user_ids(
        ordered_user_ids=ordered_user_ids,
        eligible_user_ids=tuple(eligible_by_candidate_order),
        requested_user_ids=requested_user_ids,
        selected_user_count=selected_user_count,
        candidate_seed=int(model_config.candidate_seed),
    )


def build_train_candidate_frame(
    task_data: BaseTaskDataset,
    *,
    model_config: STARecConfig,
    selected_user_ids: Iterable[int] = (),
) -> pd.DataFrame:
    train_dataset = task_data.get_train_dataset()
    if not isinstance(train_dataset, FrameDataset):
        raise TypeError(f"STARec requires a FrameDataset train split, got {type(train_dataset).__name__}.")
    train_frame = train_dataset.frame.copy().reset_index(drop=True)
    selected_user_set = {int(user_id) for user_id in selected_user_ids}
    if selected_user_set:
        train_frame = train_frame.loc[train_frame[USER_ID].isin(selected_user_set)].reset_index(drop=True)
    if train_frame.empty:
        result = train_frame.copy()
        result[SEEN_ITEM_IDS] = pd.Series(dtype=object)
        result[CANDIDATE_ITEM_IDS] = pd.Series(dtype=object)
        return result

    result = train_frame.copy()
    result[SEEN_ITEM_IDS] = _build_seen_item_ids(result)
    result[CANDIDATE_ITEM_IDS] = _generate_train_random_candidates(
        task_data,
        result,
        model_config=model_config,
        split="train",
    )
    return result


def finalize_starec_candidate_row(
    *,
    candidate_item_ids,
    target_item_id: int,
    split: str,
    row_index: int,
    recall_budget: int,
    has_gt: bool,
    fix_pos: int,
    shuffle: bool,
    candidate_seed: int,
    source_name: str,
) -> tuple[int, ...]:
    backbone_candidates = [int(item_id) for item_id in (candidate_item_ids or ())]
    recall_budget = int(recall_budget)
    has_gt = bool(has_gt)
    fix_pos = int(fix_pos)
    required = recall_budget - 1 if has_gt else recall_budget

    if has_gt and target_item_id in backbone_candidates:
        backbone_candidates.remove(target_item_id)
    if len(backbone_candidates) < required:
        raise ValueError(
            f"{source_name} candidate row {row_index} for split '{split}' only has {len(backbone_candidates)} items, "
            f"but recall_budget={recall_budget} requires {required} non-ground-truth candidates."
        )

    final_candidates = backbone_candidates[:required]
    if has_gt:
        if fix_pos == -1 or fix_pos == recall_budget - 1:
            final_candidates.append(target_item_id)
        elif fix_pos == 0:
            final_candidates = [target_item_id, *final_candidates]
        else:
            if not 0 <= fix_pos < recall_budget:
                raise ValueError(f"fix_pos={fix_pos} is out of range for recall_budget={recall_budget}.")
            final_candidates = [
                *final_candidates[:fix_pos],
                target_item_id,
                *final_candidates[fix_pos:],
            ]

    if bool(shuffle):
        rng = np.random.default_rng(_candidate_seed(candidate_seed, split=split, row_index=row_index))
        rng.shuffle(final_candidates)

    return tuple(int(item_id) for item_id in final_candidates)


def _build_seen_item_ids(frame: pd.DataFrame) -> list[tuple[int, ...]]:
    item_ids = frame[ITEM_ID].to_numpy()
    seen_item_ids: list[tuple[int, ...]] = [()] * len(frame)
    for _, positions in frame.groupby(USER_ID, sort=False).indices.items():
        history: list[int] = []
        seen_item_set: set[int] = set()
        for position in positions:
            row_position = int(position)
            seen_item_ids[row_position] = tuple(history)
            item_id = int(item_ids[row_position])
            if item_id not in seen_item_set:
                history.append(item_id)
                seen_item_set.add(item_id)
    return seen_item_ids


def _generate_train_random_candidates(
    task_data: BaseTaskDataset,
    frame: pd.DataFrame,
    *,
    model_config: STARecConfig,
    split: str,
) -> list[tuple[int, ...]]:
    num_items = int(task_data.get_num_items())
    all_item_ids = np.arange(num_items, dtype=np.int64)
    required = int(model_config.backbone_topk)
    if required <= 0:
        raise ValueError("model.backbone_topk must be a positive integer.")

    candidate_rows: list[tuple[int, ...]] = []
    for row_index, record in enumerate(frame.to_dict(orient="records")):
        if not is_positive_or_unlabeled_record(record, model_config=model_config):
            candidate_rows.append(())
            continue
        user_id = int(record[USER_ID])
        masked_item_ids = set(record.get(SEEN_ITEM_IDS, ()))
        available_item_ids = [int(item_id) for item_id in all_item_ids.tolist() if int(item_id) not in masked_item_ids]
        if len(available_item_ids) < required:
            raise ValueError(
                f"STARec train random candidate generation only has {len(available_item_ids)} unmasked items "
                f"for user {user_id}, but backbone_topk={required} is required."
            )
        rng = np.random.default_rng(
            _candidate_seed(model_config.candidate_seed, split=split, row_index=row_index, user_id=user_id)
        )
        sampled = rng.choice(
            np.asarray(available_item_ids, dtype=np.int64),
            size=required,
            replace=False,
        ).tolist()
        candidate_rows.append(tuple(int(item_id) for item_id in sampled))
    return candidate_rows


def _split_offset(split: str) -> int:
    if split == "train":
        return 20_000
    if split == "valid":
        return 0
    return 10_000


def _candidate_seed(candidate_seed: int, *, split: str, row_index: int, user_id: int = 0) -> int:
    return (int(candidate_seed) + int(user_id) + _split_offset(split) + int(row_index)) % (2**32)


def _frame_dataset_frame(dataset, *, split: str) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"STARec requires FrameDataset for split '{split}', got {type(dataset).__name__}.")
    return dataset.frame.copy().reset_index(drop=True)


def _ordered_user_ids(frame: pd.DataFrame) -> tuple[int, ...]:
    user_ids: list[int] = []
    seen_user_ids: set[int] = set()
    for user_id in frame[USER_ID].tolist():
        normalized_user_id = int(user_id)
        if normalized_user_id in seen_user_ids:
            continue
        user_ids.append(normalized_user_id)
        seen_user_ids.add(normalized_user_id)
    return tuple(user_ids)


def _candidate_user_ids(
    ordered_user_ids: tuple[int, ...],
    *,
    selected_user_count: int,
    candidate_seed: int,
) -> tuple[int, ...]:
    if selected_user_count == -1:
        return ordered_user_ids
    shuffled_user_ids = list(ordered_user_ids)
    random.Random(int(candidate_seed)).shuffle(shuffled_user_ids)
    return tuple(shuffled_user_ids)


def _candidate_history_user_ids(
    test_frame: pd.DataFrame,
    *,
    model_config: STARecConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, int]:
    selected_user_count = int(model_config.selected_user_count)
    if selected_user_count != -1 and selected_user_count <= 0:
        raise ValueError("selected_user_count must be -1 or a positive integer.")

    ordered_user_ids = _ordered_user_ids(test_frame)
    requested_user_ids = _load_selected_user_ids(model_config.selected_user_ids_path)
    if requested_user_ids is not None:
        if selected_user_count != -1:
            raise ValueError("selected_user_count must be -1 when selected_user_ids_path is set.")
        test_user_set = set(ordered_user_ids)
        missing_user_ids = [user_id for user_id in requested_user_ids if user_id not in test_user_set]
        if missing_user_ids:
            raise ValueError(
                "selected_user_ids_path contains user ids that are not present in the positive test split: "
                f"{missing_user_ids[:10]}"
            )
        ordered_user_ids = requested_user_ids

    candidate_user_ids = _candidate_user_ids(
        ordered_user_ids,
        selected_user_count=selected_user_count,
        candidate_seed=int(model_config.candidate_seed),
    )
    return ordered_user_ids, candidate_user_ids, requested_user_ids, selected_user_count


def _load_selected_user_ids(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"selected_user_ids_path does not exist: {path}")

    user_ids: list[int] = []
    seen_user_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            user_id = _parse_user_id_line(line, path=path, line_number=line_number)
            if user_id in seen_user_ids:
                raise ValueError(f"Duplicate user_id={user_id} in selected_user_ids_path: {path}")
            user_ids.append(user_id)
            seen_user_ids.add(user_id)
    if not user_ids:
        raise ValueError(f"selected_user_ids_path did not contain any user ids: {path}")
    return tuple(user_ids)


def _parse_user_id_line(line: str, *, path: Path, line_number: int) -> int:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        raw = line
    if isinstance(raw, dict):
        if USER_ID not in raw:
            raise ValueError(f"Line {line_number} in {path} must contain '{USER_ID}'.")
        raw = raw[USER_ID]
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Line {line_number} in {path} does not contain a valid integer user id.") from exc


def _group_positions_by_user(frame: pd.DataFrame) -> dict[int, list[int]]:
    if frame.empty:
        return {}
    return {
        int(user_id): [int(position) for position in positions]
        for user_id, positions in frame.groupby(USER_ID, sort=False).indices.items()
    }


def _user_frame(frame: pd.DataFrame, positions_by_user: dict[int, list[int]], user_id: int) -> pd.DataFrame:
    positions = positions_by_user.get(int(user_id))
    if positions is None:
        return frame.iloc[0:0].copy()
    return frame.iloc[positions].copy().reset_index(drop=True)


def _history_limited_user_frames(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    history_max_length: int | None,
    model_config: STARecConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for split_index, (split, frame) in enumerate((("train", train_frame), ("valid", valid_frame), ("test", test_frame))):
        if frame.empty:
            continue
        part = frame.copy().reset_index(drop=True)
        part["_starec_source_split"] = split
        part["_starec_source_order"] = range(len(part))
        part["_starec_split_order"] = split_index
        parts.append(part)
    if not parts:
        empty = train_frame.iloc[0:0].copy()
        return empty, valid_frame.iloc[0:0].copy(), test_frame.iloc[0:0].copy()

    timeline = pd.concat(parts, ignore_index=True, sort=False)
    timeline = timeline.sort_values(["_starec_split_order", "_starec_source_order"], kind="mergesort").reset_index(drop=True)
    positive_positions = timeline.index[positive_or_unlabeled_mask(timeline, model_config=model_config)].tolist()
    if not positive_positions:
        return train_frame.iloc[0:0].copy(), valid_frame.iloc[0:0].copy(), test_frame.iloc[0:0].copy()
    last_positive_position = int(positive_positions[-1])
    timeline = timeline.iloc[: last_positive_position + 1].copy()
    if history_max_length is not None:
        timeline = timeline.iloc[-int(history_max_length) :].copy()

    retained_train = _retained_split_frame(timeline, source_frame=train_frame, split="train")
    retained_valid = _retained_split_frame(timeline, source_frame=valid_frame, split="valid")
    retained_test = _retained_split_frame(timeline, source_frame=test_frame, split="test")
    return _with_recomputed_seen_item_ids(retained_train, retained_valid, retained_test)


def _retained_split_frame(timeline: pd.DataFrame, *, source_frame: pd.DataFrame, split: str) -> pd.DataFrame:
    retained = timeline.loc[timeline["_starec_source_split"] == split].copy()
    drop_columns = [column for column in retained.columns if column.startswith("_starec_")]
    retained = retained.drop(columns=drop_columns)
    if retained.empty:
        return source_frame.iloc[0:0].copy()
    return retained.reset_index(drop=True)


def _with_recomputed_seen_item_ids(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for split_index, (split, frame) in enumerate((("train", train_frame), ("valid", valid_frame), ("test", test_frame))):
        if frame.empty:
            continue
        part = frame.copy().reset_index(drop=True)
        part["_starec_source_split"] = split
        part["_starec_source_order"] = range(len(part))
        part["_starec_split_order"] = split_index
        parts.append(part)
    if not parts:
        return train_frame, valid_frame, test_frame

    timeline = pd.concat(parts, ignore_index=True, sort=False)
    timeline = timeline.sort_values(["_starec_split_order", "_starec_source_order"], kind="mergesort").reset_index(drop=True)
    seen_item_ids: list[tuple[int, ...]] = []
    history: list[int] = []
    seen_item_set: set[int] = set()
    for item_id in timeline[ITEM_ID].tolist():
        seen_item_ids.append(tuple(history))
        normalized_item_id = int(item_id)
        if normalized_item_id not in seen_item_set:
            history.append(normalized_item_id)
            seen_item_set.add(normalized_item_id)
    timeline[SEEN_ITEM_IDS] = seen_item_ids

    train_result = _retained_split_frame(timeline, source_frame=train_frame, split="train")
    valid_result = _retained_split_frame(timeline, source_frame=valid_frame, split="valid")
    test_result = _retained_split_frame(timeline, source_frame=test_frame, split="test")
    return train_result, valid_result, test_result


def _is_user_eligible(
    *,
    retained_train: pd.DataFrame,
    retained_valid: pd.DataFrame,
    retained_test: pd.DataFrame,
    train_init_interactions: int,
    history_min_length: int,
    require_train_warmup: bool,
    model_config: STARecConfig,
) -> bool:
    retained_history_length = len(retained_train) + len(retained_valid) + len(retained_test)
    if retained_history_length < int(history_min_length):
        return False
    if (
        positive_or_unlabeled(retained_valid, model_config=model_config).empty
        or positive_or_unlabeled(retained_test, model_config=model_config).empty
    ):
        return False
    if not require_train_warmup:
        return True
    if len(retained_train) < int(train_init_interactions) + 1:
        return False
    train_after_init = retained_train.iloc[int(train_init_interactions) :].copy()
    return not positive_or_unlabeled(train_after_init, model_config=model_config).empty


def _is_user_eligible_by_positions(
    *,
    train_positions: Iterable[int],
    valid_positions: Iterable[int],
    test_positions: Iterable[int],
    train_positive_mask: np.ndarray,
    history_max_length: int | None,
    train_init_interactions: int,
    history_min_length: int,
    require_train_warmup: bool,
) -> bool:
    train_position_list = [int(position) for position in train_positions]
    valid_count = len(tuple(valid_positions))
    test_count = len(tuple(test_positions))
    if valid_count == 0 or test_count == 0:
        return False

    retained_train_count, retained_valid_count, retained_test_count = _retained_split_lengths(
        train_count=len(train_position_list),
        valid_count=valid_count,
        test_count=test_count,
        history_max_length=history_max_length,
    )
    retained_history_length = retained_train_count + retained_valid_count + retained_test_count
    if retained_history_length < int(history_min_length):
        return False
    if retained_valid_count == 0 or retained_test_count == 0:
        return False
    if not require_train_warmup:
        return True
    if retained_train_count < int(train_init_interactions) + 1:
        return False

    retained_train_positions = train_position_list[-retained_train_count:]
    train_after_init_positions = retained_train_positions[int(train_init_interactions) :]
    return any(bool(train_positive_mask[position]) for position in train_after_init_positions)


def _retained_split_lengths(
    *,
    train_count: int,
    valid_count: int,
    test_count: int,
    history_max_length: int | None,
) -> tuple[int, int, int]:
    total = int(train_count) + int(valid_count) + int(test_count)
    retained = total if history_max_length is None else min(total, int(history_max_length))
    retained_test = min(int(test_count), retained)
    retained -= retained_test
    retained_valid = min(int(valid_count), retained)
    retained -= retained_valid
    retained_train = min(int(train_count), retained)
    return retained_train, retained_valid, retained_test


def _requires_train_warmup(model_config: STARecConfig) -> bool:
    if not bool(model_config.run_warmup):
        return False
    return not (bool(model_config.memory_load_path) and bool(model_config.skip_warmup_when_memory_loaded))


def _finalize_selected_user_ids(
    *,
    ordered_user_ids: tuple[int, ...],
    eligible_user_ids: tuple[int, ...],
    requested_user_ids: tuple[int, ...] | None,
    selected_user_count: int,
    candidate_seed: int,
) -> tuple[int, ...]:
    eligible_user_set = set(int(user_id) for user_id in eligible_user_ids)
    ordered_eligible_user_ids = tuple(user_id for user_id in ordered_user_ids if user_id in eligible_user_set)
    selected_user_ids = _select_eligible_user_ids(
        eligible_user_ids=ordered_eligible_user_ids,
        selected_user_count=selected_user_count,
        candidate_seed=int(candidate_seed),
    )
    if not selected_user_ids:
        raise ValueError(
            "STARec could not find any eligible users after applying positive target, train warmup, "
            "history_min_length, and history_max_length constraints."
        )
    if requested_user_ids is not None and tuple(selected_user_ids) != tuple(requested_user_ids):
        selected_user_set = set(selected_user_ids)
        missing_after_filters = [user_id for user_id in requested_user_ids if user_id not in selected_user_set]
        raise ValueError(
            "selected_user_ids_path contains user ids that are not eligible after STARec history filters: "
            f"{missing_after_filters[:10]}"
        )
    return selected_user_ids


def _select_eligible_user_ids(
    *,
    eligible_user_ids: tuple[int, ...],
    selected_user_count: int,
    candidate_seed: int,
) -> tuple[int, ...]:
    if selected_user_count == -1:
        return eligible_user_ids
    if selected_user_count <= 0:
        raise ValueError("selected_user_count must be -1 or a positive integer.")
    if len(eligible_user_ids) < selected_user_count:
        raise ValueError(
            f"STARec selected_user_count={selected_user_count} requested more users than the "
            f"{len(eligible_user_ids)} users eligible after history filters."
        )
    if selected_user_count == len(eligible_user_ids):
        return eligible_user_ids
    randomizer = random.Random(int(candidate_seed))
    sampled_user_ids = randomizer.sample(list(eligible_user_ids), selected_user_count)
    sampled_user_set = set(sampled_user_ids)
    return tuple(user_id for user_id in eligible_user_ids if user_id in sampled_user_set)


def _concat_like(frames: list[pd.DataFrame], template: pd.DataFrame) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return template.iloc[0:0].copy()
    return pd.concat(non_empty_frames, ignore_index=True, sort=False).reset_index(drop=True)


__all__ = [
    "build_train_candidate_frame",
    "build_history_limited_frames",
    "finalize_starec_candidate_row",
    "select_history_eligible_user_ids",
]
