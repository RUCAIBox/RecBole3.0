from __future__ import annotations

from typing import Any

import pandas as pd

from recbole3.dataset import LABEL


def positive_or_unlabeled(frame: pd.DataFrame, *, model_config: Any) -> pd.DataFrame:
    return frame.loc[positive_or_unlabeled_mask(frame, model_config=model_config)].copy()


def positive_or_unlabeled_mask(frame: pd.DataFrame, *, model_config: Any) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    label_mask = _label_positive_or_unlabeled_mask(frame)
    score_field = str(getattr(model_config, "feedback_score_field", "") or "").strip()
    if not score_field or score_field not in frame.columns:
        return label_mask

    raw_scores = frame[score_field]
    scores = pd.to_numeric(raw_scores, errors="coerce")
    invalid_mask = raw_scores.notna() & scores.isna()
    if invalid_mask.any():
        bad_index = int(invalid_mask[invalid_mask].index[0])
        raise ValueError(f"Configured feedback_score_field '{score_field}' has a non-numeric value at row {bad_index}.")

    score_mask = scores > float(getattr(model_config, "feedback_positive_threshold", 0.0))
    return score_mask.where(scores.notna(), label_mask)


def is_positive_or_unlabeled_record(record: dict[str, Any], *, model_config: Any) -> bool:
    score_field = str(getattr(model_config, "feedback_score_field", "") or "").strip()
    if score_field and score_field in record:
        score = _optional_float(record.get(score_field), field_name=score_field)
        if score is not None:
            return score > float(getattr(model_config, "feedback_positive_threshold", 0.0))
    return _label_positive_or_unlabeled_record(record)


def feedback_label(record: dict[str, Any], *, model_config: Any) -> str:
    return "liked" if is_positive_or_unlabeled_record(record, model_config=model_config) else "disliked"


def actual_feedback(record: dict[str, Any], *, model_config: Any) -> str:
    return "Actually Liked" if is_positive_or_unlabeled_record(record, model_config=model_config) else "Actually Disliked"


def feedback_numeric_value(record: dict[str, Any], *, model_config: Any) -> float | None:
    score_field = str(getattr(model_config, "feedback_score_field", "") or "").strip()
    if score_field and score_field in record:
        score = _optional_float(record.get(score_field), field_name=score_field)
        if score is not None:
            return score
    if LABEL not in record:
        return None
    return _optional_float(record.get(LABEL), field_name=LABEL)


def _label_positive_or_unlabeled_mask(frame: pd.DataFrame) -> pd.Series:
    if LABEL not in frame.columns:
        return pd.Series(True, index=frame.index)
    labels = pd.to_numeric(frame[LABEL], errors="coerce")
    return frame[LABEL].isna() | (labels > 0)


def _label_positive_or_unlabeled_record(record: dict[str, Any]) -> bool:
    if LABEL not in record:
        return True
    value = record.get(LABEL)
    if value is None or pd.isna(value):
        return True
    return float(value) > 0


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configured feedback field '{field_name}' has a non-numeric value: {value!r}.") from exc


__all__ = [
    "actual_feedback",
    "feedback_label",
    "feedback_numeric_value",
    "is_positive_or_unlabeled_record",
    "positive_or_unlabeled",
    "positive_or_unlabeled_mask",
]
