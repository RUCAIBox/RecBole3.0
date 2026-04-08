from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


EvalProtocol = Literal["labeled", "sampled", "full"]


@dataclass(slots=True)
class RankingEvalData:
    scores: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray


@dataclass(slots=True)
class RetrievalEvalData:
    pred_item_ids: np.ndarray
    target_item_ids: np.ndarray
    target_mask: np.ndarray


@dataclass(slots=True)
class MetricSpec:
    """Configuration for one metric family and its optional top-k expansion."""

    name: str = field(default="", metadata={"help": "Built-in metric name such as auc, gauc, logloss, recall, or ndcg."})
    ks: tuple[int, ...] = field(default_factory=tuple, metadata={"help": "Top-k cutoffs used by retrieval metrics."})


class BaseMetric(ABC):
    """Metric interface consumed by trainer evaluation loops."""

    name: str
    higher_is_better: bool = True

    def result_directions(self) -> dict[str, bool]:
        return {self.name: self.higher_is_better}


class BaseRankingMetric(BaseMetric):
    """Metric that consumes flat ranking predictions and labels."""

    @abstractmethod
    def compute(self, eval_data: RankingEvalData) -> dict[str, float]:
        """Compute one or more scalar metric values from labeled evaluation data."""


class BaseRetrievalMetric(BaseMetric):
    """Metric that consumes ordered top-k retrieval predictions."""

    def __init__(self, name: str, ks: tuple[int, ...], *, higher_is_better: bool = True) -> None:
        self.name = name
        self.ks = _require_ks(name, ks)
        self.higher_is_better = higher_is_better

    def result_directions(self) -> dict[str, bool]:
        return {f"{self.name}@{k}": self.higher_is_better for k in self.ks}

    @abstractmethod
    def compute(self, eval_data: RetrievalEvalData) -> dict[str, float]:
        """Compute one or more scalar metric values from retrieval evaluation data."""


class AUCMetric(BaseRankingMetric):
    """Area under the ROC curve for binary labels."""

    name = "auc"
    higher_is_better = True

    def compute(self, eval_data: RankingEvalData) -> dict[str, float]:
        scores = np.asarray(eval_data.scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(eval_data.labels, dtype=np.float64).reshape(-1)
        return {self.name: _compute_auc(scores, labels)}


class GAUCMetric(BaseRankingMetric):
    """Group AUC averaged over groups with both positive and negative labels."""

    name = "gauc"
    higher_is_better = True

    def compute(self, eval_data: RankingEvalData) -> dict[str, float]:
        scores = np.asarray(eval_data.scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(eval_data.labels, dtype=np.float64).reshape(-1)
        group_ids = np.asarray(eval_data.group_ids).reshape(-1)
        if len(scores) != len(labels) or len(scores) != len(group_ids):
            raise ValueError("Metric 'gauc' requires scores, labels, and group_ids with matching lengths.")

        if len(scores) == 0:
            return {self.name: 0.0}

        order = np.argsort(group_ids, kind="mergesort")
        sorted_group_ids = group_ids[order]
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        starts, ends = _segment_boundaries(sorted_group_ids)
        lengths = ends - starts
        positives = np.add.reduceat((sorted_labels > 0).astype(np.int64), starts)
        valid_mask = (positives > 0) & (positives < lengths)
        if not np.any(valid_mask):
            return {self.name: 0.0}

        valid_starts = starts[valid_mask]
        valid_ends = ends[valid_mask]
        aucs = np.fromiter(
            (_compute_auc(sorted_scores[start:end], sorted_labels[start:end]) for start, end in zip(valid_starts, valid_ends, strict=True)),
            dtype=np.float64,
            count=len(valid_starts),
        )
        weights = lengths[valid_mask].astype(np.float64)
        return {self.name: float(np.average(aucs, weights=weights))}


class LogLossMetric(BaseRankingMetric):
    """Binary log loss that treats scores as probabilities."""

    name = "logloss"
    higher_is_better = False

    def compute(self, eval_data: RankingEvalData) -> dict[str, float]:
        scores = np.asarray(eval_data.scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(eval_data.labels, dtype=np.float64).reshape(-1)
        if len(scores) != len(labels):
            raise ValueError("Metric 'logloss' requires scores and labels with matching lengths.")
        if len(scores) == 0:
            return {self.name: 0.0}
        eps = 1e-12
        probabilities = np.clip(scores, eps, 1.0 - eps)
        losses = -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
        return {self.name: float(np.mean(losses))}


class RecallMetric(BaseRetrievalMetric):
    """Recall@K averaged over grouped retrieval examples."""

    def __init__(self, ks: tuple[int, ...]) -> None:
        super().__init__(name="recall", ks=ks)

    def compute(self, eval_data: RetrievalEvalData) -> dict[str, float]:
        pred_item_ids, target_item_ids, target_mask = _normalize_retrieval_eval_data(eval_data, self.name)
        _require_prediction_width(pred_item_ids, max(self.ks), self.name)
        relevant_count = np.sum(target_mask, axis=1)
        valid_rows = relevant_count > 0
        if not np.any(valid_rows):
            return {f"{self.name}@{k}": 0.0 for k in self.ks}

        results: dict[str, float] = {}
        for k in self.ks:
            hit_count = _compute_hit_count(pred_item_ids[:, :k], target_item_ids, target_mask)
            recall = hit_count[valid_rows] / relevant_count[valid_rows]
            results[f"{self.name}@{k}"] = float(np.mean(recall))
        return results


class NDCGMetric(BaseRetrievalMetric):
    """Normalized discounted cumulative gain at K for binary relevance."""

    def __init__(self, ks: tuple[int, ...]) -> None:
        super().__init__(name="ndcg", ks=ks)

    def compute(self, eval_data: RetrievalEvalData) -> dict[str, float]:
        pred_item_ids, target_item_ids, target_mask = _normalize_retrieval_eval_data(eval_data, self.name)
        _require_prediction_width(pred_item_ids, max(self.ks), self.name)
        relevant_count = np.sum(target_mask, axis=1)
        valid_rows = relevant_count > 0
        if not np.any(valid_rows):
            return {f"{self.name}@{k}": 0.0 for k in self.ks}

        results: dict[str, float] = {}
        for k in self.ks:
            relevance = _compute_binary_relevance(pred_item_ids[:, :k], target_item_ids, target_mask)
            discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
            dcg = np.sum(relevance * discounts, axis=1)

            ideal_limit = np.minimum(relevant_count, k).astype(np.int64)
            discount_prefix = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(discounts)))
            ideal_dcg = discount_prefix[ideal_limit]

            ndcg = np.divide(dcg, ideal_dcg, out=np.zeros_like(dcg), where=ideal_dcg > 0)
            results[f"{self.name}@{k}"] = float(np.mean(ndcg[valid_rows]))
        return results


def create_builtin_metrics(metric_specs: tuple[MetricSpec, ...], protocol: EvalProtocol) -> list[BaseMetric]:
    """Instantiate built-in metrics for one evaluation protocol."""

    metrics: list[BaseMetric] = []
    for spec in metric_specs:
        metric_name = spec.name.strip().lower()
        if not metric_name:
            raise ValueError("Metric names cannot be empty.")

        if metric_name == "auc":
            _require_protocol(metric_name, protocol, {"labeled"})
            metrics.append(AUCMetric())
            continue
        if metric_name == "gauc":
            _require_protocol(metric_name, protocol, {"labeled"})
            metrics.append(GAUCMetric())
            continue
        if metric_name == "logloss":
            _require_protocol(metric_name, protocol, {"labeled"})
            metrics.append(LogLossMetric())
            continue
        if metric_name == "recall":
            _require_protocol(metric_name, protocol, {"sampled", "full"})
            metrics.append(RecallMetric(spec.ks))
            continue
        if metric_name == "ndcg":
            _require_protocol(metric_name, protocol, {"sampled", "full"})
            metrics.append(NDCGMetric(spec.ks))
            continue
        raise ValueError(f"Unsupported metric '{spec.name}'.")
    return metrics


def _require_protocol(metric_name: str, protocol: EvalProtocol, supported_protocols: set[str]) -> None:
    if protocol not in supported_protocols:
        supported = ", ".join(sorted(supported_protocols))
        raise ValueError(f"Metric '{metric_name}' does not support eval protocol '{protocol}'. Supported: {supported}.")


def _require_ks(metric_name: str, ks: tuple[int, ...]) -> tuple[int, ...]:
    if not ks:
        raise ValueError(f"Metric '{metric_name}' requires at least one value in `ks`.")
    normalized = tuple(int(k) for k in ks)
    if any(k <= 0 for k in normalized):
        raise ValueError(f"Metric '{metric_name}' requires all `ks` values to be positive.")
    return normalized


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) != len(labels):
        raise ValueError("AUC requires scores and labels with matching lengths.")
    if len(scores) == 0:
        return 0.0

    positive_mask = labels > 0
    positive_count = int(np.sum(positive_mask))
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0

    ranks = _average_ranks(scores)
    positive_rank_sum = float(np.sum(ranks[positive_mask]))
    numerator = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    denominator = float(positive_count * negative_count)
    return numerator / denominator


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    if len(sorted_values) == 0:
        return ranks

    starts, ends = _segment_boundaries(sorted_values)
    average_ranks = 0.5 * (starts + 1 + ends)
    ranks[order] = np.repeat(average_ranks, ends - starts)
    return ranks


def _segment_boundaries(sorted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(sorted_values) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    change_points = np.flatnonzero(np.diff(sorted_values)) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change_points.astype(np.int64, copy=False)))
    ends = np.concatenate((change_points.astype(np.int64, copy=False), np.array([len(sorted_values)], dtype=np.int64)))
    return starts, ends


def _normalize_retrieval_eval_data(
    eval_data: RetrievalEvalData,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_item_ids = _ensure_2d_array(np.asarray(eval_data.pred_item_ids), name="pred_item_ids")
    target_item_ids = _ensure_2d_array(np.asarray(eval_data.target_item_ids), name="target_item_ids")
    target_mask = _ensure_2d_array(np.asarray(eval_data.target_mask, dtype=bool), name="target_mask")

    if len(pred_item_ids) != len(target_item_ids) or len(pred_item_ids) != len(target_mask):
        raise ValueError(
            f"Metric '{metric_name}' requires pred_item_ids, target_item_ids, and target_mask with matching batch sizes."
        )
    if target_item_ids.shape != target_mask.shape:
        raise ValueError(
            f"Metric '{metric_name}' requires target_item_ids and target_mask with matching shapes."
        )
    return pred_item_ids, target_item_ids, target_mask


def _ensure_2d_array(value: np.ndarray, *, name: str) -> np.ndarray:
    if value.ndim == 2:
        return value
    if value.ndim == 1:
        if value.size == 0:
            return value.reshape(0, 0)
        return value.reshape(1, -1)
    raise ValueError(f"Expected '{name}' to be a 1D or 2D array, got shape {value.shape}.")


def _require_prediction_width(pred_item_ids: np.ndarray, required_k: int, metric_name: str) -> None:
    if pred_item_ids.shape[1] < required_k:
        raise ValueError(
            f"Metric '{metric_name}' requires pred_item_ids with at least {required_k} columns, got {pred_item_ids.shape[1]}."
        )


def _compute_binary_relevance(
    pred_item_ids: np.ndarray,
    target_item_ids: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    return np.any((pred_item_ids[:, :, None] == target_item_ids[:, None, :]) & target_mask[:, None, :], axis=2).astype(
        np.float64
    )


def _compute_hit_count(
    pred_item_ids: np.ndarray,
    target_item_ids: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    return np.sum(_compute_binary_relevance(pred_item_ids, target_item_ids, target_mask), axis=1)


__all__ = [
    "AUCMetric",
    "BaseMetric",
    "BaseRankingMetric",
    "BaseRetrievalMetric",
    "EvalProtocol",
    "GAUCMetric",
    "LogLossMetric",
    "MetricSpec",
    "NDCGMetric",
    "RankingEvalData",
    "RecallMetric",
    "RetrievalEvalData",
    "create_builtin_metrics",
]
