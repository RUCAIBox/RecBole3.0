from __future__ import annotations

import numpy as np
import pytest

from recbole3.evaluation import AUCMetric, GAUCMetric, NDCGMetric, RankingEvalData, RecallMetric, RetrievalEvalData



def test_gauc_weights_valid_groups_and_skips_single_class_groups() -> None:
    metric = GAUCMetric()
    eval_data = RankingEvalData(
        scores=np.array([0.65, 0.10, 0.20, 0.90, 0.70, 0.10]),
        labels=np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        group_ids=np.array([20, 10, 10, 30, 20, 20]),
    )

    result = metric.compute(eval_data)

    assert result["gauc"] == pytest.approx(0.7)



def test_gauc_returns_zero_when_no_group_has_both_classes() -> None:
    metric = GAUCMetric()
    eval_data = RankingEvalData(
        scores=np.array([0.20, 0.30, 0.40]),
        labels=np.array([1.0, 1.0, 0.0]),
        group_ids=np.array([1, 1, 2]),
    )

    result = metric.compute(eval_data)

    assert result == {"gauc": 0.0}



def test_auc_handles_tied_scores_consistently() -> None:
    metric = AUCMetric()
    eval_data = RankingEvalData(
        scores=np.array([0.8, 0.8, 0.2, 0.2]),
        labels=np.array([1.0, 0.0, 1.0, 0.0]),
        group_ids=np.array([0, 0, 0, 0]),
    )

    result = metric.compute(eval_data)

    assert result["auc"] == pytest.approx(0.5)



def test_retrieval_metrics_ignore_rows_without_targets() -> None:
    eval_data = RetrievalEvalData(
        pred_item_ids=np.array([[1, 2, 3], [4, 5, 6]]),
        target_item_ids=np.array([[2, 9], [0, 0]]),
        target_mask=np.array([[True, False], [False, False]]),
    )

    recall = RecallMetric((1, 2, 3)).compute(eval_data)
    ndcg = NDCGMetric((1, 2, 3)).compute(eval_data)

    discount = 1.0 / np.log2(3.0)
    assert recall == {"recall@1": 0.0, "recall@2": 1.0, "recall@3": 1.0}
    assert ndcg["ndcg@1"] == pytest.approx(0.0)
    assert ndcg["ndcg@2"] == pytest.approx(discount)
    assert ndcg["ndcg@3"] == pytest.approx(discount)



def test_retrieval_metrics_reject_predictions_shorter_than_requested_k() -> None:
    eval_data = RetrievalEvalData(
        pred_item_ids=np.array([[1, 2]]),
        target_item_ids=np.array([[2]]),
        target_mask=np.array([[True]]),
    )

    with pytest.raises(ValueError, match="at least 3 columns"):
        RecallMetric((3,)).compute(eval_data)

    with pytest.raises(ValueError, match="at least 3 columns"):
        NDCGMetric((3,)).compute(eval_data)



def test_retrieval_metrics_require_matching_target_shapes() -> None:
    eval_data = RetrievalEvalData(
        pred_item_ids=np.array([[1, 2, 3]]),
        target_item_ids=np.array([[2, 0]]),
        target_mask=np.array([[True]]),
    )

    with pytest.raises(ValueError, match="matching shapes"):
        RecallMetric((1,)).compute(eval_data)
