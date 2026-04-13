from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import torch

from recbole3.dataset import CANDIDATE_ITEM_IDS, PAD_ITEM_ID
from recbole3.evaluation.metric import MetricSpec, RetrievalEvalData
from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod
from recbole3.model.base import BaseModel, BaseRetrievalModel


class SampledEvaluationMethod(BaseRetrievalEvaluationMethod):
    protocol = "sampled"

    def __init__(self, *, metric_specs: tuple[MetricSpec, ...]) -> None:
        super().__init__(metric_specs=metric_specs)

    def _collect_retrieval_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Sequence[Any] | pd.DataFrame,
        max_k: int,
    ) -> RetrievalEvalData:
        if not isinstance(model, BaseRetrievalModel):
            raise TypeError("Sampled evaluation requires BaseRetrievalModel.")
        if isinstance(records, pd.DataFrame):
            missing_candidates = CANDIDATE_ITEM_IDS not in records.columns or records[CANDIDATE_ITEM_IDS].isna().any()
        else:
            missing_candidates = any(self._record_value(record, CANDIDATE_ITEM_IDS) is None for record in records)
        if missing_candidates:
            raise TypeError("Sampled evaluation requires retrieval records with non-null candidate_item_ids.")

        device = self._infer_device(model_inputs)
        target_item_ids, target_mask = self._single_target_tensors(records)
        if len(records) == 0:
            pred_item_ids = torch.empty((0, max(0, max_k)), dtype=torch.long)
        elif max_k <= 0:
            pred_item_ids = torch.empty((len(records), 0), dtype=torch.long)
        else:
            candidate_item_ids, candidate_mask = self._pad_int_lists(records, CANDIDATE_ITEM_IDS, device=device)
            valid_candidate_count = torch.sum(candidate_mask, dim=1)
            if torch.any(valid_candidate_count != valid_candidate_count[0]):
                raise ValueError("Sampled evaluation requires equal candidate counts in every row.")
            candidate_count = int(valid_candidate_count[0].item()) if len(valid_candidate_count) > 0 else 0
            if candidate_count < max_k:
                raise ValueError(
                    "Sampled evaluation requires at least k candidates per row. "
                    f"Got k={max_k} with candidate count {candidate_count}."
                )
            if candidate_count > 0 and torch.any(candidate_item_ids == PAD_ITEM_ID):
                raise ValueError("Sampled evaluation candidate_item_ids must not contain PAD_ITEM_ID=0.")
            pred_item_ids = model.predict(
                model_inputs,
                k=max_k,
                candidate_item_ids=candidate_item_ids,
            )
            pred_item_ids = self._normalize_pred_item_ids(pred_item_ids, len(records), max_k)

        return RetrievalEvalData(
            pred_item_ids=self._to_numpy(pred_item_ids),
            target_item_ids=self._to_numpy(target_item_ids),
            target_mask=self._to_numpy(target_mask),
        )

    @staticmethod
    def _normalize_pred_item_ids(pred_item_ids: torch.Tensor, batch_size: int, width: int) -> torch.Tensor:
        if pred_item_ids.ndim != 2 or tuple(pred_item_ids.shape) != (batch_size, width):
            raise ValueError(
                "Retrieval predict() must return top-k item ids with shape [batch, k]. "
                f"Got {tuple(pred_item_ids.shape)} for expected {(batch_size, width)}."
            )
        return pred_item_ids.to(dtype=torch.long)
