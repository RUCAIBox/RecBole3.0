from __future__ import annotations

from typing import Any, Sequence

import torch

from recbole3.dataset import RetrievalEvalRequest
from recbole3.evaluation.metric import MetricSpec, RetrievalEvalData
from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod
from recbole3.model import BaseModel, BaseRetrievalModel


class FullEvaluationMethod(BaseRetrievalEvaluationMethod):
    protocol = "full"

    def __init__(self, *, metric_specs: tuple[MetricSpec, ...], exclude_history: bool) -> None:
        super().__init__(metric_specs=metric_specs)
        self.exclude_history = bool(exclude_history)

    def _collect_retrieval_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Sequence[RetrievalEvalRequest],
        max_k: int,
    ) -> RetrievalEvalData:
        if not isinstance(model, BaseRetrievalModel):
            raise TypeError("Full evaluation requires BaseRetrievalModel.")

        target_item_ids, target_mask = self._single_target_tensors(records)
        if max_k <= 0:
            pred_item_ids = torch.empty((len(records), 0), dtype=torch.long)
        else:
            device = self._infer_device(model_inputs)
            exclude_item_ids = None
            exclude_mask = None
            if self.exclude_history:
                exclude_item_ids, exclude_mask = self._pad_int_lists(records, "seen_item_ids", device=device)
            pred_item_ids = model.predict(
                model_inputs,
                k=max_k,
                candidate_item_ids=None,
                exclude_item_ids=exclude_item_ids,
                exclude_mask=exclude_mask,
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
