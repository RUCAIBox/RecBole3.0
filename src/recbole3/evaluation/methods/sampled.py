from __future__ import annotations

from typing import Any, Sequence

import torch

from recbole3.dataset import RetrievalEvalRequest
from recbole3.evaluation.metric import MetricSpec, RetrievalEvalData
from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod
from recbole3.model import BaseModel, BaseRetrievalModel


class SampledEvaluationMethod(BaseRetrievalEvaluationMethod):
    protocol = "sampled"

    def __init__(self, *, metric_specs: tuple[MetricSpec, ...]) -> None:
        super().__init__(metric_specs=metric_specs)

    def _collect_retrieval_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Sequence[RetrievalEvalRequest],
        max_k: int,
    ) -> RetrievalEvalData:
        if not isinstance(model, BaseRetrievalModel):
            raise TypeError("Sampled evaluation requires BaseRetrievalModel.")
        if any(record.candidate_item_ids is None for record in records):
            raise TypeError("Sampled evaluation requires retrieval records with non-null candidate_item_ids.")

        device = self._infer_device(model_inputs)
        target_item_ids, target_mask = self._single_target_tensors(records)
        if max_k <= 0:
            pred_item_ids = torch.empty((len(records), 0), dtype=torch.long)
        else:
            candidate_item_ids, candidate_mask = self._pad_int_lists(records, "candidate_item_ids", device=device)
            candidate_count = int(candidate_item_ids.shape[1])
            prediction_k = min(max_k, candidate_count)
            if prediction_k == 0:
                pred_item_ids = torch.empty((len(records), 0), dtype=torch.long)
            else:
                pred_item_ids = model.predict(
                    model_inputs,
                    k=prediction_k,
                    candidate_item_ids=candidate_item_ids,
                )
                pred_item_ids = self._normalize_pred_item_ids(pred_item_ids, len(records), prediction_k)
                pred_item_ids = self._pad_prediction_width(pred_item_ids, max_k)

                if candidate_count > 0:
                    valid_candidate_count = torch.sum(candidate_mask, dim=1)
                    if torch.any(valid_candidate_count < prediction_k):
                        raise ValueError(
                            "Sampled evaluation requires at least k valid candidates per row. "
                            f"Got k={prediction_k} with candidate counts {valid_candidate_count.tolist()}."
                        )

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

    @staticmethod
    def _pad_prediction_width(pred_item_ids: torch.Tensor, width: int) -> torch.Tensor:
        if pred_item_ids.shape[1] == width:
            return pred_item_ids
        padded = torch.full((pred_item_ids.shape[0], width), -1, dtype=torch.long, device=pred_item_ids.device)
        padded[:, : pred_item_ids.shape[1]] = pred_item_ids
        return padded
