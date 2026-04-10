from __future__ import annotations

from typing import Any

import numpy as np

from recbole3.dataset import Interaction
from recbole3.evaluation.methods.base import BaseRankingEvaluationMethod
from recbole3.evaluation.metric import RankingEvalData
from recbole3.model.base import BaseModel, BaseRankingModel


class LabeledEvaluationMethod(BaseRankingEvaluationMethod):
    protocol = "labeled"

    def collect_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: list[Interaction],
    ) -> RankingEvalData:
        if not isinstance(model, BaseRankingModel):
            raise TypeError("Labeled evaluation requires BaseRankingModel.")

        labels = np.empty(len(records), dtype=np.float64)
        group_ids = np.empty(len(records), dtype=np.int64)
        for index, record in enumerate(records):
            if record.label is None:
                raise TypeError("Labeled evaluation requires row-based labeled records with non-null label values.")
            labels[index] = float(record.label)
            group_ids[index] = int(record.user_id)

        scores = model.predict(model_inputs).reshape(-1)
        if scores.numel() != len(labels):
            raise ValueError(
                "Labeled evaluation requires predict() to return one score per record. "
                f"Got {tuple(scores.shape)} for batch size {len(labels)}."
            )
        return RankingEvalData(
            scores=self._to_numpy(scores),
            labels=labels,
            group_ids=group_ids,
        )
