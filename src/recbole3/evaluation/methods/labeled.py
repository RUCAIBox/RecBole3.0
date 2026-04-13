from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recbole3.dataset import LABEL, USER_ID
from recbole3.evaluation.methods.base import BaseRankingEvaluationMethod
from recbole3.evaluation.metric import RankingEvalData
from recbole3.model.base import BaseModel, BaseRankingModel


class LabeledEvaluationMethod(BaseRankingEvaluationMethod):
    protocol = "labeled"

    def collect_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: pd.DataFrame,
    ) -> RankingEvalData:
        if not isinstance(model, BaseRankingModel):
            raise TypeError("Labeled evaluation requires BaseRankingModel.")

        if records[LABEL].isna().any():
            raise TypeError("Labeled evaluation requires row-based labeled records with non-null label values.")
        labels = records[LABEL].to_numpy(dtype=np.float64, copy=False)
        group_ids = records[USER_ID].to_numpy(dtype=np.int64, copy=False)

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
