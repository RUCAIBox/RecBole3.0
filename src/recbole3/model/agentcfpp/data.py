from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch

from recbole3.dataset.base import BaseTaskDataset
from recbole3.dataset.utils import ITEM_ID, USER_ID
from recbole3.model.base import BaseCollator, ModelDatasets
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS


class AgentCFPPTrainCollator(BaseCollator):
    """Produces (user, pos_item) batches for AgentCF++ training.

    Negative sampling is domain-aware and handled inside the model's train_step
    (which has access to per-domain candidate pools), so this collator only
    surfaces the positive interactions plus optional history.
    """

    def __init__(self, config: Any, prepared_data: BaseTaskDataset):
        super().__init__(config, prepared_data)

    def __call__(self, feature_records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(feature_records, pd.DataFrame):
            records = feature_records.reset_index(drop=True)
        else:
            records = pd.DataFrame(feature_records)

        user_ids = torch.tensor(records[USER_ID].tolist(), dtype=torch.long)
        pos_item_ids = torch.tensor(records[ITEM_ID].tolist(), dtype=torch.long)

        history_item_ids = None
        if HISTORY_ITEM_IDS in records.columns:
            history_item_ids = records[HISTORY_ITEM_IDS].tolist()

        return {
            "user_ids": user_ids,
            "pos_item_ids": pos_item_ids,
            "history_item_ids": history_item_ids,
        }


class AgentCFPPEvalCollator(BaseCollator):
    """Produces evaluation batches with user ids, history, and the raw records."""

    def __init__(self, config: Any, prepared_data: BaseTaskDataset):
        super().__init__(config, prepared_data)

    def __call__(self, feature_records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(feature_records, pd.DataFrame):
            records = feature_records.reset_index(drop=True)
        else:
            records = pd.DataFrame(feature_records)

        user_ids = torch.tensor(records[USER_ID].tolist(), dtype=torch.long)

        history_item_ids = None
        if HISTORY_ITEM_IDS in records.columns:
            history_item_ids = records[HISTORY_ITEM_IDS].tolist()

        return {
            "user_ids": user_ids,
            "history_item_ids": history_item_ids,
            "records": records,
        }


class AgentCFPPModelDataset(BaseSequentialModelDataset):
    """Adds history_item_ids for sequential context in AgentCF++."""

    def _build_model_datasets(self, *, model_config: Any) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        return super()._build_model_datasets(model_config=model_config)


__all__ = [
    "AgentCFPPTrainCollator",
    "AgentCFPPEvalCollator",
    "AgentCFPPModelDataset",
]
