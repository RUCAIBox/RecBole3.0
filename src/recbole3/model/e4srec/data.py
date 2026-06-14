from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.e4srec.config import E4SRecConfig
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS

ITEM_ID_OFFSET = 1
PAD_TOKEN = 0


def _records(
    feature_records: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(feature_records, pd.DataFrame):
        return feature_records.to_dict("records")
    return list(feature_records)


class E4SRecModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds history_item_ids and filters empty histories."""

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets:
        model_datasets = super()._build_model_datasets(model_config=model_config)
        return ModelDatasets(
            train_dataset=FrameDataset(_filter_empty(_frame(model_datasets.train_dataset))),
            valid_dataset=FrameDataset(_filter_empty(_frame(model_datasets.valid_dataset))),
            test_dataset=FrameDataset(_filter_empty(_frame(model_datasets.test_dataset))),
        )


def _filter_empty(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[HISTORY_ITEM_IDS].apply(len) > 0].reset_index(drop=True)


def _frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"E4SRec requires FrameDataset, got {type(dataset).__name__}.")
    return dataset.frame.copy()


class E4SRecCollator(BaseCollator):
    """Collate records into right-padded item sequences with optional labels.

    Item IDs are 1-indexed internally so that 0 is reserved for padding.
    Set ``include_labels=True`` for training, ``False`` for evaluation.
    """

    config: E4SRecConfig

    def __init__(
        self,
        config: E4SRecConfig,
        prepared_data: Any,
        *,
        include_labels: bool = True,
    ) -> None:
        super().__init__(config, prepared_data)
        self.history_max_length = int(config.history_max_length)
        self.include_labels = include_labels

    def __call__(
        self,
        feature_records: pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        rows = _records(feature_records)
        all_input_ids: list[list[int]] = []
        all_attention_mask: list[list[int]] = []
        all_labels: list[int] = []

        for record in rows:
            history = tuple(
                int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ())
            )
            item_seq = list(history)
            if len(item_seq) > self.history_max_length:
                item_seq = item_seq[-self.history_max_length:]

            seq_len = len(item_seq)
            # 1-indexed item IDs (0 = padding)
            all_input_ids.append([item_id + ITEM_ID_OFFSET for item_id in item_seq])
            all_attention_mask.append([1] * seq_len)
            if self.include_labels:
                all_labels.append(int(record[ITEM_ID]) + ITEM_ID_OFFSET)

        # Right-pad to batch max length
        max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 0
        padded_input_ids = [
            ids + [PAD_TOKEN] * (max_len - len(ids)) for ids in all_input_ids
        ]
        padded_mask = [
            mask + [0] * (max_len - len(mask)) for mask in all_attention_mask
        ]

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }
        if self.include_labels:
            result["labels"] = torch.tensor(all_labels, dtype=torch.long)
        return result


__all__ = [
    "E4SRecCollator",
    "E4SRecModelDataset",
    "ITEM_ID_OFFSET",
    "PAD_TOKEN",
]
