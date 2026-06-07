from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import torch

from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.lsrm.config import LSRMConfig
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS

ITEM_ID_OFFSET = 1
PAD_TOKEN = 0
LABEL_IGNORE = -100


class LSRMModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds history_item_ids for LSRM, filtering out empty-history records."""

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets:
        model_datasets = super()._build_model_datasets(model_config=model_config)
        train_frame = _filter_empty(_dataset_frame(model_datasets.train_dataset))
        valid_frame = _filter_empty(_dataset_frame(model_datasets.valid_dataset))
        test_frame = _filter_empty(_dataset_frame(model_datasets.test_dataset))
        return ModelDatasets(
            train_dataset=FrameDataset(train_frame),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )


def _filter_empty(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[HISTORY_ITEM_IDS].apply(len) > 0].reset_index(drop=True)


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"LSRM requires FrameDataset, got {type(dataset).__name__}.")
    return dataset.frame.copy()


class _LSRMBaseCollator(BaseCollator):
    config: LSRMConfig

    def __init__(self, config: LSRMConfig, prepared_data: Any):
        super().__init__(config, prepared_data)
        self.history_max_length = int(config.history_max_length)

    def _records(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if isinstance(feature_records, pd.DataFrame):
            return feature_records.to_dict("records")
        return list(feature_records)


class LSRMTrainCollator(_LSRMBaseCollator):
    """Collate LSRM training records into next-token prediction batches."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        all_input_ids: list[list[int]] = []
        all_attention_mask: list[list[int]] = []
        all_labels: list[list[int]] = []
        all_history_lengths: list[int] = []

        for record in rows:
            history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
            target_item_id = int(record[ITEM_ID])

            item_seq = list(history)
            if len(item_seq) > self.history_max_length:
                item_seq = item_seq[-self.history_max_length:]

            seq_len = len(item_seq)
            # offset item ids by ITEM_ID_OFFSET (0 is padding)
            input_ids = [item_id + ITEM_ID_OFFSET for item_id in item_seq]
            attention_mask = [1] * seq_len

            # Only compute loss at the last position (target item)
            labels = [LABEL_IGNORE] * (seq_len - 1) + [target_item_id + ITEM_ID_OFFSET]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_history_lengths.append(seq_len)

        # pad to max length in batch
        max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 0
        padded_input_ids = [ids + [PAD_TOKEN] * (max_len - len(ids)) for ids in all_input_ids]
        padded_attention_mask = [mask + [0] * (max_len - len(mask)) for mask in all_attention_mask]
        padded_labels = [labs + [LABEL_IGNORE] * (max_len - len(labs)) for labs in all_labels]

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "history_lengths": torch.tensor(all_history_lengths, dtype=torch.long),
        }


class LSRMEvalCollator(_LSRMBaseCollator):
    """Collate LSRM evaluation records into padded history tensors."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        all_input_ids: list[list[int]] = []
        all_attention_mask: list[list[int]] = []
        all_history_lengths: list[int] = []

        for record in rows:
            history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
            if len(history) > self.history_max_length:
                history = history[-self.history_max_length:]

            seq_len = len(history)
            input_ids = [item_id + ITEM_ID_OFFSET for item_id in history]
            attention_mask = [1] * seq_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_history_lengths.append(seq_len)

        max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 0
        padded_input_ids = [ids + [PAD_TOKEN] * (max_len - len(ids)) for ids in all_input_ids]
        padded_attention_mask = [mask + [0] * (max_len - len(mask)) for mask in all_attention_mask]

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "history_lengths": torch.tensor(all_history_lengths, dtype=torch.long),
        }


__all__ = [
    "LSRMModelDataset",
    "LSRMTrainCollator",
    "LSRMEvalCollator",
]
