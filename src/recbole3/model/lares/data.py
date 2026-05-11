from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID
from recbole3.model.base import BaseCollator, BaseModelDataset, ModelDatasets
from recbole3.model.lares.config import LARESConfig
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.model.sequential import BaseSequentialModelDataset


class LARESModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds same-target-item augmentation mapping."""

    def _build_model_datasets(
        self,
        *,
        model_config: LARESConfig,
    ) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        model_datasets = super()._build_model_datasets(model_config=model_config)

        train_frame = _filter_empty_histories(_dataset_frame(model_datasets.train_dataset))
        valid_frame = _filter_empty_histories(_dataset_frame(model_datasets.valid_dataset))
        test_frame = _filter_empty_histories(_dataset_frame(model_datasets.test_dataset))

        self._same_target_index = _build_same_target_index(train_frame)
        train_frame["aug_history_item_ids"] = _precompute_augmentations(train_frame, self._same_target_index)
        self._full_train_frame = train_frame

        return ModelDatasets(
            train_dataset=FrameDataset(train_frame),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )

    @property
    def same_target_index(self) -> dict[int, list[int]]:
        return getattr(self, "_same_target_index", {})

    @property
    def full_train_frame(self) -> pd.DataFrame | None:
        return getattr(self, "_full_train_frame", None)


def _filter_empty_histories(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[HISTORY_ITEM_IDS].apply(len) > 0].reset_index(drop=True)


def _build_same_target_index(train_frame: pd.DataFrame) -> dict[int, list[int]]:
    """Build mapping from target item_id to list of row indices in training frame."""
    same_target_index: dict[int, list[int]] = {}
    for idx, (_, row) in enumerate(train_frame.iterrows()):
        item_id = int(row[ITEM_ID])
        same_target_index.setdefault(item_id, []).append(idx)
    return {k: v for k, v in same_target_index.items() if len(v) >= 2}


def _precompute_augmentations(
    train_frame: pd.DataFrame,
    same_target_index: dict[int, list[int]],
) -> list[tuple[int, ...]]:
    aug_seqs: list[tuple[int, ...]] = []
    for idx, (_, row) in enumerate(train_frame.iterrows()):
        item_id = int(row[ITEM_ID])
        candidates = same_target_index.get(item_id, [])
        valid = [c for c in candidates if c != idx]
        if valid:
            pick = valid[np.random.randint(0, len(valid))]
            aug_seqs.append(tuple(train_frame.iloc[pick][HISTORY_ITEM_IDS]))
        else:
            aug_seqs.append(tuple(row[HISTORY_ITEM_IDS]))
    return aug_seqs


class LARESTrainCollator(BaseCollator):
    """Collate LARES training records with semantic augmentation."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        batch = _pad_history_batch(feature_records)
        batch[ITEM_ID] = torch.as_tensor(feature_records[ITEM_ID].to_numpy(), dtype=torch.long)
        aug_padded, aug_lengths = _pad_sequence_column(feature_records, "aug_history_item_ids")
        batch["aug_history_item_ids"] = aug_padded
        batch["aug_history_lengths"] = aug_lengths
        return batch


class LARESEvalCollator(BaseCollator):
    """Collate LARES evaluation records into padded history tensors."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _pad_history_batch(feature_records)


def _pad_sequence_column(records: pd.DataFrame, column: str) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [tuple(values) for values in records[column].tolist()]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    B = len(seqs)
    max_len = int(torch.max(lengths).item()) if B > 0 else 0
    padded = torch.zeros((B, max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        if len(seq) > 0:
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, lengths


def _pad_history_batch(records: pd.DataFrame) -> dict[str, torch.Tensor]:
    padded, lengths = _pad_sequence_column(records, HISTORY_ITEM_IDS)
    return {HISTORY_ITEM_IDS: padded, "history_lengths": lengths}


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(
            f"LARES model datasets require FrameDataset, got {type(dataset).__name__}."
        )
    return dataset.frame.copy()
