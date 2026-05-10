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
        train_frame = _dataset_frame(model_datasets.train_dataset)
        self._same_target_index = _build_same_target_index(train_frame)
        self._full_train_frame = train_frame
        return model_datasets

    @property
    def same_target_index(self) -> dict[int, list[int]]:
        return getattr(self, "_same_target_index", {})

    @property
    def full_train_frame(self) -> pd.DataFrame | None:
        return getattr(self, "_full_train_frame", None)


def _build_same_target_index(train_frame: pd.DataFrame) -> dict[int, list[int]]:
    """Build mapping from target item_id to list of row indices in training frame."""
    same_target_index: dict[int, list[int]] = {}
    for idx, (_, row) in enumerate(train_frame.iterrows()):
        item_id = int(row[ITEM_ID])
        same_target_index.setdefault(item_id, []).append(idx)
    return {k: v for k, v in same_target_index.items() if len(v) >= 2}


class LARESTrainCollator(BaseCollator):
    """Collate LARES training records with semantic augmentation."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        batch = _pad_history_batch(feature_records)
        batch[ITEM_ID] = torch.as_tensor(feature_records[ITEM_ID].to_numpy(), dtype=torch.long)

        model_dataset = self.prepared_data
        if isinstance(model_dataset, LARESModelDataset):
            aug_ids, aug_lengths = _sample_augmentations(
                feature_records,
                model_dataset.same_target_index,
                model_dataset.full_train_frame,
                feature_records[HISTORY_ITEM_IDS],
                feature_records[ITEM_ID],
            )
        else:
            aug_ids, aug_lengths = _fallback_augmentations(feature_records)

        batch["aug_history_item_ids"] = aug_ids
        batch["aug_history_lengths"] = aug_lengths
        return batch


class LARESEvalCollator(BaseCollator):
    """Collate LARES evaluation records into padded history tensors."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _pad_history_batch(feature_records)


def _pad_history_batch(records: pd.DataFrame) -> dict[str, torch.Tensor]:
    history_items = [tuple(values) for values in records[HISTORY_ITEM_IDS].tolist()]
    history_lengths = torch.tensor([len(values) for values in history_items], dtype=torch.long)
    batch_size = len(records)
    max_length = int(torch.max(history_lengths).item()) if batch_size > 0 else 0
    padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row_index, item_history in enumerate(history_items):
        if len(item_history) > 0:
            padded[row_index, : len(item_history)] = torch.tensor(item_history, dtype=torch.long)
    return {
        HISTORY_ITEM_IDS: padded,
        "history_lengths": history_lengths,
    }


def _sample_augmentations(
    feature_records: pd.DataFrame,
    same_target_index: dict[int, list[int]],
    full_train_frame: pd.DataFrame | None,
    history_column: pd.Series,
    item_id_column: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample augmented sequences (same target item, different record)."""
    if full_train_frame is None:
        return _fallback_augmentations(feature_records)

    aug_seqs: list[tuple[int, ...]] = []
    for i, (_, row) in enumerate(feature_records.iterrows()):
        item_id = int(row[ITEM_ID])
        candidates = same_target_index.get(item_id, [])
        current_idx = i
        valid_candidates = [c for c in candidates if c != current_idx]
        if valid_candidates:
            pick_idx = valid_candidates[np.random.randint(0, len(valid_candidates))]
            aug_seqs.append(tuple(full_train_frame.iloc[pick_idx][HISTORY_ITEM_IDS]))
        else:
            aug_seqs.append(tuple(row[HISTORY_ITEM_IDS]))

    aug_lengths = torch.tensor([len(seq) for seq in aug_seqs], dtype=torch.long)
    batch_size = len(aug_seqs)
    max_length = int(torch.max(aug_lengths).item()) if batch_size > 0 else 0
    padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row_index, seq in enumerate(aug_seqs):
        if len(seq) > 0:
            padded[row_index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, aug_lengths


def _fallback_augmentations(
    feature_records: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback: use original sequence as augmentation."""
    history_items = [tuple(values) for values in feature_records[HISTORY_ITEM_IDS].tolist()]
    history_lengths = torch.tensor([len(values) for values in history_items], dtype=torch.long)
    batch_size = len(history_items)
    max_length = int(torch.max(history_lengths).item()) if batch_size > 0 else 0
    padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row_index, item_history in enumerate(history_items):
        if len(item_history) > 0:
            padded[row_index, : len(item_history)] = torch.tensor(item_history, dtype=torch.long)
    return padded, history_lengths


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(
            f"LARES model datasets require FrameDataset, got {type(dataset).__name__}."
        )
    return dataset.frame.copy()
