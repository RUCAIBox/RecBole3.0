from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID
from recbole3.model.base import BaseCollator, ModelDatasets
from recbole3.model.lares.config import LARESConfig, LARES_PADDING_ITEM_ID, ITEM_ID_OFFSET
from recbole3.model.sequential import HISTORY_ITEM_IDS, BaseSequentialModelDataset


class LARESModelDataset(BaseSequentialModelDataset):

    def _build_model_datasets(self, *, model_config: LARESConfig) -> ModelDatasets:
        model_datasets = super()._build_model_datasets(model_config=model_config)

        train_frame = _filter_empty(_dataset_frame(model_datasets.train_dataset))
        valid_frame = _filter_empty(_dataset_frame(model_datasets.valid_dataset))
        test_frame = _filter_empty(_dataset_frame(model_datasets.test_dataset))

        same_target = _build_item_maps(train_frame)
        train_frame["_row_idx"] = range(len(train_frame))

        self._same_target_index = same_target
        self._full_train_frame = train_frame

        return ModelDatasets(
            train_dataset=LARESFrameDataset(train_frame, same_target),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )

    @property
    def same_target_index(self) -> dict[int, list[int]]:
        return getattr(self, "_same_target_index", {})

    @property
    def full_train_frame(self) -> pd.DataFrame | None:
        return getattr(self, "_full_train_frame", None)


class LARESFrameDataset(FrameDataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        same_target: dict[int, list[int]],
    ) -> None:
        super().__init__(frame)
        self._same_target = same_target
        self._full_frame = self.frame

    def __getitems__(self, indices: list[int]) -> pd.DataFrame:
        batch = self.frame.take(indices).reset_index(drop=True)
        batch["aug_history_item_ids"] = _sample_augmentations(
            batch, self._same_target, self._full_frame,
        )
        return batch


class LARESTrainCollator(BaseCollator):

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        records = feature_records
        if "aug_history_item_ids" not in records.columns:
            ds = self.prepared_data
            records = records.copy()
            if isinstance(ds, LARESModelDataset) and ds.full_train_frame is not None:
                records["aug_history_item_ids"] = _sample_augmentations(
                    records, ds.same_target_index, ds.full_train_frame,
                )
            else:
                records["aug_history_item_ids"] = [
                    tuple(row[HISTORY_ITEM_IDS]) for _, row in records.iterrows()
                ]

        batch = _pad_history(records)
        batch[ITEM_ID] = torch.as_tensor(records[ITEM_ID].to_numpy(), dtype=torch.long) + ITEM_ID_OFFSET
        aug, aug_len = _pad_column(records, "aug_history_item_ids")
        batch["aug_history_item_ids"] = aug
        batch["aug_history_lengths"] = aug_len
        return batch


class LARESEvalCollator(BaseCollator):

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _pad_history(feature_records)


def _filter_empty(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[HISTORY_ITEM_IDS].apply(len) > 0].reset_index(drop=True)


def _build_item_maps(frame: pd.DataFrame) -> dict[int, list[int]]:
    """Return same_target_index mapping item_id -> row indices with repeated targets."""
    item_rows: dict[int, list[int]] = {}
    for i, (_, row) in enumerate(frame.iterrows()):
        item_rows.setdefault(int(row[ITEM_ID]), []).append(i)
    same_target = {k: v for k, v in item_rows.items() if len(v) >= 2}
    return same_target


def _sample_augmentations(
    batch: pd.DataFrame,
    same_target: dict[int, list[int]],
    full_frame: pd.DataFrame,
) -> list[tuple[int, ...]]:
    aug_seqs: list[tuple[int, ...]] = []
    for _, row in batch.iterrows():
        item_id = int(row[ITEM_ID])
        my_idx = int(row["_row_idx"])
        candidates = same_target.get(item_id, [])
        valid = [c for c in candidates if c != my_idx]
        if valid:
            pick = valid[np.random.randint(0, len(valid))]
            aug_seqs.append(tuple(full_frame.iloc[pick][HISTORY_ITEM_IDS]))
        else:
            aug_seqs.append(tuple(row[HISTORY_ITEM_IDS]))
    return aug_seqs


def _pad_column(records: pd.DataFrame, column: str) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [tuple(values) for values in records[column].tolist()]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    B = len(seqs)
    max_len = int(torch.max(lengths).item()) if B > 0 else 0
    padded = torch.full((B, max_len), LARES_PADDING_ITEM_ID, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if len(seq) > 0:
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long) + ITEM_ID_OFFSET
    return padded, lengths


def _pad_history(records: pd.DataFrame) -> dict[str, torch.Tensor]:
    padded, lengths = _pad_column(records, HISTORY_ITEM_IDS)
    return {HISTORY_ITEM_IDS: padded, "history_lengths": lengths}


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"LARES requires FrameDataset, got {type(dataset).__name__}.")
    return dataset.frame.copy()
