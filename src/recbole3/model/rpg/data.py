from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from recbole3.dataset import FrameDataset, ITEM_ID, SEEN_ITEM_IDS, USER_ID
from recbole3.model.base import BaseCollator, BaseModelDataset, ModelDatasets
from recbole3.model.rpg.config import RPGConfig
from recbole3.model.rpg.tokenizer import RPGSemanticTokenizer, ensure_rpg_config


RPG_INPUT_IDS = "input_ids"
RPG_ATTENTION_MASK = "attention_mask"
RPG_LABELS = "labels"
RPG_SEQ_LENS = "seq_lens"


class RPGModelDataset(BaseModelDataset[pd.DataFrame, pd.DataFrame]):
    """Model-side dataset that converts RecBole interactions into RPG sequences."""

    def _build_model_datasets(self, *, model_config: RPGConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        config = ensure_rpg_config(model_config)
        tokenizer = RPGSemanticTokenizer(config, self)

        train_frame = self._build_train_frame(_dataset_frame(self.get_train_dataset()), tokenizer)
        valid_frame = self._build_eval_frame(_dataset_frame(self.get_eval_dataset("valid")), tokenizer)
        test_frame = self._build_eval_frame(_dataset_frame(self.get_eval_dataset("test")), tokenizer)
        return ModelDatasets(
            train_dataset=FrameDataset(train_frame),
            valid_dataset=FrameDataset(valid_frame),
            test_dataset=FrameDataset(test_frame),
        )

    def _build_train_frame(self, records: pd.DataFrame, tokenizer: RPGSemanticTokenizer) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if records.empty:
            return self._empty_tokenized_frame(records)

        for user_id, user_records in records.groupby(USER_ID, sort=False):
            item_seq = [int(item_id) for item_id in user_records[ITEM_ID].tolist()]
            for example in tokenizer.tokenize_train_sequence(item_seq):
                row = {
                    USER_ID: int(user_id),
                    ITEM_ID: item_seq[-1],
                    **example,
                }
                rows.append(row)

        if not rows:
            return self._empty_tokenized_frame(records)
        return pd.DataFrame(rows)

    def _build_eval_frame(self, records: pd.DataFrame, tokenizer: RPGSemanticTokenizer) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in records.to_dict("records"):
            seen_item_ids = record.get(SEEN_ITEM_IDS) or ()
            item_seq = [int(item_id) for item_id in seen_item_ids] + [int(record[ITEM_ID])]
            rows.append({**record, **tokenizer.tokenize_eval_sequence(item_seq)})
        if not rows:
            return self._empty_tokenized_frame(records)
        return pd.DataFrame(rows)

    @staticmethod
    def _empty_tokenized_frame(like: pd.DataFrame) -> pd.DataFrame:
        columns = list(like.columns)
        for column in (RPG_INPUT_IDS, RPG_ATTENTION_MASK, RPG_LABELS, RPG_SEQ_LENS):
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)


class RPGTrainCollator(BaseCollator):
    """Collate RPG training rows into tensors expected by the original model."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _build_rpg_batch(feature_records, include_labels=True)


class RPGEvalCollator(BaseCollator):
    """Collate RPG evaluation rows into tensors for generation."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _build_rpg_batch(feature_records, include_labels=RPG_LABELS in feature_records.columns)


def _build_rpg_batch(records: pd.DataFrame, *, include_labels: bool) -> dict[str, torch.Tensor]:
    batch = {
        RPG_INPUT_IDS: _tensor_from_list_column(records, RPG_INPUT_IDS, dtype=torch.long),
        RPG_ATTENTION_MASK: _tensor_from_list_column(records, RPG_ATTENTION_MASK, dtype=torch.long),
        RPG_SEQ_LENS: torch.as_tensor(records[RPG_SEQ_LENS].to_numpy(), dtype=torch.long),
    }
    if include_labels:
        batch[RPG_LABELS] = _tensor_from_list_column(records, RPG_LABELS, dtype=torch.long)
    return batch


def _tensor_from_list_column(records: pd.DataFrame, column: str, *, dtype: torch.dtype) -> torch.Tensor:
    values = records[column].tolist()
    if not values:
        return torch.empty((0, 0), dtype=dtype)
    return torch.tensor([list(value) for value in values], dtype=dtype)


def _dataset_frame(dataset: Dataset[Any]) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"RPG model datasets require FrameDataset, got {type(dataset).__name__}.")
    return dataset.frame.copy()


__all__ = [
    "RPG_ATTENTION_MASK",
    "RPG_INPUT_IDS",
    "RPG_LABELS",
    "RPGModelDataset",
    "RPGEvalCollator",
    "RPGTrainCollator",
    "RPG_SEQ_LENS",
]
