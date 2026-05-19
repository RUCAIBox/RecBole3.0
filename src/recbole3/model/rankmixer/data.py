from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import torch

from recbole3.dataset import ITEM_ID, LABEL, USER_ID
from recbole3.dataset import FrameDataset
from recbole3.model.base import BaseCollator


RANKMIXER_FEATURES = "features"


@dataclass(slots=True)
class RankMixerPreparedData:
    """Minimal prepared dataset contract for point-wise RankMixer pipelines."""

    config: Any
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    _train_dataset: FrameDataset = field(init=False, repr=False)
    _valid_dataset: FrameDataset = field(init=False, repr=False)
    _test_dataset: FrameDataset = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._train_dataset = FrameDataset(self.train_frame)
        self._valid_dataset = FrameDataset(self.valid_frame)
        self._test_dataset = FrameDataset(self.test_frame)

    def get_train_dataset(self) -> FrameDataset:
        return self._train_dataset

    def get_eval_dataset(self, split: Literal["valid", "test"]) -> FrameDataset:
        return self._valid_dataset if split == "valid" else self._test_dataset

    def get_num_users(self) -> int:
        return self._count_optional_entities(USER_ID)

    def get_num_items(self) -> int:
        return self._count_optional_entities(ITEM_ID)

    def _count_optional_entities(self, column: str) -> int:
        frames = [self.train_frame, self.valid_frame, self.test_frame]
        available = [frame[column] for frame in frames if column in frame.columns]
        if not available:
            return 0
        return int(len(pd.unique(pd.concat(available, ignore_index=True))))


def build_rankmixer_feature_columns(num_features: int) -> tuple[str, ...]:
    if int(num_features) <= 0:
        raise ValueError("RankMixer requires num_features to be a positive integer.")
    return tuple(f"feature_{index}" for index in range(int(num_features)))


def resolve_rankmixer_feature_columns(config) -> tuple[str, ...]:
    configured_columns = tuple(str(column) for column in getattr(config, "feature_columns", ()) if str(column).strip())
    if configured_columns:
        if len(configured_columns) != int(config.num_features):
            raise ValueError(
                "RankMixer feature_columns length must match num_features, "
                f"got len(feature_columns)={len(configured_columns)} and num_features={config.num_features}."
            )
        return configured_columns
    return build_rankmixer_feature_columns(config.num_features)


class _BaseRankMixerCollator(BaseCollator):
    def __init__(self, config, prepared_data) -> None:
        super().__init__(config, prepared_data)
        self._feature_columns = resolve_rankmixer_feature_columns(config)

    def _build_feature_tensor(self, records: pd.DataFrame) -> torch.Tensor:
        missing = [column for column in self._feature_columns if column not in records.columns]
        if missing:
            raise ValueError(
                f"RankMixer expected feature columns {self._feature_columns}, missing {missing} in batch frame."
            )
        feature_matrix = records.loc[:, list(self._feature_columns)].to_numpy(dtype="int64", copy=False)
        return torch.as_tensor(feature_matrix, dtype=torch.long)


class RankMixerTrainCollator(_BaseRankMixerCollator):
    """Collator that packs hashed feature columns and binary labels for training."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        labels = pd.to_numeric(feature_records[LABEL], errors="coerce").fillna(0.0).to_numpy(dtype="float32", copy=False)
        return {
            RANKMIXER_FEATURES: self._build_feature_tensor(feature_records),
            LABEL: torch.as_tensor(labels, dtype=torch.float32),
        }


class RankMixerEvalCollator(_BaseRankMixerCollator):
    """Collator that packs hashed feature columns for labeled evaluation."""

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return {
            RANKMIXER_FEATURES: self._build_feature_tensor(feature_records),
        }


__all__ = [
    "RANKMIXER_FEATURES",
    "RankMixerEvalCollator",
    "RankMixerPreparedData",
    "RankMixerTrainCollator",
    "build_rankmixer_feature_columns",
    "resolve_rankmixer_feature_columns",
]
