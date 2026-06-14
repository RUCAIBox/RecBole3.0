from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from recbole3.dataset.base import BaseTaskDataset

from .base import AvazuBaseConfig, AvazuBaseParser
from .utils import AVAZU_SPLIT_COLUMN


@dataclass(slots=True)
class AvazuCTRConfig(AvazuBaseConfig):
    name: str = field(default="avazu_ctr", metadata={"help": "Registered Avazu CTR dataset name."})


class AvazuCTRParser(AvazuBaseParser):
    config_cls = AvazuCTRConfig
    config: AvazuCTRConfig


class AvazuCTRDataset(BaseTaskDataset):
    config_cls = AvazuCTRConfig
    parser_cls = AvazuCTRParser

    def _build_split_frames(
        self,
        ordered_interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        protocol = self._require_eval_config().protocol
        if protocol != "labeled":
            raise ValueError(
                "AvazuCTRDataset only supports labeled evaluation."
            )
        return (
            self._split_frame(ordered_interactions, split_name="train"),
            self._split_frame(ordered_interactions, split_name="valid"),
            self._split_frame(ordered_interactions, split_name="test"),
        )

    @staticmethod
    def _split_frame(records: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
        frame = records.loc[records[AVAZU_SPLIT_COLUMN] == split_name].copy()
        if AVAZU_SPLIT_COLUMN in frame.columns:
            frame = frame.drop(columns=[AVAZU_SPLIT_COLUMN])
        return frame.reset_index(drop=True)


__all__ = [
    "AvazuCTRConfig",
    "AvazuCTRDataset",
    "AvazuCTRParser",
]