from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from recbole3.dataset.base import BaseTaskDataset, FrameDataset
from recbole3.dataset.config import DatasetConfig, SplitConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import ITEM_ID, LABEL, SEEN_ITEM_IDS, TIMESTAMP, USER_ID


ITEM_ID_LIST = "item_id_list"


@dataclass(slots=True)
class AgentCFDatasetConfig(DatasetConfig):
    """Configuration for the AgentCF local pre-split dataset.

    For how to manually construct the dataset and download the data files, refer to
    https://github.com/RUCAIBox/AgentCF
    """

    name: str = field(default="agentcf", metadata={"help": "Dataset name."})
    data_dir: str = field(default="", metadata={"help": "Path to the dataset directory containing .inter and .item files."})
    dataset_name: str = field(default="CDs-100-user-dense", metadata={"help": "Dataset folder name."})
    item_file: str = field(default="CDs.item", metadata={"help": "Item metadata filename."})
    candidate_file_suffix: str = field(default="random", metadata={"help": "Suffix for candidate file."})
    split: SplitConfig = field(
        default_factory=lambda: SplitConfig(strategy="leave_one_out", order="chronological", per_user=True),
        metadata={"help": "Split config (unused since data is pre-split)."},
    )


class AgentCFDatasetParser(BaseDatasetParser):
    """Parser for AgentCF's pre-split .inter and .item files."""

    config_cls = AgentCFDatasetConfig
    config: AgentCFDatasetConfig

    @property
    def data_dir(self) -> Path:
        return Path(self.config.data_dir)

    def parse(self) -> ParsedData:
        dataset_dir = self.data_dir / self.config.dataset_name
        train_inter = self._read_inter_file(dataset_dir / f"{self.config.dataset_name}.train.inter")
        valid_inter = self._read_inter_file(dataset_dir / f"{self.config.dataset_name}.valid.inter")
        test_inter = self._read_inter_file(dataset_dir / f"{self.config.dataset_name}.test.inter")

        # Assign synthetic timestamps to preserve ordering across splits
        train_inter[TIMESTAMP] = range(len(train_inter))
        offset = len(train_inter)
        valid_inter[TIMESTAMP] = range(offset, offset + len(valid_inter))
        offset += len(valid_inter)
        test_inter[TIMESTAMP] = range(offset, offset + len(test_inter))

        # Mark splits for the dataset to use
        train_inter["_split"] = "train"
        valid_inter["_split"] = "valid"
        test_inter["_split"] = "test"

        interactions = pd.concat([train_inter, valid_inter, test_inter], ignore_index=True)

        item_table = self._read_item_file(dataset_dir / self.config.item_file)

        return ParsedData(
            interactions=interactions,
            item_table=item_table,
        )

    def _read_inter_file(self, path: Path) -> pd.DataFrame:
        """Read a .inter file with format: user_id:token  item_id_list:token_seq  item_id:token"""
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    user_id = parts[0]
                    item_id_list = parts[1]
                    item_id = parts[2]
                    rows.append({
                        USER_ID: user_id,
                        ITEM_ID: item_id,
                        ITEM_ID_LIST: item_id_list,
                        LABEL: None,
                    })
                elif len(parts) == 2:
                    user_id = parts[0]
                    item_id = parts[1]
                    rows.append({
                        USER_ID: user_id,
                        ITEM_ID: item_id,
                        ITEM_ID_LIST: "",
                        LABEL: None,
                    })
        return pd.DataFrame(rows)

    def _read_item_file(self, path: Path) -> pd.DataFrame:
        """Read item metadata file with format: item_id  title  category"""
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    rows.append({ITEM_ID: parts[0], "title": parts[1], "category": parts[2]})
                elif len(parts) == 2:
                    rows.append({ITEM_ID: parts[0], "title": parts[1], "category": ""})
        return pd.DataFrame(rows)


class AgentCFDataset(BaseTaskDataset):
    """AgentCF local dataset with pre-split train/valid/test."""

    config_cls = AgentCFDatasetConfig
    parser_cls = AgentCFDatasetParser

    def _build_prepared_datasets(self) -> None:
        """Override to use pre-split data from the _split column."""
        interactions = self._interactions

        if "_split" in interactions.columns:
            train_frame = interactions[interactions["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
            valid_interactions = interactions[interactions["_split"] == "valid"].drop(columns=["_split"]).reset_index(drop=True)
            test_interactions = interactions[interactions["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

            protocol = self._require_eval_config().protocol
            if protocol in {"full", "sampled"}:
                valid_frame = self._build_eval_frame(
                    valid_interactions,
                    seen_history_interactions=train_frame,
                    split="valid",
                )
                seen_for_test = self._concat_like(
                    [train_frame, self._positive_interactions(valid_interactions)],
                    interactions.drop(columns=["_split"], errors="ignore"),
                )
                test_frame = self._build_eval_frame(
                    test_interactions,
                    seen_history_interactions=seen_for_test,
                    split="test",
                )
            else:
                valid_frame = valid_interactions
                test_frame = test_interactions

            self._train_dataset = FrameDataset(train_frame)
            self._valid_dataset = FrameDataset(valid_frame)
            self._test_dataset = FrameDataset(test_frame)
        else:
            super()._build_prepared_datasets()

    @staticmethod
    def _concat_like(frames: list[pd.DataFrame], reference: pd.DataFrame) -> pd.DataFrame:
        if not frames:
            return reference.iloc[:0].copy()
        return pd.concat(frames, ignore_index=True, sort=False)
