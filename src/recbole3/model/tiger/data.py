from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from recbole3.dataset import ITEM_ID, USER_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS
from recbole3.model.tiger.config import TIGERConfig


@dataclass(frozen=True, slots=True)
class TIGERSIDCodec:
    """Map RecBole item ids to TIGER semantic tokens and back."""

    item_to_sid: dict[int, tuple[int, ...]]
    item_to_tokens: dict[int, tuple[int, ...]]
    tokens_to_item: dict[tuple[int, ...], int]
    n_digit: int
    semantic_vocab_size: int
    fallback_item_ids: tuple[int, ...]

    @classmethod
    def from_file(cls, sid_file: str, *, num_items: int) -> "TIGERSIDCodec":
        path = Path(sid_file)
        if not sid_file:
            raise ValueError("TIGERConfig.sid_file must point to an item_sids.json file.")
        if not path.exists():
            raise FileNotFoundError(f"TIGER sid_file does not exist: {path}")
        with open(path, "r", encoding="utf-8") as file:
            raw_sids = json.load(file)
        if not isinstance(raw_sids, dict):
            raise ValueError("TIGER item_sids.json must be a JSON object.")

        item_to_sid: dict[int, tuple[int, ...]] = {}
        expected_width: int | None = None
        max_sid = -1
        for raw_item_id, raw_sid in raw_sids.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"TIGER SID key must be a remapped item_id string, got {raw_item_id!r}.") from exc
            if item_id < 0 or item_id >= num_items:
                raise ValueError(f"TIGER SID item_id {item_id} is outside dataset range [0, {num_items - 1}].")
            if not isinstance(raw_sid, list) or not raw_sid:
                raise ValueError(f"TIGER SID for item_id {item_id} must be a non-empty list.")
            sid = tuple(int(value) for value in raw_sid)
            if any(value < 0 for value in sid):
                raise ValueError(f"TIGER SID for item_id {item_id} contains negative values: {sid}.")
            if expected_width is None:
                expected_width = len(sid)
            elif len(sid) != expected_width:
                raise ValueError("All TIGER SID lists must have the same length.")
            item_to_sid[item_id] = sid
            max_sid = max(max_sid, *sid)

        missing = sorted(set(range(num_items)) - set(item_to_sid))
        if missing:
            preview = ", ".join(str(item_id) for item_id in missing[:10])
            raise ValueError(f"TIGER SID file is missing {len(missing)} item ids. First missing ids: {preview}.")

        item_to_tokens = {
            item_id: tuple(value + 1 for value in sid)
            for item_id, sid in item_to_sid.items()
        }
        tokens_to_item: dict[tuple[int, ...], int] = {}
        for item_id, tokens in item_to_tokens.items():
            if tokens in tokens_to_item:
                raise ValueError(
                    "TIGER SID file contains duplicate SID tuples after offsetting. "
                    f"Items {tokens_to_item[tokens]} and {item_id} both map to {tokens}."
                )
            tokens_to_item[tokens] = item_id

        return cls(
            item_to_sid=item_to_sid,
            item_to_tokens=item_to_tokens,
            tokens_to_item=tokens_to_item,
            n_digit=int(expected_width or 0),
            semantic_vocab_size=max_sid + 1,
            fallback_item_ids=tuple(sorted(item_to_sid)),
        )

    def item_tokens(self, item_id: int) -> tuple[int, ...]:
        return self.item_to_tokens[int(item_id)]

    def token_tuple_to_item(self, tokens: Sequence[int]) -> int | None:
        return self.tokens_to_item.get(tuple(int(token) for token in tokens))


class TIGERModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds TIGER prefix histories and SID mappings."""

    tiger_codec: TIGERSIDCodec

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        if not isinstance(model_config, TIGERConfig):
            raise TypeError(f"TIGERModelDataset requires TIGERConfig, got {type(model_config).__name__}.")
        self.tiger_codec = TIGERSIDCodec.from_file(
            model_config.sid_file,
            num_items=int(self.get_num_items()),
        )
        return super()._build_model_datasets(model_config=model_config)


class _TIGERBaseCollator(BaseCollator):
    config: TIGERConfig

    def __init__(self, config: TIGERConfig, prepared_data: TIGERModelDataset):
        super().__init__(config, prepared_data)
        if not hasattr(prepared_data, "tiger_codec"):
            raise RuntimeError("TIGER prepared_data is missing tiger_codec. Use TIGERModelDataset as model_data_cls.")
        self.codec = prepared_data.tiger_codec
        self.pad_token = 0
        if int(config.n_user_tokens) <= 0:
            raise ValueError("TIGERConfig.n_user_tokens must be a positive integer.")
        self.base_user_token = self.codec.semantic_vocab_size + 1
        self.eos_token = self.base_user_token + int(config.n_user_tokens)
        self.max_token_seq_len = int(config.history_max_length or 0) * self.codec.n_digit + 2

    def _user_token(self, user_id: int) -> int:
        return self.base_user_token + int(user_id) % int(self.config.n_user_tokens)

    def _input_for_record(self, record: Mapping[str, Any]) -> tuple[list[int], list[int]]:
        history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
        max_items = int(self.config.history_max_length or 0)
        if max_items > 0:
            history = history[-max_items:]
        else:
            history = ()

        input_ids = [self._user_token(int(record[USER_ID]))]
        for item_id in history:
            input_ids.extend(self.codec.item_tokens(item_id))
        input_ids.append(self.eos_token)

        attention_mask = [1] * len(input_ids)
        pad_width = self.max_token_seq_len - len(input_ids)
        if pad_width < 0:
            raise ValueError("TIGER input sequence exceeded max_token_seq_len.")
        input_ids.extend([self.pad_token] * pad_width)
        attention_mask.extend([0] * pad_width)
        return input_ids, attention_mask

    def _records(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if isinstance(feature_records, pd.DataFrame):
            return feature_records.to_dict("records")
        return list(feature_records)


class TIGERTrainCollator(_TIGERBaseCollator):
    """Collate sequential training records into T5 teacher-forcing batches."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for record in rows:
            cur_input_ids, cur_attention_mask = self._input_for_record(record)
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            labels.append(list(self.codec.item_tokens(int(record[ITEM_ID]))) + [self.eos_token])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TIGEREvalCollator(_TIGERBaseCollator):
    """Collate evaluation records into T5 generation inputs."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for record in rows:
            cur_input_ids, cur_attention_mask = self._input_for_record(record)
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


__all__ = [
    "TIGEREvalCollator",
    "TIGERModelDataset",
    "TIGERSIDCodec",
    "TIGERTrainCollator",
]
