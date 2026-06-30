from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from recbole3.config import instantiate_dataclass
from recbole3.dataset import ITEM_ID, USER_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.care.config import CAREConfig
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS

_LEVEL_PREFIXES = tuple("abcdefghijklmnopqrstuvwxyz")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CARETokenCodec:
    """Map RecBole item ids to CARE textual SID tokens and back."""

    item_to_codes: dict[int, tuple[str, ...]]
    code_text_to_item: dict[str, int]
    identifier_len: int
    fallback_item_ids: tuple[int, ...]

    @classmethod
    def from_file(cls, sid_file: str, *, num_items: int) -> "CARETokenCodec":
        if not str(sid_file or "").strip():
            raise ValueError("CAREConfig.sid_file must point to an item SID JSON file.")
        path = Path(sid_file)
        if not path.exists():
            raise FileNotFoundError(f"CARE sid_file does not exist: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("CARE sid_file must be a JSON object keyed by RecBole remapped item ids.")

        item_to_codes: dict[int, tuple[str, ...]] = {}
        width: int | None = None
        for raw_item_id, raw_codes in raw.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CAR SID key must be an integer-like item id, got {raw_item_id!r}.") from exc
            if item_id < 0 or item_id >= int(num_items):
                raise ValueError(f"CAR SID item_id {item_id} is outside RecBole range [0, {int(num_items) - 1}].")
            if not isinstance(raw_codes, list) or not raw_codes:
                raise ValueError(f"CAR SID for item_id {item_id} must be a non-empty list.")
            codes = _normalize_codes(raw_codes, item_id=item_id)
            width = len(codes) if width is None else width
            if len(codes) != width:
                raise ValueError(f"All CAR SID lists must have length {width}; item_id {item_id} has {len(codes)}.")
            item_to_codes[item_id] = codes

        missing = sorted(set(range(int(num_items))) - set(item_to_codes))
        if missing:
            preview = ", ".join(str(x) for x in missing[:10])
            raise ValueError(f"CAR sid_file is missing {len(missing)} RecBole item ids. First missing ids: {preview}.")

        code_text_to_item: dict[str, int] = {}
        for item_id, codes in item_to_codes.items():
            text = _normalize_code_text("".join(codes))
            if text in code_text_to_item:
                raise ValueError(f"Duplicate CAR textual SID {text!r} for items {code_text_to_item[text]} and {item_id}.")
            code_text_to_item[text] = item_id
        return cls(item_to_codes, code_text_to_item, int(width or 0), tuple(sorted(item_to_codes)))

    @property
    def all_new_tokens(self) -> list[str]:
        return sorted({token for codes in self.item_to_codes.values() for token in codes})

    @property
    def all_items(self) -> list[str]:
        """All CARE textual identifiers, matching original inference.py all_items."""
        return [self.item_code_text(item_id) for item_id in self.fallback_item_ids]

    @property
    def all_item_ids(self) -> tuple[int, ...]:
        """All RecBole remapped item ids in deterministic order."""
        return self.fallback_item_ids

    def item_codes(self, item_id: int) -> tuple[str, ...]:
        return self.item_to_codes[int(item_id)]

    def item_code_text(self, item_id: int) -> str:
        return "".join(self.item_codes(item_id))

    def code_text_to_id(self, code_text: str) -> int | None:
        return self.code_text_to_item.get(_normalize_code_text(code_text))


def _normalize_code_text(code_text: Any) -> str:
    return str(code_text).strip().replace(" ", "").replace("\n", "").replace("\t", "")


def _normalize_codes(raw_codes: Sequence[Any], *, item_id: int) -> tuple[str, ...]:
    if all(isinstance(v, int) and not isinstance(v, bool) for v in raw_codes):
        if len(raw_codes) > len(_LEVEL_PREFIXES):
            raise ValueError(f"CAR integer SID for item_id {item_id} has too many levels: {len(raw_codes)}.")
        if any(int(v) < 0 for v in raw_codes):
            raise ValueError(f"CAR integer SID for item_id {item_id} contains negative values: {raw_codes}.")
        return tuple(f"<{_LEVEL_PREFIXES[i]}_{int(v)}>" for i, v in enumerate(raw_codes))
    if all(isinstance(v, str) for v in raw_codes):
        return tuple(str(v) for v in raw_codes)
    raise ValueError(f"CAR SID for item_id {item_id} must contain either all integers or all strings.")


class CAREModelDataset(BaseSequentialModelDataset):
    care_codec: CARETokenCodec

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        if not isinstance(model_config, CAREConfig):
            model_config = instantiate_dataclass(CAREConfig, model_config)
        self.care_codec = CARETokenCodec.from_file(model_config.sid_file, num_items=int(self.get_num_items()))
        if len(model_config.query_list) != self.care_codec.identifier_len:
            raise ValueError("CAREConfig.query_list length must equal SID identifier length.")
        if bool(model_config.progressive_attn):
            if str(model_config.history_separator) != "" or bool(model_config.add_history_prefix):
                raise ValueError(
                    "CAR progressive_attn requires compact CARE history: "
                    "history_separator must be '' and add_history_prefix must be false."
                )
        datasets = super()._build_model_datasets(model_config=model_config)
        train_frame = datasets.train_dataset.frame
        keep_mask = train_frame[HISTORY_ITEM_IDS].map(lambda history: len(history or ()) > 0)
        skipped = int((~keep_mask).sum())
        if skipped:
            logger.info("CAR train skipped %d empty-history samples to match CARE sliding-window training from i=1.", skipped)
        return ModelDatasets(
            train_dataset=type(datasets.train_dataset)(train_frame.loc[keep_mask].reset_index(drop=True)),
            valid_dataset=datasets.valid_dataset,
            test_dataset=datasets.test_dataset,
        )


class _CAREBaseCollator(BaseCollator):
    def __init__(self, config: CAREConfig, prepared_data: CAREModelDataset, *, tokenizer: Any, codec: CARETokenCodec) -> None:
        super().__init__(config, prepared_data)
        self.tokenizer = tokenizer
        self.codec = codec

    def _records(self, records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return records.to_dict("records") if isinstance(records, pd.DataFrame) else list(records)

    def _input_text(self, record: Mapping[str, Any]) -> str:
        history = tuple(int(x) for x in (record.get(HISTORY_ITEM_IDS) or ()))
        if self.config.history_max_length is not None:
            history = history[-int(self.config.history_max_length):]
        codes = [self.codec.item_code_text(item_id) for item_id in history]
        if self.config.add_history_prefix:
            codes = [f"{idx + 1}. {code}" for idx, code in enumerate(codes)]
        return f"{self.config.history_separator.join(codes)}{self.config.special_token_for_answer}"

    def _tokenize_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(texts, return_tensors="pt", padding="longest", max_length=int(self.config.model_max_length), truncation=True, return_attention_mask=True)

    def _tokenize_labels(self, texts: list[str]) -> torch.Tensor:
        labels = self.tokenizer([f"{text}{self.tokenizer.eos_token}" for text in texts], return_tensors="pt", padding="longest", max_length=self.codec.identifier_len + 1, truncation=True)["input_ids"]
        if int(labels.shape[1]) != self.codec.identifier_len + 1:
            raise ValueError("CAR labels must tokenize to identifier_len + 1 tokens; check SID special tokens.")
        return labels


class CARETrainCollator(_CAREBaseCollator):
    def __call__(self, records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(records)
        batch = self._tokenize_inputs([self._input_text(row) for row in rows])
        batch["labels"] = self._tokenize_labels([self.codec.item_code_text(int(row[ITEM_ID])) for row in rows])
        return batch


class CAREEvalCollator(_CAREBaseCollator):
    def __call__(self, records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(records)
        batch = self._tokenize_inputs([self._input_text(row) for row in rows])
        batch["user_ids"] = torch.tensor([int(row[USER_ID]) for row in rows], dtype=torch.long)
        return batch


__all__ = ["CAREEvalCollator", "CAREModelDataset", "CARETokenCodec", "CARETrainCollator"]