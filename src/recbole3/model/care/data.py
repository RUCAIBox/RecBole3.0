from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_CARE_LEVEL_PREFIXES = tuple("abcdefghijklmnopqrstuvwxyz")

import pandas as pd
import torch

from recbole3.config import instantiate_dataclass
from recbole3.dataset import ITEM_ID, USER_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.care.config import CAREConfig
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS


@dataclass(frozen=True, slots=True)
class CARETokenCodec:
    """Map RecBole item ids to CARE textual identifier tokens and back."""

    item_to_codes: dict[int, tuple[str, ...]]
    code_text_to_item: dict[str, int]
    identifier_len: int
    fallback_item_ids: tuple[int, ...]

    @classmethod
    def from_file(cls, sid_file: str, *, num_items: int) -> "CARETokenCodec":
        if not str(sid_file or "").strip():
            raise ValueError("CAREConfig.sid_file must point to a CARE/TIGER SID JSON file.")
        path = Path(sid_file)
        if not path.exists():
            raise FileNotFoundError(f"CARE sid_file does not exist: {path}")

        with open(path, "r", encoding="utf-8") as file:
            raw_indices = json.load(file)
        if not isinstance(raw_indices, dict):
            raise ValueError("CARE sid_file must be a JSON object: {item_id: [code_token, ...]} or {item_id: [int_sid, ...]}.")

        item_to_codes: dict[int, tuple[str, ...]] = {}
        expected_width: int | None = None
        for raw_item_id, raw_codes in raw_indices.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CARE index key must be an integer-like item id, got {raw_item_id!r}.") from exc
            if item_id < 0 or item_id >= int(num_items):
                raise ValueError(
                    f"CARE index item_id {item_id} is outside RecBole item id range [0, {int(num_items) - 1}]. "
                    "If CARE uses raw ids, convert the index file to RecBole remapped item ids first."
                )
            if not isinstance(raw_codes, list) or not raw_codes:
                raise ValueError(f"CARE sid_file value for item_id {item_id} must be a non-empty list.")
            codes = _normalize_sid_codes(raw_codes, item_id=item_id)
            if expected_width is None:
                expected_width = len(codes)
            elif len(codes) != expected_width:
                raise ValueError(
                    "All CARE item identifiers must have the same length. "
                    f"Expected {expected_width}, got {len(codes)} for item_id {item_id}."
                )
            item_to_codes[item_id] = codes

        missing = sorted(set(range(int(num_items))) - set(item_to_codes))
        if missing:
            preview = ", ".join(str(item_id) for item_id in missing[:10])
            raise ValueError(f"CARE sid_file is missing {len(missing)} RecBole item ids. First missing ids: {preview}.")

        code_text_to_item: dict[str, int] = {}
        for item_id, codes in item_to_codes.items():
            code_text = "".join(codes)
            if code_text in code_text_to_item:
                raise ValueError(
                    "CARE sid_file contains duplicate textual identifiers: "
                    f"items {code_text_to_item[code_text]} and {item_id} both map to {code_text!r}."
                )
            code_text_to_item[code_text] = item_id

        return cls(
            item_to_codes=item_to_codes,
            code_text_to_item=code_text_to_item,
            identifier_len=int(expected_width or 0),
            fallback_item_ids=tuple(sorted(item_to_codes)),
        )

    @property
    def all_new_tokens(self) -> list[str]:
        tokens: set[str] = set()
        for codes in self.item_to_codes.values():
            tokens.update(codes)
        return sorted(tokens)

    @property
    def all_item_ids(self) -> tuple[int, ...]:
        return self.fallback_item_ids

    def item_codes(self, item_id: int) -> tuple[str, ...]:
        return self.item_to_codes[int(item_id)]

    def item_code_text(self, item_id: int) -> str:
        return "".join(self.item_codes(item_id))

    def code_text_to_id(self, code_text: str) -> int | None:
        return self.code_text_to_item.get(str(code_text))


def _normalize_sid_codes(raw_codes: Sequence[Any], *, item_id: int) -> tuple[str, ...]:
    if all(isinstance(value, int) and not isinstance(value, bool) for value in raw_codes):
        if len(raw_codes) > len(_CARE_LEVEL_PREFIXES):
            raise ValueError(
                f"CARE integer SID for item_id {item_id} has {len(raw_codes)} levels, "
                f"but only {len(_CARE_LEVEL_PREFIXES)} level prefixes are available."
            )
        if any(int(value) < 0 for value in raw_codes):
            raise ValueError(f"CARE integer SID for item_id {item_id} contains negative values: {raw_codes}.")
        return tuple(f"<{_CARE_LEVEL_PREFIXES[idx]}_{int(value)}>" for idx, value in enumerate(raw_codes))
    if all(isinstance(value, str) for value in raw_codes):
        return tuple(str(value) for value in raw_codes)
    raise ValueError(
        f"CARE sid_file value for item_id {item_id} must contain either all integers or all strings, got {raw_codes!r}."
    )


class CAREModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds CARE sequential histories and identifier mapping."""

    care_codec: CARETokenCodec

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        if not isinstance(model_config, CAREConfig):
            model_config = instantiate_dataclass(CAREConfig, model_config)

        self.care_codec = CARETokenCodec.from_file(
            model_config.sid_file,
            num_items=int(self.get_num_items()),
        )
        if len(model_config.query_list) != self.care_codec.identifier_len:
            raise ValueError(
                "CAREConfig.query_list length must equal identifier length from sid_file. "
                f"Got query_list={model_config.query_list}, identifier_len={self.care_codec.identifier_len}."
            )
        return super()._build_model_datasets(model_config=model_config)


class _CAREBaseCollator(BaseCollator):
    config: CAREConfig

    def __init__(
        self,
        config: CAREConfig,
        prepared_data: CAREModelDataset,
        *,
        tokenizer: Any,
        codec: CARETokenCodec,
    ) -> None:
        super().__init__(config, prepared_data)
        self.tokenizer = tokenizer
        self.codec = codec

    def _records(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if isinstance(feature_records, pd.DataFrame):
            return feature_records.to_dict("records")
        return list(feature_records)

    def _history_codes(self, record: Mapping[str, Any]) -> list[str]:
        history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
        max_len = self.config.history_max_length
        if max_len is not None:
            history = history[-int(max_len):]
        codes = [self.codec.item_code_text(item_id) for item_id in history]
        if self.config.add_history_prefix:
            codes = [f"{index + 1}. {code}" for index, code in enumerate(codes)]
        return codes

    def _input_text(self, record: Mapping[str, Any]) -> str:
        history = self.config.history_separator.join(self._history_codes(record))
        return f"{history}{self.config.special_token_for_answer}"

    def _label_text(self, item_id: int) -> str:
        return self.codec.item_code_text(int(item_id))

    def _tokenize_inputs(self, input_texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=int(self.config.model_max_length),
            truncation=True,
            return_attention_mask=True,
        )

    def _tokenize_labels(self, label_texts: list[str]) -> torch.Tensor:
        output_texts = [f"{text}{self.tokenizer.eos_token}" for text in label_texts]
        labels = self.tokenizer(
            output_texts,
            return_tensors="pt",
            padding="longest",
            max_length=int(self.codec.identifier_len + 1),
            truncation=True,
            return_attention_mask=False,
        )["input_ids"]
        expected_len = self.codec.identifier_len + 1
        if int(labels.shape[1]) != expected_len:
            raise ValueError(
                "CARE labels must tokenize to identifier_len + 1 tokens. "
                f"Expected {expected_len}, got {int(labels.shape[1])}. "
                "Check that every code token was added as a tokenizer special/new token."
            )
        return labels


class CARETrainCollator(_CAREBaseCollator):
    """Collate sequential records into CARE teacher-forcing batches."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        input_texts = [self._input_text(record) for record in rows]
        label_texts = [self._label_text(int(record[ITEM_ID])) for record in rows]
        batch = self._tokenize_inputs(input_texts)
        batch["labels"] = self._tokenize_labels(label_texts)
        return batch


class CAREEvalCollator(_CAREBaseCollator):
    """Collate evaluation records into CARE scoring inputs."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        input_texts = [self._input_text(record) for record in rows]
        batch = self._tokenize_inputs(input_texts)
        batch["user_ids"] = torch.tensor([int(record[USER_ID]) for record in rows], dtype=torch.long)
        return batch


__all__ = [
    "CAREEvalCollator",
    "CAREModelDataset",
    "CARETokenCodec",
    "CARETrainCollator",
]