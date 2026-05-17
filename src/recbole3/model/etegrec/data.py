from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recbole3.config import instantiate_dataclass
from recbole3.dataset import ITEM_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.etegrec.config import ETEGRecConfig
from recbole3.model.sequential import HISTORY_ITEM_IDS, BaseSequentialModelDataset


class ETEGRecModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds sequential histories for ETEGRec."""

    semantic_embeddings: torch.Tensor

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[Any, Any]:
        if not isinstance(model_config, ETEGRecConfig):
            model_config = instantiate_dataclass(ETEGRecConfig, model_config)
        self.semantic_embeddings = _load_semantic_embeddings(
            model_config.semantic_emb_file,
            data_dir=_parser_data_dir(self._parser, model_config.semantic_emb_file),
            num_items=int(self.get_num_items()),
            semantic_hidden_size=int(model_config.semantic_hidden_size),
        )
        return super()._build_model_datasets(model_config=model_config)


class ETEGRecTrainCollator(BaseCollator):
    """Collate sequential records into ETEGRec history/target batches."""

    config: ETEGRecConfig

    def __init__(self, config: ETEGRecConfig, prepared_data: ETEGRecModelDataset):
        super().__init__(config, prepared_data)
        self.history_max_length = _normalize_history_max_length(config.history_max_length)

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = _records(feature_records)
        histories = [_history_tokens(row, history_max_length=self.history_max_length) for row in rows]
        input_ids = _pad_histories(histories)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0),
            "targets": torch.tensor([[int(row[ITEM_ID]) + 1] for row in rows], dtype=torch.long),
        }


class ETEGRecEvalCollator(BaseCollator):
    """Collate sequential evaluation records into ETEGRec generation batches."""

    config: ETEGRecConfig

    def __init__(self, config: ETEGRecConfig, prepared_data: ETEGRecModelDataset):
        super().__init__(config, prepared_data)
        self.history_max_length = _normalize_history_max_length(config.history_max_length)

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = _records(feature_records)
        histories = [_history_tokens(row, history_max_length=self.history_max_length) for row in rows]
        input_ids = _pad_histories(histories)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0),
        }


def _records(feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(feature_records, pd.DataFrame):
        return feature_records.to_dict("records")
    return list(feature_records)


def _history_tokens(record: Mapping[str, Any], *, history_max_length: int | None) -> list[int]:
    history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
    if history_max_length is not None:
        history = history[-history_max_length:]
    # RecBole task data uses 0-based remapped item ids. ETEGRec reserves token
    # 0 for padding, so model-side item tokens are shifted by one.
    return [item_id + 1 for item_id in history]


def _pad_histories(histories: Sequence[Sequence[int]]) -> torch.Tensor:
    max_length = max((len(history) for history in histories), default=0)
    max_length = max(max_length, 1)
    input_ids = torch.zeros((len(histories), max_length), dtype=torch.long)
    for row_index, history in enumerate(histories):
        if history:
            input_ids[row_index, : len(history)] = torch.tensor(history, dtype=torch.long)
    return input_ids


def _normalize_history_max_length(history_max_length: int | None) -> int | None:
    if history_max_length is None:
        return None
    value = int(history_max_length)
    if value <= 0:
        raise ValueError("ETEGRecConfig.history_max_length must be None or a positive integer.")
    return value


def _load_semantic_embeddings(
    semantic_emb_file: str,
    *,
    data_dir: Path | None = None,
    num_items: int,
    semantic_hidden_size: int,
) -> torch.Tensor:
    if not str(semantic_emb_file or "").strip():
        raise ValueError("ETEGRecConfig.semantic_emb_file must point to a semantic embedding .npy file.")
    path = _resolve_semantic_embedding_path(semantic_emb_file, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"ETEGRec semantic_emb_file does not exist: {path}")
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"ETEGRec semantic embeddings must be a 2D array, got shape {embeddings.shape}.")
    if int(embeddings.shape[0]) != int(num_items):
        raise ValueError(
            "ETEGRec semantic embeddings must contain one row per remapped item id. "
            f"Expected {num_items} rows, got {int(embeddings.shape[0])}."
        )
    if int(embeddings.shape[1]) != int(semantic_hidden_size):
        raise ValueError(
            "ETEGRec semantic embedding dimension does not match model.semantic_hidden_size. "
            f"Expected {semantic_hidden_size}, got {int(embeddings.shape[1])}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("ETEGRec semantic embeddings contain NaN or infinite values.")
    return torch.as_tensor(embeddings, dtype=torch.float32)


def _resolve_semantic_embedding_path(semantic_emb_file: str, *, data_dir: Path | None = None) -> Path:
    path = Path(semantic_emb_file)
    if path.is_absolute() or path.parent != Path("."):
        return path
    if data_dir is not None:
        return data_dir / path
    return path


def _parser_data_dir(parser: Any, semantic_emb_file: str) -> Path | None:
    if not str(semantic_emb_file or "").strip():
        return None
    path = Path(semantic_emb_file)
    if path.is_absolute() or path.parent != Path("."):
        return None
    return Path(parser.data_dir)


__all__ = [
    "ETEGRecEvalCollator",
    "ETEGRecModelDataset",
    "ETEGRecTrainCollator",
]
