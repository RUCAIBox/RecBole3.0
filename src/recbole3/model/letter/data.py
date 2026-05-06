from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from recbole3.dataset.utils import ITEM_ID
from recbole3.model.base import BaseCollator, ModelDatasets
from recbole3.model.letter.config import LETTERConfig
from recbole3.model.rqvae.data import RQVAEModelDataset


class _ItemEmbeddingDataset(Dataset):
    """In-memory dataset of item semantic/collaborative embeddings."""

    def __init__(self, item_ids: list[int], sem_embeddings: torch.Tensor, cf_embeddings: torch.Tensor):
        self.item_ids = item_ids
        self.sem_embeddings = sem_embeddings
        self.cf_embeddings = cf_embeddings

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            ITEM_ID: self.item_ids[index],
            "embedding": self.sem_embeddings[index],
            "cf_embedding": self.cf_embeddings[index],
        }


class LETTERModelDataset(RQVAEModelDataset):
    """Model-side dataset for LETTER tokenizer training/evaluation."""

    def _build_model_datasets(self, *, model_config: LETTERConfig) -> ModelDatasets:
        sem_embeddings = self._load_or_generate_semantic_embeddings(model_config)
        cf_embeddings = self._load_collaborative_embeddings(model_config)

        train_item_ids = self._get_training_item_ids()
        train_dataset = self._create_dataset(sem_embeddings, cf_embeddings, train_item_ids)
        valid_dataset = self._create_dataset(sem_embeddings, cf_embeddings)
        test_dataset = self._create_dataset(sem_embeddings, cf_embeddings)

        return ModelDatasets(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    def _load_collaborative_embeddings(self, config: LETTERConfig) -> torch.Tensor:
        if not config.cf_emb_file:
            raise ValueError("model.cf_emb_file is required for LETTER.")

        cf_path = Path(config.cf_emb_file)
        if not cf_path.is_absolute():
            cf_path = Path(self._parser.data_dir) / cf_path

        if not cf_path.exists():
            raise ValueError(
                f"Collaborative embedding file not found: {cf_path}. "
                "Run `python -m recbole3.tools.generate_letter_cf_hstu ...` to generate it first."
            )

        return (
            torch.load(str(cf_path), map_location="cpu")
            .squeeze()
            .detach()
            .to(dtype=torch.float32)
        )

    def _create_dataset(
        self,
        sem_embeddings: torch.Tensor,
        cf_embeddings: torch.Tensor,
        item_ids: list[int] | None = None,
    ) -> Dataset:
        if item_ids is None:
            item_ids = list(range(len(sem_embeddings)))

        return _ItemEmbeddingDataset(
            item_ids=item_ids,
            sem_embeddings=sem_embeddings[item_ids],
            cf_embeddings=cf_embeddings[item_ids],
        )


class LETTERTrainCollator(BaseCollator):
    """Collator for LETTER training."""

    def __call__(self, feature_records: Any) -> dict[str, torch.Tensor]:
        sem_embeddings = torch.stack([record["embedding"] for record in feature_records])
        cf_embeddings = torch.stack([record["cf_embedding"] for record in feature_records])
        item_ids = torch.tensor([record[ITEM_ID] for record in feature_records], dtype=torch.long)
        return {
            "item_embeddings": sem_embeddings,
            "cf_embeddings": cf_embeddings,
            "item_ids": item_ids,
        }


class LETTEREvalCollator(BaseCollator):
    """Collator for LETTER evaluation."""

    def __call__(self, feature_records: Any) -> dict[str, torch.Tensor]:
        sem_embeddings = torch.stack([record["embedding"] for record in feature_records])
        cf_embeddings = torch.stack([record["cf_embedding"] for record in feature_records])
        item_ids = torch.tensor([record[ITEM_ID] for record in feature_records], dtype=torch.long)
        return {
            "item_embeddings": sem_embeddings,
            "cf_embeddings": cf_embeddings,
            "item_ids": item_ids,
        }


__all__ = ["LETTEREvalCollator", "LETTERModelDataset", "LETTERTrainCollator"]
