from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from recbole3.dataset.utils import ITEM_ID
from recbole3.model.base import BaseCollator, BaseModelDataset, ModelDatasets
from recbole3.model.rqvae.config import RQVAEConfig


class _ItemEmbeddingDataset(Dataset):
    """In-memory Dataset backed by item IDs and a stacked embedding tensor."""

    def __init__(self, item_ids: list[int], embeddings: torch.Tensor):
        self.item_ids = item_ids
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {ITEM_ID: self.item_ids[index], "embedding": self.embeddings[index]}


class RQVAEModelDataset(BaseModelDataset):
    """Model-side dataset for RQ-VAE that handles dynamic embedding generation.

    This dataset:
    1. Checks if pre-computed semantic embeddings exist in the data directory
    2. If not, generates them using sentence transformers from item metadata
    3. Filters to training items only (following SIDv1 pattern)
    4. Supports PCA reduction if configured
    """

    def _build_model_datasets(
        self, *, model_config: RQVAEConfig
    ) -> ModelDatasets:
        """Build model datasets for RQ-VAE training and evaluation.

        Returns:
            ModelDatasets containing train, valid, and test datasets.
        """
        # Load or generate semantic embeddings
        embeddings = self._load_or_generate_semantic_embeddings(model_config)

        # Filter to training items only for training dataset
        train_item_ids = self._get_training_item_ids()
        train_dataset = self._create_dataset(embeddings, train_item_ids)

        # Use all items for validation and test
        valid_dataset = self._create_dataset(embeddings)
        test_dataset = self._create_dataset(embeddings)

        return ModelDatasets(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    def _load_or_generate_semantic_embeddings(self, config: RQVAEConfig) -> torch.Tensor:
        """Load or generate semantic embeddings for all items.

        Args:
            config: RQVAEConfig containing embedding file path and generation settings.

        Returns:
            Tensor of shape (num_items, embedding_dim) containing semantic embeddings.
        """
        data_dir = self._parser.data_dir
        sem_emb_path = os.path.join(data_dir, config.sem_emb_file)

        # Check if embeddings exist
        if os.path.exists(sem_emb_path):
            embeddings = np.load(sem_emb_path)
        else:
            # Generate embeddings from item metadata
            embeddings = self._generate_semantic_embeddings(config)
            # Save for future use
            os.makedirs(os.path.dirname(sem_emb_path), exist_ok=True)
            np.save(sem_emb_path, embeddings)

        # Apply PCA if configured
        if config.sem_emb_pca > 0:
            pca = PCA(n_components=config.sem_emb_pca, whiten=True)
            embeddings = pca.fit_transform(embeddings)

        return torch.FloatTensor(embeddings)

    def _generate_semantic_embeddings(self, config: RQVAEConfig) -> np.ndarray:
        """Generate semantic embeddings using sentence transformers.

        Args:
            config: RQVAEConfig containing text field and model settings.

        Returns:
            NumPy array of embeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required to generate embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        # Get text from item table
        # sorted by item_id
        item_table = self.get_item_table()
        texts = [""] * len(item_table)
        for idx in range(len(item_table)):
            item_id = item_table.iloc[idx].get(ITEM_ID, idx)
            text = item_table.iloc[idx].get(config.text_field, "")
            texts[int(item_id)] = str(text)

        # Load model and encode
        model = SentenceTransformer(config.sent_emb_model).cuda()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=config.sent_emb_batch_size,
            show_progress_bar=True,
            device="cuda",
        )

        return embeddings

    def _get_training_item_ids(self) -> list[int]:
        """Get item IDs that appear in training interactions.

        Returns:
            Sorted list of item IDs used in training.
        """
        train_items = set()

        # Iterate over training dataset to get item IDs
        train_dataset = self.get_train_dataset()
        for interaction in train_dataset:
            train_items.add(int(interaction[ITEM_ID]))

        return sorted(list(train_items))

    def _create_dataset(
        self, embeddings: torch.Tensor, item_ids: list[int] | None = None
    ) -> Dataset:
        """Create a dataset from embeddings.

        Args:
            embeddings: Tensor of embeddings for all items.
            item_ids: List of item IDs to include (None means all items).

        Returns:
            _ItemEmbeddingDataset containing item embedding records.
        """
        if item_ids is None:
            item_ids = list(range(len(embeddings)))

        return _ItemEmbeddingDataset(
            item_ids=item_ids,
            embeddings=embeddings,
        )


class RQVAETrainCollator(BaseCollator):
    """Collator for RQ-VAE training batches."""

    def __call__(self, feature_records: Any) -> dict[str, torch.Tensor]:
        """Build a training batch from item embedding records.

        Args:
            feature_records: List of dicts with 'embedding' keys.

        Returns:
            Dictionary containing 'item_embeddings' tensor.
        """
        embeddings = torch.stack([record["embedding"] for record in feature_records])
        return {"item_embeddings": embeddings}


class RQVAEEvalCollator(BaseCollator):
    """Collator for RQ-VAE evaluation batches."""

    def __call__(self, feature_records: Any) -> dict[str, torch.Tensor]:
        """Build an evaluation batch from item embedding records.

        Args:
            feature_records: List of dicts with 'embedding' keys.

        Returns:
            Dictionary containing 'item_embeddings' tensor.
        """
        embeddings = torch.stack([record["embedding"] for record in feature_records])
        return {"item_embeddings": embeddings}


__all__ = ["RQVAEModelDataset", "RQVAEEvalCollator", "RQVAETrainCollator"]
