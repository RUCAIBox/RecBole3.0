from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.base import BaseModel, BaseCollator, ModelConfig
from recbole3.model.rqvae.config import RQVAEConfig
from recbole3.model.rqvae.layers import MLP, RQLayer


class RQVAEModel(BaseModel):
    """Residual Quantized Variational AutoEncoder (RQ-VAE) model.

    This model learns to quantize semantic item embeddings into discrete tokens
    using residual vector quantization. The model outputs token assignments that
    can be used downstream for item representation tasks.

    Args:
        config: RQVAEConfig object containing model parameters.
    """

    def __init__(self, config: RQVAEConfig):
        super().__init__(config)
        self._embedding_dim = config.sem_emb_pca if config.sem_emb_pca > 0 else config.sem_emb_dim
        # Build encoder: [emb_dim, hidden_sizes..., codebook_dim]
        encoder_sizes = (self._embedding_dim,) + self.config.hidden_sizes + (self.config.codebook_dim,)
        self._encoder = MLP(list(encoder_sizes), dropout=self.config.dropout)

        # Build RQ layer
        self._rq_layer = RQLayer(self.config)

        # Build decoder: reverse of encoder
        decoder_sizes = encoder_sizes[::-1]
        self._decoder = MLP(list(decoder_sizes), dropout=self.config.dropout)

        self._initted: bool = False

    def build_train_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        """Return the collator used for training batches."""
        from recbole3.model.rqvae.data import RQVAETrainCollator
        return RQVAETrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        """Return the collator used to pack evaluation model inputs."""
        from recbole3.model.rqvae.data import RQVAEEvalCollator
        return RQVAEEvalCollator(self.config, prepared_data)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the forward pass on a prepared batch.

        Args:
            batch: Dictionary containing 'item_embeddings' tensor.

        Returns:
            Dictionary containing reconstruction, quantization loss, unused codes, and tokens.
        """
        embeddings = batch["item_embeddings"].to(next(self.parameters()).device)

        encoded = self._encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self._rq_layer(encoded)
        reconstructed = self._decoder(quantized)

        return {
            "reconstruction": reconstructed,
            "quant_loss": quant_loss,
            "unused_codes": unused_codes,
            "tokens": tokens,
        }

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        """Compute training loss from a batch and model outputs.

        Args:
            batch: Dictionary containing 'item_embeddings' tensor.
            outputs: Dictionary containing model outputs.

        Returns:
            Dictionary containing total loss, reconstruction loss, and quantization loss.
        """
        recon_loss = F.mse_loss(
            outputs["reconstruction"],
            batch["item_embeddings"].to(outputs["reconstruction"].device),
        )
        quant_loss = outputs["quant_loss"]
        total_loss = recon_loss + quant_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "quant_loss": quant_loss,
        }

    def predict(self, embeddings: torch.Tensor, infer_use_sk: bool = False) -> torch.Tensor:
        """Return token assignments for all items.

        Args:
            embeddings: Input embeddings tensor.

        Returns:
            Token indices tensor of shape (batch_size, codebook_num).
        """
        embeddings = embeddings.to(next(self.parameters()).device)

        encoded = self._encoder(embeddings)
        _, _, _, tokens = self._rq_layer(encoded, infer_use_sk)
        return tokens


    def init_codebook(self, x: torch.Tensor) -> None:
        """Initialize codebooks using K-means clustering.

        Args:
            x: Input tensor for codebook initialization.
        """
        encoded = self._encoder(x)
        self._rq_layer.init_codebook(encoded, x.device)
        self._initted = True


__all__ = ["RQVAEModel"]
