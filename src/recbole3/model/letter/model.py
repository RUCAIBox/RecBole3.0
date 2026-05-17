from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.base import BaseCollator, BaseModel
from recbole3.model.letter.config import LETTERConfig
from recbole3.model.letter.layers import LetterRQLayer, MLP


class LETTERModel(BaseModel):
    """LETTER tokenizer model (semantic + collaborative regularization)."""

    def __init__(self, config: LETTERConfig):
        super().__init__(config)
        self._embedding_dim = config.sem_emb_pca if config.sem_emb_pca > 0 else config.sem_emb_dim

        encoder_sizes = (self._embedding_dim,) + tuple(self.config.layers) + (self.config.codebook_dim,)
        self._encoder = MLP(list(encoder_sizes), dropout=self.config.dropout, use_bn=self.config.bn)
        self._rq_layer = LetterRQLayer(self.config)
        self._decoder = MLP(list(encoder_sizes[::-1]), dropout=self.config.dropout, use_bn=self.config.bn)

        self._initted: bool = False
        self._diversity_labels: dict[str, list[int]] | None = None

    def build_train_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        from recbole3.model.letter.data import LETTERTrainCollator

        return LETTERTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        from recbole3.model.letter.data import LETTEREvalCollator

        return LETTEREvalCollator(self.config, prepared_data)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        embeddings = batch["item_embeddings"]
        encoded = self._encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self._rq_layer(encoded, self._diversity_labels)
        reconstructed = self._decoder(quantized)
        return {
            "reconstruction": reconstructed,
            "quantized": quantized,
            "quant_loss": quant_loss,
            "unused_codes": unused_codes,
            "tokens": tokens,
        }

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        if self.config.loss_type == "mse":
            recon_loss = F.mse_loss(outputs["reconstruction"], batch["item_embeddings"])
        elif self.config.loss_type == "l1":
            recon_loss = F.l1_loss(outputs["reconstruction"], batch["item_embeddings"])
        else:
            raise ValueError(f"Unsupported loss_type: {self.config.loss_type}. Expected 'mse' or 'l1'.")
        quant_loss = outputs["quant_loss"] * self.config.quant_loss_weight
        cf_loss = self._compute_cf_loss(outputs["quantized"], batch["cf_embeddings"])

        total_loss = recon_loss + quant_loss + self.config.cf_loss_weight * cf_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "quant_loss": quant_loss,
            "cf_loss": cf_loss,
        }

    def _compute_cf_loss(self, quantized_repr: torch.Tensor, cf_embeddings: torch.Tensor) -> torch.Tensor:
        if quantized_repr.size(-1) != cf_embeddings.size(-1):
            raise ValueError(
                "LETTER CF loss requires quantized representations and collaborative embeddings "
                "to have the same last dimension, "
                f"got {quantized_repr.size(-1)} and {cf_embeddings.size(-1)}. "
                f"Set HSTU model.embedding_dim to LETTER codebook_dim={self.config.codebook_dim} "
                "when generating model.cf_emb_file."
            )
        labels = torch.arange(quantized_repr.size(0), dtype=torch.long, device=quantized_repr.device)
        similarities = torch.matmul(quantized_repr, cf_embeddings.transpose(0, 1))
        return F.cross_entropy(similarities, labels)

    def predict(self, embeddings: torch.Tensor, infer_use_sk: bool = False) -> torch.Tensor:
        embeddings = embeddings.to(next(self.parameters()).device)
        encoded = self._encoder(embeddings)
        _, _, _, tokens = self._rq_layer(encoded, self._diversity_labels, infer_use_sk=infer_use_sk)
        return tokens

    def init_codebook(self, x: torch.Tensor) -> None:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                encoded = self._encoder(x)
                self._rq_layer.init_codebook(encoded, x.device)
        finally:
            self.train(was_training)
        self._initted = True

    def set_diversity_labels(self, labels: dict[str, list[int]]) -> None:
        self._diversity_labels = labels


__all__ = ["LETTERModel"]
