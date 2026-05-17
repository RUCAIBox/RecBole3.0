from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole3.model.etegrec.config import ETEGRecConfig
from recbole3.model.etegrec.layers import MLPLayers


class RQVAE(nn.Module):
    """ETEGRec residual-quantization tokenizer.

    This mirrors the original ETEGRec RQVAE structure closely enough for model
    forward/loss and later trainer migration, while keeping data loading outside
    the model.
    """

    def __init__(self, config: ETEGRecConfig, *, in_dim: int):
        super().__init__()
        if config.vq_type != "vq":
            raise NotImplementedError("ETEGRec stage 4 supports only vq_type='vq'.")

        self.in_dim = int(in_dim)
        self.e_dim = int(config.e_dim)
        self.encode_layer_dims = [self.in_dim, *[int(size) for size in config.layers], self.e_dim]
        self.decode_layer_dims = list(reversed(self.encode_layer_dims))
        self.encoder = MLPLayers(self.encode_layer_dims, dropout=float(config.dropout_prob), bn=bool(config.bn))
        self.rq = ResidualVectorQuantizer(config)
        self.decoder = MLPLayers(self.decode_layer_dims, dropout=float(config.dropout_prob), bn=bool(config.bn))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        x_q, rq_loss, indices, code_one_hot, logits = self.rq(latent)
        out = self.decoder(x_q)
        return out, rq_loss, indices, code_one_hot, logits

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor) -> torch.Tensor:
        return self.rq.get_indices(self.encoder(xs))

    def get_codebook(self) -> torch.Tensor:
        return self.rq.get_codebook()


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config: ETEGRecConfig):
        super().__init__()
        self.n_e_list = tuple(int(size) for size in config.num_emb_list)
        self.vq_layers = nn.ModuleList([VectorQuantizer(config, n_e=n_e) for n_e in self.n_e_list])

    def get_codebook(self) -> torch.Tensor:
        return torch.stack([quantizer.get_codebook().detach().cpu() for quantizer in self.vq_layers])

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor) -> torch.Tensor:
        all_indices: list[torch.Tensor] = []
        residual = x
        for quantizer in self.vq_layers:
            x_res, _, indices, _, _ = quantizer(residual)
            residual = residual - x_res
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_losses: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        all_one_hots: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []

        x_q: torch.Tensor | int = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, one_hot, logits = quantizer(residual)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
            all_one_hots.append(one_hot)
            all_logits.append(logits)

        return (
            x_q,
            torch.stack(all_losses).mean(),
            torch.stack(all_indices, dim=-1),
            torch.stack(all_one_hots, dim=1),
            torch.stack(all_logits, dim=1),
        )


class VectorQuantizer(nn.Module):
    def __init__(self, config: ETEGRecConfig, *, n_e: int):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(config.e_dim)
        self.beta = float(config.beta)
        self.dist = str(config.dist).lower()
        self.kmeans_init = bool(config.kmeans_init)
        if self.kmeans_init:
            raise NotImplementedError("ETEGRec stage 4 does not initialize RQVAE codebooks with k-means.")

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.e_dim)
        if self.dist == "l2":
            distances = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        elif self.dist == "dot":
            distances = -torch.matmul(latent, self.embedding.weight.t())
        elif self.dist == "cos":
            distances = -torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t())
        else:
            raise NotImplementedError(f"Unsupported ETEGRec RQVAE distance: {self.dist}")

        indices = torch.argmin(distances, dim=-1)
        one_hot = F.one_hot(indices, self.n_e).float()
        x_q = self.embedding(indices).view(x.shape)

        if self.dist == "l2":
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        else:
            loss = self.beta * F.cross_entropy(-distances, indices.detach())

        x_q = x + (x_q - x).detach()
        return x_q, loss, indices.view(x.shape[:-1]), one_hot, distances


__all__ = ["RQVAE", "ResidualVectorQuantizer", "VectorQuantizer"]
