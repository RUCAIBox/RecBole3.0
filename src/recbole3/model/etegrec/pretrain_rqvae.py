from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole3.model.etegrec.pretrain_layers import (
    MLPLayers,
    kmeans,
    laplace_smoothing,
    moving_average,
    sinkhorn_algorithm,
)


@dataclass(slots=True)
class ETEGRecRQVAEPretrainConfig:
    """Configuration for the original ETEGRec standalone RQVAE pretraining."""

    num_emb_list: tuple[int, ...] = (256, 256, 256)
    e_dim: int = 128
    layers: tuple[int, ...] = (512, 256)
    quant_loss_weight: float = 1.0
    beta: float = 0.25
    vq_type: str = "vq"
    dropout_prob: float = 0.0
    bn: bool = False
    loss_type: str = "mse"
    dist: str = "l2"
    tau: float = 0.1
    h_dim: int = 2048
    temperature: float = 0.9
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sk_epsilons: tuple[float, ...] = (0.0, 0.0, 0.0)
    sk_iters: int = 50
    moving_avg_decay: float = 0.99


def build_pretrain_config(model_config: object, **overrides: object) -> ETEGRecRQVAEPretrainConfig:
    values = {
        "num_emb_list": tuple(int(value) for value in getattr(model_config, "num_emb_list", (256, 256, 256))),
        "e_dim": int(getattr(model_config, "e_dim", 128)),
        "layers": tuple(int(value) for value in getattr(model_config, "layers", (512, 256))),
        "quant_loss_weight": float(getattr(model_config, "quant_loss_weight", getattr(model_config, "alpha", 1.0))),
        "beta": float(getattr(model_config, "beta", 0.25)),
        "vq_type": str(getattr(model_config, "vq_type", "vq")),
        "dropout_prob": float(getattr(model_config, "dropout_prob", 0.0)),
        "bn": bool(getattr(model_config, "bn", False)),
        "loss_type": str(getattr(model_config, "loss_type", "mse")),
        "dist": str(getattr(model_config, "dist", "l2")),
        "tau": float(getattr(model_config, "tau", 0.1)),
        "h_dim": int(getattr(model_config, "h_dim", 2048)),
        "temperature": float(getattr(model_config, "temperature", 0.9)),
        "kmeans_init": bool(getattr(model_config, "kmeans_init", True)),
        "kmeans_iters": int(getattr(model_config, "kmeans_iters", 100)),
        "sk_epsilons": tuple(float(value) for value in getattr(model_config, "sk_epsilons", (0.0, 0.0, 0.0))),
        "sk_iters": int(getattr(model_config, "sk_iters", 50)),
        "moving_avg_decay": float(getattr(model_config, "moving_avg_decay", 0.99)),
    }
    values.update(overrides)
    config = ETEGRecRQVAEPretrainConfig(**values)
    if len(config.sk_epsilons) != len(config.num_emb_list):
        raise ValueError("ETEGRec RQVAE pretraining expects len(sk_epsilons) == len(num_emb_list).")
    return config


class RQVAE(nn.Module):
    """Standalone ETEGRec RQVAE pretraining model."""

    def __init__(self, config: ETEGRecRQVAEPretrainConfig, *, in_dim: int):
        super().__init__()
        self.config = config
        self.in_dim = int(in_dim)
        self.e_dim = int(config.e_dim)
        self.dropout_prob = float(config.dropout_prob)
        self.bn = bool(config.bn)
        self.loss_type = str(config.loss_type)
        self.quant_loss_weight = float(config.quant_loss_weight)
        self.beta = float(config.beta)
        self.vq_type = str(config.vq_type)
        self.tau = float(config.tau)

        if self.vq_type in {"vq", "ema"}:
            self.encode_layer_dims = [self.in_dim, *[int(size) for size in config.layers], self.e_dim]
            self.decode_layer_dims = list(reversed(self.encode_layer_dims))
        elif self.vq_type == "gumbel":
            self.encode_layer_dims = [self.in_dim, *[int(size) for size in config.layers], int(config.h_dim)]
            self.decode_layer_dims = [self.e_dim, *[int(size) for size in reversed(config.layers)], self.in_dim]
        else:
            raise NotImplementedError(f"Unsupported ETEGRec RQVAE vq_type: {self.vq_type}")

        self.encoder = MLPLayers(self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn)
        self.rq = ResidualVectorQuantizer(config)
        self.decoder = MLPLayers(self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        quantized, rq_loss, indices = self.rq(encoded)
        out = self.decoder(quantized)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, *, conflict: bool = False) -> torch.Tensor:
        return self.rq.get_indices(self.encoder(xs), conflict=conflict)

    @torch.no_grad()
    def get_maxk_indices(self, xs: torch.Tensor, *, maxk: int = 1, used: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rq.get_maxk_indices(self.encoder(xs), maxk=maxk, used=used)

    def get_codebook(self) -> torch.Tensor:
        return self.rq.get_codebook()

    @staticmethod
    def compute_contrastive_loss(query_embeds: torch.Tensor, semantic_embeds: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
        query_embeds = F.normalize(query_embeds, dim=-1)
        semantic_embeds = F.normalize(semantic_embeds, dim=-1)
        labels = torch.arange(query_embeds.size(0), dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(query_embeds, semantic_embeds.transpose(0, 1)) / float(temperature)
        return F.cross_entropy(similarities, labels)

    def compute_loss(
        self,
        out: torch.Tensor,
        quant_loss: torch.Tensor,
        *,
        xs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            recon_loss = F.l1_loss(out, xs, reduction="mean")
        elif self.loss_type == "infonce":
            recon_loss = self.compute_contrastive_loss(out, xs, temperature=self.tau)
        else:
            raise ValueError(f"incompatible ETEGRec RQVAE loss_type: {self.loss_type}")
        return recon_loss + self.quant_loss_weight * quant_loss, recon_loss


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config: ETEGRecRQVAEPretrainConfig):
        super().__init__()
        self.n_e_list = tuple(int(value) for value in config.num_emb_list)
        self.vq_type = str(config.vq_type)
        if self.vq_type == "vq":
            self.vq_layers = nn.ModuleList(
                [VectorQuantizer(config, n_e=n_e, sk_epsilon=sk_epsilon) for n_e, sk_epsilon in zip(self.n_e_list, config.sk_epsilons, strict=True)]
            )
        elif self.vq_type == "ema":
            self.vq_layers = nn.ModuleList(
                [EMAVectorQuantizer(config, n_e=n_e, sk_epsilon=sk_epsilon) for n_e, sk_epsilon in zip(self.n_e_list, config.sk_epsilons, strict=True)]
            )
        elif self.vq_type == "gumbel":
            self.vq_layers = nn.ModuleList([GumbelVectorQuantizer(config, n_e=n_e) for n_e in self.n_e_list])
        else:
            raise NotImplementedError(f"Unsupported ETEGRec RQVAE vq_type: {self.vq_type}")

    def get_codebook(self) -> torch.Tensor:
        return torch.stack([quantizer.get_codebook().detach().cpu() for quantizer in self.vq_layers])

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor, *, conflict: bool = False) -> torch.Tensor:
        all_indices = []
        residual = x
        for index, quantizer in enumerate(self.vq_layers):
            x_res, _, indices = quantizer(residual, conflict=conflict and index == len(self.vq_layers) - 1)
            residual = residual - x_res
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)

    @torch.no_grad()
    def get_maxk_indices(self, x: torch.Tensor, *, maxk: int = 1, used: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        all_indices = []
        residual = x
        fix = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        for index, quantizer in enumerate(self.vq_layers):
            if index == len(self.vq_layers) - 1:
                indices, fix = quantizer.get_maxk_indices(residual, maxk=maxk, used=used)
                x_res = 0
            else:
                x_res, _, indices = quantizer(residual, conflict=False)
            residual = residual - x_res
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1), fix

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = []
        indices = []
        quantized: torch.Tensor | int = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, index = quantizer(residual)
            residual = residual - x_res
            quantized = quantized + x_res
            losses.append(loss)
            indices.append(index)
        return quantized, torch.stack(losses).mean(), torch.stack(indices, dim=-1)


class VectorQuantizer(nn.Module):
    def __init__(self, config: ETEGRecRQVAEPretrainConfig, *, n_e: int, sk_epsilon: float = 0.003):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(config.e_dim)
        self.beta = float(config.beta)
        self.dist = str(config.dist).lower()
        self.kmeans_init = bool(config.kmeans_init)
        self.kmeans_iters = int(config.kmeans_iters)
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(config.sk_iters)
        self.tau = float(config.tau)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.initted = not self.kmeans_init
        if self.initted:
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.embedding.weight.data.zero_()

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    def get_codebook_entry(self, indices: torch.Tensor, shape: Sequence[int] | None = None) -> torch.Tensor:
        quantized = self.embedding(indices)
        if shape is not None:
            quantized = quantized.view(*shape)
        return quantized

    def init_emb(self, data: torch.Tensor) -> None:
        self.embedding.weight.data.copy_(kmeans(data, self.n_e, self.kmeans_iters))
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (distances - middle) / amplitude

    def forward(self, x: torch.Tensor, *, conflict: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)

        if self.dist == "l2":
            distances = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        elif self.dist == "dot":
            distances = -torch.matmul(latent, self.embedding.weight.t()) / self.tau
        elif self.dist == "cos":
            distances = -torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t()) / self.tau
        else:
            raise NotImplementedError(f"Unsupported ETEGRec RQVAE distance: {self.dist}")

        if self.sk_epsilon > 0 and (self.training or conflict):
            q = sinkhorn_algorithm(self.center_distance_for_constraint(distances).double(), self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                raise ValueError("ETEGRec RQVAE Sinkhorn assignment returned NaN or infinite values.")
            indices = torch.argmax(q, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        if self.dist == "l2":
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        elif self.dist in {"dot", "cos"}:
            logits = -torch.matmul(F.normalize(latent.detach(), dim=-1), F.normalize(self.embedding.weight, dim=-1).t()) / self.tau
            loss = self.beta * F.cross_entropy(logits, indices.detach())
        else:
            raise NotImplementedError(f"Unsupported ETEGRec RQVAE distance: {self.dist}")
        x_q = x + (x_q - x).detach()
        return x_q, loss, indices.view(x.shape[:-1])

    @torch.no_grad()
    def get_maxk_indices(self, x: torch.Tensor, *, maxk: int = 1, used: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.e_dim)
        distances = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        scores = -distances
        topk_scores, topk_indices = scores.topk(int(maxk) + 1, dim=-1)
        if used:
            indices = topk_indices[:, int(maxk)]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = topk_scores[:, int(maxk) - 1] == topk_scores[:, int(maxk) - 1].max()
            indices = torch.where(fix, topk_indices[:, int(maxk) - 1], topk_indices[:, int(maxk)])
        return indices.view(x.shape[:-1]), fix.view(x.shape[:-1])


class EMAVectorQuantizer(VectorQuantizer):
    def __init__(self, config: ETEGRecRQVAEPretrainConfig, *, n_e: int, sk_epsilon: float = 0.003):
        nn.Module.__init__(self)
        self.n_e = int(n_e)
        self.e_dim = int(config.e_dim)
        self.beta = float(config.beta)
        self.kmeans_init = bool(config.kmeans_init)
        self.kmeans_iters = int(config.kmeans_iters)
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(config.sk_iters)
        self.decay = float(config.moving_avg_decay)
        embedding = torch.randn(self.n_e, self.e_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("embedding_avg", embedding.clone())
        self.register_buffer("cluster_size", torch.ones(self.n_e))
        self.initted = not self.kmeans_init

    def get_codebook(self) -> torch.Tensor:
        return self.embedding

    def init_emb(self, data: torch.Tensor) -> None:
        self.embedding.data.copy_(kmeans(data, self.n_e, self.kmeans_iters))
        self.initted = True

    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        num_samples, dim = x.shape
        if num_samples < self.n_e:
            repeats = (self.n_e + num_samples - 1) // num_samples
            std = 0.01 / np.sqrt(dim)
            x = x.repeat(repeats, 1) + torch.randn_like(x.repeat(repeats, 1)) * std
        return x

    def forward(self, x: torch.Tensor, *, conflict: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        distances = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.t())
        )
        if self.sk_epsilon > 0 and (self.training or conflict):
            q = sinkhorn_algorithm(self.center_distance_for_constraint(distances).double(), self.sk_epsilon, self.sk_iters)
            indices = torch.argmax(q, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)
        x_q = F.embedding(indices, self.embedding).view(x.shape)
        if self.training:
            one_hot = F.one_hot(indices, self.n_e).type(latent.dtype)
            embedding_sum = one_hot.t() @ latent
            moving_average(self.cluster_size, one_hot.sum(0), self.decay)
            moving_average(self.embedding_avg, embedding_sum, self.decay)
            n = self.cluster_size.sum()
            cluster_size = laplace_smoothing(self.cluster_size, self.n_e) * n
            self.embedding.data.copy_(self.embedding_avg / cluster_size.unsqueeze(1))
            temp = self._tile(latent)
            temp = temp[torch.randperm(temp.size(0))][: self.n_e]
            usage = (self.cluster_size.view(self.n_e, 1) >= 1).float()
            self.embedding.data.mul_(usage).add_(temp * (1 - usage))
        loss = self.beta * F.mse_loss(x_q.detach(), x)
        x_q = x + (x_q - x).detach()
        return x_q, loss, indices.view(x.shape[:-1])


class GumbelVectorQuantizer(nn.Module):
    def __init__(self, config: ETEGRecRQVAEPretrainConfig, *, n_e: int):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(config.e_dim)
        self.h_dim = int(config.h_dim)
        self.tau = float(config.temperature)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.proj = nn.Linear(self.h_dim, self.n_e, bias=False)

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor, *, conflict: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.h_dim)
        logits = self.proj(latent)
        if self.training or conflict:
            soft_onehot = F.gumbel_softmax(logits, tau=self.tau, dim=-1, hard=False)
        else:
            soft_onehot = F.softmax(logits, dim=-1)
        indices = soft_onehot.argmax(dim=-1)
        x_q = torch.matmul(soft_onehot, self.embedding.weight)
        log_logits = F.log_softmax(logits, dim=-1)
        log_uniform = torch.full_like(log_logits, -torch.log(torch.tensor(self.n_e, device=logits.device)))
        loss = F.kl_div(log_logits, log_uniform, reduction="batchmean", log_target=True)
        return x_q, loss, indices.view(x.shape[:-1])

    @torch.no_grad()
    def get_maxk_indices(self, x: torch.Tensor, *, maxk: int = 1, used: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.h_dim)
        scores = F.softmax(self.proj(latent), dim=-1)
        topk_scores, topk_indices = scores.topk(int(maxk) + 1, dim=-1)
        if used:
            indices = topk_indices[:, int(maxk)]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = topk_scores[:, int(maxk) - 1] == topk_scores[:, int(maxk) - 1].max()
            indices = torch.where(fix, topk_indices[:, int(maxk) - 1], topk_indices[:, int(maxk)])
        return indices.view(x.shape[:-1]), fix.view(x.shape[:-1])


__all__ = [
    "ETEGRecRQVAEPretrainConfig",
    "EMAVectorQuantizer",
    "GumbelVectorQuantizer",
    "RQVAE",
    "ResidualVectorQuantizer",
    "VectorQuantizer",
    "build_pretrain_config",
]
