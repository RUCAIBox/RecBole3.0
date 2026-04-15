from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class MLP(nn.Module):
    """Multi-Layer Perceptron module.

    Args:
        hidden_sizes: List of layer sizes (including input and output).
        dropout: Dropout probability.

    Attributes:
        mlp: Sequential container for the MLP layers.
    """

    def __init__(self, hidden_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())

        # Remove the last ReLU
        if mlp_modules:
            mlp_modules.pop()
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.mlp(x)


class VQLayer(nn.Module):
    """Vector Quantization layer.

    Args:
        codebook_size: Number of code vectors in the codebook.
        codebook_dim: Dimension of each code vector.
        beta: Commitment loss coefficient.
    """

    def __init__(self, codebook_size: int,
                 codebook_dim: int,
                 beta: float = 0.25,
                 sk_epsilon: float = -1,
                 sk_iters: int = -1,
                 ):
        super().__init__()
        self.dim = codebook_dim
        self.n_embed = codebook_size
        self.beta = beta
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self) -> nn.Parameter:
        """Get the codebook embeddings."""
        return self.embed.weight

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        """Initialize codebook embeddings with given values."""
        self.embed.weight.data.copy_(init_embed)

    @staticmethod
    def center_distance(distances: torch.Tensor) -> torch.Tensor:
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    @torch.no_grad()
    def sinkhorn(self, distances: torch.Tensor, epsilon: float = 0.003, iterations: int = 5) -> torch.Tensor:
        Q = torch.exp(- distances / epsilon)

        B = Q.shape[0]  # number of samples to assign
        K = Q.shape[1]  # how many centroids per block (usually set to 256)

        # make the matrix sums to 1
        sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)

        Q /= sum_Q
        for _ in range(iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_0 = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_0
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q

    def forward(
        self, x: torch.Tensor, infer_use_sk: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Forward pass of VQ layer.

        Args:
            x: Input tensor.

        Returns:
            Quantized tensor, quantization loss, number of unused codes, and code indices.
        """
        latent = x.view(-1, self.dim)
        code_embs = self.get_code_embs()
        dist = (
            latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )

        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            dist = self.center_distance(dist)
            dist = dist.double()
            Q = self.sinkhorn(dist, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise RuntimeError(f"Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(Q, dim=-1)
        else:
            embed_ind = torch.argmin(dist, dim=-1)

        embed_onehot = F.one_hot(embed_ind, self.n_embed)
        embed_onehot_sum = embed_onehot.sum(0)
        if distributed.is_initialized():
            distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
        unused_codes = (embed_onehot_sum == 0).sum().item()

        x_q = F.embedding(embed_ind, code_embs).view(x.shape)

        quant_loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()

        embed_ind = embed_ind.view(*x.shape[:-1])

        return x_q, quant_loss, unused_codes, embed_ind

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Embed code indices to vectors."""
        code_embs = self.get_code_embs()
        return F.embedding(embed_id, code_embs)

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Initialize codebook using K-means clustering.

        Args:
            x: Input tensor for clustering.
            device: Device for codebook.

        Returns:
            Residual tensor (x - quantized(x)).
        """
        kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(x.detach().cpu().numpy())

        centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float, device=device
        ).view(self.n_embed, self.dim)

        if distributed.is_initialized():
            distributed.broadcast(centers, 0)

        self._copy_init_embed(centers.clone())

        code_embs = self.get_code_embs()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        embed_ind = torch.argmin(dist, dim=-1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        x_q = self.embed_code(embed_ind)

        return x - x_q


class EMAVQLayer(VQLayer):
    """EMA (Exponential Moving Average) VQ layer.

    Args:
        codebook_size: Number of code vectors in the codebook.
        codebook_dim: Dimension of each code vector.
        beta: Commitment loss coefficient.
        decay: EMA decay rate.
        eps: Small epsilon for numerical stability.
    """

    def __init__(
        self, codebook_size: int,
            codebook_dim: int,
            beta: float = 0.25,
            sk_epsilon: float = -1,
            sk_iters: int = -1,
            decay: float = 0.99,
            eps: float = 1e-5
    ):
        super().__init__(codebook_size, codebook_dim, beta, sk_epsilon, sk_iters)

        self.decay = decay
        self.eps = eps

        embed = torch.zeros(self.n_embed, self.dim)
        self.embed = nn.Parameter(embed, requires_grad=False)
        nn.init.xavier_normal_(self.embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(self.n_embed))

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        """Initialize codebook embeddings and buffers."""
        self.embed.data.copy_(init_embed)
        self.embed_avg.data.copy_(init_embed)
        self.cluster_size.data.copy_(torch.ones(self.n_embed, device=init_embed.device))

    def get_code_embs(self) -> nn.Parameter:
        """Get the codebook embeddings."""
        return self.embed

    def forward(
        self, x: torch.Tensor, infer_use_sk: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Forward pass of EMA VQ layer.

        Args:
            x: Input tensor.

        Returns:
            Quantized tensor, quantization loss, number of unused codes, and code indices.
        """
        latent = x.view(-1, self.dim)
        code_embs = self.get_code_embs()
        dist = (
            latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )

        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            dist = self.center_distance(dist)
            dist = dist.double()
            Q = self.sinkhorn(dist, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise RuntimeError(f"Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(Q, dim=-1)
        else:
            embed_ind = torch.argmin(dist, dim=-1)

        x_q = F.embedding(embed_ind, code_embs).view(x.shape)

        if self.training:
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = embed_onehot.t() @ latent

            if distributed.is_initialized():
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)

            unused_codes = (embed_onehot_sum == 0).sum().item()

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            norm_w = n * (self.cluster_size + self.eps) / (n + self.n_embed * self.eps)
            embed_normalized = self.embed_avg / norm_w.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        else:
            embed_onehot = F.one_hot(embed_ind, self.n_embed)
            embed_onehot_sum = embed_onehot.sum(0)
            if distributed.is_initialized():
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
            unused_codes = (embed_onehot_sum == 0).sum().item()

        quant_loss = self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()

        embed_ind = embed_ind.view(*x.shape[:-1])

        return x_q, quant_loss, unused_codes, embed_ind


class SimVQLayer(VQLayer):
    """Simulated VQ layer with projection.

    Args:
        codebook_size: Number of code vectors in the codebook.
        codebook_dim: Dimension of each code vector.
        beta: Commitment loss coefficient.
        fix_code_embs: Whether to fix codebook embeddings during training.
    """

    def __init__(
        self, codebook_size: int,
            codebook_dim: int,
            beta: float = 0.25,
            sk_epsilon: float = -1,
            sk_iters: int = -1,
            fix_code_embs: bool = False
    ):
        super().__init__(codebook_size, codebook_dim, beta, sk_epsilon, sk_iters)

        nn.init.xavier_normal_(self.embed.weight)

        self.embed_proj = nn.Linear(self.dim, self.dim, bias=False)
        if fix_code_embs:
            for param in self.embed.parameters():
                param.requires_grad = False

    def get_code_embs(self) -> Any:
        """Get the projected codebook embeddings."""
        return self.embed_proj(self.embed.weight)

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        """No-op initialization (codebook initialized via projection)."""
        pass


class RQLayer(nn.Module):
    """Residual Quantization layer.

    Args:
        config: RQVAEConfig object containing codebook parameters.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.codebook_num = config.codebook_num
        self.codebook_dim = config.codebook_dim

        # Check if codebook_size is an int and convert it to a list of the same size for each level
        if isinstance(config.codebook_size, int):
            self.codebook_sizes = [config.codebook_size] * self.codebook_num
        elif isinstance(config.codebook_size, list) or isinstance(config.codebook_size, tuple):
            if len(config.codebook_size) == self.codebook_num:
                self.codebook_sizes = list(config.codebook_size)
            else:
                raise ValueError(
                    "codebook_size must be an int or a list of int with length equal to codebook_num"
                )

        self.vq_type = config.vq_type
        self.vq_beta = config.beta
        self.sk_epsilon = config.sk_epsilon
        self.sk_iters = config.sk_iters

        if self.vq_type == "vq":
            self.vq_layers = nn.ModuleList(
                [
                    VQLayer(
                        codebook_size=size,
                        codebook_dim=self.codebook_dim,
                        beta=self.vq_beta
                    )
                    for size in self.codebook_sizes[:-1]
                ] + [
                    VQLayer(
                        codebook_size=self.codebook_sizes[-1],
                        codebook_dim=self.codebook_dim,
                        beta=self.vq_beta,
                        sk_epsilon=self.sk_epsilon,
                        sk_iters=self.sk_iters
                    )
                ]
            )
        elif self.vq_type == "ema":
            self.vq_layers = nn.ModuleList(
                [
                    EMAVQLayer(
                        codebook_size=size,
                        codebook_dim=self.codebook_dim,
                        beta=self.vq_beta,
                        decay=config.ema_decay,
                    )
                    for size in self.codebook_sizes[:-1]
                ] + [
                    EMAVQLayer(
                        codebook_size=self.codebook_sizes[-1],
                        codebook_dim=self.codebook_dim,
                        beta=self.vq_beta,
                        sk_epsilon=self.sk_epsilon,
                        sk_iters=self.sk_iters,
                        decay=config.ema_decay,
                    )
                ]
            )
        elif self.vq_type == "simvq":
            self.vq_layers = nn.ModuleList(
                [
                    SimVQLayer(
                        codebook_size=size,
                        codebook_dim=self.codebook_dim,
                        beta=self.vq_beta,
                        fix_code_embs=config.fix_code_embs,
                    )
                    for size in self.codebook_sizes[:-1]
                ] + [
                    SimVQLayer(
                        codebook_size=self.codebook_sizes[-1],
                        codebook_dim=self.codebook_dim,
                        beta=self.vq_beta,
                        sk_epsilon=self.sk_epsilon,
                        sk_iters=self.sk_iters,
                        fix_code_embs=config.fix_code_embs,
                    )
                ]
            )

    def forward(
        self, x: torch.Tensor, infer_use_sk: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Forward pass of RQ layer.

        Args:
            x: Input tensor.

        Returns:
            Quantized tensor, mean quantization loss, number of unused codes, and code indices.
        """
        batch_size, _ = x.shape
        quantized_x = torch.zeros(batch_size, self.codebook_dim, device=x.device)
        sum_quant_loss = 0.0
        num_unused_codes = 0.0
        output = torch.empty(batch_size, self.codebook_num, dtype=torch.long, device=x.device)

        for vq_layer, level in zip(self.vq_layers, range(self.codebook_num)):
            quant, quant_loss, unused_codes, output[:, level] = vq_layer(x, infer_use_sk)
            x = x - quant
            quantized_x += quant
            sum_quant_loss += quant_loss
            num_unused_codes += unused_codes

        mean_quant_loss = sum_quant_loss / self.codebook_num

        return quantized_x, mean_quant_loss, num_unused_codes, output

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Initialize all codebook levels using K-means clustering.

        Args:
            x: Input tensor for clustering.
            device: Device for codebooks.

        Returns:
            Residual tensor after all quantization levels.
        """
        for vq_layer in self.vq_layers:
            x = vq_layer.init_codebook(x, device)
        return x


__all__ = ["EMAVQLayer", "MLP", "RQLayer", "SimVQLayer", "VQLayer"]
