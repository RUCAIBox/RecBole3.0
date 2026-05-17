from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.init import xavier_normal_


class MLPLayers(nn.Module):
    """MLP helper used by the original ETEGRec RQVAE pretraining code."""

    def __init__(
        self,
        layers: list[int] | tuple[int, ...],
        *,
        dropout: float = 0.0,
        activation: str | type[nn.Module] | None = "relu",
        bn: bool = False,
    ):
        super().__init__()
        self.layers = list(layers)
        self.dropout = float(dropout)
        self.activation = activation
        self.use_bn = bool(bn)

        modules: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:], strict=False)):
            is_last = index == len(self.layers) - 2
            modules.append(nn.Dropout(p=self.dropout))
            modules.append(nn.Linear(input_size, output_size))
            if self.use_bn and not is_last:
                modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_module = activation_layer(self.activation)
            if activation_module is not None and not is_last:
                modules.append(activation_module)

        self.mlp_layers = nn.Sequential(*modules)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(input_feature)


def activation_layer(activation_name: str | type[nn.Module] | None = "relu") -> nn.Module | None:
    if activation_name is None:
        return None
    if isinstance(activation_name, str):
        name = activation_name.lower()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "leakyrelu":
            return nn.LeakyReLU()
        if name == "none":
            return None
    elif issubclass(activation_name, nn.Module):
        return activation_name()
    raise NotImplementedError(f"activation function {activation_name} is not implemented")


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10) -> torch.Tensor:
    device = samples.device
    x = samples.detach().cpu().numpy()
    cluster = KMeans(n_clusters=int(num_clusters), max_iter=int(num_iters), random_state=42).fit(x)
    return torch.from_numpy(cluster.cluster_centers_).to(device=device, dtype=samples.dtype)


@torch.no_grad()
def sinkhorn_algorithm(distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int) -> torch.Tensor:
    q = torch.exp(-distances / float(epsilon))
    batch_size = q.shape[0]
    num_codes = q.shape[1]

    q /= q.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
    for _ in range(int(sinkhorn_iterations)):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= batch_size
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= num_codes
    q *= batch_size
    return q


def moving_average(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(float(decay)).add_(new, alpha=1.0 - float(decay))


def laplace_smoothing(x: torch.Tensor, n_categories: int, eps: float = 1e-5) -> torch.Tensor:
    return (x + eps) / (x.sum() + int(n_categories) * eps)


__all__ = [
    "MLPLayers",
    "activation_layer",
    "kmeans",
    "laplace_smoothing",
    "moving_average",
    "sinkhorn_algorithm",
]
