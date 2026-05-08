from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, hidden_sizes: list[int], dropout: float = 0.0, use_bn: bool = False):
        super().__init__()
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            mlp_modules.append(nn.ReLU())

        if mlp_modules:
            mlp_modules.pop()  # remove last ReLU
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LetterVQLayer(nn.Module):
    """Vector quantizer with LETTER diversity regularization."""

    def __init__(
        self,
        *,
        codebook_size: int,
        codebook_dim: int,
        commit_loss_weight: float,
        diversity_loss_weight: float,
        sk_epsilon: float = -1.0,
        sk_iters: int = -1,
    ):
        super().__init__()
        self.dim = codebook_dim
        self.n_embed = codebook_size
        self.commit_loss_weight = commit_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self) -> nn.Parameter:
        return self.embed.weight

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        self.embed.weight.data.copy_(init_embed)

    @staticmethod
    def center_distance(distances: torch.Tensor) -> torch.Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        return (distances - middle) / amplitude

    @torch.no_grad()
    def sinkhorn(self, distances: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
        q = torch.exp(-distances / epsilon)
        b = q.shape[0]
        k = q.shape[1]
        q = q / q.sum(-1, keepdim=True).sum(-2, keepdim=True)
        for _ in range(iterations):
            q = q / torch.sum(q, dim=1, keepdim=True)
            q = q / b
            q = q / torch.sum(q, dim=0, keepdim=True)
            q = q / k
        q *= b
        return q

    def _compute_diversity_loss(self, x_q: torch.Tensor, indices: torch.Tensor, labels: list[int]) -> torch.Tensor:
        # Mathematically-equivalent vectorization of the original per-row Python loop
        # in HonghuiBao2000/LETTER (RQ-VAE/models/vq.py `diversity_loss*`).
        #   original:
        #     indices_cluster[i] = labels[indices[i]]
        #     indices_list[c]    = {code | labels[code] == c}
        #     pos_sample[i]      = random.choice(indices_list[indices_cluster[i]] \ {indices[i]})
        #     sim = x_q @ emb.T ; sim[i, indices[i]] -= 1e12 ; CE(sim, pos_sample)
        device = x_q.device
        batch_size = x_q.size(0)
        n_clusters = 10
        n_embed = int(self.n_embed)

        labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)  # (N,)
        indices_cluster = labels_t.index_select(0, indices)  # (B,)

        # cluster_mask[c, code] = 1 iff labels[code] == c  -> (n_clusters, N)
        cluster_mask = F.one_hot(labels_t, num_classes=n_clusters).t().float()
        # valid_mask[i, code] = 1 iff code in the same cluster as indices[i]  -> (B, N)
        valid_mask = cluster_mask.index_select(0, indices_cluster)
        # Exclude self, equivalent to `while random_element == indices[idx]` rejection
        row_idx = torch.arange(batch_size, device=device)
        valid_mask[row_idx, indices] = 0.0

        # Uniform sampling from the masked candidate set, equivalent to `random.choice`
        # over a uniform subset. Fall back to the uniform distribution across all codes
        # on the pathological row where the cluster only contains `indices[i]` itself
        # (the original code would infinite-loop in that case).
        row_sum = valid_mask.sum(dim=1, keepdim=True)
        sampling_probs = torch.where(row_sum > 0, valid_mask, torch.full_like(valid_mask, 1.0 / n_embed))
        y_true = torch.multinomial(sampling_probs, num_samples=1).squeeze(-1)

        emb = self.get_code_embs()
        sim = torch.matmul(x_q, emb.t())
        sim_self = torch.zeros_like(sim)
        sim_self[row_idx, indices] = 1e12
        sim = sim - sim_self
        return F.cross_entropy(sim, y_true)

    def forward(
        self,
        x: torch.Tensor,
        labels: list[int] | None,
        level: int,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        latent = x.view(-1, self.dim)
        code_embs = self.get_code_embs()
        dist = (
            latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )

        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            dist = self.center_distance(dist).double()
            q = self.sinkhorn(dist, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(q, dim=-1)
        else:
            embed_ind = torch.argmin(dist, dim=-1)

        embed_onehot = F.one_hot(embed_ind, self.n_embed)
        embed_onehot_sum = embed_onehot.sum(0)
        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
        unused_codes = (embed_onehot_sum == 0).sum().item()

        x_q = F.embedding(embed_ind, code_embs).view(x.shape)
        diversity_loss = x_q.new_zeros(())
        if labels is not None and float(self.diversity_loss_weight) > 0:
            diversity_loss = self._compute_diversity_loss(x_q.view(-1, self.dim), embed_ind.view(-1), labels)
        codebook_loss = F.mse_loss(x_q, x.detach())
        commitment_loss = F.mse_loss(x_q.detach(), x)
        quant_loss = codebook_loss + self.commit_loss_weight * commitment_loss + self.diversity_loss_weight * diversity_loss

        x_q = x + (x_q - x).detach()
        embed_ind = embed_ind.view(*x.shape[:-1])
        return x_q, quant_loss, unused_codes, embed_ind

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.get_code_embs())

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        from k_means_constrained import KMeansConstrained

        x_np = x.detach().cpu().numpy()
        n_clusters = self.n_embed
        n_samples = len(x_np)
        if n_samples < n_clusters:
            raise ValueError(
                f"Cannot initialize codebook with {n_clusters} clusters from only {n_samples} samples."
            )
        size_min = max(1, min(n_samples // (n_clusters * 2), 50))
        size_max = max(size_min, size_min * 4, (n_samples + n_clusters - 1) // n_clusters)
        km = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        km.fit(x_np)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float, device=device).view(n_clusters, self.dim)
        self._copy_init_embed(centers.clone())

        code_embs = self.get_code_embs()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        if self.use_sk:
            dist = self.center_distance(dist).double()
            q = self.sinkhorn(dist, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(q, dim=-1).view(*x.shape[:-1])
        else:
            embed_ind = torch.argmin(dist, dim=-1).view(*x.shape[:-1])
        x_q = self.embed_code(embed_ind)
        return x - x_q


class LetterRQLayer(nn.Module):
    """Residual quantization stack for LETTER."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.codebook_sizes = list(config.codebook_size)
        derived_codebook_num = len(self.codebook_sizes)
        if config.codebook_num != derived_codebook_num:
            raise ValueError(
                "config.codebook_num must match len(config.codebook_size), "
                f"got codebook_num={config.codebook_num} and "
                f"len(codebook_size)={derived_codebook_num}"
            )
        self.codebook_num = derived_codebook_num
        self.codebook_dim = config.codebook_dim
        sk_epsilons = list(config.sk_epsilons)
        if len(sk_epsilons) != self.codebook_num:
            raise ValueError(
                "config.sk_epsilons must have the same length as config.codebook_size, "
                f"got len(sk_epsilons)={len(sk_epsilons)} and len(codebook_size)={self.codebook_num}."
            )

        self.vq_layers = nn.ModuleList(
            [
                LetterVQLayer(
                    codebook_size=size,
                    codebook_dim=self.codebook_dim,
                    commit_loss_weight=config.commit_loss_weight,
                    diversity_loss_weight=config.diversity_loss_weight,
                    sk_epsilon=float(sk_epsilon),
                    sk_iters=(config.sk_iters if float(sk_epsilon) > 0 else -1),
                )
                for size, sk_epsilon in zip(self.codebook_sizes, sk_epsilons)
            ]
        )
        self.codebook_num = len(self.vq_layers)

    def forward(
        self,
        x: torch.Tensor,
        labels: dict[str, list[int]] | None,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        batch_size, _ = x.shape
        quantized_x = torch.zeros(batch_size, self.codebook_dim, device=x.device)
        sum_quant_loss = 0.0
        num_unused_codes = 0.0
        output = torch.empty(batch_size, self.codebook_num, dtype=torch.long, device=x.device)

        for level, vq_layer in enumerate(self.vq_layers):
            label = None if labels is None else labels[str(level)]
            quant, quant_loss, unused_codes, output[:, level] = vq_layer(
                x,
                label,
                level,
                infer_use_sk=infer_use_sk,
            )
            x = x - quant
            quantized_x += quant
            sum_quant_loss += quant_loss
            num_unused_codes += unused_codes

        mean_quant_loss = sum_quant_loss / self.codebook_num
        return quantized_x, mean_quant_loss, num_unused_codes, output

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        for vq_layer in self.vq_layers:
            x = vq_layer.init_codebook(x, device)
        return x


__all__ = ["LetterRQLayer", "LetterVQLayer", "MLP"]
