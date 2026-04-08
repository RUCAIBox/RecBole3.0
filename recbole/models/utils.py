import torch


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


def init_mlp_xavier_weights_zero_bias(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


def get_current_embeddings(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (lengths - 1) + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.clamp(
        torch.linalg.norm(x, ord=None, dim=-1, keepdim=True),
        min=eps,
    )

