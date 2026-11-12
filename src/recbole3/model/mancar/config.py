from dataclasses import dataclass

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class ManCARConfig(SequentialModelConfig):
    name: str = "mancar"
    history_max_length: int = 50
    embedding_dim: int = 256
    num_layers: int = 2
    num_heads: int = 2
    inner_size: int = 300
    dropout: float = 0.5
    hidden_act: str = "gelu"
    rms_norm_eps: float = 1e-12
    reason_step: int = 5
    temperature: float = 1.0
    pl_weight: float = 1.0
    noise_factor: float = 0.1
    trigger_num: int = 4
    items_per_trigger: int = 7
    swing_topk: int = 100
    swing_alpha1: float = 5.0
    swing_alpha2: float = 1.0
    swing_beta: float = 0.3
    swing_m: int = 10
    swing_min_u: int = 2
    graph_seed: int = 0
