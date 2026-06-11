from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig
from recbole3.trainer_config import TrainerConfig


@dataclass(slots=True)
class ETEGRecConfig(SequentialModelConfig):
    """Configuration for the ETEGRec generative retrieval model."""

    name: str = field(default="etegrec", metadata={"help": "Registered model name."})
    semantic_emb_file: str = field(
        default="",
        metadata={"help": "Path to item semantic embeddings aligned to remapped 0-based item ids."},
    )
    rqvae_path: str = field(default="", metadata={"help": "Optional pretrained ETEGRec RQVAE checkpoint path."})

    history_max_length: int | None = field(
        default=50,
        metadata={"help": "Maximum number of recent history items used by ETEGRec."},
    )
    semantic_hidden_size: int = field(default=256, metadata={"help": "Dimension of semantic item embeddings."})
    code_num: int = field(default=256, metadata={"help": "Number of codes in each RQVAE codebook."})
    code_length: int = field(default=4, metadata={"help": "Semantic code length including collision token."})
    num_beams: int = field(default=20, metadata={"help": "Beam width used during generation."})
    eval_topk: tuple[int, ...] = field(
        default=(5, 10),
        metadata={"help": "Top-k values expected during full retrieval evaluation."},
    )

    num_layers: int = field(default=6, metadata={"help": "Number of T5 encoder layers."})
    num_decoder_layers: int = field(default=6, metadata={"help": "Number of T5 decoder layers."})
    d_model: int = field(default=128, metadata={"help": "T5 hidden size."})
    d_ff: int = field(default=512, metadata={"help": "T5 feed-forward hidden size."})
    num_heads: int = field(default=4, metadata={"help": "Number of T5 attention heads."})
    d_kv: int = field(default=64, metadata={"help": "T5 key/value projection size per head."})
    dropout_rate: float = field(default=0.1, metadata={"help": "T5 dropout rate."})
    activation_function: str = field(default="relu", metadata={"help": "T5 activation function."})
    feed_forward_proj: str = field(default="relu", metadata={"help": "T5 feed-forward projection type."})
    max_positions: int = field(default=210, metadata={"help": "T5 maximum position budget."})

    num_emb_list: tuple[int, ...] = field(
        default=(256, 256, 256),
        metadata={"help": "ETEGRec RQVAE codebook sizes."},
    )
    e_dim: int = field(default=128, metadata={"help": "ETEGRec RQVAE code embedding dimension."})
    layers: tuple[int, ...] = field(default=(512, 256), metadata={"help": "ETEGRec RQVAE MLP hidden sizes."})
    vq_type: str = field(default="vq", metadata={"help": "Vector quantizer type."})
    dist: str = field(default="l2", metadata={"help": "RQVAE code distance metric."})
    beta: float = field(default=0.25, metadata={"help": "RQVAE commitment loss weight."})
    quant_loss_weight: float = field(default=1.0, metadata={"help": "Standalone RQVAE pretraining quantization loss weight."})
    tau: float = field(default=0.07, metadata={"help": "Contrastive loss temperature."})
    kmeans_init: bool = field(default=False, metadata={"help": "Whether to initialize codebooks with k-means."})
    kmeans_iters: int = field(default=100, metadata={"help": "Maximum k-means iterations."})
    sk_epsilons: tuple[float, ...] = field(
        default=(0.0, 0.0, 0.0),
        metadata={"help": "Sinkhorn epsilons used by standalone RQVAE pretraining."},
    )
    sk_iters: int = field(default=50, metadata={"help": "Sinkhorn iterations used by standalone RQVAE pretraining."})
    moving_avg_decay: float = field(default=0.99, metadata={"help": "EMA quantizer moving average decay."})
    h_dim: int = field(default=2048, metadata={"help": "Hidden size for gumbel RQVAE pretraining."})
    temperature: float = field(default=0.9, metadata={"help": "Gumbel softmax temperature for RQVAE pretraining."})
    dropout_prob: float = field(default=0.0, metadata={"help": "RQVAE dropout probability."})
    bn: bool = field(default=False, metadata={"help": "Whether to use batch normalization in RQVAE MLPs."})
    loss_type: str = field(default="mse", metadata={"help": "RQVAE reconstruction loss type."})


@dataclass(slots=True)
class ETEGRecTrainerConfig(TrainerConfig):
    """Trainer configuration for ETEGRec alternating optimization."""

    eval_batch_size: int | None = field(default=None, metadata={"help": "Batch size used only for ETEGRec valid/test generation."})
    eval_steps: int = field(default=1, metadata={"help": "Run ETEGRec joint validation every N epochs; the last epoch is always evaluated."})
    lr_rec: float = field(default=0.005, metadata={"help": "Learning rate for the ETEGRec recommender/T5 side."})
    lr_id: float = field(default=0.0001, metadata={"help": "Learning rate for the ETEGRec RQVAE tokenizer side."})
    cycle: int = field(default=2, metadata={"help": "Train RQVAE every N epochs and recommender otherwise."})
    lr_scheduler_type: str | None = field(default=None, metadata={"help": "ETEGRec rec/id scheduler type, e.g. cosine."})
    warmup_steps: int = field(default=0, metadata={"help": "Shared warmup steps used by the original ETEGRec schedulers."})
    rec_warmup_steps: int | None = field(default=None, metadata={"help": "Optional recommender scheduler warmup override."})
    id_warmup_steps: int | None = field(default=None, metadata={"help": "Optional tokenizer scheduler warmup override."})
    warm_epoch: int = field(default=10, metadata={"help": "Epochs before auxiliary ETEGRec losses are enabled."})
    gradient_clip_norm: float = field(default=1.0, metadata={"help": "Gradient clipping max norm."})
    alpha: float = field(default=1.0, metadata={"help": "RQVAE quantization loss weight."})
    rec_code_loss: float = field(default=1.0, metadata={"help": "Weight for recommender code prediction loss."})
    rec_kl_loss: float = field(default=0.0, metadata={"help": "Reserved KL loss weight for later ETEGRec alignment."})
    rec_dec_cl_loss: float = field(default=0.0, metadata={"help": "Reserved decoder contrastive loss weight."})
    id_vq_loss: float = field(default=1.0, metadata={"help": "Weight for RQVAE tokenizer reconstruction/commitment loss."})
    id_code_loss: float = field(default=0.0, metadata={"help": "Reserved code loss weight for tokenizer updates."})
    id_kl_loss: float = field(default=0.0, metadata={"help": "Reserved KL loss weight for tokenizer updates."})
    id_dec_cl_loss: float = field(default=0.0, metadata={"help": "Reserved decoder contrastive loss weight for tokenizer updates."})
    finetune_enabled: bool = field(default=False, metadata={"help": "Reserved flag for the original ETEGRec finetune phase."})
    finetune_epochs: int = field(default=100, metadata={"help": "Number of ETEGRec recommender-only finetune epochs."})
    finetune_lr: float = field(default=5e-4, metadata={"help": "Learning rate for ETEGRec recommender-only finetuning."})
    finetune_patience: int = field(default=10, metadata={"help": "Early stopping patience used during ETEGRec finetuning."})
    finetune_eval_steps: int = field(default=1, metadata={"help": "Validation interval used during ETEGRec finetuning."})
    finetune_lr_scheduler_type: str | None = field(
        default="cosine",
        metadata={"help": "Scheduler type used by ETEGRec recommender-only finetuning."},
    )
    finetune_warmup_steps: int = field(default=0, metadata={"help": "Warmup steps for ETEGRec finetune scheduler."})


__all__ = ["ETEGRecConfig", "ETEGRecTrainerConfig"]
