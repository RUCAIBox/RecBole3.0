from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.base import ModelConfig


@dataclass(slots=True)
class RQVAEConfig(ModelConfig):
    """Configuration for the RQ-VAE (Residual Quantized Variational AutoEncoder) model."""

    name: str = field(default="rqvae", metadata={"help": "Registered model name."})

    # RQ-VAE architecture
    hidden_sizes: tuple[int, ...] = field(
        default=(2048, 1024, 512, 256),
        metadata={"help": "Hidden layer sizes for the encoder (excluding input and output layers)."},
    )
    codebook_num: int = field(
        default=3,
        metadata={"help": "Number of residual quantization layers."},
    )
    codebook_size: int = field(
        default=256,
        metadata={"help": "Size of each codebook (number of code vectors)."},
    )
    codebook_dim: int = field(
        default=128,
        metadata={"help": "Dimension of each code vector."},
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout probability for encoder/decoder layers."})

    sk_epsilon: float = field(
        default=-1,
        metadata={"help": "Epsilon value for sinkhorn algorithm."},
    )

    sk_iters: int = field(
        default=-1,
        metadata={"help": "Number of iterations for sinkhorn algorithm."},
    )

    # Quantization parameters
    beta: float = field(default=0.25, metadata={"help": "Commitment loss coefficient."})
    vq_type: str = field(
        default="ema",
        metadata={"help": "Type of vector quantization: 'vq', 'ema', or 'simvq'."},
    )
    ema_decay: float = field(default=0.99, metadata={"help": "EMA decay rate for codebook updates (vq_type=ema)."})
    fix_code_embs: bool = field(
        default=False,
        metadata={"help": "Whether to fix codebook embeddings during training (vq_type=simvq)."},
    )

    # Semantic embedding configuration
    sem_emb_file: str = field(
        default="sentence_t5.npy",
        metadata={"help": "Path to semantic embedding file (relative to data directory)."},
    )
    sem_emb_dim: int = field(
        default=768,
        metadata={"help": "Dimension of semantic embeddings."},
    )
    sem_emb_pca: int = field(
        default=0,
        metadata={"help": "PCA dimension for embedding reduction (0 = no PCA)."},
    )

    # Text encoding configuration (for dynamic embedding generation)
    text_field: str = field(
        default="metadata_text",
        metadata={"help": "Field name in item_table for text encoding."},
    )
    sent_emb_model: str = field(
        default="sentence-transformers/sentence-t5-base",
        metadata={"help": "Sentence transformer model name for text encoding."},
    )
    sent_emb_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for sentence transformer encoding."},
    )

    # SID generation parameters
    sid_collision_handling: str = field(
        default="extend",
        metadata={"help": "Method for handling token collisions: 'sinkhorn' (iterative sinkhorn) or 'extend' (extend last digit)."},
    )
    sid_output_file: str = field(
        default="item_sids.json",
        metadata={"help": "Output file name for generated semantic IDs (saved in output directory)."},
    )


__all__ = ["RQVAEConfig"]
