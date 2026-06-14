from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.base import ModelConfig


@dataclass(slots=True)
class LETTERConfig(ModelConfig):
    """Configuration for LETTER tokenizer."""

    name: str = field(default="letter", metadata={"help": "Registered model name."})

    # Encoder / decoder architecture
    layers: tuple[int, ...] = field(
        default=(2048, 1024, 512, 256, 128, 64),
        metadata={"help": "Hidden layer sizes in original LETTER implementation before codebook_dim projection."},
    )
    codebook_num: int = field(default=4, metadata={"help": "Number of residual quantization levels."})
    codebook_size: tuple[int, ...] = field(
        default=(256, 256, 256, 256),
        metadata={
            "help": (
                "Per-level codebook sizes (LETTER num_emb_list semantics). "
                "Length must equal codebook_num; strict LETTER reproduction requires each value to be 256."
            )
        },
    )
    codebook_dim: int = field(default=32, metadata={"help": "Dimension of each code embedding."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout probability for MLP layers."})
    bn: bool = field(default=False, metadata={"help": "Whether to use BatchNorm1d in encoder/decoder MLP."})
    loss_type: str = field(default="mse", metadata={"help": "Reconstruction loss type: 'mse' or 'l1'."})

    # Quantization / sinkhorn
    commit_loss_weight: float = field(default=0.25, metadata={"help": "Commitment loss weight in VQ objective."})
    quant_loss_weight: float = field(default=1.0, metadata={"help": "Global weight for quantization loss term."})
    sk_epsilons: tuple[float, ...] = field(
        default=(0.0, 0.0, 0.0, 0.003),
        metadata={"help": "Per-level sinkhorn epsilon list in original LETTER style (0 disables that level)."},
    )
    sk_iters: int = field(default=50, metadata={"help": "Sinkhorn iterations shared by all levels."})

    # LETTER regularizers
    cf_loss_weight: float = field(default=0.01, metadata={"help": "Weight for collaborative alignment loss."})
    diversity_loss_weight: float = field(
        default=1e-4, metadata={"help": "Weight for code assignment diversity loss in each codebook."}
    )

    # Semantic embeddings
    sem_emb_file: str = field(
        default="sentence_t5.npy",
        metadata={"help": "Path to cached semantic embeddings (relative to data directory)."},
    )
    sem_emb_dim: int = field(default=768, metadata={"help": "Dimension of semantic embeddings."})
    sem_emb_pca: int = field(
        default=0,
        metadata={"help": "PCA dimension for embedding reduction (0 = no PCA)."},
    )
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

    # Collaborative embeddings for CF loss
    cf_emb_file: str = field(
        default="",
        metadata={"help": "Path to cached collaborative embedding file loaded by torch.load (LETTER style)."},
    )

    # SID generation
    sid_collision_handling: str = field(
        default="sinkhorn",
        metadata={"help": "Collision handling for SIDs: 'sinkhorn' or 'extend'."},
    )
    sid_output_file: str = field(
        default="item_sids.json",
        metadata={"help": "Output file name for generated SIDs (saved under output directory)."},
    )


__all__ = ["LETTERConfig"]
