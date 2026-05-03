from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.base import ModelConfig


@dataclass(slots=True)
class RPGConfig(ModelConfig):
    """Configuration for the RPG sequential retrieval model."""

    name: str = field(default="rpg", metadata={"help": "Registered model name."})

    # Semantic ID source. If semantic_id_file is empty, RPG can build and cache
    # OPQ/PQ semantic IDs from item metadata, matching the original implementation.
    semantic_id_file: str = field(
        default="",
        metadata={
            "help": (
                "Optional JSON file mapping item_id (string) to raw OPQ codes "
                "(list of n_codebook ints in [0, codebook_size)). RPG always interprets "
                "semantic id files as raw codes; pre-offset token ids are not accepted."
            )
        },
    )
    cache_dir: str = field(
        default="",
        metadata={
            "help": (
                "Optional cache directory for generated RPG assets. "
                "Defaults to <parser.data_dir>/rpg."
            )
        },
    )
    semantic_id_cache_file: str = field(
        default="",
        metadata={
            "help": (
                "Optional explicit cache file name for generated OPQ/PQ semantic IDs. "
                "If empty, RPG derives a fingerprinted name from "
                "(sent_emb_model_basename, n_codebook, codebook_size, sent_emb_pca, semantic_embedding_file) "
                "so different generation configurations never share a cache."
            )
        },
    )
    semantic_embedding_file: str = field(
        default="",
        metadata={"help": "Optional .npy semantic embedding file. If absent, embeddings are generated from item metadata."},
    )
    metadata_text_field: str = field(
        default="metadata_text",
        metadata={"help": "Item table column containing prebuilt metadata text."},
    )
    item_title_field: str = field(default="title", metadata={"help": "Fallback item title column."})
    item_description_field: str = field(default="description", metadata={"help": "Fallback item description column."})

    # Sentence embedding and OPQ/PQ tokenizer settings from the original RPG code.
    sent_emb_model: str = field(default="sentence-transformers/sentence-t5-base")
    sent_emb_dim: int = field(default=768)
    sent_emb_pca: int = field(default=128)
    sent_emb_batch_size: int = field(default=512)
    openai_api_key: str | None = field(default=None)
    n_codebook: int = field(default=32)
    codebook_size: int = field(default=256)
    opq_use_gpu: bool = field(default=False)
    opq_gpu_id: int = field(default=0)
    faiss_omp_num_threads: int = field(default=32)

    # GPT2 backbone settings.
    max_item_seq_len: int = field(default=50)
    n_embd: int = field(default=448)
    n_layer: int = field(default=2)
    n_head: int = field(default=4)
    n_inner: int = field(default=1024)
    activation_function: str = field(default="gelu_new")
    resid_pdrop: float = field(default=0.0)
    embd_pdrop: float = field(default=0.5)
    attn_pdrop: float = field(default=0.5)
    layer_norm_epsilon: float = field(default=1e-12)
    initializer_range: float = field(default=0.02)

    # RPG scoring and graph-constrained decoding settings.
    temperature: float = field(default=0.07)
    use_decoding_graph: bool = field(
        default=False,
        metadata={"help": "Whether predict() uses RPG graph-constrained decoding for full retrieval when possible."},
    )
    num_beams: int = field(default=50)
    n_edges: int = field(default=50)
    propagation_steps: int = field(default=3)
    chunk_size: int = field(default=1024)


__all__ = ["RPGConfig"]
