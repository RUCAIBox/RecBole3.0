"""BIGRec configuration dataclass.

BIGRec (Bi-step Grounding Paradigm for Recommendation) uses a two-step approach:
  Step 1 - Fine-tune a causal LLM (LLaMA + LoRA) to generate item title text
            given a user's interaction history.
  Step 2 - Ground the generated text to actual items by computing L2 distance
            between the LLM-derived oracle embedding and pre-computed item
            embeddings, then returning the closest items as recommendations.

Reference:
    Bao et al., "A Bi-Step Grounding Paradigm for Large Language Models in
    Recommendation Systems", arXiv:2308.08434 (2023).
    Official code: https://github.com/SAI990323/Grounding4Rec
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class BIGRecConfig(SequentialModelConfig):
    """Full configuration for the BIGRec model.

    Parameters are grouped by functional area to keep the config readable.
    """

    name: str = field(default="bigrec", metadata={"help": "Registered model name."})

    # ── LLM Backbone ──────────────────────────────────────────────────────────
    llm_path: str = field(
        default="",
        metadata={"help": "Local path (or HuggingFace hub identifier) to the pretrained LLaMA model."},
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": (
                "CUDA device index used in single-process (non-DDP) mode. "
                "Ignored when torchrun sets LOCAL_RANK. "
                "Set CUDA_VISIBLE_DEVICES before launching to pick a specific physical GPU, "
                "then leave device_id=0 (the first visible device)."
            )
        },
    )
    torch_dtype: str = field(
        default="float16",
        metadata={"help": "Model weight dtype loaded by from_pretrained(). 'float16' or 'bfloat16'."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in INT8 quantization via bitsandbytes. Reduces VRAM by ~50%%."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention backend. 'eager' (default) or 'flash_attention_2' (needs package)."},
    )

    # ── LoRA Fine-tuning ──────────────────────────────────────────────────────
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to apply LoRA (Parameter-Efficient Fine-Tuning)."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank. Paper default: 8."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor. Paper default: 16."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout applied to LoRA layers."},
    )
    lora_target_modules: tuple[str, ...] = field(
        default=("q_proj", "v_proj"),
        metadata={"help": "Names of linear sub-modules to replace with LoRA adapters."},
    )

    # ── Tokenization ──────────────────────────────────────────────────────────
    max_input_length: int = field(
        default=512,
        metadata={"help": "Maximum token length for the instruction + input portion of the prompt."},
    )
    max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of new tokens to generate during beam-search inference."},
    )

    # ── Training (HuggingFace Trainer Arguments) ──────────────────────────────
    train_batch_size: int = field(
        default=4,
        metadata={"help": "Per-device training batch size."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of gradient accumulation micro-steps before each optimizer update."},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs."},
    )
    learning_rate: float = field(
        default=3e-4,
        metadata={"help": "Peak learning rate for the optimizer. Official BIGRec default: 3e-4."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "L2 weight decay applied to non-bias parameters."},
    )
    warmup_steps: int | None = field(
        default=20,
        metadata={
            "help": (
                "Fixed number of warm-up steps (official BIGRec default: 20). "
                "Takes precedence over warmup_ratio when set to a non-None value."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of total steps used for linear LR warm-up. "
                "Only applied when warmup_steps is None."
            )
        },
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning-rate schedule: 'cosine', 'linear', 'constant', etc."},
    )
    fp16: bool = field(
        default=True,
        metadata={
            "help": (
                "Use float16 mixed-precision training (official BIGRec default). "
                "Mutually exclusive with bf16. Requires a CUDA device with FP16 support."
            )
        },
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bfloat16 mixed precision. Mutually exclusive with fp16 (set by torch_dtype)."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer name passed to HuggingFace TrainingArguments. Official default: 'adamw_torch'."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Recompute activations to reduce peak VRAM at the cost of extra compute."},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log training metrics every N optimizer steps."},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={
            "help": (
                "HuggingFace Trainer checkpoint save strategy: 'epoch', 'steps', or 'no'. "
                "Must match evaluation_strategy when load_best_model_at_end=True. "
                "Official BIGRec default: 'epoch'."
            )
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Maximum number of checkpoints to keep on disk. Oldest are deleted when exceeded."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": (
                "When True, restore the checkpoint with the lowest validation LM loss "
                "after training finishes. Requires save_strategy='epoch'. "
                "Official BIGRec default: True."
            )
        },
    )
    deepspeed: str | None = field(
        default=None,
        metadata={"help": "Path to a DeepSpeed JSON config for multi-GPU / ZeRO optimization."},
    )

    # ── Early Stopping (LM Validation Loss, matching official BIGRec) ────────
    early_stopping_patience: int = field(
        default=5,
        metadata={
            "help": (
                "Number of consecutive per-epoch evaluations with no improvement in "
                "validation LM loss before training stops early. "
                "Passed directly to HuggingFace EarlyStoppingCallback. "
                "Official BIGRec default: 5."
            )
        },
    )

    # ── Generation  (Eval Step 1) ─────────────────────────────────────────────
    num_beams: int = field(
        default=4,
        metadata={"help": "Beam-search width during evaluation. Paper default: 4."},
    )
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Per-device batch size used during beam-search generation at eval time."},
    )

    # ── Embedding Grounding (Eval Step 2) ─────────────────────────────────────
    history_max_length: int | None = field(
        default=10,
        metadata={"help": "Number of most-recent history items included in the prompt. Paper uses 10."},
    )
    item_text_field: str = field(
        default="title",
        metadata={"help": "Column in item_table used as the natural-language item name in prompts."},
    )
    fallback_item_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Fallback item_table column used when item_text_field is absent or empty."},
    )
    domain: str = field(
        default="product",
        metadata={
            "help": (
                "Recommendation domain controlling prompt wording. "
                "Supported: 'movie', 'product', 'item'."
            )
        },
    )
    embedding_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size used when encoding item titles to build the embedding index."},
    )
    embedding_cache_dir: str = field(
        default="outputs/bigrec/embeddings",
        metadata={"help": "Directory where pre-computed item embedding tensors are cached as .pt files."},
    )
    refresh_embedding_cache: bool = field(
        default=False,
        metadata={"help": "Re-compute item embeddings even when a cached file already exists."},
    )

    # ── Grounding Weight Injection (Eq. 3 in BIGRec paper) ───────────────────
    # After computing raw L2 distances, BIGRec optionally reweights them by
    # statistical signals to improve ranking accuracy:
    #
    #   D̂ᵢ = (Dᵢ − min D) / (max D − min D)      [per-row L2 normalisation]
    #   D̃ᵢ = D̂ᵢ × (1 + Wᵢ)^(−γ)                 [popularity / CF adjustment]
    #
    # A larger Wᵢ (popular or highly-scored item) → smaller D̃ᵢ → higher rank.
    # γ controls how strongly the weight signal bends the ranking.
    grounding_mode: str = field(
        default="none",
        metadata={
            "help": (
                "Weight injection mode for Step-2 grounding. "
                "'none': pure L2 distance (no reweighting). "
                "'popularity': inject min-max-normalised training-interaction counts. "
                "'cf': inject pre-computed CF model scores (requires cf_score_path). "
                "'popularity+cf': sum both signals then re-normalise to [0, 1]."
            )
        },
    )
    grounding_gamma: float = field(
        default=1.0,
        metadata={
            "help": (
                "γ exponent in the grounding weight formula (Eq. 3). "
                "γ=0 disables reweighting. Larger γ → stronger popularity/CF pull. "
                "Paper searches [0, 100] on the validation split per top-K."
            )
        },
    )
    cf_score_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a .pt file containing a 1-D float tensor of shape [num_items] "
                "with pre-computed CF model scores (e.g., from BPR or SASRec). "
                "Required when grounding_mode contains 'cf'."
            )
        },
    )

    # ── Training / Evaluation Behaviour Flags ────────────────────────────────
    train_on_inputs: bool = field(
        default=True,
        metadata={
            "help": (
                "When True (official BIGRec default), cross-entropy loss is computed "
                "over the full token sequence (prompt + response). "
                "When False, prompt tokens are masked with -100 so loss is computed "
                "only on the generated response (response-only supervision)."
            )
        },
    )
    embedding_use_base_model: bool = field(
        default=True,
        metadata={
            "help": (
                "When True (official BIGRec default), a fresh base CausalLM (no LoRA "
                "adapter) is loaded for both item-embedding pre-computation and oracle-"
                "embedding extraction at evaluation time, ensuring both embeddings live "
                "in the same vector space. "
                "When False, the fine-tuned LoRA model is used for embedding extraction."
            )
        },
    )
    grounding_gamma_search: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, automatically grid-search the best γ independently for each "
                "metric×K combination on the validation split (official BIGRec procedure). "
                "The best-found γ per metric@K is then applied on the test split. "
                "Has no effect when grounding_mode='none'."
            )
        },
    )
    grounding_gamma_search_values: tuple[float, ...] = field(
        default=(),
        metadata={
            "help": (
                "Custom γ candidates to evaluate when grounding_gamma_search=True. "
                "When empty (default), the official 199-value grid is used: "
                "[0.00, 0.01, …, 0.99, 1, 2, …, 99]."
            )
        },
    )

    # ── Evaluation Metrics ────────────────────────────────────────────────────
    eval_metrics: tuple[str, ...] = field(
        default=("recall", "ndcg"),
        metadata={"help": "Retrieval metrics to compute. Supported: 'recall', 'ndcg'."},
    )
    eval_topk: tuple[int, ...] = field(
        default=(1, 5, 10, 20),
        metadata={"help": "Top-K cutoffs reported for each metric."},
    )
    eval_protocol: str = field(
        default="sampled",
        metadata={
            "help": (
                "Evaluation protocol passed to BaseTaskDataset.prepare(). "
                "'sampled': rank pre-defined candidate sets. "
                "'full': rank all items (expensive but matches the paper's all-rank setting)."
            )
        },
    )

    # ── Pipeline Control ──────────────────────────────────────────────────────
    pipeline_stage: str = field(
        default="training",
        metadata={
            "help": (
                "Which pipeline stage to execute. "
                "'training': LoRA fine-tuning followed by test evaluation. "
                "'evaluation': load a trained checkpoint and run evaluation only."
            )
        },
    )
    checkpoint_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a saved model/LoRA checkpoint. "
                "Required when pipeline_stage='evaluation'."
            )
        },
    )


__all__ = ["BIGRecConfig"]
