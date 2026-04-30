from __future__ import annotations

from dataclasses import dataclass, field
from recbole3.model.base import ModelConfig

@dataclass(slots=True)
class LCRecConfig(ModelConfig):
    """Configuration for LCRec model (multi-task SFT of LLM for sequential recommendation)."""

    name: str = field(default="lcrec", metadata={"help": "LCRec model name."})

    # --- SID mapping ---
    sid_file: str = field(default="", metadata={"help": "Path to item_sids.json (item -> SID list mapping)."})

    # --- LLM backbone ---
    model_name_or_path: str = field(default="", metadata={"help": "HuggingFace model path or identifier."})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Model weight dtype."})

    # --- LoRA ---
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA fine-tuning."})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_target_modules: tuple[str, ...] = field(
        default=("q_proj", "v_proj"), metadata={"help": "LoRA target module names."}
    )
    lora_modules_to_save: tuple[str, ...] = field(
        default=("embed_tokens", "lm_head"), metadata={"help": "Modules to save with LoRA."}
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate."})

    # --- SFT data format ---
    max_source_length: int = field(default=512, metadata={"help": "Max source (instruction) token length."})
    max_target_length: int = field(default=64, metadata={"help": "Max target token length."})
    max_item_seq_len: int = field(default=20, metadata={"help": "Max history item sequence length."})
    his_sep: str = field(default=", ", metadata={"help": "Separator between history items."})

    # --- HF Trainer parameters ---
    train_batch_size: int = field(default=4, metadata={"help": "Per-device training batch size."})
    eval_batch_size: int = field(default=4, metadata={"help": "Per-device evaluation batch size."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio."})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing."})
    deepspeed: str | None = field(default=None, metadata={"help": "DeepSpeed config path."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "LR scheduler type."})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer type."})
    logging_steps: int = field(default=10, metadata={"help": "Logging interval."})
    bf16: bool = field(default=True, metadata={"help": "Whether to use bf16 mixed precision training."})

    # --- Evaluation ---
    num_beams: int = field(default=20, metadata={"help": "Beam search width for evaluation."})
    test_prompt_ids: str = field(default="all", metadata={"help": "Prompt IDs to test: 'all' or comma-separated IDs (e.g. '0,5,10')."})
    metrics: tuple[str, ...] = field(default=("recall", "ndcg"), metadata={"help": "Evaluation metrics."})
    topk: tuple[int, ...] = field(default=(5, 10, 20), metadata={"help": "Top-K values for evaluation."})

    # --- Pipeline control ---
    pipeline_stage: str = field(
        default="training", metadata={"help": "Pipeline stage: 'training' or 'evaluation'."}
    )
    model_checkpoint_path: str | None = field(
        default=None, metadata={"help": "Model checkpoint path for evaluation."}
    )

    # --- Item metadata fields (matching item_table column names) ---
    item_title_field: str = field(default="title", metadata={"help": "Column name for item title in item_table."})
    item_description_field: str = field(
        default="description", metadata={"help": "Column name for item description in item_table."}
    )
