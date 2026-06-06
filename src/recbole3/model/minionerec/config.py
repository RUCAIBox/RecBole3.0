from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from recbole3.model.base import ModelConfig


MiniOneRecRewardType = Literal["rule", "ranking", "ranking_only"]
MiniOneRecSIDFileItemIDSpace = Literal["recbole", "raw"]
MiniOneRecStage = Literal["sft", "grpo", "evaluation"]


@dataclass(slots=True)
class MiniOneRecConfig(ModelConfig):
    """Configuration for MiniOneRec SFT and constrained generation evaluation."""

    name: str = field(default="minionerec", metadata={"help": "Registered model name."})

    # SID mapping.
    sid_file: str = field(
        default="",
        metadata={"help": "Path to an RQ-VAE-generated MiniOneRec item.index.json file."},
    )
    sid_file_item_id_space: MiniOneRecSIDFileItemIDSpace = field(
        default="recbole",
        metadata={"help": "Whether sid_file keys are RecBole-remapped item ids or raw dataset item ids."},
    )
    require_complete_sid_file: bool = field(
        default=True,
        metadata={"help": "Whether sid_file must contain every remapped item id in the prepared dataset."},
    )
    allow_duplicate_sid_aliases: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow multiple item ids to share one SID. Keep this false for RecBole item-level evaluation; "
                "SID string collision / aliases make generated SID strings ambiguous."
            )
        },
    )
    # LLM backbone.
    model_name_or_path: str = field(default="", metadata={"help": "HuggingFace base model path or identifier."})
    model_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Checkpoint path used when pipeline_stage='grpo' or 'evaluation'."},
    )
    trust_remote_code: bool = field(default=True, metadata={"help": "Passed to AutoTokenizer/AutoModel loaders."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Model dtype such as bfloat16, float16, float32, or auto."})
    attn_implementation: str | None = field(
        default=None,
        metadata={"help": "Optional attention implementation passed to AutoModelForCausalLM.from_pretrained."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Initialize the CausalLM from base config instead of pretrained weights for SFT."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze original LLM parameters and train only newly added SID token rows."},
    )
    add_sid_tokens: bool = field(
        default=True,
        metadata={"help": "Add all SID vocabulary tokens from sid_file to the tokenizer before training/evaluation."},
    )

    # MiniOneRec prompt/data semantics.
    history_max_length: int = field(default=20, metadata={"help": "Most recent history items retained in prompts."})
    min_history_length: int = field(default=1, metadata={"help": "Minimum history length required for SFT examples."})
    max_len: int = field(default=512, metadata={"help": "Maximum tokenized SFT sequence length."})
    eval_max_len: int = field(default=2560, metadata={"help": "Maximum tokenized generation-evaluation prompt length."})
    add_item_alignment_tasks: bool = field(
        default=True,
        metadata={"help": "Include MiniOneRec title<->SID item-alignment SFT tasks when item titles are available."},
    )
    add_fusion_seqrec_task: bool = field(
        default=True,
        metadata={"help": "Include MiniOneRec SID-history-to-title fusion SFT task when item titles are available."},
    )
    item_title_field: str = field(default="title", metadata={"help": "Item-table field used as item title."})
    item_description_field: str = field(default="description", metadata={"help": "Item-table field used as item description."})
    fallback_item_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Fallback item-table text field used when title/description are unavailable."},
    )

    # HF Trainer parameters for SFT.
    pipeline_stage: MiniOneRecStage = field(
        default="sft",
        metadata={"help": "Pipeline stage: 'sft', 'grpo', or 'evaluation'."},
    )
    train_batch_size: int = field(default=16, metadata={"help": "Per-device SFT train batch size."})
    eval_batch_size: int = field(default=16, metadata={"help": "Per-device SFT/evaluation batch size."})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps for SFT."})
    num_train_epochs: int = field(default=10, metadata={"help": "Number of SFT epochs."})
    learning_rate: float = field(default=3e-4, metadata={"help": "SFT learning rate."})
    warmup_steps: int = field(default=20, metadata={"help": "Number of SFT warmup steps."})
    weight_decay: float = field(default=0.0, metadata={"help": "SFT weight decay."})
    logging_steps: int = field(default=1, metadata={"help": "HF Trainer logging interval."})
    eval_steps: float = field(default=0.05, metadata={"help": "HF Trainer eval_steps; recent transformers accept ratios."})
    save_steps: float = field(default=0.05, metadata={"help": "HF Trainer save_steps; recent transformers accept ratios."})
    save_total_limit: int = field(default=1, metadata={"help": "Maximum checkpoints retained by HF Trainer."})
    group_by_length: bool = field(default=False, metadata={"help": "Whether HF Trainer groups examples by length."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing during SFT."})
    bf16: bool = field(default=True, metadata={"help": "Use bf16 training in HF Trainer."})
    fp16: bool = field(default=False, metadata={"help": "Use fp16 training in HF Trainer."})
    optim: str = field(default="adamw_torch", metadata={"help": "HF Trainer optimizer name."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "HF Trainer LR scheduler type."})
    deepspeed: str | None = field(default=None, metadata={"help": "Optional DeepSpeed config path."})
    report_to: str = field(default="none", metadata={"help": "HF Trainer report_to value."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "HF Trainer dataloader worker count."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best SFT checkpoint at the end."})
    early_stopping_patience: int = field(default=3, metadata={"help": "SFT early stopping patience. Use 0 to disable."})
    evaluate_after_training: bool = field(
        default=True,
        metadata={"help": "Run constrained generation on valid/test after SFT completes."},
    )

    # Recommendation-oriented RL (GRPO) parameters. This intentionally keeps
    # only the MiniOneRec paper/codepath: SID prompts, rule/ranking rewards,
    # source-code constrained beam rollout, reference-model KL, and group-normalized rewards.
    rl_train_batch_size: int = field(default=64, metadata={"help": "Per-device GRPO train batch size."})
    rl_eval_batch_size: int = field(default=128, metadata={"help": "Per-device GRPO eval batch size."})
    rl_gradient_accumulation_steps: int = field(default=2, metadata={"help": "GRPO gradient accumulation steps."})
    rl_num_train_epochs: int = field(default=2, metadata={"help": "Number of GRPO epochs."})
    rl_learning_rate: float = field(default=1e-5, metadata={"help": "GRPO learning rate."})
    rl_beta: float = field(default=1e-3, metadata={"help": "KL coefficient in the GRPO objective."})
    rl_sync_ref_model: bool = field(
        default=True,
        metadata={"help": "Use TRL SyncRefModelCallback during GRPO; keep false if TRL/torch FSDP symbols are incompatible."},
    )
    rl_temperature: float = field(default=1.0, metadata={"help": "Sampling temperature used by GRPO generation."})
    rl_num_generations: int = field(default=16, metadata={"help": "Number of generated completions per prompt group."})
    rl_max_completion_length: int = field(default=128, metadata={"help": "Maximum generated tokens during GRPO."})
    rl_max_prompt_length: int | None = field(default=None, metadata={"help": "Optional left truncation length for GRPO prompts."})
    rl_exclude_history: bool = field(
        default=False,
        metadata={
            "help": (
                "Exclude prompt-history items from GRPO constrained rollout. "
                "False preserves the original MiniOneRec rollout; true aligns RecBole exclude_history evaluation."
            )
        },
    )
    rl_reward_type: MiniOneRecRewardType = field(
        default="ranking",
        metadata={"help": "MiniOneRec reward: 'rule', 'ranking', or 'ranking_only'."},
    )
    rl_warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio for the GRPO scheduler."})
    rl_max_grad_norm: float = field(default=0.3, metadata={"help": "Max grad norm used by HF Trainer during GRPO."})
    rl_eval_steps: float = field(default=0.0999, metadata={"help": "GRPO eval_steps; recent transformers accept ratios."})
    rl_save_steps: float = field(default=0.1, metadata={"help": "GRPO save_steps; recent transformers accept ratios."})
    rl_save_total_limit: int = field(default=20, metadata={"help": "Maximum GRPO checkpoints retained."})
    rl_optim: str = field(default="paged_adamw_32bit", metadata={"help": "HF Trainer optimizer used for GRPO."})
    rl_lr_scheduler_type: str = field(default="cosine", metadata={"help": "HF Trainer LR scheduler type for GRPO."})
    rl_report_to: str = field(default="none", metadata={"help": "HF Trainer report_to value for GRPO."})
    rl_add_item_alignment_tasks: bool = field(
        default=True,
        metadata={"help": "Include original active RL title/description-to-SID alignment tasks."},
    )
    rl_add_title_sequence_task: bool = field(
        default=True,
        metadata={"help": "Include original active RL title-history-to-SID task."},
    )
    rl_title_sequence_sample_size: int = field(
        default=10000,
        metadata={"help": "Sample size for the title-history RL task. Use -1 to keep all examples."},
    )
    evaluate_after_rl: bool = field(
        default=False,
        metadata={"help": "Quick valid/test eval after GRPO on rank 0 only; prefer offline pipeline_stage=evaluation for Recall@K."},
    )
    # Constrained decoding evaluation.
    num_beams: int = field(default=50, metadata={"help": "Beam width used in constrained SID generation."})
    max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum generated tokens during evaluation (RQVAE SIDs are short; large values mainly slow beam search)."},
    )
    length_penalty: float = field(default=0.0, metadata={"help": "Beam-search length penalty."})
    topk: tuple[int, ...] = field(default=(5, 10, 20), metadata={"help": "Top-K values for final metrics."})
    metrics: tuple[str, ...] = field(default=("recall", "ndcg"), metadata={"help": "RecBole retrieval metrics reported after generation."})
    exclude_history: bool = field(default=True, metadata={"help": "Pass seen_item_ids through RecBole full evaluation exclusion."})
    constraint_prefix_token_count: int | None = field(
        default=None,
        metadata={"help": "Override MiniOneRec's model-family-specific Response-prefix token count."},
    )
    constraint_cache_size: int = field(
        default=32,
        metadata={"help": "Bounded LRU size for evaluation-time constrained decoding prefix tables. Use 0 to disable."},
    )
    large_eval_warning_threshold: int = field(
        default=10000,
        metadata={"help": "Warn before generation full evaluation when a split has at least this many rows. Use 0 to disable."},
    )
    save_evaluation_predictions: bool = field(
        default=False,
        metadata={"help": "Write RecBole-protocol evaluation predictions as JSON artifacts."},
    )
    evaluation_prediction_dir: str | None = field(
        default=None,
        metadata={"help": "Optional directory for prediction JSON files. Defaults to output_dir/evaluation_predictions."},
    )


__all__ = [
    "MiniOneRecConfig",
    "MiniOneRecRewardType",
    "MiniOneRecSIDFileItemIDSpace",
    "MiniOneRecStage",
]
