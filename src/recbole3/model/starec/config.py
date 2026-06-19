from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recbole3.model.sequential import SequentialModelConfig


STARecBackend = Literal["deterministic", "openai"]
STARecReflectionMode = Literal["discrepancy-only", "always", "none"]


@dataclass(slots=True)
class STARecConfig(SequentialModelConfig):
    """Configuration for the STARec architecture-only agent model."""

    name: str = field(default="starec", metadata={"help": "Registered model name."})
    history_max_length: int | None = field(
        default=40,
        metadata={"help": "Maximum number of mixed-feedback history rows retained per selected user."},
    )
    history_min_length: int = field(
        default=30,
        metadata={"help": "Minimum retained train/valid/test mixed-feedback history rows required per selected user."},
    )
    candidate_source: str = field(
        default="random",
        metadata={"help": "STARec MVP only supports random candidates."},
    )
    backbone_topk: int = field(
        default=20,
        metadata={"help": "Number of raw random candidates generated before ground-truth insertion."},
    )
    recall_budget: int = field(
        default=20,
        metadata={"help": "Number of candidates finally passed into STARec ranking."},
    )
    candidate_seed: int = field(default=42, metadata={"help": "Seed for random candidates and candidate shuffling."})
    candidate_cache_dir: str = field(
        default="outputs/candidate_cache",
        metadata={"help": "Root directory used to cache generated candidate sets."},
    )
    candidate_file_dir: str = field(
        default="outputs/candidate_files",
        metadata={"help": "Root directory used to read/write external candidate files."},
    )
    refresh_candidate_cache: bool = field(
        default=False,
        metadata={"help": "Whether to rebuild cached candidate sets even if cache files already exist."},
    )
    use_candidate_file: bool = field(
        default=True,
        metadata={"help": "Whether to read external candidate files from disk before generating candidates."},
    )
    selected_user_count: int = field(
        default=1000,
        metadata={"help": "Number of users evaluated by STARec. Use -1 to keep all evaluable users."},
    )
    selected_user_ids_path: str | None = field(
        default=None,
        metadata={"help": "Optional JSONL user-id artifact that restricts STARec to an explicit user set."},
    )
    has_gt: bool = field(
        default=True,
        metadata={"help": "STARec MVP requires inserting the ground-truth item into every candidate set."},
    )
    fix_pos: int = field(
        default=-1,
        metadata={"help": "Ground-truth insertion position. Use -1 to append before optional shuffle."},
    )
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the final candidate list."})
    item_text_field: str = field(default="title", metadata={"help": "Preferred item-table column for item text."})
    fallback_item_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Fallback item-table column for item text."},
    )
    item_text_template: str | None = field(
        default=None,
        metadata={
            "help": "Optional format string rendered from item-table fields, e.g. '{title}. Artist/brand: {brand}'."
        },
    )
    item_domain_singular: str | None = field(
        default=None,
        metadata={"help": "Optional singular item-domain wording used in STARec prompts."},
    )
    item_domain_plural: str | None = field(
        default=None,
        metadata={"help": "Optional plural item-domain wording used in STARec prompts."},
    )
    bm25_item_text_field: str = field(
        default="title",
        metadata={"help": "Compatibility field for LLMRank candidate cache signatures; unused by STARec MVP."},
    )
    bm25_fallback_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Compatibility field for LLMRank candidate cache signatures; unused by STARec MVP."},
    )
    backbone_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Compatibility field for LLMRank candidate cache signatures; unused by STARec MVP."},
    )
    backbone_model: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Compatibility field for LLMRank candidate cache signatures; unused by STARec MVP."},
    )
    backbone_trainer: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Compatibility field for LLMRank candidate cache signatures; unused by STARec MVP."},
    )
    user_profile_fields: tuple[str, ...] = field(
        default=("gender", "age", "occupation"),
        metadata={"help": "Optional user-table columns rendered into the profile when present."},
    )
    reflection_mode: STARecReflectionMode = field(
        default="discrepancy-only",
        metadata={"help": "When to update the user description: discrepancy-only, always, or none."},
    )
    prediction_liked_threshold: int = field(
        default=5,
        metadata={"help": "Target ranks at or below this value are treated as Predicted Liked."},
    )
    feedback_score_field: str | None = field(
        default=None,
        metadata={"help": "Optional interaction field used as raw feedback score before falling back to label."},
    )
    feedback_positive_threshold: float = field(
        default=0.0,
        metadata={"help": "Feedback scores greater than this threshold are treated as liked."},
    )
    backend: STARecBackend = field(
        default="deterministic",
        metadata={"help": "LLM backend. deterministic is for smoke tests; openai uses the OpenAI Python SDK."},
    )
    api_model_name: str = field(default="gpt-4o-mini", metadata={"help": "OpenAI-compatible chat model name."})
    api_base_url: str = field(
        default="https://api.openai.com/v1",
        metadata={"help": "OpenAI-compatible SDK base URL, not a chat-completions endpoint."},
    )
    api_key_env: str = field(default="OPENAI_API_KEY", metadata={"help": "Environment variable with the API key."})
    temperature: float = field(default=1.0, metadata={"help": "Generation temperature."})
    top_p: float = field(default=1.0, metadata={"help": "Nucleus sampling value."})
    max_output_tokens: int = field(default=1200, metadata={"help": "Maximum output tokens requested from the backend."})
    api_batch: int = field(
        default=1,
        metadata={"help": "Maximum number of user sequences processed concurrently by STARec."},
    )
    async_dispatch: bool = field(
        default=False,
        metadata={"help": "Whether to process different STARec user sequences concurrently."},
    )
    request_retries: int = field(default=1, metadata={"help": "Number of retry attempts for API requests."})
    retry_backoff_sec: float = field(default=2.0, metadata={"help": "Initial retry backoff in seconds."})
    request_timeout_sec: float = field(default=60.0, metadata={"help": "Network timeout for API requests."})
    parse_retries: int = field(default=1, metadata={"help": "Ranking parse retry count for LLM backends."})
    train_init_interactions: int = field(
        default=20,
        metadata={"help": "Number of initial train interactions used only to initialize each user memory."},
    )
    run_warmup: bool = field(default=True, metadata={"help": "Whether to run train-split memory warmup."})
    skip_warmup_when_memory_loaded: bool = field(
        default=False,
        metadata={"help": "Whether a loaded memory artifact skips train-split warmup."},
    )
    memory_load_path: str | None = field(default=None, metadata={"help": "Optional JSONL memory artifact to load."})
    memory_save_path: str | None = field(
        default="starec_memories.jsonl",
        metadata={"help": "Optional JSONL memory artifact path to write, relative to runtime.output_dir when not absolute."},
    )
    sample_log_path: str | None = field(
        default="starec_samples.jsonl",
        metadata={"help": "Optional JSONL per-sample audit log path, relative to runtime.output_dir when not absolute."},
    )
    teacher_trace_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional JSONL teacher rollout trace path. When set, STARec writes train-only init/ranking/"
                "reflection prompt-response traces for SFT/RL data export."
            )
        },
    )

    def __post_init__(self) -> None:
        if str(self.candidate_source).strip().lower() != "random":
            raise ValueError("STARec MVP requires model.candidate_source=random.")
        if not bool(self.has_gt):
            raise ValueError("STARec MVP requires model.has_gt=true.")
        if self.reflection_mode not in {"discrepancy-only", "always", "none"}:
            raise ValueError("model.reflection_mode must be one of: discrepancy-only, always, none.")
        if int(self.api_batch) < 1:
            raise ValueError("model.api_batch must be >= 1.")
        if int(self.train_init_interactions) < 0:
            raise ValueError("model.train_init_interactions must be >= 0.")
        if self.history_max_length is not None and int(self.history_max_length) <= 0:
            raise ValueError("model.history_max_length must be a positive integer or null.")
        if int(self.history_min_length) < 0:
            raise ValueError("model.history_min_length must be >= 0.")
        if self.history_max_length is not None and int(self.history_max_length) < int(self.history_min_length):
            raise ValueError("model.history_max_length must be >= model.history_min_length when both are set.")
        if bool(str(self.item_domain_singular or "").strip()) != bool(str(self.item_domain_plural or "").strip()):
            raise ValueError("model.item_domain_singular and model.item_domain_plural must be set together.")


__all__ = [
    "STARecBackend",
    "STARecConfig",
    "STARecReflectionMode",
]
