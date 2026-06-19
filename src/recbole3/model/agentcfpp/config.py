from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


def _default_domain_main_category() -> dict[str, str]:
    """Domain -> Amazon `main_category` label (ported from AgentCF-plus/config.py)."""
    return {
        "Books": "Books",
        "Movies_and_TV": "Movies & TV",
        "Beauty_and_Personal_Care": "All Beauty",
        "Electronics": "All Electronics",
        "Sports_and_Outdoors": "Sports & Outdoors",
        "CDs_and_Vinyl": "Digital Music",
        "Video_Games": "Video Games",
    }


@dataclass(slots=True)
class AgentCFPPConfig(SequentialModelConfig):
    """Configuration for the AgentCF++ cross-domain model."""

    name: str = field(default="agentcfpp", metadata={"help": "Registered model name."})
    history_max_length: int = field(default=50, metadata={"help": "Max history length for sequential context."})

    # LLM settings
    api_model_name: str = field(default="gpt-4o-mini", metadata={"help": "Chat model name."})
    api_base_url: str = field(default="https://api.openai.com/v1", metadata={"help": "API base URL."})
    api_key_env: str = field(default="OPENAI_API_KEY", metadata={"help": "Environment variable for API key."})
    embedding_model: str = field(default="text-embedding-3-large", metadata={"help": "Embedding model name."})
    temperature: float = field(default=0.7, metadata={"help": "Temperature for training LLM calls."})
    temperature_eval: float = field(default=0.0, metadata={"help": "Temperature for evaluation LLM calls."})
    max_tokens_chat: int = field(default=800, metadata={"help": "Max tokens for chat completions."})
    embedding_dim: int = field(default=128, metadata={"help": "Truncated embedding dim for user-tag clustering."})

    # Batching and concurrency
    chat_api_batch: int = field(default=10, metadata={"help": "Batch size / concurrency for chat API calls."})
    request_retries: int = field(default=3, metadata={"help": "Number of retries for API requests."})
    retry_backoff_sec: float = field(default=20.0, metadata={"help": "Backoff seconds between retries."})
    request_timeout_sec: float = field(default=120.0, metadata={"help": "Request timeout in seconds."})

    # Training behavior
    update_neg_item: bool = field(default=False, metadata={"help": "Whether to update negative item descriptions."})

    # Cross-domain
    domain_list: tuple[str, ...] = field(
        default_factory=lambda: ("Books", "Video_Games", "Movies_and_TV"),
        metadata={"help": "Ordered list of domains used in this cross-domain run."},
    )
    domain_main_category_dict: dict[str, str] = field(
        default_factory=_default_domain_main_category,
        metadata={"help": "Mapping from domain name to Amazon main_category label."},
    )
    use_intermediate_node: bool = field(
        default=True,
        metadata={"help": "Whether to include private single-domain memory in eval user description."},
    )

    # Evaluation
    prompt_strategy: str = field(default="B", metadata={"help": "Eval prompt strategy: B, B+H, or B+R."})
    candidate_num: int = field(default=10, metadata={"help": "Number of candidates to rank in evaluation."})
    match_rule: str = field(default="fuzzy", metadata={"help": "Output matching rule: fuzzy or exact."})
    has_gt: bool = field(default=True, metadata={"help": "Whether candidate set includes ground truth."})
    shuffle_candidates: bool = field(default=True, metadata={"help": "Whether to shuffle candidate set."})

    # Group / shared memory
    use_group_memory: bool = field(default=False, metadata={"help": "Whether to build and inject group memory."})
    group_n_cluster: int = field(default=384, metadata={"help": "Base number of KMeans clusters for tag embeddings."})
    group_num_groups: int = field(default=10, metadata={"help": "Number of top groups to keep."})
    group_mem_length: int = field(default=5, metadata={"help": "Number of recent interactions per domain in group memory."})

    # State persistence
    record_path: str = field(default="outputs/agentcfpp_records", metadata={"help": "Path to save interaction records."})
    save_agent_state: bool = field(default=True, metadata={"help": "Whether to save agent states after training."})
    load_agent_state_path: str = field(default="", metadata={"help": "Path to load pre-trained agent states."})
    group_state_path: str = field(default="", metadata={"help": "Path to load/save group memory state."})
