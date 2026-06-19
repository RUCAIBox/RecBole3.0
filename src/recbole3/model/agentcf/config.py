from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class AgentCFConfig(SequentialModelConfig):
    """Configuration for the AgentCF model."""

    name: str = field(default="agentcf", metadata={"help": "Registered model name."})
    history_max_length: int = field(default=50, metadata={"help": "Max history length for sequential context."})

    # LLM settings
    api_model_name: str = field(default="gpt-3.5-turbo", metadata={"help": "Chat model name."})
    api_base_url: str = field(default="https://api.openai.com/v1", metadata={"help": "API base URL."})
    api_key_env: str = field(default="OPENAI_API_KEY", metadata={"help": "Environment variable for API key."})
    embedding_model: str = field(default="text-embedding-ada-002", metadata={"help": "Embedding model name."})
    temperature: float = field(default=0.2, metadata={"help": "Temperature for training LLM calls."})
    temperature_eval: float = field(default=0.0, metadata={"help": "Temperature for evaluation LLM calls."})
    max_tokens: int = field(default=2000, metadata={"help": "Max tokens for completion model."})
    max_tokens_chat: int = field(default=3000, metadata={"help": "Max tokens for chat model."})

    # Batching and concurrency
    api_batch: int = field(default=20, metadata={"help": "Batch size for embedding API calls."})
    chat_api_batch: int = field(default=10, metadata={"help": "Batch size for chat API calls."})
    request_retries: int = field(default=3, metadata={"help": "Number of retries for API requests."})
    retry_backoff_sec: float = field(default=20.0, metadata={"help": "Backoff seconds between retries."})
    request_timeout_sec: float = field(default=120.0, metadata={"help": "Request timeout in seconds."})

    # Training behavior
    all_update_rounds: int = field(default=2, metadata={"help": "Number of forward-backward rounds per batch."})
    update_neg_item: bool = field(default=False, metadata={"help": "Whether to update negative item descriptions."})

    # Evaluation
    evaluation_mode: str = field(default="basic", metadata={"help": "Evaluation mode: basic, sequential, or rag."})
    item_representation: str = field(default="direct", metadata={"help": "Item representation: direct or rag."})
    recall_budget: int = field(default=20, metadata={"help": "Number of candidates to rank in evaluation."})
    match_rule: str = field(default="fuzzy", metadata={"help": "Output matching rule: fuzzy or exact."})
    has_gt: bool = field(default=True, metadata={"help": "Whether candidate set includes ground truth."})
    fix_pos: int = field(default=-1, metadata={"help": "Fixed position for ground truth (-1 for shuffle)."})
    shuffle_candidates: bool = field(default=True, metadata={"help": "Whether to shuffle candidate set."})

    # Data paths
    candidate_file_suffix: str = field(default="random", metadata={"help": "Suffix for candidate file (dataset.random)."})
    record_path: str = field(default="outputs/agentcf_records", metadata={"help": "Path to save interaction records."})
    save_agent_state: bool = field(default=True, metadata={"help": "Whether to save agent states after training."})
    load_agent_state_path: str = field(default="", metadata={"help": "Path to load pre-trained agent states."})

    # Domain
    domain: str = field(default="cd", metadata={"help": "Domain for prompt selection: cd or movie."})
