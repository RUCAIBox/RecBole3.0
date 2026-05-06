from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recbole3.model.sequential import SequentialModelConfig


LLMRankBackend = Literal["identity", "openai", "heuristic_overlap"]
LLMRankCandidateSource = str
LLMRankDomain = Literal["item", "movie", "product"]
LLMRankParsingStrategy = Literal["title", "index"]
LLMRankPromptStrategy = Literal["sequential", "recency_focused", "in_context_learning"]


@dataclass(slots=True)
class LLMRankConfig(SequentialModelConfig):
    """Configuration for the prompt-based LLM candidate reranker."""

    name: str = field(default="llmrank", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=50,
        metadata={"help": "Maximum number of recent interactions retained in prompt history."},
    )
    backbone_topk: int = field(
        default=100,
        metadata={"help": "Number of items produced by the backbone candidate generator before recall-budget truncation."},
    )
    recall_budget: int = field(
        default=20,
        metadata={"help": "Number of candidate items finally passed into the LLM reranker."},
    )
    candidate_source: LLMRankCandidateSource = field(
        default="bm25",
        metadata={"help": "Source used to build the candidate set before LLM reranking. Use 'random', 'bm25', or one registered retrieval backbone name such as 'hstu'."},
    )
    candidate_seed: int = field(
        default=42,
        metadata={"help": "Deterministic seed used by candidate generation and prompt shuffling."},
    )
    candidate_cache_dir: str = field(
        default="outputs/candidate_cache",
        metadata={"help": "Root directory used to cache generated candidate sets and auto-trained backbones."},
    )
    candidate_file_dir: str = field(
        default="outputs/candidate_files",
        metadata={"help": "Root directory used to read/write external backbone candidate files."},
    )
    refresh_candidate_cache: bool = field(
        default=False,
        metadata={"help": "Whether to rebuild cached candidate sets even if cache files already exist."},
    )
    use_candidate_file: bool = field(
        default=True,
        metadata={"help": "Whether to read backbone candidate files from disk before generating them on the fly."},
    )
    selected_user_count: int = field(
        default=200,
        metadata={"help": "Number of users evaluated by LLMRank. Use -1 to keep all evaluable users, matching the official full-user option."},
    )
    has_gt: bool = field(
        default=True,
        metadata={"help": "Whether to force the evaluation target item into the candidate set before LLM reranking."},
    )
    fix_pos: int = field(
        default=-1,
        metadata={"help": "Ground-truth insertion position inside the candidate set. Use -1 to match the official shuffled placement."},
    )
    shuffle: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the candidate list."},
    )
    item_text_field: str = field(
        default="title",
        metadata={"help": "Preferred item-table column used as the natural-language item text."},
    )
    fallback_item_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Optional fallback item-table column used when item_text_field is empty."},
    )
    domain: LLMRankDomain = field(
        default="product",
        metadata={"help": "Prompt domain used to choose movie/product/item wording."},
    )
    backend: LLMRankBackend = field(
        default="openai",
        metadata={"help": "Inference backend used to obtain ranked candidate outputs. 'identity' keeps the backbone candidate order unchanged."},
    )
    parsing_strategy: LLMRankParsingStrategy = field(
        default="title",
        metadata={"help": "How to parse LLM outputs back into ranked candidate ids."},
    )
    prompt_strategy: LLMRankPromptStrategy = field(
        default="sequential",
        metadata={"help": "Prompt construction strategy used before querying the LLM."},
    )
    boots: int = field(
        default=0,
        metadata={"help": "Number of bootstrapping rounds used to alleviate position bias. Official default is 0."},
    )
    bm25_item_text_field: str = field(
        default="title",
        metadata={"help": "Primary item text field used to build BM25 documents."},
    )
    bm25_fallback_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Fallback item text field used when bm25_item_text_field is empty."},
    )
    backbone_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Optional checkpoint path used by model-backed candidate sources. If unset, the backbone is trained automatically."},
    )
    backbone_model: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Optional overrides merged into the selected backbone model config."},
    )
    backbone_trainer: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Optional overrides merged into the selected backbone trainer config."},
    )
    api_model_name: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "Remote chat-completions model name used by the openai backend."},
    )
    api_base_url: str = field(
        default="https://api.openai.com/v1/chat/completions",
        metadata={"help": "OpenAI-compatible chat-completions endpoint for the openai backend."},
    )
    api_key_env: str = field(
        default="OPENAI_API_KEY",
        metadata={"help": "Environment variable that stores the OpenAI-compatible API key."},
    )
    temperature: float = field(
        default=0.2,
        metadata={"help": "Generation temperature passed to the configured LLM backend."},
    )
    max_output_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of output tokens requested from the openai backend."},
    )
    request_retries: int = field(
        default=3,
        metadata={"help": "Number of retry attempts for OpenAI-compatible API requests."},
    )
    retry_backoff_sec: float = field(
        default=2.0,
        metadata={"help": "Initial retry backoff in seconds for OpenAI-compatible API requests."},
    )
    request_timeout_sec: float = field(
        default=60.0,
        metadata={"help": "Network timeout used by the openai backend."},
    )
    api_batch: int = field(
        default=8,
        metadata={"help": "Maximum number of requests launched together for one asynchronous OpenAI-compatible batch."},
    )
    async_dispatch: bool = field(
        default=True,
        metadata={"help": "Whether to dispatch OpenAI-compatible requests in parallel batches like the official implementation."},
    )
    api_response_cache_path: str = field(
        default="outputs/candidate_cache/llmrank_api_responses.jsonl",
        metadata={"help": "JSONL cache file used to reuse OpenAI-compatible responses for identical prompts."},
    )
    refresh_api_response_cache: bool = field(
        default=False,
        metadata={"help": "Whether to ignore cached OpenAI-compatible responses and rebuild the cache."},
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Optional system instruction prepended to the user prompt."},
    )
    mock_responses: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Deprecated legacy field from the removed mock backend; ignored and kept only for backward compatibility."},
    )

    def __post_init__(self) -> None:
        if str(self.backend).strip().lower() == "mock":
            self.backend = "identity"  # type: ignore[assignment]


__all__ = [
    "LLMRankBackend",
    "LLMRankCandidateSource",
    "LLMRankConfig",
    "LLMRankDomain",
    "LLMRankParsingStrategy",
    "LLMRankPromptStrategy",
]
