from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from recbole3.model.sequential import SequentialModelConfig


LLMRankBackend = Literal["heuristic_overlap", "mock", "openai"]
LLMRankPromptStrategy = Literal["sequential", "recency_focused", "in_context_learning"]
LLMRankDomain = Literal["item", "movie", "product"]
LLMRankParsingStrategy = Literal["title", "index"]


@dataclass(slots=True)
class LLMRankConfig(SequentialModelConfig):
    """Configuration for the prompt-based LLM candidate reranker."""

    name: str = field(default="llmrank", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=5,
        metadata={"help": "Maximum number of recent interactions retained in prompt history."},
    )
    item_text_field: str = field(
        default="metadata_text",
        metadata={"help": "Preferred item-table column used as the natural-language item text."},
    )
    fallback_item_text_field: str | None = field(
        default="raw_item_id",
        metadata={"help": "Optional fallback item-table column used when item_text_field is empty."},
    )
    domain: LLMRankDomain = field(
        default="item",
        metadata={"help": "Prompt domain used to choose movie/product/item wording."},
    )
    prompt_strategy: LLMRankPromptStrategy = field(
        default="recency_focused",
        metadata={"help": "Prompting strategy inspired by the original LLMRank implementation."},
    )
    backend: LLMRankBackend = field(
        default="heuristic_overlap",
        metadata={"help": "Inference backend used to obtain ranked candidate outputs."},
    )
    parsing_strategy: LLMRankParsingStrategy = field(
        default="title",
        metadata={"help": "How to parse LLM outputs back into ranked candidate ids."},
    )
    bootstrap_rounds: int = field(
        default=1,
        metadata={"help": "Number of repeated shuffled ranking rounds used for bootstrapping."},
    )
    candidate_shuffle: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle candidates before prompting even without bootstrapping."},
    )
    random_seed: int = field(
        default=42,
        metadata={"help": "Random seed used by candidate shuffling and mock backends."},
    )
    api_model_name: str = field(
        default="gpt-4o-mini",
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
    api_concurrency: int = field(
        default=4,
        metadata={"help": "Maximum number of concurrent OpenAI-compatible ranking requests."},
    )
    enforce_candidate_constraint: bool = field(
        default=True,
        metadata={"help": "Whether prompts explicitly forbid outputs outside the candidate list."},
    )
    include_reasoning_instruction: bool = field(
        default=True,
        metadata={"help": "Whether prompts ask the LLM to think step by step before ranking."},
    )
    require_order_numbers: bool = field(
        default=True,
        metadata={"help": "Whether prompts ask the LLM to return order numbers with item names."},
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Optional system instruction prepended to the user prompt."},
    )
    mock_responses: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Optional fixed response texts consumed by the mock backend."},
    )


__all__ = [
    "LLMRankBackend",
    "LLMRankConfig",
    "LLMRankDomain",
    "LLMRankParsingStrategy",
    "LLMRankPromptStrategy",
]
