from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recbole3.model.sequential import SequentialModelConfig


LLMRankBackend = Literal["heuristic_overlap", "mock", "openai", "local_hf"]
LLMRankCandidateSource = Literal["random", "bm25", "hstu"]
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
    candidate_source: LLMRankCandidateSource = field(
        default="bm25",
        metadata={"help": "Source used to build the candidate set before LLM reranking."},
    )
    candidate_topk: int = field(
        default=100,
        metadata={"help": "Candidate set size passed into the LLM reranker, including the target item."},
    )
    candidate_seed: int = field(
        default=42,
        metadata={"help": "Deterministic seed used by candidate generation and prompt shuffling."},
    )
    candidate_cache_dir: str = field(
        default="outputs/candidate_cache",
        metadata={"help": "Root directory used to cache generated candidate sets and auto-trained backbones."},
    )
    refresh_candidate_cache: bool = field(
        default=False,
        metadata={"help": "Whether to rebuild cached candidate sets even if cache files already exist."},
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
        default=True,
        metadata={"help": "Whether to shuffle candidates before prompting even without bootstrapping."},
    )
    random_seed: int = field(
        default=42,
        metadata={"help": "Random seed used by candidate shuffling and mock backends."},
    )
    bm25_item_text_field: str = field(
        default="title",
        metadata={"help": "Primary item text field used to build BM25 documents."},
    )
    bm25_fallback_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Fallback item text field used when bm25_item_text_field is empty."},
    )
    hstu_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Optional checkpoint path used by candidate_source='hstu'. If unset, HSTU is trained automatically."},
    )
    hstu_model_overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Optional overrides merged into the default HSTU model config when candidate_source='hstu'."},
    )
    hstu_trainer_overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Optional overrides merged into the default HSTU trainer config when candidate_source='hstu'."},
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
    api_response_cache_path: str = field(
        default="outputs/candidate_cache/llmrank_api_responses.jsonl",
        metadata={"help": "JSONL cache file used to reuse OpenAI-compatible responses for identical prompts."},
    )
    refresh_api_response_cache: bool = field(
        default=False,
        metadata={"help": "Whether to ignore cached OpenAI-compatible responses and rebuild the cache."},
    )
    local_model_path: str | None = field(
        default=None,
        metadata={"help": "Filesystem path to one local Hugging Face causal LM used when backend='local_hf'."},
    )
    local_tokenizer_path: str | None = field(
        default=None,
        metadata={"help": "Optional tokenizer path; defaults to local_model_path when backend='local_hf'."},
    )
    local_device: str | None = field(
        default=None,
        metadata={"help": "Optional explicit device such as 'cuda:0' or 'cpu' for single-device local inference."},
    )
    local_device_map: str | None = field(
        default="auto",
        metadata={"help": "Optional Hugging Face device_map; use 'auto' for multi-GPU sharding or null for single-device placement."},
    )
    local_dtype: Literal["auto", "bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype used when loading one local Hugging Face model."},
    )
    local_batch_size: int = field(
        default=8,
        metadata={"help": "Prompt batch size used for one local Hugging Face generation pass."},
    )
    local_max_input_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum prompt token length for one local Hugging Face generation pass."},
    )
    local_trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether local Hugging Face model loading may execute custom modeling code."},
    )
    local_attn_implementation: str | None = field(
        default="flash_attention_2",
        metadata={"help": "Optional attn_implementation passed to local Hugging Face models; set null to disable."},
    )
    local_use_chat_template: bool = field(
        default=True,
        metadata={"help": "Whether to wrap prompts with tokenizer.apply_chat_template when available for local Hugging Face models."},
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
    "LLMRankCandidateSource",
    "LLMRankConfig",
    "LLMRankDomain",
    "LLMRankParsingStrategy",
    "LLMRankPromptStrategy",
]
