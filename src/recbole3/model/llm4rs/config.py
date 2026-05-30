from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from recbole3.model.sequential import SequentialModelConfig


LLM4RSBackend = Literal["identity", "openai"]
LLM4RSDomain = Literal["Movie", "Music", "Book", "News", "agnostic"]
LLM4RSPolicy = Literal["point", "pair", "list"]

LLM4RS_DEFAULT_SYSTEM_PROMPT = (
    "You are a recommender. Reply with ONLY the ranking letters requested in the user message "
    "(for example A B C D E), separated by spaces. No explanation or other text."
)


@dataclass(slots=True)
class LLM4RSConfig(SequentialModelConfig):
    """Configuration for Dai et al.'s prompt-based LLM recommendation probe."""

    name: str = field(default="llm4rs", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=5,
        metadata={"help": "Number of latest positively interacted items shown in the prompt."},
    )
    candidate_num: int = field(
        default=5,
        metadata={"help": "Number of target-plus-negative items evaluated per request, as in the paper."},
    )
    ranking_policy: LLM4RSPolicy = field(
        default="list",
        metadata={"help": "Original LLM4RS recommendation policy: point, pair, or list."},
    )
    domain: LLM4RSDomain = field(
        default="agnostic",
        metadata={
            "help": "Prompt-domain wording; native datasets default to agnostic, while official reproduction sets its domain explicitly."
        },
    )
    no_instruction: bool = field(
        default=False,
        metadata={"help": "Whether to omit the recommender-system instruction line."},
    )
    example_num: int = field(
        default=1,
        metadata={"help": "Number of official-style few-shot examples, selected from the first five rows."},
    )
    example_pool_size: int = field(
        default=5,
        metadata={"help": "Number of leading rows reserved as the paper's example pool."},
    )
    begin_index: int = field(
        default=5,
        metadata={"help": "First eval-row index, matching the official default that reserves five examples."},
    )
    end_index: int | None = field(
        default=None,
        metadata={"help": "Exclusive ending eval-row index; None evaluates every remaining row."},
    )
    shuffle_candidates: bool = field(
        default=True,
        metadata={"help": "Whether to deterministically shuffle target-plus-negative candidate positions."},
    )
    item_text_field: str = field(
        default="title",
        metadata={"help": "Item-table column used as the natural-language item name."},
    )
    fallback_item_text_field: str | None = field(
        default="metadata_text",
        metadata={"help": "Fallback text column when item_text_field is absent or empty."},
    )

    # Candidate generation is delegated to the framework's shared candidate stage.
    candidate_source: str = field(
        default="random",
        metadata={"help": "Candidate source: random, prepared, bm25, or a registered retrieval backbone."},
    )
    backbone_topk: int = field(
        default=100,
        metadata={"help": "Number of pre-LLM candidates generated before official target injection."},
    )
    recall_budget: int = field(
        default=5,
        metadata={"help": "Compatibility field for the shared candidate cache signature; set to candidate_num."},
    )
    candidate_seed: int = field(default=2023, metadata={"help": "Seed for candidate generation and shuffling."})
    candidate_cache_dir: str = field(default="outputs/candidate_cache", metadata={"help": "Generated candidate cache root."})
    candidate_file_dir: str = field(default="outputs/candidate_files", metadata={"help": "External candidate file root."})
    refresh_candidate_cache: bool = field(default=False, metadata={"help": "Whether to rebuild generated candidate caches."})
    use_candidate_file: bool = field(default=True, metadata={"help": "Whether to use an external candidate file when present."})
    selected_user_count: int = field(default=-1, metadata={"help": "Number of evaluated users; -1 keeps all users."})
    bm25_item_text_field: str = field(default="title", metadata={"help": "BM25 item text column."})
    bm25_fallback_text_field: str | None = field(default="metadata_text", metadata={"help": "BM25 fallback text column."})
    backbone_checkpoint_path: str | None = field(default=None, metadata={"help": "Optional candidate-backbone checkpoint."})
    backbone_model: dict[str, Any] = field(default_factory=dict, metadata={"help": "Candidate-backbone model overrides."})
    backbone_trainer: dict[str, Any] = field(default_factory=dict, metadata={"help": "Candidate-backbone trainer overrides."})

    backend: LLM4RSBackend = field(
        default="openai",
        metadata={"help": "Chat-completions backend. identity is an offline deterministic smoke-test backend."},
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
    api_extra_body: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Optional extra fields merged into OpenAI-compatible chat-completions requests."},
    )
    temperature: float = field(default=0.0, metadata={"help": "Generation temperature; official requests use zero."})
    max_output_tokens: int = field(default=10, metadata={"help": "Maximum generated tokens for one policy response."})
    request_retries: int = field(default=3, metadata={"help": "API attempts before a response is marked failed."})
    retry_backoff_sec: float = field(default=2.0, metadata={"help": "Initial retry delay in seconds."})
    request_timeout_sec: float = field(default=60.0, metadata={"help": "Network timeout per request in seconds."})
    api_batch: int = field(default=1, metadata={"help": "Maximum concurrent chat requests."})
    async_dispatch: bool = field(
        default=False,
        metadata={"help": "Whether independent policy prompts are sent concurrently."},
    )
    api_response_cache_path: str = field(
        default="outputs/candidate_cache/llm4rs_api_responses.jsonl",
        metadata={"help": "Prompt-response JSONL cache path."},
    )
    refresh_api_response_cache: bool = field(default=False, metadata={"help": "Whether cached API responses are ignored."})
    system_prompt: str | None = field(
        default=LLM4RS_DEFAULT_SYSTEM_PROMPT,
        metadata={"help": "OpenAI-compatible system message; keeps list-wise answers to letters only."},
    )

    def __post_init__(self) -> None:
        if int(self.candidate_num) <= 0:
            raise ValueError("candidate_num must be a positive integer.")
        if int(self.candidate_num) > 26:
            raise ValueError("candidate_num must be at most 26 because official prompts use letter choices.")
        if self.ranking_policy == "pair" and int(self.candidate_num) < 2:
            raise ValueError("Pair-wise LLM4RS requires candidate_num to be at least two.")
        if self.ranking_policy == "point" and int(self.example_num) > 0 and int(self.candidate_num) < 2:
            raise ValueError("Point-wise official examples require candidate_num to be at least two.")
        if int(self.history_max_length) <= 0:
            raise ValueError("history_max_length must be a positive integer.")
        if not 0 <= int(self.example_num) <= int(self.example_pool_size):
            raise ValueError("example_num must be between zero and example_pool_size.")
        if int(self.begin_index) < 0:
            raise ValueError("begin_index must be non-negative.")
        if self.end_index is not None and int(self.end_index) < int(self.begin_index):
            raise ValueError("end_index must be None or no smaller than begin_index.")
        self.recall_budget = int(self.candidate_num)


__all__ = [
    "LLM4RSBackend",
    "LLM4RSConfig",
    "LLM4RSDomain",
    "LLM4RSPolicy",
]
