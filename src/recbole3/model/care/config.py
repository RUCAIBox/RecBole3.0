from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class CAREConfig(SequentialModelConfig):
    """Configuration for the CARE generative retrieval model (faithful CARE port)."""

    name: str = field(default="car", metadata={"help": "Registered model name."})

    sid_file: str = field(
        default="",
        metadata={
            "help": "Path to item SID JSON keyed by RecBole remapped item ids. "
            "Values may be integer SID lists or CARE textual code-token lists."
        },
    )

    base_model: str = field(
        default="",
        metadata={"help": "Path or HuggingFace id of the pretrained Qwen2/Qwen2.5 causal LM."},
    )
    model_max_length: int = field(default=2048, metadata={"help": "Tokenizer max length."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "float32, float16, bfloat16, or auto."})
    attn_implementation: str | None = field(default=None, metadata={"help": "Optional HF attention implementation."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Forwarded to from_pretrained."})
    low_cpu_mem_usage: bool = field(default=True, metadata={"help": "Forwarded to from_pretrained."})

    query_list: tuple[int, ...] = field(
        default=(1, 1, 1, 1),
        metadata={"help": "CARE args.query_list: number of learnable query tokens per identifier code."},
    )
    progressive_list: tuple[bool, ...] = field(
        default=(True, True, True, True),
        metadata={"help": "CARE args.progressive_list: per-stage progressive history visibility."},
    )
    progressive_attn: bool = field(default=True, metadata={"help": "CARE args.progressive_attn."})
    attention_strategy: str = field(default="hard", metadata={"help": "CARE args.attention_strategy."})
    query_div_scale: float = field(default=0.3, metadata={"help": "CARE args.query_div_scale."})

    special_token_for_answer: str = field(
        default="|start_of_answer|", metadata={"help": "CARE separator token before the target code."}
    )
    history_separator: str = field(default="", metadata={"help": "CARE args.his_sep used between history identifiers."})
    add_history_prefix: bool = field(default=False, metadata={"help": "CARE args.add_prefix (1., 2., ... prefixes)."})

    num_beams: int = field(default=20, metadata={"help": "CARE args.num_beams used by constrained generation."})
    max_new_token: int = field(default=10, metadata={"help": "CARE args.max_new_token for generation."})
    filter_items: bool = field(
        default=True,
        metadata={"help": "CARE inference.py args.filter_items: filter generated identifiers not in all_items."},
    )

    def __post_init__(self) -> None:
        self.query_list = tuple(int(value) for value in self.query_list)
        self.progressive_list = tuple(bool(value) for value in self.progressive_list)
        if not self.query_list:
            raise ValueError("CAREConfig.query_list must contain at least one stage.")
        if len(self.query_list) != len(self.progressive_list):
            raise ValueError(
                "CAREConfig.query_list and CAREConfig.progressive_list must have the same length, "
                f"got {len(self.query_list)} and {len(self.progressive_list)}."
            )
        if any(value < 0 for value in self.query_list):
            raise ValueError(f"CAREConfig.query_list must contain non-negative integers, got {self.query_list}.")
        if str(self.attention_strategy).lower() != "hard":
            raise ValueError(
                "CARE only implements the original hard progressive attention mask. "
                f"Unsupported attention_strategy={self.attention_strategy!r}."
            )
        self.attention_strategy = "hard"
        if int(self.num_beams) <= 0:
            raise ValueError("CAREConfig.num_beams must be positive.")
        if int(self.max_new_token) <= 0:
            raise ValueError("CAREConfig.max_new_token must be positive.")


__all__ = ["CAREConfig"]