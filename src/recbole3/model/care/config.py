from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class CAREConfig(SequentialModelConfig):
    """Configuration for CARE generative retrieval.

    CARE is a Qwen2-based generative retrieval model. It represents each item by
    a fixed-length semantic identifier and ranks items by the likelihood of their
    identifier conditioned on the user's sequential history.
    """

    name: str = field(default="care", metadata={"help": "Registered model name."})

    # Item identifier mapping.
    sid_file: str = field(
        default="",
        metadata={
            "help": "Path to item_sids.json. Keys must be RecBole remapped item ids; values may be TIGER/RQVAE integer SID lists or CARE textual code-token lists."
        },
    )

    # Qwen2 backbone.
    base_model: str = field(
        default="",
        metadata={"help": "Path or HuggingFace id of the pretrained Qwen2/Qwen2.5 causal LM."},
    )
    model_max_length: int = field(default=2048, metadata={"help": "Tokenizer max length."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "float32, float16, bfloat16, or auto."})
    attn_implementation: str | None = field(default=None, metadata={"help": "Optional HF attention implementation."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Forwarded to AutoTokenizer/from_pretrained."})
    low_cpu_mem_usage: bool = field(default=True, metadata={"help": "Forwarded to from_pretrained."})

    # CARE reasoning.
    query_list: tuple[int, ...] = field(
        default=(1, 1, 1, 1),
        metadata={"help": "Number of learnable query tokens inserted before each identifier code token."},
    )
    progressive_list: tuple[bool, ...] = field(
        default=(True, True, True, True),
        metadata={"help": "Whether each identifier stage uses progressive history visibility."},
    )
    progressive_attn: bool = field(
        default=True,
        metadata={"help": "Enable CARE progressive history encoding through a 4D attention mask."},
    )
    attention_strategy: str = field(default="hard", metadata={"help": "CARE progressive attention strategy."})
    query_div_scale: float = field(default=0.3, metadata={"help": "Weight of query diversity regularization."})

    # Text format.
    special_token_for_answer: str = field(default="|start_of_answer|", metadata={"help": "Separator before target code."})
    history_separator: str = field(default="", metadata={"help": "Separator used between historical item identifiers."})
    add_history_prefix: bool = field(default=False, metadata={"help": "Prefix history items with 1., 2., ... ."})

    # Evaluation.
    num_beams: int = field(
        default=20,
        metadata={"help": "Beam width used by CARE SID-constrained generation."},
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
                "CARE currently implements the original hard progressive attention mask only. "
                f"Unsupported attention_strategy={self.attention_strategy!r}."
            )
        self.attention_strategy = "hard"
        if int(self.num_beams) <= 0:
            raise ValueError("CAREConfig.num_beams must be positive.")


__all__ = ["CAREConfig"]