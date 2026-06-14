from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class E4SRecConfig(SequentialModelConfig):
    """Configuration for E4SRec: LLM-based sequential recommendation with LoRA.

    E4SRec wraps a pre-trained causal LM (e.g. Llama, Qwen, Mistral) as a feature
    extractor, applies LoRA fine-tuning, and projects pre-trained collaborative
    filtering item embeddings into the LLM hidden space.  The final prediction head
    scores all items via a single linear layer.
    """

    name: str = field(default="e4srec", metadata={"help": "Registered model name."})
    history_max_length: int = field(
        default=50,
        metadata={"help": "Maximum number of recent history items fed to the model."},
    )

    # ---- LLM backbone ---------------------------------------------------------
    base_model: str = field(
        default="",
        metadata={"help": "HuggingFace model identifier (e.g. 'meta-llama/Llama-2-7b-hf')."},
    )
    cache_dir: str = field(
        default="",
        metadata={"help": "Directory to cache downloaded HF models."},
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model parallelism ('auto', 'sequential', or specific mappings)."},
    )
    load_in_8bit: bool = field(
        default=True,
        metadata={"help": "Load the LLM backbone in 8-bit quantized mode."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load the LLM backbone in 4-bit quantized mode."},
    )
    torch_dtype: str = field(
        default="float16",
        metadata={"help": "Torch dtype for model weights: 'float16', 'bfloat16', or 'float32'."},
    )
    use_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )

    # ---- LoRA -----------------------------------------------------------------
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling factor."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate."})
    lora_target_modules: tuple[str, ...] = field(
        default=("gate_proj", "down_proj", "up_proj"),
        metadata={"help": "Names of LoRA target modules in the backbone."},
    )

    # ---- HF Trainer settings --------------------------------------------------
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps for learning rate scheduler."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "HF learning rate scheduler type (cosine, linear, constant, …)."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "HF optimizer."},
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use the model's past key/values cache (if supported) during training."},
    )

    # ---- Pipeline stage -------------------------------------------------------
    pipeline_stage: str = field(
        default="training",
        metadata={"help": "Pipeline stage: 'training' or 'evaluation'."},
    )
    checkpoint_path: str = field(
        default="",
        metadata={"help": "Path to checkpoint directory for evaluation / inference stage."},
    )

    # ---- Pre-trained item embeddings ------------------------------------------
    item_embed_path: str = field(
        default="",
        metadata={"help": "Path to pre-trained item embeddings (.pkl, shape [num_items, embed_dim])."},
    )
    item_embed_dim: int = field(
        default=64,
        metadata={"help": "Dimensionality of the pre-trained CF item embeddings."},
    )

    # ---- Prompt template ------------------------------------------------------
    instruction_text: str = field(
        default="Given the user's purchase history, predict next possible item to be purchased.",
        metadata={"help": "Instruction text placed inside the prompt template."},
    )
    prompt_template: str = field(
        default=(
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the "
            "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n"
        ),
        metadata={"help": "Prompt template with an '{instruction}' placeholder."},
    )
    response_split: str = field(
        default="\n### Response:\n",
        metadata={"help": "Text appended after the item sequence to mark the response position."},
    )


__all__ = ["E4SRecConfig"]
