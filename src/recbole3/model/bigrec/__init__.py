"""BIGRec — Bi-Step Grounding Paradigm for Large Language Models in Recommendation.

Reference:
    Bao et al., "A Bi-Step Grounding Paradigm for Large Language Models in
    Recommendation Systems", arXiv:2308.08434 (2023).
"""

from recbole3.model.bigrec.config import BIGRecConfig
from recbole3.model.bigrec.data import (
    BIGRecModelDataset,
    BIGRecSFTDataset,
    batchify,
    build_eval_prompts,
    build_input_block,
    build_instruction,
    build_item_text_lookup,
    build_prompt,
    select_sft_training_records,
)
from recbole3.model.bigrec.generator import BIGRecGenerator
from recbole3.model.bigrec.grounding import BIGRecGrounder
from recbole3.model.bigrec.pipeline import BIGRecPipeline
from recbole3.model.bigrec.trainer import BIGRecTrainer

__all__ = [
    "BIGRecConfig",
    "BIGRecGenerator",
    "BIGRecGrounder",
    "BIGRecModelDataset",
    "BIGRecPipeline",
    "BIGRecSFTDataset",
    "BIGRecTrainer",
    "batchify",
    "build_eval_prompts",
    "build_input_block",
    "build_instruction",
    "build_item_text_lookup",
    "build_prompt",
    "select_sft_training_records",
]
