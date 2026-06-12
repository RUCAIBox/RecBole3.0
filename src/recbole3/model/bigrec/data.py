"""BIGRec data utilities.

This module provides:

- ``BIGRecModelDataset`` — model-side prepared dataset that adds
  ``history_item_ids`` to every split via ``BaseSequentialModelDataset``.

- ``BIGRecSFTDataset`` — HuggingFace-compatible ``Dataset`` that formats
  (history, target) pairs into Alpaca-style instruction-following samples
  for supervised fine-tuning of the LLM backbone.

- Domain-aware prompt helpers that produce the exact prompt template used
  in the official BIGRec implementation.

- ``build_item_text_lookup`` — build a ``list[str]`` that maps framework
  ``item_id`` → natural-language item name, reading from the prepared
  dataset's ``item_table``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from recbole3.dataset.utils import ITEM_ID
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS

if TYPE_CHECKING:
    from recbole3.model.bigrec.config import BIGRecConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain vocabulary for prompt construction
# ---------------------------------------------------------------------------

# Maps domain name → wording dict used to fill the Alpaca instruction template.
_DOMAIN_VOCAB: dict[str, dict[str, str]] = {
    "movie": {
        "item":         "movie",
        "items":        "movies",
        "action":       "watched",
        "action_past":  "watched",
        "action_list":  "watched the following movies",
    },
    "product": {
        "item":         "product",
        "items":        "products",
        "action":       "purchased",
        "action_past":  "purchased",
        "action_list":  "purchased the following products",
    },
    "item": {
        "item":         "item",
        "items":        "items",
        "action":       "interacted with",
        "action_past":  "interacted with",
        "action_list":  "interacted with the following items",
    },
}

# Alpaca-style system preamble (identical to the official implementation).
_PROMPT_PREAMBLE: str = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
)


def _resolve_domain_vocab(domain: str) -> dict[str, str]:
    """Return the wording dict for *domain*, falling back to 'item'."""
    normalized = domain.strip().lower()
    if normalized not in _DOMAIN_VOCAB:
        logger.warning(
            "BIGRec: unknown domain '%s'. Supported: %s. Falling back to 'item'.",
            domain,
            ", ".join(_DOMAIN_VOCAB),
        )
        normalized = "item"
    return _DOMAIN_VOCAB[normalized]


def build_instruction(domain: str) -> str:
    """Build the ``### Instruction:`` line for *domain*.

    Matches the official BIGRec prompt template::

        "Given a list of <items> the user has <action> before, please recommend
         a new <item> that the user likes to the user."

    Args:
        domain: Recommendation domain ('movie', 'product', 'item').

    Returns:
        The instruction string (without the ``### Instruction:`` prefix).
    """
    v = _resolve_domain_vocab(domain)
    return (
        f"Given a list of {v['items']} the user has {v['action']} before, "
        f"please recommend a new {v['item']} that the user likes to the user."
    )


def build_input_block(domain: str, history_texts: list[str]) -> str:
    """Build the ``### Input:`` block from a list of item title strings.

    Args:
        domain: Recommendation domain.
        history_texts: Ordered list of history item titles.

    Returns:
        The input block string (without the ``### Input:`` prefix).
    """
    v = _resolve_domain_vocab(domain)
    quoted = ", ".join(f'"{t}"' for t in history_texts)
    # Trailing "\n " matches the official BIGRec prompt format (process.ipynb):
    #   "input": f"{history}\n "
    # This produces the separator "...<input>\n \n\n### Response:\n" in the
    # final prompt, consistent with how official training data was generated.
    return f"The user has {v['action_list']} before:{quoted}\n "


def build_prompt(
    domain: str,
    history_texts: list[str],
    *,
    include_response_prefix: bool = True,
) -> str:
    """Build a complete Alpaca-format prompt string.

    During training, the caller appends the target title + EOS after the
    returned string.  During evaluation, ``include_response_prefix=True``
    (the default) adds ``### Response:\\n`` so generation starts there.

    Args:
        domain: Recommendation domain.
        history_texts: Ordered list of history item text strings.
        include_response_prefix: Whether to append ``### Response:\\n``.

    Returns:
        The complete prompt string.
    """
    instruction = build_instruction(domain)
    input_block = build_input_block(domain, history_texts)
    prompt = (
        f"{_PROMPT_PREAMBLE}"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_block}\n\n"
    )
    if include_response_prefix:
        prompt += "### Response:\n"
    return prompt


# ---------------------------------------------------------------------------
# Item text lookup
# ---------------------------------------------------------------------------


def build_item_text_lookup(
    prepared_data: Any,
    config: "BIGRecConfig",
) -> list[str]:
    """Build an indexed list of item text strings from *prepared_data*'s item table.

    The returned list has length ``num_items``; entry ``i`` is the text for
    framework ``item_id == i``.  When a preferred field is absent or empty
    the fallback field is tried; if still absent the raw item_id is used.

    Args:
        prepared_data: A prepared ``BaseTaskDataset`` (or ``BIGRecModelDataset``).
        config: ``BIGRecConfig`` supplying ``item_text_field`` and
            ``fallback_item_text_field``.

    Returns:
        List of item text strings indexed by framework item_id.
    """
    num_items: int = int(prepared_data.get_num_items())
    item_table: pd.DataFrame = prepared_data.get_item_table()

    # Default placeholder text — will be overwritten for items that have metadata.
    text_lookup: list[str] = [f"item_{i}" for i in range(num_items)]

    primary_col: str = config.item_text_field
    fallback_col: str | None = config.fallback_item_text_field

    if primary_col not in item_table.columns and (
        fallback_col is None or fallback_col not in item_table.columns
    ):
        logger.warning(
            "BIGRec: item_table has neither column '%s' nor '%s'. "
            "Using placeholder text for all items.",
            primary_col,
            fallback_col,
        )
        return text_lookup

    for row in item_table.itertuples(index=False):
        item_id = int(getattr(row, ITEM_ID))
        if not 0 <= item_id < num_items:
            continue

        # Resolve text: primary → fallback → placeholder.
        text: str = ""
        if primary_col in item_table.columns:
            raw = getattr(row, primary_col, None)
            if raw is not None:
                text = str(raw).strip()

        if not text and fallback_col and fallback_col in item_table.columns:
            raw = getattr(row, fallback_col, None)
            if raw is not None:
                text = str(raw).strip()

        if text:
            text_lookup[item_id] = text

    logger.info(
        "BIGRec: built item text lookup for %d items (field=%s).",
        num_items,
        primary_col,
    )
    return text_lookup


# ---------------------------------------------------------------------------
# Model-side dataset (adds history_item_ids to every split)
# ---------------------------------------------------------------------------


class BIGRecModelDataset(BaseSequentialModelDataset):
    """Model-side prepared dataset for BIGRec.

    Extends ``BaseSequentialModelDataset`` to add ``history_item_ids`` to the
    train, valid, and test ``FrameDataset`` splits.  The base class already
    implements the full cross-split history accumulation logic; this subclass
    needs no additional overrides for the basic case.

    The ``history_item_ids`` column is used by ``BIGRecSFTDataset`` (training)
    and by ``BIGRecTrainer._evaluate_split()`` (inference prompt construction).
    """

    # Inherits _build_model_datasets from BaseSequentialModelDataset.
    # Override _include_target_item_in_history here if domain-specific logic
    # is needed (e.g., exclude negative interactions from history for certain
    # datasets).  The base-class default keeps only positive / unlabeled items.


# ---------------------------------------------------------------------------
# SFT training dataset
# ---------------------------------------------------------------------------


class BIGRecSFTDataset(Dataset):
    """Alpaca-format supervised fine-tuning dataset for BIGRec.

    Each sample is derived from one (user, history, target) triple produced
    by the BIGRecModelDataset training split.  The full-text prompt is
    tokenized into ``input_ids`` / ``labels`` pairs where the instruction and
    input parts are masked with ``-100`` so that cross-entropy loss is computed
    only on the generated response (the target item title).

    Args:
        records: Pandas DataFrame with columns at least ``history_item_ids``
            (tuple of int) and ``item_id`` (int), as produced by
            ``BIGRecModelDataset.get_train_dataset()``.
        tokenizer: HuggingFace tokenizer loaded from the LLM backbone.
        item_text_lookup: Index list mapping framework item_id → title string.
        config: ``BIGRecConfig`` controlling history length, domain, etc.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        tokenizer: Any,
        item_text_lookup: list[str],
        config: "BIGRecConfig",
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._item_text_lookup = item_text_lookup
        self._config = config
        # Pre-process all records into (input_ids, labels) tensors once.
        self._samples: list[dict[str, list[int]]] = self._build_samples(records)
        logger.info("BIGRecSFTDataset: prepared %d training samples.", len(self._samples))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_samples(self, records: pd.DataFrame) -> list[dict[str, list[int]]]:
        """Tokenize all (history, target) pairs into SFT samples.

        Args:
            records: DataFrame with ``history_item_ids`` and ``item_id``.

        Returns:
            List of ``{"input_ids": [...], "labels": [...]}`` dicts.
        """
        samples: list[dict[str, list[int]]] = []
        history_max = self._config.history_max_length
        domain = self._config.domain
        max_len = self._config.max_input_length + self._config.max_new_tokens

        # Iterate once over the DataFrame — avoid repeated pandas overhead.
        for row in tqdm(
            records.itertuples(index=False),
            total=len(records),
            desc="Tokenising SFT samples",
            unit="sample",
        ):
            history_ids: tuple[int, ...] = getattr(row, HISTORY_ITEM_IDS, ())
            if history_ids is None:
                history_ids = ()
            target_id: int = int(getattr(row, ITEM_ID))

            # Truncate history to the most recent N items.
            if history_max is not None and len(history_ids) > history_max:
                history_ids = history_ids[-history_max:]

            history_texts = [self._item_text_lookup[int(iid)] for iid in history_ids]
            target_text = self._item_text_lookup[target_id]

            sample = self._format_sample(
                domain=domain,
                history_texts=history_texts,
                target_text=target_text,
                max_length=max_len,
            )
            samples.append(sample)
        return samples

    def _format_sample(
        self,
        domain: str,
        history_texts: list[str],
        target_text: str,
        max_length: int,
    ) -> dict[str, list[int]]:
        """Build one tokenized training sample.

        The prompt part (everything before ``### Response:\\n``) gets labels
        masked to ``-100``.  The response part (target title + EOS) is
        supervised.

        Args:
            domain: Recommendation domain for prompt wording.
            history_texts: List of history item title strings.
            target_text: Target item title string.
            max_length: Maximum total token length (truncates if exceeded).

        Returns:
            Dict with ``input_ids`` and ``labels`` lists.
        """
        tok = self._tokenizer

        # Build prompt prefix (everything up to and including "### Response:\n").
        prompt = build_prompt(domain, history_texts, include_response_prefix=True)

        # Build the response suffix: the quoted target title + EOS.
        response = f'"{target_text}"{tok.eos_token}'

        # Tokenize each part separately so we can compute the boundary length.
        # Prompts are truncated from the LEFT so that if the history is too long,
        # old items are dropped while "### Response:\n" is always preserved.
        # We encode WITHOUT add_special_tokens so BOS is not included in the text
        # before truncation; BOS is then prepended manually afterwards.  This
        # ensures BOS survives even when the prompt body fills max_input_length.
        bos_id: int | None = getattr(tok, "bos_token_id", None)
        # Reserve one slot for BOS when the tokenizer uses one.
        prompt_budget: int = (
            self._config.max_input_length - 1 if bos_id is not None
            else self._config.max_input_length
        )
        orig_truncation_side = getattr(tok, "truncation_side", "right")
        tok.truncation_side = "left"
        prompt_ids: list[int] = tok.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_budget,
        )
        tok.truncation_side = orig_truncation_side
        if bos_id is not None:
            prompt_ids = [bos_id] + prompt_ids
        response_ids: list[int] = tok.encode(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=self._config.max_new_tokens,
        )

        # LLaMA tokenizers sometimes prepend a spurious space token for
        # sequences that do not start with a special token.  Remove it.
        if (
            len(response_ids) > 0
            and tok.convert_ids_to_tokens(response_ids[:1]) == ["▁"]
        ):
            response_ids = response_ids[1:]

        input_ids: list[int] = (prompt_ids + response_ids)[:max_length]
        # Apply train_on_inputs supervision mask (official BIGRec default: True).
        if self._config.train_on_inputs:
            # Full-sequence supervision: identical to the official BIGRec training.
            labels: list[int] = list(input_ids)
        else:
            # Response-only supervision: mask the prompt portion from loss.
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {"input_ids": input_ids, "labels": labels}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._samples[index]


# ---------------------------------------------------------------------------
# Helpers for the inference path
# ---------------------------------------------------------------------------


def build_eval_prompts(
    batch_df: pd.DataFrame,
    item_text_lookup: list[str],
    config: "BIGRecConfig",
) -> list[str]:
    """Build inference prompt strings for one evaluation batch.

    Each row in *batch_df* must contain a ``history_item_ids`` column (tuple).
    The prompt ends with ``### Response:\\n`` so beam-search generation starts
    immediately after it.

    Args:
        batch_df: DataFrame batch from the eval ``FrameDataset``.
        item_text_lookup: Index list mapping framework item_id → title string.
        config: ``BIGRecConfig`` controlling domain and history length.

    Returns:
        List of prompt strings, one per row.
    """
    prompts: list[str] = []
    history_max = config.history_max_length

    for row in batch_df.itertuples(index=False):
        history_ids: tuple[int, ...] = getattr(row, HISTORY_ITEM_IDS, ())
        if history_ids is None:
            history_ids = ()
        if history_max is not None and len(history_ids) > history_max:
            history_ids = history_ids[-history_max:]
        history_texts = [item_text_lookup[int(iid)] for iid in history_ids]
        prompts.append(build_prompt(config.domain, history_texts, include_response_prefix=True))
    return prompts


def batchify(items: list[Any], batch_size: int):
    """Yield successive fixed-size batches from *items*.

    Args:
        items: Any list.
        batch_size: Maximum items per batch.

    Yields:
        Sub-lists of *items* of length at most *batch_size*.
    """
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


__all__ = [
    "BIGRecModelDataset",
    "BIGRecSFTDataset",
    "batchify",
    "build_eval_prompts",
    "build_input_block",
    "build_instruction",
    "build_item_text_lookup",
    "build_prompt",
]
