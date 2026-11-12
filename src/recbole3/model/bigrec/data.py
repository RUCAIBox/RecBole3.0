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

from recbole3.dataset.utils import ITEM_ID, USER_ID
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

# Alpaca-style system preamble.  The trailing space after "request." matches
# official BIGRec train.py (generate_prompt) — BPE/SentencePiece merges around
# whitespace are sensitive, so a missing space desyncs the full tokenization
# after this point.  Official inference.py uses two trailing spaces; we mirror
# train.py to keep training tokenization aligned.
_PROMPT_PREAMBLE: str = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request. \n\n"
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
    """Build an indexed list of item title strings from *prepared_data*'s item table.

    The returned list has length ``num_items``; entry ``i`` is the title for
    framework ``item_id == i``. Missing or empty titles fail fast because BIGRec
    grounding compares generated title text against item-title embeddings.

    Args:
        prepared_data: A prepared ``BaseTaskDataset`` (or ``BIGRecModelDataset``).
        config: ``BIGRecConfig`` supplying ``item_text_field``.

    Returns:
        List of item title strings indexed by framework item_id.
    """
    num_items: int = int(prepared_data.get_num_items())
    item_table: pd.DataFrame = prepared_data.get_item_table()

    primary_col: str = config.item_text_field
    text_lookup: list[str | None] = [None for _ in range(num_items)]
    missing_titles: list[int] = []

    if primary_col not in item_table.columns:
        available = ", ".join(str(col) for col in item_table.columns)
        raise ValueError(
            "BIGRec requires item titles for grounding, but item_table is missing "
            f"column '{primary_col}'. Use Amazon Reviews 2023 with metadata_mode='fields' "
            f"so a title column is materialized. Available columns: {available}"
        )

    for row in item_table.itertuples(index=False):
        item_id = int(getattr(row, ITEM_ID))
        if not 0 <= item_id < num_items:
            continue

        text: str = ""
        raw = getattr(row, primary_col, None)
        if raw is not None:
            text = str(raw).strip()
        if text:
            text_lookup[item_id] = text
        else:
            missing_titles.append(item_id)

    unresolved = [idx for idx, text in enumerate(text_lookup) if text is None]
    if missing_titles or unresolved:
        missing_count = len(set(missing_titles + unresolved))
        examples = ", ".join(str(i) for i in (missing_titles or unresolved)[:10])
        raise ValueError(
            "BIGRec requires non-empty item titles for every item. "
            f"Found {missing_count} item(s) without a usable '{primary_col}' value; "
            f"example item_id(s): {examples}. Use Amazon Reviews 2023 with "
            "metadata_mode='fields' and verify the raw metadata contains titles "
            "for all interacted items."
        )

    logger.info(
        "BIGRec: built item text lookup for %d items (field=%s).",
        num_items,
        primary_col,
    )
    return [str(text) for text in text_lookup]


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

    def _build_model_datasets(self, *, model_config: Any):
        """Filter untitled items, then inject sequential histories.

        BIGRec uses item titles as the only natural-language item identity in
        both SFT targets and embedding grounding. Items without usable titles
        are removed together with their interactions; remaining item ids are
        compacted before RecBole3 rebuilds train/valid/test splits.
        """
        self._filter_items_without_titles(model_config=model_config)
        return super()._build_model_datasets(model_config=model_config)

    def _filter_items_without_titles(self, *, model_config: Any) -> None:
        title_col = str(getattr(model_config, "item_text_field", "title"))
        item_table = self._item_table.copy()
        if title_col not in item_table.columns:
            available = ", ".join(str(col) for col in item_table.columns)
            raise ValueError(
                "BIGRec requires item titles before model-data construction, "
                f"but item_table is missing column '{title_col}'. Available columns: {available}"
            )

        title_values = item_table[title_col].map(self._normalize_title_value)
        keep_mask = title_values != ""
        dropped_item_count = int((~keep_mask).sum())
        if dropped_item_count == 0:
            return

        kept_item_table = item_table.loc[keep_mask].copy()
        old_item_ids = kept_item_table[ITEM_ID].astype("int64").tolist()
        item_id_map = {old_id: new_id for new_id, old_id in enumerate(old_item_ids)}

        original_interaction_count = len(self._interactions)
        filtered_interactions = self._interactions.loc[
            self._interactions[ITEM_ID].isin(item_id_map)
        ].copy()
        dropped_interaction_count = original_interaction_count - len(filtered_interactions)

        kept_item_table[ITEM_ID] = kept_item_table[ITEM_ID].map(item_id_map).astype("int64")
        filtered_interactions[ITEM_ID] = (
            filtered_interactions[ITEM_ID].map(item_id_map).astype("int64")
        )

        self._item_table = kept_item_table.reset_index(drop=True)
        self._num_items = int(len(self._item_table))
        self._interactions = filtered_interactions.reset_index(drop=True)
        self._build_prepared_datasets()

        logger.warning(
            "BIGRec: filtered %d item(s) without non-empty '%s' and %d related interaction(s); "
            "%d item(s) and %d interaction(s) remain.",
            dropped_item_count,
            title_col,
            dropped_interaction_count,
            self._num_items,
            len(self._interactions),
        )

    @staticmethod
    def _normalize_title_value(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()


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

        Tokenization strategy mirrors the official BIGRec ``train.py``:
        the full prompt (instruction + input + response) is encoded once as a
        single string so BPE/SentencePiece merges across the ``### Response:\\n``
        boundary are identical to the official implementation.  For
        ``train_on_inputs=False`` the boundary is recovered by independently
        encoding the same prompt with an empty ``output`` field — matching
        official ``generate_and_tokenize_prompt``.
        """
        tok = self._tokenizer
        eos_id: int | None = getattr(tok, "eos_token_id", None)

        # Build the two strings the official code encodes:
        #   full_prompt = "...### Response:\n\"<target>\""
        #   user_prompt = "...### Response:\n"        (output field empty)
        full_prompt = build_prompt(
            domain, history_texts, include_response_prefix=True
        ) + f'"{target_text}"'
        user_prompt = build_prompt(
            domain, history_texts, include_response_prefix=True
        )

        # Left-truncate the full prompt so that if history exceeds the budget,
        # older items are dropped while "### Response:\n\"<target>\"" is kept.
        # Encode WITH special tokens so the tokenizer's BOS/EOS behaviour
        # matches its default (the official code relies on this default).
        orig_truncation_side = getattr(tok, "truncation_side", "right")
        tok.truncation_side = "left"
        input_ids: list[int] = tok.encode(
            full_prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        # Append EOS if the tokenizer did not already add one and there is room
        # (matches official train.py:tokenize add_eos_token=True branch).
        if (
            eos_id is not None
            and (len(input_ids) == 0 or input_ids[-1] != eos_id)
            and len(input_ids) < max_length
        ):
            input_ids.append(eos_id)
        tok.truncation_side = orig_truncation_side

        if self._config.train_on_inputs:
            labels: list[int] = list(input_ids)
        else:
            # Boundary = length of "user_prompt" tokenized identically (no EOS
            # appended, matching official add_eos_token=False for the boundary
            # probe).
            user_ids: list[int] = tok.encode(
                user_prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            prompt_len = min(len(user_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {"input_ids": input_ids, "labels": labels}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._samples[index]


def select_sft_training_records(
    records: pd.DataFrame,
    history_max_length: int | None,
) -> pd.DataFrame:
    """Select the autoregressive rows used for BIGRec fine-tuning.

    Users that reach the configured history length contribute every full
    sliding-window row.  Users that never reach it contribute exactly their
    latest row, whose history is also their longest one.  Evaluation frames are
    deliberately not filtered by this helper.

    Args:
        records: Autoregressive training frame in chronological row order.
        history_max_length: Required full history length. ``None`` keeps all
            rows for backwards-compatible unbounded-history experiments.

    Returns:
        A new frame preserving the original row order and columns.
    """
    if records.empty or history_max_length is None:
        return records.copy().reset_index(drop=True)
    if history_max_length <= 0:
        raise ValueError("history_max_length must be None or a positive integer.")
    required_columns = {USER_ID, HISTORY_ITEM_IDS}
    missing = required_columns.difference(records.columns)
    if missing:
        raise ValueError(
            "BIGRec SFT record selection requires columns: "
            + ", ".join(sorted(missing))
        )

    history_lengths = records[HISTORY_ITEM_IDS].map(lambda value: len(value or ()))
    full_mask = history_lengths.eq(int(history_max_length))
    full_records = records.loc[full_mask]

    users_with_full_history = set(full_records[USER_ID].tolist())
    short_records = records.loc[~records[USER_ID].isin(users_with_full_history)]
    # Histories are non-decreasing within a user in the framework-generated
    # autoregressive frame, so the final row is the unique desired sample (or
    # the latest row in the presence of non-positive interactions).
    short_records = short_records.groupby(USER_ID, sort=False, group_keys=False).tail(1)

    selected = pd.concat([full_records, short_records], axis=0).sort_index(kind="stable")
    return selected.reset_index(drop=True)


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
    "select_sft_training_records",
]
