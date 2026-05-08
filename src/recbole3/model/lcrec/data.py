from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from typing import Any

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from recbole3.dataset.utils import ITEM_ID, USER_ID, SEEN_ITEM_IDS
from recbole3.model.lcrec.config import LCRecConfig
from recbole3.model.lcrec.prompts import index2item_prompts, item2index_prompts, seqrec_prompts

logger = logging.getLogger(__name__)


class LCRecItemTokenizer:
    """Load existing item SID mappings and convert item_id to SID token strings."""

    def __init__(self, config: LCRecConfig):
        self._load_sid_file(config.sid_file)

    def _load_sid_file(self, sid_file: str) -> None:
        if not sid_file:
            raise ValueError("sid_file must be specified for LCRec model.")
        with open(sid_file, "r") as f:
            raw: dict[str, list[int]] = json.load(f)

        if not raw:
            raise ValueError(f"sid_file '{sid_file}' is empty — no item SID mappings found.")

        self._id2tokens: dict[int, tuple[str, ...]] = {
            int(k): self._add_prefix(v) for k, v in raw.items()
        }
        self._n_codebooks: int = len(next(iter(self._id2tokens.values())))
        logger.info("LCRecItemTokenizer: loaded %d item SID mappings from %s (n_codebooks=%d)", len(self._id2tokens), sid_file, self._n_codebooks)

    def _add_prefix(self, sids: list[int]) -> tuple[str, ...]:
        return tuple(f"<{level + 1}_{code}>" for level, code in enumerate(sids))

    def tokenize(self, item_id: int) -> str:
        tokens = self._id2tokens.get(item_id)
        if tokens is None:
            raise KeyError(f"No SID mapping for item_id={item_id}")
        return "".join(tokens)

    def tokenize_tuple(self, item_id: int) -> tuple[str, ...]:
        tokens = self._id2tokens.get(item_id)
        if tokens is None:
            raise KeyError(f"No SID mapping for item_id={item_id}")
        return tokens

    def __call__(self, item_id: int) -> str:
        return self.tokenize(item_id)

    @property
    def all_tokens(self) -> list[str]:
        token_set: set[str] = set()
        for tokens in self._id2tokens.values():
            token_set.update(tokens)
        return sorted(token_set)

    @property
    def n_digit(self) -> int:
        return self._n_codebooks


# ---------------------------------------------------------------------------
# Base SFT Dataset
# ---------------------------------------------------------------------------


class _SFTDataset(Dataset):
    """Base dataset for supervised fine-tuning."""

    def __init__(
        self,
        config: LCRecConfig,
        item_tokenizer: LCRecItemTokenizer,
        llm_tokenizer: PreTrainedTokenizer,
        prompts: list[dict[str, str]],
    ):
        super().__init__()
        self.config = config
        self.item_tokenizer = item_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.prompts = prompts
        self.prompt_id: int | None = None
        self._is_llama = "llama" in config.model_name_or_path.lower()
        self.data: list[dict[str, str]] = self._get_sft_data()

    def _get_sft_data(self) -> list[dict[str, str]]:
        raise NotImplementedError

    def set_prompt_id(self, prompt_id: int) -> None:
        """Set the prompt to use for all subsequent __getitem__ calls.

        NOTE: This mutates dataset state in-place. It relies on the DataLoader
        using num_workers=0 (the default) so that the main-process mutation is
        visible to the data-loading loop. Do NOT use with num_workers > 0.
        """
        self.prompt_id = prompt_id

    def __len__(self) -> int:
        return len(self.data)

    def _get_inputs_data(self, example: dict[str, str], prompt: dict[str, str]) -> tuple[list[int], list[int]]:
        instruction = prompt["instruction"].format(**example)
        target = prompt["target"].format(**example)

        a_ids = self.llm_tokenizer.encode(
            text=instruction, add_special_tokens=True, truncation=True, max_length=self.config.max_source_length
        )
        b_ids = self.llm_tokenizer.encode(
            text=target, add_special_tokens=False, truncation=True, max_length=self.config.max_target_length
        )

        # Fix leading space token issue with LLaMA models
        if self._is_llama and len(b_ids) > 0 and b_ids[0] == self.llm_tokenizer.convert_tokens_to_ids(" "):
            b_ids = b_ids[1:]

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.llm_tokenizer.eos_token_id]
        labels = [-100] * context_length + b_ids + [self.llm_tokenizer.eos_token_id]
        return input_ids, labels

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.data[index]
        if self.prompt_id is not None:
            prompt = self.prompts[self.prompt_id]
        else:
            prompt = self.prompts[random.randint(0, len(self.prompts) - 1)]
        input_ids, labels = self._get_inputs_data(example, prompt)
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# SeqRec SFT Dataset
# ---------------------------------------------------------------------------


class SeqRecSFTDataset(_SFTDataset):
    """Sequential recommendation SFT dataset.

    Train: sliding-window samples from each user's training item sequence.
    Val/Test: use history (seen_item_ids) as input, target item_id as label.
    """

    def __init__(
        self,
        config: LCRecConfig,
        item_tokenizer: LCRecItemTokenizer,
        llm_tokenizer: PreTrainedTokenizer,
        split: "train | val | test",
        task_data: Any,
    ):
        self.split = split
        self._task_data = task_data
        super().__init__(config, item_tokenizer, llm_tokenizer, seqrec_prompts)

    def _get_sft_data(self) -> list[dict[str, str]]:
        if self.split == "train":
            return self._process_train_data()
        return self._process_eval_data()

    def _process_train_data(self) -> list[dict[str, str]]:
        train_ds = self._task_data.get_train_dataset()
        # Data is already chronologically sorted by BaseTaskDataset, just group by user
        user_seqs: dict[int, list[int]] = defaultdict(list)
        for i in range(len(train_ds)):
            inter = train_ds[i]
            user_seqs[inter[USER_ID]].append(inter[ITEM_ID])

        data: list[dict[str, str]] = []
        for user_id, item_seq in user_seqs.items():
            tokenized_seq = [self.item_tokenizer(item_id) for item_id in item_seq]
            for i in range(1, len(tokenized_seq)):
                example: dict[str, str] = {}
                example["item"] = tokenized_seq[i]
                history = tokenized_seq[:i][- self.config.max_item_seq_len :]
                example["inters"] = self.config.his_sep.join(history)
                data.append(example)
        return data

    def _process_eval_data(self) -> list[dict[str, str]]:
        eval_split = "valid" if self.split == "val" else self.split
        eval_ds = self._task_data.get_eval_dataset(eval_split)
        data: list[dict[str, str]] = []
        for i in range(len(eval_ds)):
            record = eval_ds[i]
            tokenized_history = [self.item_tokenizer(iid) for iid in record[SEEN_ITEM_IDS]]
            history = tokenized_history[- self.config.max_item_seq_len :]
            example = {
                "item": self.item_tokenizer(record[ITEM_ID]),
                "inters": self.config.his_sep.join(history),
            }
            data.append(example)
        return data

    def _prepare_generation_inputs(self, inputs: dict[str, list[int]]) -> dict[str, list[int]]:
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        new_input_ids: list[int] = []
        new_labels: list[int] = []
        for token, label in zip(input_ids, labels):
            if label == -100:
                new_input_ids.append(token)
            else:
                new_labels.append(label)
        return {"input_ids": new_input_ids, "labels": new_labels}

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        inputs = super().__getitem__(index)
        if self.split == "test":
            inputs = self._prepare_generation_inputs(inputs)
        return inputs


# ---------------------------------------------------------------------------
# ItemFeat SFT Dataset (Item2Index + Index2Item)
# ---------------------------------------------------------------------------


class ItemFeatSFTDataset(_SFTDataset):
    """SFT dataset for item-to-index and index-to-item alignment tasks."""

    def __init__(
        self,
        config: LCRecConfig,
        item_tokenizer: LCRecItemTokenizer,
        llm_tokenizer: PreTrainedTokenizer,
        prompts: list[dict[str, str]],
        train_item_ids: set[int],
        item_table: pd.DataFrame,
    ):
        self._train_item_ids = train_item_ids
        self._item_table = item_table
        super().__init__(config, item_tokenizer, llm_tokenizer, prompts)

    def _get_sft_data(self) -> list[dict[str, str]]:
        title_col = self.config.item_title_field
        desc_col = self.config.item_description_field
        data: list[dict[str, str]] = []
        for row in self._item_table.itertuples():
            item_id = int(row.item_id)
            if item_id not in self._train_item_ids:
                continue
            title = getattr(row, title_col, "") or ""
            description = getattr(row, desc_col, "") or ""
            item_token_str = self.item_tokenizer(item_id)
            data.append({
                "item": item_token_str,
                "title": str(title),
                "description": str(description),
            })
        return data


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_lcrec_sft_datasets(
    config: LCRecConfig,
    item_tokenizer: LCRecItemTokenizer,
    llm_tokenizer: PreTrainedTokenizer,
    task_data: Any,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build train/val/test SFT datasets from RecBole3.0 prepared task data.

    Args:
        config: LCRecConfig instance.
        item_tokenizer: LCRecItemTokenizer instance.
        llm_tokenizer: HuggingFace tokenizer with item tokens added.
        task_data: Prepared retrieval-style BaseTaskDataset.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from recbole3.dataset.base import BaseTaskDataset

    assert isinstance(task_data, BaseTaskDataset)

    # SeqRec datasets - each fetches data from task_data based on split
    seqrec_train = SeqRecSFTDataset(
        config=config,
        item_tokenizer=item_tokenizer,
        llm_tokenizer=llm_tokenizer,
        split="train",
        task_data=task_data,
    )
    seqrec_val = SeqRecSFTDataset(
        config=config,
        item_tokenizer=item_tokenizer,
        llm_tokenizer=llm_tokenizer,
        split="val",
        task_data=task_data,
    )
    seqrec_test = SeqRecSFTDataset(
        config=config,
        item_tokenizer=item_tokenizer,
        llm_tokenizer=llm_tokenizer,
        split="test",
        task_data=task_data,
    )

    # ItemFeat datasets
    item_table = task_data.get_item_table()
    has_item_feats = (
        config.item_title_field in item_table.columns and config.item_description_field in item_table.columns
    )

    if has_item_feats:
        train_ds = task_data.get_train_dataset()
        train_item_ids: set[int] = {train_ds[i][ITEM_ID] for i in range(len(train_ds))}

        item2index_ds = ItemFeatSFTDataset(
            config=config,
            item_tokenizer=item_tokenizer,
            llm_tokenizer=llm_tokenizer,
            prompts=item2index_prompts,
            train_item_ids=train_item_ids,
            item_table=item_table,
        )
        index2item_ds = ItemFeatSFTDataset(
            config=config,
            item_tokenizer=item_tokenizer,
            llm_tokenizer=llm_tokenizer,
            prompts=index2item_prompts,
            train_item_ids=train_item_ids,
            item_table=item_table,
        )
        train_dataset: Dataset = ConcatDataset([seqrec_train, item2index_ds, index2item_ds])
    else:
        logger.warning("Item table missing title/description columns; skipping ItemFeat tasks.")
        train_dataset = seqrec_train

    logger.info(
        "LCRec SFT datasets: train=%d, val=%d, test=%d",
        len(train_dataset), len(seqrec_val), len(seqrec_test),
    )
    return train_dataset, seqrec_val, seqrec_test
