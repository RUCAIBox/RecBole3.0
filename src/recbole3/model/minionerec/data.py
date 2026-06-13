from __future__ import annotations

import ast
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from recbole3.dataset import FrameDataset, ITEM_ID, LABEL, SEEN_ITEM_IDS
from recbole3.model.minionerec.config import MiniOneRecConfig
from recbole3.model.sequential import build_history_item_ids


logger = logging.getLogger(__name__)


MINIONEREC_SEQREC_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""

MINIONEREC_ITEM_ALIGNMENT_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""

MINIONEREC_FUSION_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""


@dataclass(frozen=True, slots=True)
class MiniOneRecSIDCodec:
    """Map RecBole remapped item ids to MiniOneRec semantic ID strings."""

    item_to_tokens: dict[int, tuple[str, ...]]
    item_to_sid: dict[int, str]
    sid_to_item: dict[str, int]
    sid_to_items: dict[str, tuple[int, ...]]
    all_tokens: tuple[str, ...]
    item_ids: tuple[int, ...]

    @classmethod
    def from_file(
        cls,
        sid_file: str,
        *,
        num_items: int | None = None,
        require_complete: bool = True,
        allow_duplicate_sid_aliases: bool = False,
    ) -> "MiniOneRecSIDCodec":
        raw_sids = _load_sid_mapping(sid_file)
        return cls.from_mapping(
            raw_sids,
            num_items=num_items,
            require_complete=require_complete,
            allow_duplicate_sid_aliases=allow_duplicate_sid_aliases,
        )

    @classmethod
    def from_mapping(
        cls,
        raw_sids: Mapping[Any, Any],
        *,
        num_items: int | None = None,
        require_complete: bool = True,
        allow_duplicate_sid_aliases: bool = False,
    ) -> "MiniOneRecSIDCodec":
        item_to_tokens: dict[int, tuple[str, ...]] = {}
        token_set: set[str] = set()
        for raw_item_id, raw_tokens in raw_sids.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MiniOneRec SID key must be an integer-like item id, got {raw_item_id!r}.") from exc
            if item_id < 0:
                raise ValueError(f"MiniOneRec SID item_id must be non-negative, got {item_id}.")
            if num_items is not None and item_id >= int(num_items):
                raise ValueError(f"MiniOneRec SID item_id {item_id} is outside dataset range [0, {int(num_items) - 1}].")
            tokens = _normalize_sid_tokens(raw_tokens)
            if not tokens:
                raise ValueError(f"MiniOneRec SID for item_id {item_id} must contain at least one token.")
            item_to_tokens[item_id] = tokens
            token_set.update(tokens)

        if num_items is not None and require_complete:
            missing = sorted(set(range(int(num_items))) - set(item_to_tokens))
            if missing:
                preview = ", ".join(str(item_id) for item_id in missing[:10])
                raise ValueError(f"MiniOneRec sid_file is missing {len(missing)} item ids. First missing ids: {preview}.")

        item_to_sid = {item_id: "".join(tokens) for item_id, tokens in item_to_tokens.items()}
        sid_to_item: dict[str, int] = {}
        sid_to_items_list: dict[str, list[int]] = {}
        for item_id, sid in item_to_sid.items():
            sid_to_items_list.setdefault(sid, []).append(int(item_id))

        sid_to_items = {sid: tuple(sorted(item_ids)) for sid, item_ids in sid_to_items_list.items()}
        duplicate_sids = {sid: item_ids for sid, item_ids in sid_to_items.items() if len(item_ids) > 1}
        if duplicate_sids:
            first_sid, first_items = next(iter(sorted(duplicate_sids.items())))
            if not bool(allow_duplicate_sid_aliases):
                raise ValueError(
                    "MiniOneRec item.index.json contains duplicate SID strings, which makes item-level decoding ambiguous. "
                    f"Example: {first_sid!r} -> {list(first_items)}."
                )
            logger.warning(
                "MiniOneRec item.index.json contains %d duplicate SID string aliases; keeping item aliases for decoding. "
                "Example: %s -> %s",
                len(duplicate_sids),
                first_sid,
                list(first_items),
            )
        for sid, item_ids in sid_to_items.items():
            sid_to_item[sid] = int(item_ids[0])

        return cls(
            item_to_tokens=item_to_tokens,
            item_to_sid=item_to_sid,
            sid_to_item=sid_to_item,
            sid_to_items=sid_to_items,
            all_tokens=tuple(sorted(token_set)),
            item_ids=tuple(sorted(item_to_sid)),
        )

    def item_sid(self, item_id: int) -> str:
        try:
            return self.item_to_sid[int(item_id)]
        except KeyError as exc:
            raise KeyError(f"No MiniOneRec SID mapping for item_id={int(item_id)}.") from exc

    def decode_sid(self, text: str) -> int | None:
        return self.sid_to_item.get(str(text).strip())

    def decode_sid_candidates(self, text: str) -> tuple[int, ...]:
        return self.sid_to_items.get(str(text).strip(), ())


def load_minionerec_sid_codec(config: MiniOneRecConfig, task_data: Any) -> MiniOneRecSIDCodec:
    """Load MiniOneRec SIDs in the item-id namespace requested by config."""

    item_id_space = str(config.sid_file_item_id_space).strip().lower()
    if item_id_space == "recbole":
        return MiniOneRecSIDCodec.from_file(
            config.sid_file,
            num_items=int(task_data.get_num_items()),
            require_complete=bool(config.require_complete_sid_file),
            allow_duplicate_sid_aliases=bool(config.allow_duplicate_sid_aliases),
        )
    if item_id_space == "raw":
        raw_sids = _load_sid_mapping(config.sid_file)
        remapped_sids = remap_minionerec_sid_mapping_to_recbole(raw_sids, task_data)
        return MiniOneRecSIDCodec.from_mapping(
            remapped_sids,
            num_items=int(task_data.get_num_items()),
            require_complete=bool(config.require_complete_sid_file),
            allow_duplicate_sid_aliases=bool(config.allow_duplicate_sid_aliases),
        )
    raise ValueError("MiniOneRecConfig.sid_file_item_id_space must be either 'recbole' or 'raw'.")


def remap_minionerec_sid_mapping_to_recbole(raw_sids: Mapping[Any, Any], task_data: Any) -> dict[int, Any]:
    """Remap a source MiniOneRec item.index.json keyed by raw item ids into RecBole item ids."""

    item_id_map = getattr(task_data, "_item_id_map", None)
    if not isinstance(item_id_map, Mapping):
        raise ValueError(
            "sid_file_item_id_space='raw' requires a prepared RecBole dataset with an _item_id_map. "
            "Use sid_file_item_id_space='recbole' if the SID file is already remapped."
        )

    remapped: dict[int, Any] = {}
    missing_raw_ids: list[Any] = []
    for raw_item_id, tokens in raw_sids.items():
        recbole_item_id = _lookup_recbole_item_id(raw_item_id, item_id_map)
        if recbole_item_id is None:
            missing_raw_ids.append(raw_item_id)
            continue
        if recbole_item_id in remapped:
            raise ValueError(
                f"Multiple raw MiniOneRec SID keys map to RecBole item_id={recbole_item_id}; "
                "the SID file and dataset item table are inconsistent."
            )
        remapped[int(recbole_item_id)] = tokens

    if missing_raw_ids:
        preview = ", ".join(repr(item_id) for item_id in missing_raw_ids[:10])
        raise ValueError(
            f"MiniOneRec sid_file contains {len(missing_raw_ids)} raw item ids that are absent from the "
            f"RecBole item id map. First missing ids: {preview}."
        )
    return remapped


class MiniOneRecTokenizerAdapter:
    """Tokenizer wrapper matching MiniOneRec's bos/eos stripping behavior."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

    def encode(self, text: str, *, bos: bool, eos: bool) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"MiniOneRec tokenizer expects str input, got {type(text).__name__}.")
        token_ids = list(self.tokenizer.encode(text))
        while token_ids and self.bos_id is not None and token_ids[0] == self.bos_id:
            token_ids = token_ids[1:]
        while token_ids and self.eos_id is not None and token_ids[-1] == self.eos_id:
            token_ids = token_ids[:-1]
        if bos and self.bos_id is not None:
            token_ids = [int(self.bos_id), *token_ids]
        if eos and self.eos_id is not None:
            token_ids = [*token_ids, int(self.eos_id)]
        return token_ids


class MiniOneRecSFTDataset(Dataset[dict[str, list[int]]]):
    """Tokenized MiniOneRec SFT examples preserving the original prompt format."""

    def __init__(
        self,
        examples: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        max_len: int,
        instruction: str = MINIONEREC_SEQREC_INSTRUCTION,
        test: bool = False,
    ) -> None:
        super().__init__()
        self.examples = list(examples)
        self.tokenizer = MiniOneRecTokenizerAdapter(tokenizer)
        self.max_len = int(max_len)
        self.instruction = instruction
        self.test = bool(test)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[int(index)]
        prompt = _generate_prompt(str(example["input"]), output="")
        target = "" if self.test else str(example["output"])
        return self._encode_prompt_and_target(prompt=prompt, target=target)

    def _encode_prompt_and_target(self, *, prompt: str, target: str) -> dict[str, list[int]]:
        tokens = self.tokenizer.encode(self.instruction, bos=True, eos=False)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {
                "input_ids": tokens[-self.max_len :],
                "attention_mask": attention_mask[-self.max_len :],
            }

        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
        }


class MiniOneRecPromptDataset(Dataset[dict[str, Any]]):
    """Prompt/completion records for MiniOneRec GRPO."""

    def __init__(self, examples: Sequence[dict[str, Any]]) -> None:
        super().__init__()
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[int(index)]


@dataclass(frozen=True, slots=True)
class MiniOneRecRLDatasets:
    train_dataset: MiniOneRecPromptDataset
    eval_dataset: MiniOneRecPromptDataset
    prompt2history: dict[str, str]
    history2target: dict[str, str]
    prompt2excluded_item_ids: dict[str, tuple[int, ...]]


def build_minionerec_sft_datasets(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    tokenizer: Any,
    task_data: Any,
) -> tuple[Dataset[Any], Dataset[Any]]:
    """Build MiniOneRec SFT train/valid datasets from prepared RecBole data."""

    train_examples = build_sequence_sft_examples(config, codec, task_data, split="train", eval_prompt=False)
    valid_examples = build_sequence_sft_examples(config, codec, task_data, split="valid", eval_prompt=False)

    train_parts: list[Dataset[Any]] = [
        MiniOneRecSFTDataset(train_examples, tokenizer, max_len=config.max_len),
    ]
    if config.add_item_alignment_tasks:
        item_examples = build_item_alignment_examples(config, codec, task_data)
        if item_examples:
            train_parts.append(
                MiniOneRecSFTDataset(
                    item_examples,
                    tokenizer,
                    max_len=config.max_len,
                    instruction=MINIONEREC_ITEM_ALIGNMENT_INSTRUCTION,
                )
            )
    if config.add_fusion_seqrec_task:
        fusion_examples = build_fusion_seqrec_examples(config, codec, task_data)
        if fusion_examples:
            train_parts.append(
                MiniOneRecSFTDataset(
                    fusion_examples,
                    tokenizer,
                    max_len=config.max_len,
                    instruction=MINIONEREC_FUSION_INSTRUCTION,
                )
            )

    train_dataset: Dataset[Any]
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)
    valid_dataset = MiniOneRecSFTDataset(valid_examples, tokenizer, max_len=config.max_len)
    return train_dataset, valid_dataset


def build_minionerec_rl_datasets(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
) -> MiniOneRecRLDatasets:
    """Build original MiniOneRec GRPO prompt/completion datasets."""

    prompt2history: dict[str, str] = {}
    history2target: dict[str, str] = {}
    prompt2excluded_item_ids: dict[str, tuple[int, ...]] = {}
    train_examples: list[dict[str, Any]] = []
    eval_examples: list[dict[str, Any]] = []

    train_examples.extend(
        _sequence_rl_examples(
            config,
            codec,
            task_data,
            split="train",
            prompt2history=prompt2history,
            history2target=history2target,
            prompt2excluded_item_ids=prompt2excluded_item_ids,
        )
    )
    eval_examples.extend(
        _sequence_rl_examples(
            config,
            codec,
            task_data,
            split="valid",
            prompt2history=prompt2history,
            history2target=history2target,
            prompt2excluded_item_ids=prompt2excluded_item_ids,
        )
    )

    if config.rl_add_item_alignment_tasks:
        train_examples.extend(
            _item_alignment_rl_examples(
                config,
                codec,
                task_data,
                prompt2history=prompt2history,
                history2target=history2target,
                prompt2excluded_item_ids=prompt2excluded_item_ids,
            )
        )
    if config.rl_add_title_sequence_task:
        title_examples = _title_sequence_rl_examples(
            config,
            codec,
            task_data,
            prompt2history=prompt2history,
            history2target=history2target,
            prompt2excluded_item_ids=prompt2excluded_item_ids,
        )
        train_examples.extend(_sample_examples(title_examples, sample_size=int(config.rl_title_sequence_sample_size), seed=0))

    return MiniOneRecRLDatasets(
        train_dataset=MiniOneRecPromptDataset(train_examples),
        eval_dataset=MiniOneRecPromptDataset(eval_examples),
        prompt2history=prompt2history,
        history2target=history2target,
        prompt2excluded_item_ids=prompt2excluded_item_ids,
    )


def build_sequence_sft_examples(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
    *,
    split: str,
    eval_prompt: bool,
) -> list[dict[str, Any]]:
    if split == "train":
        frame = _dataset_frame(task_data.get_train_dataset())
        histories, _ = build_history_item_ids(frame, history_max_length=int(config.history_max_length))
        examples: list[dict[str, Any]] = []
        for record, history_item_ids in zip(frame.to_dict("records"), histories, strict=True):
            if not _is_positive_record(record):
                continue
            if len(history_item_ids) < int(config.min_history_length):
                continue
            examples.append(
                _sequence_example(
                    config,
                    codec,
                    history_item_ids,
                    int(record[ITEM_ID]),
                    eval_prompt=False,
                    seen_item_ids=history_item_ids,
                )
            )
        return examples

    eval_frame = _dataset_frame(task_data.get_eval_dataset("valid" if split == "valid" else "test"))
    if SEEN_ITEM_IDS not in eval_frame.columns:
        raise ValueError("MiniOneRec retrieval evaluation requires eval records with seen_item_ids.")
    examples = []
    for record in eval_frame.to_dict("records"):
        history_item_ids = tuple(int(item_id) for item_id in (record.get(SEEN_ITEM_IDS) or ()))
        if len(history_item_ids) < int(config.min_history_length):
            continue
        examples.append(
            _sequence_example(
                config,
                codec,
                history_item_ids,
                int(record[ITEM_ID]),
                eval_prompt=eval_prompt,
                seen_item_ids=history_item_ids,
            )
        )
    return examples


def _sequence_rl_examples(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
    *,
    split: str,
    prompt2history: dict[str, str],
    history2target: dict[str, str],
    prompt2excluded_item_ids: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    examples = build_sequence_sft_examples(config, codec, task_data, split=split, eval_prompt=False)
    records: list[dict[str, Any]] = []
    skipped_excluded_targets = 0
    for example in examples:
        excluded_item_ids = tuple(int(item_id) for item_id in example.get("seen_item_ids", ()))
        if _rl_target_is_excluded(config, int(example["target_item_id"]), excluded_item_ids):
            skipped_excluded_targets += 1
            continue
        prompt = _generate_prompt(str(example["input"]), output="")
        completion = str(example["output"])
        history_key = _history_key_from_prompt_input(str(example["input"]))
        prompt2history[prompt] = history_key
        history2target[history_key] = completion
        prompt2excluded_item_ids[prompt] = excluded_item_ids
        records.append({"prompt": prompt, "completion": completion, "excluded_item_ids": excluded_item_ids})
    if skipped_excluded_targets:
        logger.warning(
            "Skipped %d MiniOneRec %s GRPO sequence examples because the target item is in "
            "excluded history under rl_exclude_history=true.",
            skipped_excluded_targets,
            split,
        )
    return records


def _item_alignment_rl_examples(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
    *,
    prompt2history: dict[str, str],
    history2target: dict[str, str],
    prompt2excluded_item_ids: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    item_table = task_data.get_item_table()
    records: list[dict[str, Any]] = []
    for item_record in item_table.to_dict("records"):
        item_id = int(item_record[ITEM_ID])
        if item_id not in codec.item_to_sid:
            continue
        sid = codec.item_sid(item_id)
        title = _field_text(item_record.get(config.item_title_field)).strip()
        description = _description_text(item_record.get(config.item_description_field)).strip()
        if title:
            records.append(
                _register_rl_record(
                    prompt=f"""### User Input: 
Which item has the title: {title}?

### Response:\n""",
                    history_key=title,
                    completion=sid + "\n",
                    prompt2history=prompt2history,
                    history2target=history2target,
                    prompt2excluded_item_ids=prompt2excluded_item_ids,
                )
            )
        if description:
            records.append(
                _register_rl_record(
                    prompt=f"""### User Input: 
An item can be described as follows: "{description}". Which item is it describing?

### Response:\n""",
                    history_key=description,
                    completion=sid + "\n",
                    prompt2history=prompt2history,
                    history2target=history2target,
                    prompt2excluded_item_ids=prompt2excluded_item_ids,
                )
            )
    return records


def _title_sequence_rl_examples(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
    *,
    prompt2history: dict[str, str],
    history2target: dict[str, str],
    prompt2excluded_item_ids: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    item_lookup = _item_text_lookup(config, task_data)
    frame = _dataset_frame(task_data.get_train_dataset())
    histories, _ = build_history_item_ids(frame, history_max_length=int(config.history_max_length))
    records: list[dict[str, Any]] = []
    skipped_excluded_targets = 0
    for record, history_item_ids in zip(frame.to_dict("records"), histories, strict=True):
        if not _is_positive_record(record):
            continue
        if len(history_item_ids) < int(config.min_history_length):
            continue
        target_item_id = int(record[ITEM_ID])
        if _rl_target_is_excluded(config, target_item_id, history_item_ids):
            skipped_excluded_targets += 1
            continue
        titles = [item_lookup.get(int(item_id), f"Item_{int(item_id)}") for item_id in history_item_ids]
        inter_titles = ", ".join(f'"{title}"' for title in titles)
        prompt = f"""### User Input: 
Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?

### Response:\n"""
        history_key = "::".join(titles)
        completion = codec.item_sid(target_item_id) + "\n"
        records.append(
            _register_rl_record(
                prompt=prompt,
                history_key=history_key,
                completion=completion,
                prompt2history=prompt2history,
                history2target=history2target,
                prompt2excluded_item_ids=prompt2excluded_item_ids,
                excluded_item_ids=history_item_ids,
            )
        )
    if skipped_excluded_targets:
        logger.warning(
            "Skipped %d MiniOneRec title-sequence GRPO examples because the target item is in "
            "excluded history under rl_exclude_history=true.",
            skipped_excluded_targets,
        )
    return records


def build_item_alignment_examples(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
) -> list[dict[str, str]]:
    item_lookup = _item_text_lookup(config, task_data)
    examples: list[dict[str, str]] = []
    for item_id in codec.item_ids:
        title = item_lookup.get(int(item_id), "").strip()
        if not title:
            continue
        sid = codec.item_sid(int(item_id))
        examples.append(
            {
                "input": f"Which item has the title: {title}?",
                "output": sid + "\n",
            }
        )
        examples.append(
            {
                "input": f"What is the title of item \"{sid}\"?",
                "output": title + "\n",
            }
        )
    return examples


def build_fusion_seqrec_examples(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    task_data: Any,
) -> list[dict[str, str]]:
    item_lookup = _item_text_lookup(config, task_data)
    frame = _dataset_frame(task_data.get_train_dataset())
    histories, _ = build_history_item_ids(frame, history_max_length=int(config.history_max_length))
    examples: list[dict[str, str]] = []
    for record, history_item_ids in zip(frame.to_dict("records"), histories, strict=True):
        if not _is_positive_record(record):
            continue
        if len(history_item_ids) < int(config.min_history_length):
            continue
        target_title = item_lookup.get(int(record[ITEM_ID]), "").strip()
        if not target_title:
            continue
        history = ", ".join(codec.item_sid(int(item_id)) for item_id in history_item_ids[-int(config.history_max_length) :])
        examples.append(
            {
                "input": (
                    f"The user has sequentially interacted with items {history}. "
                    "Can you recommend the next item for him? Tell me the title of the item"
                ),
                "output": target_title + "\n",
            }
        )
    return examples


def _sequence_example(
    config: MiniOneRecConfig,
    codec: MiniOneRecSIDCodec,
    history_item_ids: Sequence[int],
    target_item_id: int,
    *,
    eval_prompt: bool,
    seen_item_ids: Sequence[int] = (),
) -> dict[str, Any]:
    truncated_history = tuple(int(item_id) for item_id in history_item_ids[-int(config.history_max_length) :])
    history = ", ".join(codec.item_sid(item_id) for item_id in truncated_history)
    if eval_prompt:
        user_input = f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}"
    else:
        user_input = (
            f"The user has interacted with items {history} in chronological order. "
            "Can you predict the next possible item that the user may expect?"
        )
    return {
        "input": user_input,
        "output": codec.item_sid(int(target_item_id)) + "\n",
        "target_item_id": int(target_item_id),
        "seen_item_ids": tuple(int(item_id) for item_id in seen_item_ids),
    }


def _register_rl_record(
    *,
    prompt: str,
    history_key: str,
    completion: str,
    prompt2history: dict[str, str],
    history2target: dict[str, str],
    prompt2excluded_item_ids: dict[str, tuple[int, ...]],
    excluded_item_ids: Sequence[int] = (),
) -> dict[str, Any]:
    prompt2history[prompt] = history_key
    history2target[history_key] = completion
    excluded = tuple(int(item_id) for item_id in excluded_item_ids)
    prompt2excluded_item_ids[prompt] = excluded
    return {"prompt": prompt, "completion": completion, "excluded_item_ids": excluded}


def _rl_target_is_excluded(config: MiniOneRecConfig, target_item_id: int, excluded_item_ids: Sequence[int]) -> bool:
    return bool(config.rl_exclude_history) and int(target_item_id) in {int(item_id) for item_id in excluded_item_ids}


def _generate_prompt(user_input: str, *, output: str) -> str:
    return f"""### User Input: 
{user_input}

### Response:\n{output}"""


def _normalize_sid_tokens(raw_tokens: Any) -> tuple[str, ...]:
    if not isinstance(raw_tokens, (list, tuple)):
        raise ValueError("MiniOneRec item.index.json values must be token lists (list[str]).")
    tokens: list[str] = []
    for token in raw_tokens:
        if not isinstance(token, str):
            raise ValueError("MiniOneRec item.index.json tokens must be strings.")
        stripped = token.strip()
        if not stripped:
            raise ValueError("MiniOneRec SID tokens must be non-empty.")
        if not (stripped.startswith("<") and stripped.endswith(">")):
            raise ValueError(f"MiniOneRec SID token must look like '<a_0>', got {token!r}.")
        tokens.append(stripped)
    return tuple(tokens)


def _load_sid_mapping(sid_file: str) -> dict[Any, Any]:
    if not str(sid_file or "").strip():
        raise ValueError("MiniOneRecConfig.sid_file must point to a MiniOneRec item.index.json file.")
    path = Path(sid_file)
    if not path.exists():
        raise FileNotFoundError(f"MiniOneRec sid_file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_sids = json.load(file)
    if not isinstance(raw_sids, dict):
        raise ValueError("MiniOneRec sid_file must be a JSON object keyed by item ids.")
    return raw_sids


def _lookup_recbole_item_id(raw_item_id: Any, item_id_map: Mapping[Any, int]) -> int | None:
    candidates: list[Any] = [raw_item_id]
    if isinstance(raw_item_id, str):
        stripped = raw_item_id.strip()
        candidates.append(stripped)
        try:
            candidates.append(int(stripped))
        except ValueError:
            pass
    else:
        candidates.append(str(raw_item_id))

    for candidate in candidates:
        if candidate in item_id_map:
            return int(item_id_map[candidate])
    return None


def _history_key_from_prompt_input(user_input: str) -> str:
    marker = "The user has interacted with items "
    suffix = " in chronological order."
    if marker in user_input and suffix in user_input:
        history = user_input.split(marker, 1)[1].split(suffix, 1)[0]
        return "::".join(part.strip() for part in history.split(",") if part.strip())
    return user_input


def _sample_examples(examples: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size < 0 or sample_size >= len(examples):
        return examples
    return random.Random(int(seed)).sample(examples, sample_size)


def _dataset_frame(dataset: Any) -> pd.DataFrame:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"MiniOneRec requires FrameDataset splits, got {type(dataset).__name__}.")
    return dataset.frame.copy()


def _is_positive_record(record: dict[str, Any]) -> bool:
    label = record.get(LABEL)
    return label is None or pd.isna(label) or float(label) > 0


def _item_text_lookup(config: MiniOneRecConfig, task_data: Any) -> dict[int, str]:
    item_table = task_data.get_item_table()
    lookup: dict[int, str] = {}
    for record in item_table.to_dict("records"):
        item_id = int(record[ITEM_ID])
        lookup[item_id] = _resolve_item_text(config, record, item_id=item_id)
    return lookup


def _resolve_item_text(config: MiniOneRecConfig, record: dict[str, Any], *, item_id: int) -> str:
    fields = [config.item_title_field, config.item_description_field]
    if config.fallback_item_text_field:
        fields.append(config.fallback_item_text_field)
    for field_name in fields:
        value = record.get(field_name)
        if value is None:
            continue
        text = _feature_to_text(value).strip()
        if text:
            return text
    return f"Item_{item_id}"


def _feature_to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return ""
    return _feature_to_text(value)


def _description_text(value: Any) -> str:
    if isinstance(value, str) and value.startswith("['") and value.endswith("']"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list) and parsed:
            return _feature_to_text(parsed[0])
    return _field_text(value)


__all__ = [
    "MINIONEREC_FUSION_INSTRUCTION",
    "MINIONEREC_ITEM_ALIGNMENT_INSTRUCTION",
    "MINIONEREC_SEQREC_INSTRUCTION",
    "MiniOneRecPromptDataset",
    "MiniOneRecRLDatasets",
    "MiniOneRecSFTDataset",
    "MiniOneRecSIDCodec",
    "MiniOneRecTokenizerAdapter",
    "build_fusion_seqrec_examples",
    "build_item_alignment_examples",
    "build_minionerec_rl_datasets",
    "build_minionerec_sft_datasets",
    "build_sequence_sft_examples",
    "load_minionerec_sid_codec",
    "remap_minionerec_sid_mapping_to_recbole",
]
