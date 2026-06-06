from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch


def build_minionerec_prefix_allowed_tokens(
    tokenizer: Any,
    semantic_ids: list[str] | tuple[str, ...],
    *,
    base_model: str,
    prefix_token_count: int | None = None,
) -> tuple[Callable[[int, list[int]], list[int]], int]:
    """Build the original MiniOneRec hash-table prefix constraint."""

    prefix_index = int(prefix_token_count) if prefix_token_count is not None else _default_prefix_token_count(base_model)
    hash_dict: dict[str, set[int]] = {}
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for semantic_id in semantic_ids:
        text = f"### Response:\n{semantic_id}\n"
        token_ids = list(tokenizer(text).input_ids)
        if "llama" in str(base_model).lower() and token_ids:
            token_ids = token_ids[1:]
        if eos_token_id is not None:
            token_ids.append(int(eos_token_id))
        for index in range(prefix_index, len(token_ids)):
            if index == prefix_index:
                hash_key = _hash_tokens(token_ids[:index])
            else:
                hash_key = _hash_tokens(token_ids[prefix_index:index])
            hash_dict.setdefault(hash_key, set()).add(int(token_ids[index]))

    allowed_by_hash = {key: sorted(values) for key, values in hash_dict.items()}

    def prefix_allowed_tokens_fn(_batch_id: int, input_ids: list[int]) -> list[int]:
        return allowed_by_hash.get(_hash_tokens(input_ids), [])

    return prefix_allowed_tokens_fn, prefix_index


class MiniOneRecConstrainedLogitsProcessor:
    """MiniOneRec constrained decoding processor adapted from the original implementation."""

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, list[int]], list[int]],
        *,
        num_beams: int,
        prefix_token_count: int,
        eos_token_id: int | None,
    ) -> None:
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = int(num_beams)
        self._prefix_token_count = int(prefix_token_count)
        self._eos_token_id = eos_token_id
        self._count = 0
        self.total_prefix_checks = 0
        self.valid_prefix_checks = 0
        self.invalid_prefix_checks = 0
        self.forced_eos_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        beams = input_ids.view(-1, self._num_beams, input_ids.shape[-1])
        for batch_id, beam_sentences in enumerate(beams):
            for beam_id, sentence in enumerate(beam_sentences):
                if self._count == 0:
                    hash_key = sentence[-self._prefix_token_count :].tolist()
                else:
                    hash_key = sentence[-self._count :].tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                row_index = batch_id * self._num_beams + beam_id
                self.total_prefix_checks += 1
                if not prefix_allowed_tokens:
                    self.invalid_prefix_checks += 1
                    warnings.warn(
                        f"No valid MiniOneRec tokens found for prefix {hash_key} at step {self._count}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    if self._eos_token_id is not None:
                        self.forced_eos_count += 1
                        mask[row_index, int(self._eos_token_id)] = 0
                    continue
                self.valid_prefix_checks += 1
                mask[row_index, prefix_allowed_tokens] = 0
        self._count += 1
        return scores + mask

    def stats(self) -> dict[str, float | int]:
        total = int(self.total_prefix_checks)
        invalid = int(self.invalid_prefix_checks)
        valid = int(self.valid_prefix_checks)
        return {
            "constraint_total_prefix_checks": total,
            "constraint_valid_prefix_checks": valid,
            "constraint_invalid_prefix_checks": invalid,
            "constraint_forced_eos_count": int(self.forced_eos_count),
            "constraint_success_rate": (valid / total) if total else 0.0,
            "constraint_invalid_prefix_rate": (invalid / total) if total else 0.0,
        }


def _default_prefix_token_count(base_model: str) -> int:
    return 4 if "gpt2" in str(base_model).lower() else 3


def _hash_tokens(tokens: list[int] | tuple[int, ...]) -> str:
    return "-".join(str(int(token)) for token in tokens)


__all__ = [
    "MiniOneRecConstrainedLogitsProcessor",
    "build_minionerec_prefix_allowed_tokens",
]
