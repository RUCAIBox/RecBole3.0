from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from random import Random
from typing import Any

import pandas as pd
import torch

from recbole3.dataset import CANDIDATE_ITEM_IDS
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.llmrank.config import LLMRankConfig
from recbole3.model.sequential import HISTORY_ITEM_IDS


class LLMRankTrainCollator(BaseCollator):
    """Placeholder train collator for inference-only LLM reranking."""

    def __call__(self, feature_records: Sequence[Any]) -> dict[str, Any]:
        return {"records": list(feature_records)}


class LLMRankEvalCollator(BaseCollator):
    """Collect prompt-ready history texts for one evaluation batch."""

    def __init__(self, config: LLMRankConfig, prepared_data, item_text_lookup: Sequence[str]):
        super().__init__(config, prepared_data=prepared_data)
        self.item_text_lookup = tuple(item_text_lookup)

    def __call__(self, feature_records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(feature_records, pd.DataFrame):
            records = feature_records.reset_index(drop=True)
            history_rows = records[HISTORY_ITEM_IDS].tolist() if HISTORY_ITEM_IDS in records.columns else [()] * len(records)
            candidate_rows = (
                records[CANDIDATE_ITEM_IDS].tolist()
                if CANDIDATE_ITEM_IDS in records.columns
                else [()] * len(records)
            )
        else:
            records = list(feature_records)
            history_rows = [_record_value(record, HISTORY_ITEM_IDS, default=()) for record in records]
            candidate_rows = [_record_value(record, CANDIDATE_ITEM_IDS, default=()) for record in records]
        history_texts = [
            [self.item_text_lookup[int(item_id)] for item_id in (history_item_ids or ())]
            for history_item_ids in history_rows
        ]
        candidate_item_ids = [list(candidate_item_ids or ()) for candidate_item_ids in candidate_rows]
        return {
            "history_texts": history_texts,
            "candidate_item_ids": candidate_item_ids,
        }


class LLMRankModel(BaseRetrievalModel):
    """Prompt-based candidate reranker inspired by RUCAIBox/LLMRank."""

    def __init__(self, config: LLMRankConfig):
        super().__init__(config)
        self.config = config
        self._item_text_lookup: tuple[str, ...] = ()
        self._mock_response_cursor = 0

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_item_text_lookup(prepared_data)
        return LLMRankTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_item_text_lookup(prepared_data)
        return LLMRankEvalCollator(self.config, prepared_data=prepared_data, item_text_lookup=self._item_text_lookup)

    def forward(self, batch: Any) -> dict[str, Any]:
        return {}

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        raise RuntimeError("LLMRankModel is inference-only and does not define a training loss.")

    def predict(
        self,
        model_inputs: Any,
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del exclude_item_ids
        del exclude_mask
        if candidate_item_ids is None:
            raise NotImplementedError("LLMRankModel currently supports sampled candidate reranking only.")
        if not isinstance(model_inputs, dict) or "history_texts" not in model_inputs:
            raise TypeError("LLMRankModel expects eval collator outputs with a 'history_texts' field.")

        history_text_batches = list(model_inputs["history_texts"])
        candidate_matrix = candidate_item_ids.detach().cpu().tolist()
        ranked_batches = self.rank_candidate_batches(history_text_batches, candidate_matrix)
        pred_item_ids: list[list[int]] = []
        for ranked_candidate_ids in ranked_batches:
            top_ids = ranked_candidate_ids[: max(0, int(k))]
            if len(top_ids) < int(k):
                top_ids.extend([-1] * (int(k) - len(top_ids)))
            pred_item_ids.append(top_ids)

        return torch.tensor(pred_item_ids, dtype=torch.long)

    def rank_candidates(self, history_texts: Sequence[str], candidate_item_ids: Sequence[int]) -> list[int]:
        ranked_batches = self.rank_candidate_batches([history_texts], [list(candidate_item_ids)])
        return ranked_batches[0] if ranked_batches else []

    def rank_candidate_batches(
        self,
        history_text_batches: Sequence[Sequence[str]],
        candidate_batches: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        tasks: list[dict[str, Any]] = []
        score_tables: list[dict[int, float]] = []
        original_positions: list[dict[int, int]] = []
        for batch_index, (history_texts, candidate_ids) in enumerate(
            zip(history_text_batches, candidate_batches, strict=True)
        ):
            filtered_candidate_ids = [int(item_id) for item_id in candidate_ids if int(item_id) >= 0]
            score_tables.append({int(item_id): 0.0 for item_id in filtered_candidate_ids})
            original_positions.append({int(item_id): index for index, item_id in enumerate(filtered_candidate_ids)})
            if not filtered_candidate_ids:
                continue
            rounds = max(1, int(self.config.bootstrap_rounds))
            randomizer = Random(self.config.random_seed + batch_index + len(history_texts) + len(filtered_candidate_ids))
            for round_index in range(rounds):
                ordered_candidate_ids = list(filtered_candidate_ids)
                should_shuffle = bool(self.config.candidate_shuffle) or rounds > 1
                if should_shuffle and len(ordered_candidate_ids) > 1:
                    randomizer.shuffle(ordered_candidate_ids)
                tasks.append(
                    {
                        "batch_index": batch_index,
                        "round_index": round_index,
                        "history_texts": list(history_texts),
                        "candidate_item_ids": ordered_candidate_ids,
                        "prompt": self.build_prompt(history_texts, ordered_candidate_ids),
                    }
                )

        responses = self._generate_batch_responses(tasks)
        for task, response in zip(tasks, responses, strict=True):
            parsed_candidate_ids = self.parse_response(response, task["candidate_item_ids"])
            score_by_item_id = score_tables[int(task["batch_index"])]
            for rank_index, item_id in enumerate(parsed_candidate_ids):
                score_by_item_id[int(item_id)] += float(len(parsed_candidate_ids) - rank_index)

        ranked_batches: list[list[int]] = []
        for score_by_item_id, position_by_item_id in zip(score_tables, original_positions, strict=True):
            ranked_batches.append(
                sorted(
                    score_by_item_id,
                    key=lambda item_id: (-score_by_item_id[item_id], position_by_item_id[item_id]),
                )
            )
        return ranked_batches

    def build_prompt(self, history_texts: Sequence[str], candidate_item_ids: Sequence[int]) -> str:
        domain_terms = _domain_terms(self.config.domain)
        history_block = _format_prompt_list(history_texts)
        candidate_texts = [self._item_text(int(item_id)) for item_id in candidate_item_ids]
        candidate_block = _format_prompt_list(candidate_texts)
        recent_text = history_texts[-1] if history_texts else None
        count = len(candidate_item_ids)
        candidate_constraint = ""
        if self.config.enforce_candidate_constraint:
            candidate_constraint = (
                f" You must rank the given candidate {domain_terms['plural']} only. "
                f"Do not output any {domain_terms['plural']} that are not in the candidate list."
            )
        reasoning_instruction = " Please think step by step." if self.config.include_reasoning_instruction else ""
        output_instruction = (
            f" Please show me your ranking results with order numbers. We now only need the names of the "
            f"{domain_terms['plural']}."
            if self.config.require_order_numbers
            else f" Please output the ranked {domain_terms['plural']} only."
        )

        if self.config.prompt_strategy == "sequential":
            return (
                f"I've {domain_terms['past_tense']} the following {domain_terms['plural']} in the past in order:\n"
                f"{history_block}\n"
                f"Now there are {count} candidate {domain_terms['plural']} that I can {domain_terms['verb']} next: "
                f"{candidate_block}\n"
                f"Please rank these {count} candidate {domain_terms['plural']} by measuring the possibilities that I would "
                f"like to {domain_terms['verb']} next most, according to my {domain_terms['history_noun']} history."
                f"{reasoning_instruction}{output_instruction}{candidate_constraint}"
            )

        if self.config.prompt_strategy == "in_context_learning" and len(history_texts) >= 2:
            prefix_examples = []
            for history_end in range(1, len(history_texts)):
                prefix = _format_prompt_list(history_texts[:history_end])
                successor = history_texts[history_end]
                prefix_examples.append(
                    f"If I've {domain_terms['past_tense']} the following {domain_terms['plural']} in the past in order:\n"
                    f"{prefix}\nthen you should recommend {successor} to me."
                )
            example_block = "\n".join(prefix_examples)
            current_history = _format_prompt_list(history_texts)
            return (
                f"{example_block}\n"
                f"Now I've {domain_terms['past_tense']} the following {domain_terms['plural']} in the past in order:\n"
                f"{current_history}\n"
                f"Now there are {count} candidate {domain_terms['plural']} that I can {domain_terms['verb']} next:\n"
                f"{candidate_block}\n"
                f"Please rank these {count} candidate {domain_terms['plural']} by measuring the possibilities that I would "
                f"like to {domain_terms['verb']} next most, according to my {domain_terms['history_noun']} history."
                f"{reasoning_instruction}{output_instruction}{candidate_constraint}"
            )

        recent_sentence = ""
        if recent_text:
            recent_sentence = (
                f" Note that my most recently {domain_terms['past_tense']} {domain_terms['singular']} is {recent_text}."
            )
        return (
            f"I've {domain_terms['past_tense']} the following {domain_terms['plural']} in the past in order:\n"
            f"{history_block}{recent_sentence}\n"
            f"Now there are {count} candidate {domain_terms['plural']} that I can {domain_terms['verb']} next:\n"
            f"{candidate_block}\n"
            f"Please rank these {count} candidate {domain_terms['plural']} by measuring the possibilities that I would "
            f"like to {domain_terms['verb']} next most, according to my {domain_terms['history_noun']} history."
            f"{reasoning_instruction}{output_instruction}{candidate_constraint}"
        )

    def parse_response(self, response: str, candidate_item_ids: Sequence[int]) -> list[int]:
        ordered_candidate_ids = list(int(item_id) for item_id in candidate_item_ids)
        if not ordered_candidate_ids:
            return []
        if self.config.parsing_strategy == "index":
            parsed = self._parse_indices(response, ordered_candidate_ids)
        else:
            parsed = self._parse_titles(response, ordered_candidate_ids)
        seen_item_ids = set(parsed)
        parsed.extend(item_id for item_id in ordered_candidate_ids if item_id not in seen_item_ids)
        return parsed

    def _parse_indices(self, response: str, candidate_item_ids: Sequence[int]) -> list[int]:
        parsed_item_ids: list[int] = []
        seen_item_ids: set[int] = set()
        for line in response.splitlines():
            match = re.search(r"^\s*(\d+)\s*[\).:\-]", line.strip())
            if match is None:
                continue
            candidate_index = int(match.group(1))
            if 0 <= candidate_index < len(candidate_item_ids):
                item_id = int(candidate_item_ids[candidate_index])
                if item_id not in seen_item_ids:
                    parsed_item_ids.append(item_id)
                    seen_item_ids.add(item_id)
        return parsed_item_ids

    def _parse_titles(self, response: str, candidate_item_ids: Sequence[int]) -> list[int]:
        parsed_item_ids: list[int] = []
        seen_item_ids: set[int] = set()
        for raw_line in _candidate_response_lines(response):
            line = _normalize_text(_strip_leading_numbering(raw_line))
            if not line:
                continue
            for item_id in candidate_item_ids:
                if _line_matches_candidate(line, self._candidate_aliases(int(item_id))):
                    normalized_item_id = int(item_id)
                    if normalized_item_id not in seen_item_ids:
                        parsed_item_ids.append(normalized_item_id)
                        seen_item_ids.add(normalized_item_id)
                    break
        if parsed_item_ids:
            return parsed_item_ids
        response_text = _normalize_text(response)
        ranked_by_position = []
        for item_id in candidate_item_ids:
            first_position = _first_candidate_position(response_text, self._candidate_aliases(int(item_id)))
            if first_position is None:
                continue
            ranked_by_position.append((first_position, int(item_id)))
        ranked_by_position.sort()
        return [item_id for _, item_id in ranked_by_position]

    def _generate_batch_responses(self, tasks: Sequence[dict[str, Any]]) -> list[str]:
        if not tasks:
            return []
        if self.config.backend != "openai":
            return [
                self._generate_response(
                    str(task["prompt"]),
                    task["history_texts"],
                    task["candidate_item_ids"],
                    round_index=int(task["round_index"]),
                )
                for task in tasks
            ]
        return self._request_openai_responses(
            [str(task["prompt"]) for task in tasks],
            [int(task["round_index"]) for task in tasks],
        )

    def _generate_response(
        self,
        prompt: str,
        history_texts: Sequence[str],
        candidate_item_ids: Sequence[int],
        *,
        round_index: int,
    ) -> str:
        if self.config.backend == "mock":
            if self._mock_response_cursor >= len(self.config.mock_responses):
                raise RuntimeError("mock backend requires enough configured mock_responses for every ranking call.")
            response = self.config.mock_responses[self._mock_response_cursor]
            self._mock_response_cursor += 1
            return response
        if self.config.backend == "heuristic_overlap":
            return self._build_overlap_response(history_texts, candidate_item_ids)
        if self.config.backend == "openai":
            return self._request_openai_response(prompt, round_index=round_index)
        raise ValueError(f"Unsupported llmrank backend '{self.config.backend}'.")

    def _build_overlap_response(self, history_texts: Sequence[str], candidate_item_ids: Sequence[int]) -> str:
        history_tokens = _tokenize_text(" ".join(history_texts))
        recent_tokens = _tokenize_text(history_texts[-1]) if history_texts else set()

        def candidate_score(item_id: int) -> tuple[float, int]:
            candidate_tokens = _tokenize_text(self._item_text(item_id))
            score = float(len(history_tokens & candidate_tokens)) + 0.5 * float(len(recent_tokens & candidate_tokens))
            return score, -int(item_id)

        ranked_candidate_ids = sorted(candidate_item_ids, key=candidate_score, reverse=True)
        return "\n".join(
            f"{index}. {self._item_text(int(item_id))}"
            for index, item_id in enumerate(ranked_candidate_ids)
        )

    def _request_openai_response(self, prompt: str, *, round_index: int) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"LLMRank openai backend requires the API key in environment variable '{self.config.api_key_env}'."
            )

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.api_model_name,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_output_tokens),
            "user": f"llmrank-round-{round_index}",
        }
        request = urllib.request.Request(
            self.config.api_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        backoff = max(0.1, float(self.config.retry_backoff_sec))
        last_error: Exception | None = None
        for attempt in range(1, max(1, int(self.config.request_retries)) + 1):
            try:
                with urllib.request.urlopen(request, timeout=float(self.config.request_timeout_sec)) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                try:
                    return str(response_payload["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError("OpenAI-compatible backend returned an unexpected response payload.") from exc
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                if attempt >= max(1, int(self.config.request_retries)):
                    break
                time.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError(f"Failed to call the configured openai backend after retries: {last_error}") from last_error

    def _request_openai_responses(self, prompts: Sequence[str], round_indices: Sequence[int]) -> list[str]:
        if len(prompts) != len(round_indices):
            raise ValueError("prompts and round_indices must have the same length for batched openai requests.")
        if not prompts:
            return []
        max_workers = max(1, min(int(self.config.api_concurrency), len(prompts)))
        if max_workers == 1:
            return [
                self._request_openai_response(prompt, round_index=round_index)
                for prompt, round_index in zip(prompts, round_indices, strict=True)
            ]
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_pairs = {
                executor.submit(self._request_openai_response, prompt, round_index=round_index): index
                for index, (prompt, round_index) in enumerate(zip(prompts, round_indices, strict=True))
            }
            for future, index in future_pairs.items():
                results[index] = future.result()
        return results

    def _ensure_item_text_lookup(self, prepared_data) -> None:
        if self._item_text_lookup:
            return
        item_table = prepared_data.get_item_table()
        if "item_id" not in item_table.columns:
            raise ValueError("LLMRankModel requires item_table to include an 'item_id' column.")

        num_items = int(prepared_data.get_num_items())
        item_text_lookup = [f"item {item_id}" for item_id in range(num_items)]
        records = item_table.to_dict(orient="records")
        for record in records:
            item_id = int(record["item_id"])
            if not 0 <= item_id < num_items:
                raise ValueError(f"item_table contains out-of-range item_id={item_id} for num_items={num_items}.")
            item_text_lookup[item_id] = self._resolve_item_text(record, item_id=item_id)
        self._item_text_lookup = tuple(item_text_lookup)

    def _resolve_item_text(self, item_record: dict[str, Any], *, item_id: int) -> str:
        candidate_fields = [self.config.item_text_field]
        if self.config.fallback_item_text_field:
            candidate_fields.append(self.config.fallback_item_text_field)
        candidate_fields.extend(["raw_item_id", "item_id"])
        for field_name in candidate_fields:
            if not field_name or field_name not in item_record:
                continue
            value = item_record[field_name]
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return f"item {item_id}"

    def _item_text(self, item_id: int) -> str:
        if not self._item_text_lookup:
            raise RuntimeError("LLMRankModel must build collators before prompt construction or prediction.")
        return self._item_text_lookup[int(item_id)]

    def _candidate_aliases(self, item_id: int) -> tuple[str, ...]:
        item_text = self._item_text(int(item_id))
        aliases = {_normalize_text(item_text)}
        if self.config.domain == "movie":
            aliases.update(_movie_aliases(item_text))
        return tuple(alias for alias in aliases if alias)


def _domain_terms(domain: str) -> dict[str, str]:
    if domain == "movie":
        return {
            "singular": "movie",
            "plural": "movies",
            "verb": "watch",
            "past_tense": "watched",
            "history_noun": "watching",
        }
    if domain == "product":
        return {
            "singular": "product",
            "plural": "products",
            "verb": "purchase",
            "past_tense": "purchased",
            "history_noun": "purchase",
        }
    return {
        "singular": "item",
        "plural": "items",
        "verb": "interact with",
        "past_tense": "interacted with",
        "history_noun": "interaction",
    }


def _strip_leading_numbering(text: str) -> str:
    return re.sub(r"^\s*(?:[#*\-]?\s*)?\d+\s*[\).:\-]\s*", "", text.strip())


def _format_prompt_list(items: Sequence[str]) -> str:
    if not items:
        return "[]"
    lines = [f"{index}. {text}" for index, text in enumerate(items)]
    return repr(lines)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _tokenize_text(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def _movie_aliases(title: str) -> set[str]:
    aliases = {_normalize_text(title)}
    normalized_title = title.strip()
    if normalized_title.endswith(", The"):
        aliases.add(_normalize_text(f"The {normalized_title[:-5].strip()}"))
    if normalized_title.endswith(", A"):
        aliases.add(_normalize_text(f"A {normalized_title[:-2].strip()}"))
    if normalized_title.endswith(", An"):
        aliases.add(_normalize_text(f"An {normalized_title[:-3].strip()}"))
    return {alias for alias in aliases if alias}


def _record_value(record: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(record, pd.Series):
        return record[key] if key in record else default
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _line_matches_candidate(line: str, aliases: Sequence[str]) -> bool:
    for alias in aliases:
        if not alias:
            continue
        if line == alias or line in alias or alias in line:
            return True
        if SequenceMatcher(a=line, b=alias).ratio() >= 0.9:
            return True
    return False


def _first_candidate_position(response_text: str, aliases: Sequence[str]) -> int | None:
    positions = [
        response_text.find(alias)
        for alias in aliases
        if alias and response_text.find(alias) != -1
    ]
    if not positions:
        return None
    return min(positions)


def _candidate_response_lines(response: str) -> list[str]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    numbered_chunks = re.split(r"(?:^|\s)(?=\d+\s*[\).:\-])", response.strip())
    normalized_chunks = [chunk.strip() for chunk in numbered_chunks if chunk.strip()]
    if len(normalized_chunks) > 1:
        return normalized_chunks
    return lines


__all__ = [
    "LLMRankEvalCollator",
    "LLMRankModel",
    "LLMRankTrainCollator",
]
