from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import torch

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.llmrank.config import LLMRankConfig
from recbole3.model.sequential import HISTORY_ITEM_IDS


class LLMRankTrainCollator(BaseCollator):
    """Placeholder train collator for inference-only LLM reranking."""

    def __call__(self, feature_records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(feature_records, pd.DataFrame):
            return {"records": feature_records.reset_index(drop=True)}
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
        else:
            records = list(feature_records)
            history_rows = [_record_value(record, HISTORY_ITEM_IDS, default=()) for record in records]
        history_texts = [
            [self.item_text_lookup[int(item_id)] for item_id in (history_item_ids or ())]
            for history_item_ids in history_rows
        ]
        return {"history_texts": history_texts}


class LLMRankModel(BaseRetrievalModel):
    """Official-style prompt reranker for one provided candidate set."""

    def __init__(self, config: LLMRankConfig):
        super().__init__(config)
        self.config = config
        self._item_text_lookup: tuple[str, ...] = ()
        self._item_token_lookup: tuple[frozenset[str], ...] = ()
        self._history_token_cache: dict[tuple[str, ...], tuple[frozenset[str], frozenset[str]]] = {}
        self._prompt_cache: dict[tuple[tuple[str, ...], tuple[int, ...]], str] = {}
        self._response_cache: dict[tuple[str, tuple[str, ...], tuple[int, ...]], str] = {}
        self._api_response_cache: dict[str, str] | None = None
        self._local_model: Any | None = None
        self._local_tokenizer: Any | None = None
        self._local_input_device: torch.device | None = None
        self._local_formatted_prompt_cache: dict[str, str] = {}

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
            raise NotImplementedError("LLMRankModel reranks one provided candidate set and does not implement full-sort prediction.")
        if not isinstance(model_inputs, dict) or "history_texts" not in model_inputs:
            raise TypeError("LLMRankModel expects eval collator outputs with a 'history_texts' field.")

        ranked_batches = self.rank_candidate_batches(
            list(model_inputs["history_texts"]),
            candidate_item_ids.detach().cpu().tolist(),
        )
        pred_item_ids: list[list[int]] = []
        for ranked_candidate_ids in ranked_batches:
            top_ids = ranked_candidate_ids[: max(0, int(k))]
            if len(top_ids) < int(k):
                top_ids.extend([-1] * (int(k) - len(top_ids)))
            pred_item_ids.append(top_ids)
        return torch.tensor(pred_item_ids, dtype=torch.long)

    def rank_candidate_batches(
        self,
        history_text_batches: Sequence[Sequence[str]],
        candidate_batches: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        if self.config.backend == "identity":
            return [
                [int(item_id) for item_id in candidate_ids if int(item_id) >= 0]
                for candidate_ids in candidate_batches
            ]
        if self.config.backend == "heuristic_overlap":
            return self._rank_candidate_batches_with_overlap(history_text_batches, candidate_batches)

        tasks: list[dict[str, Any]] = []
        score_tables: list[dict[int, float]] = []
        original_positions: list[dict[int, int]] = []
        boots = self._effective_boots()

        for batch_index, (history_texts, candidate_ids) in enumerate(
            zip(history_text_batches, candidate_batches, strict=True)
        ):
            filtered_candidate_ids = [int(item_id) for item_id in candidate_ids if int(item_id) > 0]
            score_tables.append({int(item_id): 0.0 for item_id in filtered_candidate_ids})
            original_positions.append({int(item_id): index for index, item_id in enumerate(filtered_candidate_ids)})
            if not filtered_candidate_ids:
                continue

            rounds = boots if boots > 0 else 1
            for round_index in range(rounds):
                ordered_candidate_ids = list(filtered_candidate_ids)
                if boots > 0 and len(ordered_candidate_ids) > 1:
                    randomizer = Random(int(self.config.candidate_seed) + batch_index * 1009 + round_index)
                    randomizer.shuffle(ordered_candidate_ids)
                history_key = tuple(str(text) for text in history_texts)
                tasks.append(
                    {
                        "batch_index": batch_index,
                        "round_index": round_index,
                        "history_texts": history_key,
                        "candidate_item_ids": tuple(ordered_candidate_ids),
                        "prompt": self.build_prompt(history_key, ordered_candidate_ids),
                    }
                )

        responses = self._generate_batch_responses(tasks)
        for task, response in zip(tasks, responses, strict=True):
            parsed_candidate_ids = self.parse_response(response, task["candidate_item_ids"])
            score_by_item_id = score_tables[int(task["batch_index"])]
            candidate_count = len(task["candidate_item_ids"])
            for rank_index, item_id in enumerate(parsed_candidate_ids):
                score_by_item_id[int(item_id)] += float(candidate_count - rank_index)

        ranked_batches: list[list[int]] = []
        for score_by_item_id, position_by_item_id in zip(score_tables, original_positions, strict=True):
            ranked_batches.append(
                sorted(
                    score_by_item_id,
                    key=lambda item_id: (-score_by_item_id[item_id], position_by_item_id[item_id]),
                )
            )
        return ranked_batches

    def _rank_candidate_batches_with_overlap(
        self,
        history_text_batches: Sequence[Sequence[str]],
        candidate_batches: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        ranked_batches: list[list[int]] = []
        for history_texts, candidate_ids in zip(history_text_batches, candidate_batches, strict=True):
            filtered_candidate_ids = [int(item_id) for item_id in candidate_ids if int(item_id) > 0]
            if not filtered_candidate_ids:
                ranked_batches.append([])
                continue
            history_tokens, recent_tokens = self._history_token_pair(history_texts)
            original_positions = {int(item_id): index for index, item_id in enumerate(filtered_candidate_ids)}
            ranked_batches.append(
                sorted(
                    filtered_candidate_ids,
                    key=lambda item_id: (
                        -self._overlap_score(int(item_id), history_tokens, recent_tokens),
                        original_positions[int(item_id)],
                    ),
                )
            )
        return ranked_batches

    def build_prompt(self, history_texts: Sequence[str], candidate_item_ids: Sequence[int]) -> str:
        history_key = tuple(str(text) for text in history_texts)
        candidate_key = tuple(int(item_id) for item_id in candidate_item_ids)
        prompt_key = (history_key, candidate_key)
        cached_prompt = self._prompt_cache.get(prompt_key)
        if cached_prompt is not None:
            return cached_prompt
        prompt = self._build_prompt_uncached(history_key, candidate_key)
        self._prompt_cache[prompt_key] = prompt
        return prompt

    def _build_prompt_uncached(self, history_texts: Sequence[str], candidate_item_ids: Sequence[int]) -> str:
        terms = _domain_terms(self.config.domain)
        max_history = max(0, int(self.config.history_max_length))
        truncated_history = tuple(history_texts[-max_history:] if max_history > 0 else ())
        history_block = _format_numbered_list(truncated_history)
        candidate_block = _format_numbered_list([self._item_text(int(item_id)) for item_id in candidate_item_ids])
        recall_budget = len(candidate_item_ids)
        prompt_strategy = str(self.config.prompt_strategy).strip().lower()
        if self.config.parsing_strategy == "index":
            ranking_instruction = (
                f"Please rank these {recall_budget} candidate {terms['plural']} by measuring the possibilities that I would "
                f"like to {terms['target_action']} next most, according to {terms['history_phrase']}\n"
            )
            output_instruction = "Please only output the order numbers after ranking. Split these order numbers with line break."
        else:
            ranking_instruction = (
                f"Please rank these {recall_budget} candidate {terms['plural']} by measuring the possibilities that I would "
                f"like to {terms['target_action']} next most, according to {terms['history_phrase']} Please think step by step.\n"
            )
            output_instruction = (
                "Please show me your ranking results with order numbers. Split your output with line break. "
                f"You MUST rank the given candidate {terms['constraint_plural']}. "
                f"You can not generate {terms['constraint_plural']} that are not in the given candidate list."
            )
        if prompt_strategy == "recency_focused":
            recent_item = _last_numbered_item_text(truncated_history)
            recent_sentence = ""
            if recent_item:
                recent_sentence = f"Note that my most recently {terms['past_tense']} {terms['singular']} is {recent_item}. "
            return (
                f"I've {terms['past_tense']} the following {terms['plural']} in the past in order:\n"
                f"{history_block}\n\n"
                f"Now there are {recall_budget} candidate {terms['plural']} that I can {terms['next_action']} next:\n"
                f"{candidate_block}\n"
                f"{ranking_instruction}"
                f"{recent_sentence}"
                f"{output_instruction}"
            )
        if prompt_strategy == "in_context_learning":
            history_prefix = _format_numbered_list(truncated_history[:-1])
            recent_item = _last_numbered_item_text(truncated_history)
            return (
                f"I've {terms['past_tense']} the following {terms['plural']} in the past in order:\n"
                f"{history_prefix}\n\n"
                f"Then if I ask you to recommend a new {terms['singular']} to me according to {terms['history_reference']}, "
                f"you should recommend {recent_item} and now that I've just {terms['past_tense']} {recent_item}, "
                f"there are {recall_budget} candidate {terms['plural']} that I can {terms['next_action']} next:\n"
                f"{candidate_block}\n"
                f"{ranking_instruction}"
                f"{output_instruction}"
            )
        return (
            f"I've {terms['past_tense']} the following {terms['plural']} in the past in order:\n"
            f"{history_block}\n\n"
            f"Now there are {recall_budget} candidate {terms['plural']} that I can {terms['next_action']} next:\n"
            f"{candidate_block}\n"
            f"{ranking_instruction}"
            f"{output_instruction}"
        )
    def parse_response(self, response: str, candidate_item_ids: Sequence[int]) -> list[int]:
        ordered_candidate_ids = [int(item_id) for item_id in candidate_item_ids]
        if not ordered_candidate_ids:
            return []
        if self.config.parsing_strategy == "index":
            parsed_item_ids = self._parse_indices(response, ordered_candidate_ids)
        else:
            parsed_item_ids = self._parse_titles(response, ordered_candidate_ids)
        seen_item_ids = set(parsed_item_ids)
        parsed_item_ids.extend(item_id for item_id in ordered_candidate_ids if item_id not in seen_item_ids)
        return parsed_item_ids

    def _parse_indices(self, response: str, candidate_item_ids: Sequence[int]) -> list[int]:
        parsed_item_ids: list[int] = []
        seen_item_ids: set[int] = set()
        recall_budget = len(candidate_item_ids)
        for item_detail in response.splitlines():
            item_detail = item_detail.strip()
            if not item_detail or not item_detail.isdigit():
                continue
            candidate_index = int(item_detail)
            if candidate_index >= recall_budget:
                continue
            item_id = int(candidate_item_ids[candidate_index])
            if item_id in seen_item_ids:
                continue
            parsed_item_ids.append(item_id)
            seen_item_ids.add(item_id)
            if len(parsed_item_ids) >= recall_budget:
                break
        return parsed_item_ids

    def _parse_titles(self, response: str, candidate_item_ids: Sequence[int]) -> list[int]:
        response_list = response.split("\n")
        candidate_text = [self._item_text(int(item_id)) for item_id in candidate_item_ids]
        parsed_item_ids: list[int] = []
        seen_item_ids: set[int] = set()
        for item_detail in response_list:
            if len(item_detail) < 1:
                continue
            if item_detail.endswith("candidate movies:") or item_detail.endswith("candidate products:"):
                continue
            item_detail = item_detail.strip()
            split_pos = item_detail.find(". ")
            if split_pos > 0 and item_detail[:split_pos].isdigit():
                item_name = item_detail[split_pos + 2 :]
            else:
                item_name = item_detail
            for candidate_index, candidate_text_single in enumerate(candidate_text):
                clean_candidate = html.unescape(candidate_text_single.strip())
                if _matches_candidate_title(item_name, clean_candidate):
                    item_id = int(candidate_item_ids[candidate_index])
                    if item_id in seen_item_ids:
                        break
                    parsed_item_ids.append(item_id)
                    seen_item_ids.add(item_id)
                    break
        return parsed_item_ids

    def _generate_batch_responses(self, tasks: Sequence[dict[str, Any]]) -> list[str]:
        if not tasks:
            return []
        if self.config.backend == "heuristic_overlap":
            return []
        prompts = [str(task["prompt"]) for task in tasks]
        round_indices = [int(task["round_index"]) for task in tasks]
        if self.config.backend == "local_hf":
            return self._request_local_hf_responses(prompts)
        return self._request_openai_responses(prompts, round_indices)

    def _generate_response(self, prompt: str, round_index: int) -> str:
        if self.config.backend == "openai":
            return self._request_openai_response(prompt, round_index=round_index)
        if self.config.backend == "local_hf":
            return self._request_local_hf_response(prompt)
        if self.config.backend in {"heuristic_overlap", "identity"}:
            raise RuntimeError(f"{self.config.backend} ranks candidates directly and does not generate textual responses.")
        raise ValueError(f"Unsupported llmrank backend '{self.config.backend}'.")

    def _request_openai_response(self, prompt: str, *, round_index: int) -> str:
        cache_key = self._api_cache_key(prompt, round_index=round_index)
        cached_response = self._lookup_api_response_cache(cache_key)
        if cached_response is not None:
            return cached_response
        response = self._request_openai_response_uncached(prompt, round_index=round_index)
        self._store_api_response_cache(cache_key, response)
        return response

    def _request_openai_response_uncached(self, prompt: str, *, round_index: int) -> str:
        api_key = os.environ.get(self.config.api_key_env)

        messages: list[dict[str, str]] = []
        system_prompt = self._effective_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.api_model_name,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_output_tokens),
            "user": f"llmrank-round-{round_index}",
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.config.api_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        backoff = max(0.1, float(self.config.retry_backoff_sec))
        last_error: Exception | None = None
        for attempt in range(1, max(1, int(self.config.request_retries)) + 1):
            try:
                with urllib.request.urlopen(request, timeout=float(self.config.request_timeout_sec)) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["choices"][0]["message"]["content"])
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError) as exc:
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

        results = [""] * len(prompts)
        pending_requests: dict[tuple[str, int], list[int]] = {}
        for index, (prompt, round_index) in enumerate(zip(prompts, round_indices, strict=True)):
            cache_key = self._api_cache_key(prompt, round_index=round_index)
            cached_response = self._lookup_api_response_cache(cache_key)
            if cached_response is not None:
                results[index] = cached_response
                continue
            pending_requests.setdefault((prompt, int(round_index)), []).append(index)
        if not pending_requests:
            return results

        request_items = list(pending_requests.items())
        if not bool(getattr(self.config, "async_dispatch", True)):
            for (prompt, round_index), indexes in request_items:
                response = self._request_openai_response(prompt, round_index=round_index)
                for index in indexes:
                    results[index] = response
            return results

        api_batch = max(1, int(getattr(self.config, "api_batch", 8)))
        for start in range(0, len(request_items), api_batch):
            batch = request_items[start : start + api_batch]
            with ThreadPoolExecutor(max_workers=min(len(batch), api_batch)) as executor:
                future_pairs = {
                    executor.submit(self._request_openai_response, prompt, round_index=round_index): indexes
                    for (prompt, round_index), indexes in batch
                }
                for future, indexes in future_pairs.items():
                    response = future.result()
                    for index in indexes:
                        results[index] = response
        return results

    def _request_local_hf_response(self, prompt: str) -> str:
        return self._request_local_hf_responses([prompt])[0]

    def _request_local_hf_responses(self, prompts: Sequence[str]) -> list[str]:
        if not prompts:
            return []
        model, tokenizer = self._ensure_local_generator_loaded()
        results = [""] * len(prompts)
        pending_prompts: dict[str, list[int]] = {}
        for index, prompt in enumerate(prompts):
            cache_key = self._local_response_cache_key(prompt)
            cached_response = self._lookup_local_response_cache(cache_key)
            if cached_response is not None:
                results[index] = cached_response
                continue
            pending_prompts.setdefault(prompt, []).append(index)
        if not pending_prompts:
            return results

        unique_prompts = list(pending_prompts)
        unique_prompts.sort(key=lambda prompt: len(self._format_local_chat_prompt(prompt, tokenizer)), reverse=True)
        batch_size = max(1, int(self.config.local_batch_size))
        for start in range(0, len(unique_prompts), batch_size):
            prompt_batch = unique_prompts[start : start + batch_size]
            prompt_text_batch = [self._format_local_chat_prompt(prompt, tokenizer) for prompt in prompt_batch]
            tokenizer_kwargs: dict[str, Any] = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
            }
            if int(self.config.local_max_input_tokens) > 0:
                tokenizer_kwargs["max_length"] = int(self.config.local_max_input_tokens)
            tokenized = tokenizer(prompt_text_batch, **tokenizer_kwargs)
            input_device = self._local_input_device or torch.device("cpu")
            tokenized = {key: value.to(input_device) for key, value in tokenized.items()}
            prompt_length = int(tokenized["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **tokenized,
                    max_new_tokens=int(self.config.local_max_output_tokens),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self._local_pad_token_id(tokenizer),
                    eos_token_id=self._local_eos_token_ids(tokenizer),
                )
            for batch_index, prompt in enumerate(prompt_batch):
                generated_ids = generated[batch_index, prompt_length:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                cache_key = self._local_response_cache_key(prompt)
                self._store_local_response_cache(cache_key, response)
                for result_index in pending_prompts[prompt]:
                    results[result_index] = response
        return results

    def _ensure_local_generator_loaded(self) -> tuple[Any, Any]:
        if self._local_model is not None and self._local_tokenizer is not None:
            return self._local_model, self._local_tokenizer
        model_path = str(self.config.local_model_path or "").strip()
        if not model_path:
            raise RuntimeError("LLMRank local_hf backend requires model.local_model_path to point to one local HF model.")
        tokenizer_path = str(self.config.local_tokenizer_path or model_path).strip()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LLMRank local_hf backend requires transformers. Install the optional Hugging Face dependencies first."
            ) from exc

        dtype = self._resolve_local_dtype()
        model_kwargs: dict[str, Any] = {"trust_remote_code": bool(self.config.local_trust_remote_code)}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.config.local_attn_implementation:
            model_kwargs["attn_implementation"] = self.config.local_attn_implementation
        if self.config.local_device_map:
            model_kwargs["device_map"] = self.config.local_device_map
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(self.config.local_trust_remote_code),
        )
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not self.config.local_device_map:
            target_device = torch.device(self.config.local_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
            model.to(target_device)
            self._local_input_device = target_device
        else:
            self._local_input_device = self._infer_local_input_device(model)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._local_model = model
        self._local_tokenizer = tokenizer
        return model, tokenizer

    def _resolve_local_dtype(self) -> torch.dtype | None:
        dtype_name = str(self.config.local_dtype).strip().lower()
        if dtype_name == "auto":
            return None
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported local_dtype '{self.config.local_dtype}'.")

    def _infer_local_input_device(self, model: Any) -> torch.device:
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for location in hf_device_map.values():
                if isinstance(location, str) and location not in {"cpu", "disk"}:
                    return torch.device(location)
                if isinstance(location, int):
                    return torch.device(f"cuda:{location}")
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _format_local_chat_prompt(self, prompt: str, tokenizer: Any) -> str:
        cached_prompt = self._local_formatted_prompt_cache.get(prompt)
        if cached_prompt is not None:
            return cached_prompt
        if not bool(self.config.local_use_chat_template) or not hasattr(tokenizer, "apply_chat_template"):
            self._local_formatted_prompt_cache[prompt] = prompt
            return prompt
        messages: list[dict[str, str]] = []
        system_prompt = self._effective_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        self._local_formatted_prompt_cache[prompt] = formatted_prompt
        return formatted_prompt

    @staticmethod
    def _local_pad_token_id(tokenizer: Any) -> int:
        if tokenizer.pad_token_id is not None:
            return int(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        raise ValueError("Local HF backend requires tokenizer.pad_token_id or tokenizer.eos_token_id.")

    @staticmethod
    def _local_eos_token_ids(tokenizer: Any) -> int | list[int] | None:
        eos_ids: list[int] = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end_id = None
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        if not eos_ids:
            return None
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    def _local_response_cache_key(self, prompt: str) -> str:
        payload = {
            "backend": "local_hf",
            "local_model_path": self.config.local_model_path,
            "local_tokenizer_path": self.config.local_tokenizer_path,
            "system_prompt": self._effective_system_prompt(),
            "local_max_output_tokens": int(self.config.local_max_output_tokens),
            "local_max_input_tokens": int(self.config.local_max_input_tokens),
            "prompt": prompt,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _lookup_local_response_cache(self, cache_key: str) -> str | None:
        return self._response_cache.get(("local_hf", (cache_key,), ()))

    def _store_local_response_cache(self, cache_key: str, response: str) -> None:
        self._response_cache[("local_hf", (cache_key,), ())] = response

    def _ensure_item_text_lookup(self, prepared_data) -> None:
        if self._item_text_lookup:
            return
        item_table = prepared_data.get_item_table()
        if "item_id" not in item_table.columns:
            raise ValueError("LLMRankModel requires item_table to include an 'item_id' column.")

        num_items = int(prepared_data.get_num_items())
        item_text_lookup = [f"item {item_id}" for item_id in range(num_items)]
        for record in item_table.to_dict(orient="records"):
            item_id = int(record["item_id"])
            if not 0 <= item_id < num_items:
                raise ValueError(f"item_table contains out-of-range item_id={item_id} for num_items={num_items}.")
            item_text_lookup[item_id] = self._resolve_item_text(record, item_id=item_id)
        self._item_text_lookup = tuple(item_text_lookup)
        self._item_token_lookup = tuple(frozenset(_tokenize_text(text)) for text in self._item_text_lookup)
        self._history_token_cache.clear()
        self._prompt_cache.clear()
        self._response_cache.clear()
        self._local_formatted_prompt_cache.clear()

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

    def _item_tokens(self, item_id: int) -> frozenset[str]:
        if not self._item_token_lookup:
            raise RuntimeError("LLMRankModel must build collators before heuristic overlap scoring.")
        return self._item_token_lookup[int(item_id)]

    def _history_token_pair(self, history_texts: Sequence[str]) -> tuple[frozenset[str], frozenset[str]]:
        history_key = tuple(str(text) for text in history_texts)
        cached_pair = self._history_token_cache.get(history_key)
        if cached_pair is not None:
            return cached_pair
        history_tokens = frozenset(_tokenize_text(" ".join(history_key)))
        recent_tokens = frozenset(_tokenize_text(history_key[-1])) if history_key else frozenset()
        cached_pair = (history_tokens, recent_tokens)
        self._history_token_cache[history_key] = cached_pair
        return cached_pair

    def _overlap_score(
        self,
        item_id: int,
        history_tokens: frozenset[str],
        recent_tokens: frozenset[str],
    ) -> float:
        candidate_tokens = self._item_tokens(item_id)
        return float(len(history_tokens & candidate_tokens)) + 0.5 * float(len(recent_tokens & candidate_tokens))

    def _api_cache_key(self, prompt: str, *, round_index: int) -> str:
        payload = {
            "api_base_url": self.config.api_base_url,
            "api_model_name": self.config.api_model_name,
            "system_prompt": self._effective_system_prompt(),
            "temperature": float(self.config.temperature),
            "max_output_tokens": int(self.config.max_output_tokens),
            "round_index": int(round_index),
            "prompt": prompt,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _effective_system_prompt(self) -> str:
        return str(self.config.system_prompt or "").strip()

    def _api_cache_path(self) -> Path:
        return Path(self.config.api_response_cache_path)

    def _ensure_api_response_cache_loaded(self) -> dict[str, str]:
        if self._api_response_cache is not None:
            return self._api_response_cache
        cache: dict[str, str] = {}
        cache_path = self._api_cache_path()
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = str(record.get("key", "")).strip()
                    response = record.get("response")
                    if key and isinstance(response, str):
                        cache[key] = response
        self._api_response_cache = cache
        return cache

    def _lookup_api_response_cache(self, cache_key: str) -> str | None:
        if self.config.refresh_api_response_cache:
            return None
        return self._ensure_api_response_cache_loaded().get(cache_key)

    def _store_api_response_cache(self, cache_key: str, response: str) -> None:
        cache = self._ensure_api_response_cache_loaded()
        cache[cache_key] = response
        cache_path = self._api_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": cache_key, "response": response}, ensure_ascii=False) + "\n")

    def _effective_boots(self) -> int:
        return max(0, int(self.config.boots))


def _domain_terms(domain: str) -> dict[str, str]:
    if domain == "movie":
        return {
            "singular": "movie",
            "plural": "movies",
            "past_tense": "watched",
            "next_action": "watch",
            "target_action": "watch",
            "history_phrase": "my watching history.",
            "history_reference": "my watching history",
            "constraint_plural": "movies",
        }
    if domain == "product":
        return {
            "singular": "product",
            "plural": "products",
            "past_tense": "purchased",
            "next_action": "consider to purchase",
            "target_action": "purchase",
            "history_phrase": "the given purchasing records.",
            "history_reference": "the given purchasing history",
            "constraint_plural": "products",
        }
    return {
        "singular": "item",
        "plural": "items",
        "past_tense": "interacted with",
        "next_action": "interact with",
        "target_action": "interact with",
        "history_phrase": "my interaction history.",
        "history_reference": "my interaction history",
        "constraint_plural": "items",
    }


def _format_numbered_list(items: Sequence[str]) -> str:
    if not items:
        return "[]"
    return repr([f"{index}. {text}" for index, text in enumerate(items)])


def _last_numbered_item_text(items: Sequence[str]) -> str:
    if not items:
        return ""
    last_item = str(items[-1])
    split_pos = last_item.find(". ")
    if split_pos > 0 and last_item[:split_pos].isdigit():
        return last_item[split_pos + 2 :]
    return last_item


def _tokenize_text(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    if not normalized:
        return set()
    return set(normalized.split())


def _matches_candidate_title(item_name: str, candidate_text: str) -> bool:
    normalized_item_name = item_name.strip()
    normalized_candidate = candidate_text.strip()
    if not normalized_item_name or not normalized_candidate:
        return False
    if normalized_candidate in normalized_item_name or normalized_item_name in normalized_candidate:
        return True
    return _lcs_sequence_length(normalized_item_name, normalized_candidate) > 0.9 * len(normalized_candidate)


def _lcs_sequence_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _record_value(record: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(record, pd.Series):
        return record[key] if key in record else default
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


__all__ = [
    "LLMRankEvalCollator",
    "LLMRankModel",
    "LLMRankTrainCollator",
]
