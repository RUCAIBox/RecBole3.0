from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Any, Sequence

import pandas as pd
import torch

from recbole3.dataset import CANDIDATE_ITEM_IDS, ITEM_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.llm4rs.config import LLM4RSConfig
from recbole3.model.openai import OpenAICompatibleClient, dispatch_requests
from recbole3.model.sequential import HISTORY_ITEM_IDS


LLM4RS_RECORD_INDEX = "_llm4rs_record_index"
_LLM4RS_POLICY_SYSTEM_PROMPTS = {
    "point": "Reply with ONLY one integer from 1 to 5. No explanation or other text.",
    "pair": "Reply with ONLY A or B. No explanation or other text.",
    "list": (
        "Reply with ONLY the ranking letters requested in the user message "
        "(for example A B C D E), separated by spaces. No explanation or other text."
    ),
}


@dataclass(frozen=True, slots=True)
class LLM4RSOutcome:
    """One official-policy prediction, retaining ties for faithful evaluation."""

    candidate_item_ids: tuple[int, ...]
    ranked_item_ids: tuple[int, ...] | None = None
    scores: tuple[int, ...] | None = None
    responses: tuple[str | None, ...] = ()
    failed_subrequests: int = 0
    error: str | None = None

    def ordered_item_ids(self) -> tuple[int, ...]:
        if self.ranked_item_ids is not None:
            return self.ranked_item_ids
        if self.scores is None:
            return self.candidate_item_ids
        positions = sorted(range(len(self.candidate_item_ids)), key=lambda index: (-self.scores[index], index))
        return tuple(self.candidate_item_ids[index] for index in positions)

    def target_ranks(self, target_item_id: int) -> tuple[int, ...]:
        if self.error is not None:
            return ()
        try:
            target_position = self.candidate_item_ids.index(int(target_item_id))
        except ValueError:
            return ()
        if self.ranked_item_ids is not None:
            try:
                return (self.ranked_item_ids.index(int(target_item_id)),)
            except ValueError:
                return ()
        if self.scores is None:
            return ()
        target_score = self.scores[target_position]
        higher_count = sum(score > target_score for score in self.scores)
        tied_count = sum(score == target_score for score in self.scores)
        return tuple(range(higher_count, higher_count + tied_count))


class LLM4RSTrainCollator(BaseCollator):
    """Training is not used by LLM4RS; keep a valid model interface."""

    def __call__(self, records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        return {"records": records}


class LLM4RSEvalCollator(BaseCollator):
    """Build prompt inputs and retain row-bound evaluation metadata."""

    def __init__(self, config: LLM4RSConfig, prepared_data: Any, item_text_lookup: Sequence[str]):
        super().__init__(config, prepared_data=prepared_data)
        self.item_text_lookup = tuple(item_text_lookup)

    def __call__(self, records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(records, pd.DataFrame):
            histories = records[HISTORY_ITEM_IDS].tolist()
            candidate_batches = records[CANDIDATE_ITEM_IDS].tolist() if CANDIDATE_ITEM_IDS in records else None
            target_item_ids = records[ITEM_ID].tolist() if ITEM_ID in records else None
            record_indices = records[LLM4RS_RECORD_INDEX].tolist() if LLM4RS_RECORD_INDEX in records else None
        else:
            rows = list(records)
            histories = [_record_value(record, HISTORY_ITEM_IDS, default=()) for record in rows]
            candidate_batches = _optional_record_values(rows, CANDIDATE_ITEM_IDS)
            target_item_ids = _optional_record_values(rows, ITEM_ID)
            record_indices = _optional_record_values(rows, LLM4RS_RECORD_INDEX)
        batch: dict[str, Any] = {
            "history_texts": [
                [self.item_text_lookup[int(item_id)] for item_id in (history_item_ids or ())]
                for history_item_ids in histories
            ]
        }
        if candidate_batches is not None:
            batch["candidate_item_ids"] = [
                tuple(int(item_id) for item_id in candidate_item_ids) for candidate_item_ids in candidate_batches
            ]
        if target_item_ids is not None:
            batch["target_item_ids"] = [int(item_id) for item_id in target_item_ids]
        if record_indices is not None:
            batch["record_indices"] = [int(record_index) for record_index in record_indices]
        return batch


class LLM4RSModel(BaseRetrievalModel):
    """LLM4RS point-, pair-, and list-wise inference using the official prompt design."""

    def __init__(self, config: LLM4RSConfig):
        super().__init__(config)
        self.config = config
        self._item_text_lookup: tuple[str, ...] = ()
        self._example_text = ""
        self._openai_client = OpenAICompatibleClient.from_config(config)

    def build_train_collator(self, prepared_data: Any) -> BaseCollator:
        self._ensure_item_text_lookup(prepared_data)
        return LLM4RSTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data: Any) -> BaseCollator:
        self._ensure_item_text_lookup(prepared_data)
        return LLM4RSEvalCollator(self.config, prepared_data=prepared_data, item_text_lookup=self._item_text_lookup)

    def forward(self, batch: Any) -> dict[str, Any]:
        del batch
        return {}

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        del batch, outputs
        raise RuntimeError("LLM4RSModel is an inference-only recommendation probe and has no training loss.")

    def configure_examples(self, records: pd.DataFrame) -> None:
        """Build the paper's few-shot prefix from the first selected candidate rows."""

        if records.empty or int(self.config.example_num) == 0:
            self._example_text = ""
            return
        example_lines: list[str] = []
        upper = min(int(self.config.example_num), int(self.config.example_pool_size), len(records))
        selected = records.iloc[:upper].reset_index(drop=True)
        for index, record in selected.iterrows():
            history_texts = [self._item_text(int(item_id)) for item_id in record[HISTORY_ITEM_IDS]]
            candidate_ids = tuple(int(item_id) for item_id in record[CANDIDATE_ITEM_IDS])
            target_item_id = int(record[ITEM_ID])
            example_lines.append(
                self._build_example(
                    history_texts=history_texts,
                    candidate_item_ids=candidate_ids,
                    target_item_id=target_item_id,
                    example_index=int(index),
                    example_count=upper,
                )
            )
        self._example_text = "\n".join(example_lines)

    def build_prompt(self, history_texts: Sequence[str], candidate_item_ids: Sequence[int]) -> str:
        """Build one query prompt for the configured policy; useful for list-wise inspection."""

        candidate_ids = tuple(int(item_id) for item_id in candidate_item_ids)
        if self.config.ranking_policy == "point":
            return self._point_prompt(history_texts, candidate_ids[0])
        if self.config.ranking_policy == "pair":
            if len(candidate_ids) < 2:
                raise ValueError("Pair-wise LLM4RS prompt construction requires at least two candidates.")
            return self._pair_prompt(history_texts, candidate_ids[0], candidate_ids[1])
        return self._list_prompt(history_texts, candidate_ids)

    def rank_candidate_batches(
        self,
        history_text_batches: Sequence[Sequence[str]],
        candidate_batches: Sequence[Sequence[int]],
        *,
        target_item_ids: Sequence[int] | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> list[LLM4RSOutcome]:
        if target_item_ids is not None and len(target_item_ids) != len(candidate_batches):
            raise ValueError("target_item_ids and candidate_batches must have equal lengths.")
        if record_indices is not None and len(record_indices) != len(candidate_batches):
            raise ValueError("record_indices and candidate_batches must have equal lengths.")
        tasks: list[tuple[int, tuple[int, ...], str, str]] = []
        normalized_candidates: list[tuple[int, ...]] = []
        for batch_index, (history_texts, candidate_item_ids) in enumerate(
            zip(history_text_batches, candidate_batches, strict=True)
        ):
            candidate_ids = tuple(int(item_id) for item_id in candidate_item_ids)
            normalized_candidates.append(candidate_ids)
            record_index = int(record_indices[batch_index]) if record_indices is not None else batch_index
            if self.config.ranking_policy == "list":
                tasks.append(
                    (
                        batch_index,
                        (),
                        self._list_prompt(history_texts, candidate_ids),
                        " ".join(_candidate_letters(len(candidate_ids))),
                    )
                )
            elif self.config.ranking_policy == "point":
                for candidate_position, item_id in enumerate(candidate_ids):
                    tasks.append((batch_index, (candidate_position,), self._point_prompt(history_texts, item_id), "3"))
            elif self.config.ranking_policy == "pair":
                target_position = None
                if target_item_ids is not None:
                    target_item_id = int(target_item_ids[batch_index])
                    try:
                        target_position = candidate_ids.index(target_item_id)
                    except ValueError as exc:
                        raise ValueError(
                            f"Pair-wise LLM4RS record {record_index} requires target_item_id={target_item_id} "
                            f"to be present in candidate_item_ids={candidate_ids}."
                        ) from exc
                for left_position, right_position in combinations(range(len(candidate_ids)), 2):
                    displayed_positions = [left_position, right_position]
                    if target_position in displayed_positions:
                        randomizer = Random(
                            int(self.config.candidate_seed)
                            + record_index * 1009
                            + left_position * len(candidate_ids)
                            + right_position
                        )
                        randomizer.shuffle(displayed_positions)
                    tasks.append(
                        (
                            batch_index,
                            tuple(displayed_positions),
                            self._pair_prompt(
                                history_texts,
                                candidate_ids[displayed_positions[0]],
                                candidate_ids[displayed_positions[1]],
                            ),
                            "A",
                        )
                    )
            else:  # pragma: no cover - dataclass typing prevents normal entry
                raise ValueError(f"Unsupported LLM4RS ranking_policy '{self.config.ranking_policy}'.")

        responses = self._generate_responses([task[2] for task in tasks], [task[3] for task in tasks])
        grouped_tasks: list[list[tuple[tuple[int, ...], str | None]]] = [[] for _ in normalized_candidates]
        for task, response in zip(tasks, responses, strict=True):
            grouped_tasks[task[0]].append((task[1], response))

        return [
            self._parse_row_outcome(candidate_ids, row_tasks)
            for candidate_ids, row_tasks in zip(normalized_candidates, grouped_tasks, strict=True)
        ]

    def predict(
        self,
        model_inputs: Any,
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rank supplied candidates; pair-wise inference requires collated targets for position randomization."""

        del exclude_item_ids, exclude_mask
        if candidate_item_ids is None:
            raise NotImplementedError("LLM4RS operates only on candidate sets produced by its evaluation pipeline.")
        if not isinstance(model_inputs, dict) or "history_texts" not in model_inputs:
            raise TypeError("LLM4RSModel expects collated 'history_texts'.")
        target_item_ids = model_inputs.get("target_item_ids")
        if self.config.ranking_policy == "pair" and target_item_ids is None:
            raise ValueError("Pair-wise LLM4RS predict() requires collated target_item_ids to randomize target positions.")
        outcomes = self.rank_candidate_batches(
            list(model_inputs["history_texts"]),
            candidate_item_ids.detach().cpu().tolist(),
            target_item_ids=target_item_ids,
            record_indices=model_inputs.get("record_indices"),
        )
        return torch.tensor([list(outcome.ordered_item_ids()[: int(k)]) for outcome in outcomes], dtype=torch.long)

    def _parse_row_outcome(
        self,
        candidate_item_ids: tuple[int, ...],
        tasks: Sequence[tuple[tuple[int, ...], str | None]],
    ) -> LLM4RSOutcome:
        responses = tuple(response for _, response in tasks)
        if self.config.ranking_policy == "list":
            response = responses[0] if responses else None
            ranking = self._parse_list_response(response, candidate_item_ids)
            if ranking is None:
                return LLM4RSOutcome(
                    candidate_item_ids,
                    responses=responses,
                    failed_subrequests=1,
                    error="invalid list-wise response",
                )
            return LLM4RSOutcome(candidate_item_ids, ranked_item_ids=ranking, responses=responses)

        scores = [0] * len(candidate_item_ids)
        failed_subrequests = 0
        successful_subrequests = 0
        if self.config.ranking_policy == "point":
            for (positions, response) in tasks:
                rating = self._parse_point_response(response)
                if rating is None:
                    failed_subrequests += 1
                    continue
                scores[positions[0]] = rating
                successful_subrequests += 1
            return LLM4RSOutcome(
                candidate_item_ids,
                scores=tuple(scores),
                responses=responses,
                failed_subrequests=failed_subrequests,
                error=None if successful_subrequests else "all point-wise responses are invalid",
            )

        for positions, response in tasks:
            choice = self._parse_pair_response(response)
            if choice is None:
                failed_subrequests += 1
                continue
            scores[positions[choice]] += 1
            successful_subrequests += 1
        return LLM4RSOutcome(
            candidate_item_ids,
            scores=tuple(scores),
            responses=responses,
            failed_subrequests=failed_subrequests,
            error=None if successful_subrequests else "all pair-wise responses are invalid",
        )

    def _build_example(
        self,
        *,
        history_texts: Sequence[str],
        candidate_item_ids: tuple[int, ...],
        target_item_id: int,
        example_index: int,
        example_count: int,
    ) -> str:
        target_position = candidate_item_ids.index(int(target_item_id))
        history_text = ", ".join(str(text) for text in history_texts)
        if self.config.ranking_policy == "list":
            other_positions = [position for position in range(len(candidate_item_ids)) if position != target_position]
            Random(int(self.config.candidate_seed) + example_index).shuffle(other_positions)
            answer_positions = [target_position, *other_positions]
            answer = " ".join(_candidate_letters(len(candidate_item_ids))[position] for position in answer_positions)
            candidate_text = self._candidate_choice_text(candidate_item_ids)
            return self._example_template(history_text, candidate_text, answer=answer)

        if self.config.ranking_policy == "pair":
            negative_position = next(position for position in range(len(candidate_item_ids)) if position != target_position)
            if example_index % 2 == 0:
                left_position, right_position, answer = target_position, negative_position, "A"
            else:
                left_position, right_position, answer = negative_position, target_position, "B"
            return self._example_template(
                history_text,
                self._pair_choice_text(candidate_item_ids[left_position], candidate_item_ids[right_position]),
                answer=answer,
                left_item_id=candidate_item_ids[left_position],
                right_item_id=candidate_item_ids[right_position],
            )

        defined_scores = ((3,), (1, 5), (1, 3, 5), (1, 2, 4, 5), (1, 2, 3, 4, 5))
        rating = defined_scores[example_count - 1][example_index]
        if rating <= 3:
            negative_positions = [position for position in range(len(candidate_item_ids)) if position != target_position]
            Random(int(self.config.candidate_seed) + example_index).shuffle(negative_positions)
            selected_position = negative_positions[0]
        else:
            selected_position = target_position
        return self._example_template(history_text, self._item_text(candidate_item_ids[selected_position]), answer=str(rating))

    def _instruction(self) -> str:
        if bool(self.config.no_instruction):
            return ""
        return f"You are a {_domain_terms(self.config.domain)['singular']} recommender system now."

    def _list_prompt(self, history_texts: Sequence[str], candidate_item_ids: tuple[int, ...]) -> str:
        history = ", ".join(str(text) for text in history_texts)
        query = self._query_template(history, self._candidate_choice_text(candidate_item_ids))
        return self._prompt_text(query)

    def _pair_prompt(self, history_texts: Sequence[str], left_item_id: int, right_item_id: int) -> str:
        history = ", ".join(str(text) for text in history_texts)
        candidate_text = self._pair_choice_text(left_item_id, right_item_id)
        query = self._query_template(
            history,
            candidate_text,
            left_item_id=left_item_id,
            right_item_id=right_item_id,
        )
        return self._prompt_text(query)

    def _point_prompt(self, history_texts: Sequence[str], item_id: int) -> str:
        history = ", ".join(str(text) for text in history_texts)
        query = self._query_template(history, self._item_text(item_id))
        return self._prompt_text(query)

    def _prompt_text(self, query: str) -> str:
        return "\n".join(section for section in (self._instruction(), self._example_text, query) if section)

    def _example_template(
        self,
        history_text: str,
        candidate_text: str,
        *,
        answer: str,
        left_item_id: int | None = None,
        right_item_id: int | None = None,
    ) -> str:
        query = self._input_text(
            history_text,
            candidate_text,
            left_item_id=left_item_id,
            right_item_id=right_item_id,
        )
        if self.config.ranking_policy == "point":
            return f"{query}\nOutput: {answer}."
        return f"{query}\nOutput: The answer index is {answer}."

    def _query_template(
        self,
        history_text: str,
        candidate_text: str,
        *,
        left_item_id: int | None = None,
        right_item_id: int | None = None,
    ) -> str:
        query = self._input_text(
            history_text,
            candidate_text,
            left_item_id=left_item_id,
            right_item_id=right_item_id,
        )
        if self.config.ranking_policy == "point":
            return f"{query}\nOutput:"
        return f"{query}\nOutput: The answer index is"

    def _input_text(
        self,
        history_text: str,
        candidate_text: str,
        *,
        left_item_id: int | None = None,
        right_item_id: int | None = None,
    ) -> str:
        words = _domain_terms(self.config.domain)
        prefix = f"Input: Here is the {words['activity']} history of a user: {history_text}. Based on this history, "
        if self.config.ranking_policy == "list":
            return f"{prefix}please rank the following candidate {words['plural']}: {candidate_text}"
        if self.config.ranking_policy == "pair":
            if left_item_id is None or right_item_id is None:
                raise ValueError("Pair-wise prompts require both candidate items.")
            return (
                f"{prefix}would this user prefer {self._item_text(left_item_id)} or {self._item_text(right_item_id)}? "
                f"Answer Choices: {candidate_text}"
            )
        return (
            f"{prefix}please predict the user's rating for the following item: {candidate_text} "
            "(1 being lowest and 5 being highest)"
        )

    def _candidate_choice_text(self, candidate_item_ids: Sequence[int]) -> str:
        return " ".join(
            f"({_candidate_letters(len(candidate_item_ids))[position]}) {self._item_text(int(item_id))}"
            for position, item_id in enumerate(candidate_item_ids)
        )

    def _pair_choice_text(self, left_item_id: int, right_item_id: int) -> str:
        return f"(A) {self._item_text(left_item_id)} (B) {self._item_text(right_item_id)}"

    @staticmethod
    def _parse_list_response(response: str | None, candidate_item_ids: tuple[int, ...]) -> tuple[int, ...] | None:
        positions = [
            ord(char) - ord("A")
            for char in _extract_choice_letters(response, candidate_count=len(candidate_item_ids))
        ]
        if len(positions) != len(candidate_item_ids) or set(positions) != set(range(len(candidate_item_ids))):
            return None
        return tuple(candidate_item_ids[position] for position in positions)

    @staticmethod
    def _parse_point_response(response: str | None) -> int | None:
        if response is None:
            return None
        match = re.search(r"\b([1-5])\b", response)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_pair_response(response: str | None) -> int | None:
        letters = _extract_choice_letters(response, candidate_count=2)
        return {"A": 0, "B": 1}.get(letters[0]) if len(letters) == 1 else None

    def _generate_responses(self, prompts: Sequence[str], identity_responses: Sequence[str]) -> list[str | None]:
        if self.config.backend == "identity":
            return list(identity_responses)
        if len(prompts) != len(identity_responses):
            raise ValueError("prompts and identity_responses must have equal length.")
        max_concurrency = int(self.config.api_batch) if bool(self.config.async_dispatch) else 1
        return dispatch_requests(
            list(prompts),
            self._safe_request_openai_response,
            max_concurrency=max_concurrency,
        )

    def _safe_request_openai_response(self, prompt: str) -> str | None:
        try:
            return self._request_openai_response(prompt)
        except RuntimeError:
            return None

    def _request_openai_response(self, prompt: str) -> str:
        extra_body: dict[str, Any] = {
            "top_p": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": "\n",
        }
        if self.config.api_extra_body:
            extra_body.update(dict(self.config.api_extra_body))
        return self._openai_client.request(
            prompt,
            system_prompt=self._effective_system_prompt(),
            extra_body=extra_body,
        )

    def _effective_system_prompt(self) -> str:
        if self.config.system_prompt is not None:
            return str(self.config.system_prompt).strip()
        return _LLM4RS_POLICY_SYSTEM_PROMPTS[str(self.config.ranking_policy)]

    def _ensure_item_text_lookup(self, prepared_data: Any) -> None:
        if self._item_text_lookup:
            return
        item_table = prepared_data.get_item_table()
        num_items = int(prepared_data.get_num_items())
        lookup = [f"item {item_id}" for item_id in range(num_items)]
        for record in item_table.to_dict(orient="records"):
            item_id = int(record[ITEM_ID])
            for field_name in (self.config.item_text_field, self.config.fallback_item_text_field, ITEM_ID):
                if not field_name or field_name not in record:
                    continue
                value = record[field_name]
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    lookup[item_id] = text
                    break
        self._item_text_lookup = tuple(lookup)

    def _item_text(self, item_id: int) -> str:
        if not self._item_text_lookup:
            raise RuntimeError("Build an LLM4RS collator before constructing prompts.")
        return self._item_text_lookup[int(item_id)]


def _candidate_letters(candidate_count: int) -> tuple[str, ...]:
    return tuple(chr(ord("A") + index) for index in range(int(candidate_count)))


def _extract_choice_letters(response: str | None, *, candidate_count: int) -> tuple[str, ...]:
    if response is None:
        return ()
    valid_letters = set(_candidate_letters(candidate_count))
    tokens = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", response)
    return tuple(token.upper() for token in tokens if token.upper() in valid_letters)


def _domain_terms(domain: str) -> dict[str, str]:
    key = str(domain).strip().lower()
    terms = {
        "movie": {"singular": "movie", "plural": "movies", "activity": "watching"},
        "music": {"singular": "music", "plural": "music", "activity": "listening"},
        "book": {"singular": "book", "plural": "books", "activity": "reading"},
        "news": {"singular": "news", "plural": "news", "activity": "reading"},
        "agnostic": {"singular": "general-purpose", "plural": "items", "activity": "interaction"},
    }
    if key not in terms:
        raise ValueError(f"Unknown LLM4RS domain '{domain}'.")
    return terms[key]


def _record_value(record: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    if isinstance(record, pd.Series):
        return record[key] if key in record else default
    return getattr(record, key, default)


def _optional_record_values(records: Sequence[Any], key: str) -> list[Any] | None:
    values = [_record_value(record, key) for record in records]
    return None if any(value is None for value in values) else values


__all__ = [
    "LLM4RSEvalCollator",
    "LLM4RSModel",
    "LLM4RSOutcome",
    "LLM4RSTrainCollator",
]
