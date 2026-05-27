from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch

from recbole3.dataset import ITEM_ID, LABEL, TIMESTAMP, USER_ID
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.model.starec.config import STARecConfig
from recbole3.model.starec.memory import STARecReflectionRecord, STARecUserMemory
from recbole3.model.starec.parser import (
    STARecRankingParseResult,
    parse_current_description,
    parse_ranking_output,
    parse_updated_description,
)
from recbole3.model.starec.prompts import (
    Message,
    build_memory_init_messages,
    build_ranking_messages,
    build_reflection_messages,
    resolve_item_domain,
)


class STARecPassthroughCollator(BaseCollator):
    """Keep records as DataFrames/lists for the custom STARec trainer."""

    def __call__(self, feature_records: Sequence[Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(feature_records, pd.DataFrame):
            return {"records": feature_records.reset_index(drop=True)}
        return {"records": list(feature_records)}


class STARecModel(BaseRetrievalModel):
    """Architecture-only STARec agent model with user-scoped memory helpers."""

    def __init__(self, config: STARecConfig):
        super().__init__(config)
        self.config = config
        self._item_text_lookup: tuple[str, ...] = ()
        self._user_profile_lookup: dict[int, str] = {}
        self._item_domain_singular = "item"
        self._item_domain_plural = "items"
        self._metadata_prepared = False

    def ensure_initialized(self, prepared_data) -> None:
        self.prepare_metadata(prepared_data)

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self.prepare_metadata(prepared_data)
        return STARecPassthroughCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self.prepare_metadata(prepared_data)
        return STARecPassthroughCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Any) -> dict[str, Any]:
        return {}

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        raise RuntimeError("STARecModel is inference-only and does not define a training loss.")

    def predict(
        self,
        model_inputs: Any,
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del model_inputs, exclude_item_ids, exclude_mask
        if candidate_item_ids is None:
            raise NotImplementedError("STARecModel ranks provided candidate sets and does not implement full-sort prediction.")
        return candidate_item_ids[:, : int(k)].to(dtype=torch.long)

    def prepare_metadata(self, prepared_data) -> None:
        if self._metadata_prepared:
            return
        self._item_text_lookup = tuple(self._build_item_text_lookup(prepared_data))
        self._user_profile_lookup = self._build_user_profile_lookup(prepared_data)
        self._item_domain_singular, self._item_domain_plural = resolve_item_domain(
            dataset_name=getattr(prepared_data.config, "name", None),
            category=getattr(prepared_data.config, "category", None),
            override_singular=self.config.item_domain_singular,
            override_plural=self.config.item_domain_plural,
        )
        self._metadata_prepared = True

    def build_initial_memory(self, *, user_id: int, history_item_ids: Sequence[int]) -> STARecUserMemory:
        profile_text = self.user_profile_text(user_id)
        history_ids = [int(item_id) for item_id in history_item_ids]
        history_lines = [self.format_item_line(item_id) for item_id in history_ids]
        description = self.initialize_user_description(profile_text=profile_text, history_lines=history_lines)
        memory = STARecUserMemory(
            user_id=int(user_id),
            profile_text=profile_text,
            current_user_description=description,
        )
        for item_id in history_ids:
            memory.append_interaction(
                item_id=item_id,
                item_text=self.item_text(item_id),
                feedback="liked",
            )
        return memory

    def initialize_user_description(self, *, profile_text: str, history_lines: Sequence[str]) -> str:
        if self.config.backend == "deterministic":
            if history_lines:
                return f"Deterministic profile based on {len(history_lines)} history item(s)."
            return "Deterministic generic user profile with no prior history."
        response = self._complete_openai(
            build_memory_init_messages(
                profile_text=profile_text,
                history_lines=history_lines,
                item_domain_singular=self._item_domain_singular,
                item_domain_plural=self._item_domain_plural,
            )
        )
        return parse_current_description(response)

    def rank_candidates(
        self,
        *,
        memory: STARecUserMemory,
        candidate_item_ids: Sequence[int],
    ) -> tuple[str, STARecRankingParseResult]:
        candidate_ids = [int(item_id) for item_id in candidate_item_ids]
        raw_response = ""
        parsed: STARecRankingParseResult | None = None
        for _ in range(max(1, int(self.config.parse_retries) + 1)):
            raw_response = self._rank_candidates_once(memory=memory, candidate_item_ids=candidate_ids)
            parsed = parse_ranking_output(raw_response, candidate_ids)
            if parsed.valid:
                break
        if parsed is None:
            parsed = parse_ranking_output(raw_response, candidate_ids)
        return raw_response, parsed

    def reflect(
        self,
        *,
        memory: STARecUserMemory,
        target_item_id: int,
        system_prediction: str,
        actual_feedback: str,
    ) -> tuple[str | None, bool, str | None]:
        previous_description = memory.current_user_description
        target_line = self.format_item_line(target_item_id)
        raw_response: str | None = None
        updated_description: str | None = None
        for _ in range(max(1, int(self.config.parse_retries) + 1)):
            raw_response = self._reflect_once(
                memory=memory,
                target_line=target_line,
                system_prediction=system_prediction,
                actual_feedback=actual_feedback,
            )
            updated_description = parse_updated_description(raw_response)
            if updated_description:
                break
        if not updated_description:
            return raw_response, False, "Could not parse Updated User Description"

        memory.current_user_description = updated_description
        memory.reflection_history.append(
            STARecReflectionRecord(
                target_item_id=int(target_item_id),
                target_item_text=self.item_text(target_item_id),
                system_prediction=system_prediction,
                actual_feedback=actual_feedback,
                previous_user_description=previous_description,
                updated_user_description=updated_description,
                raw_reflection_output=raw_response,
            )
        )
        return raw_response, True, None

    def format_item_line(self, item_id: int) -> str:
        return f"- [ItemID: {int(item_id)}] {self.item_text(int(item_id))}"

    def item_text(self, item_id: int) -> str:
        if not self._item_text_lookup:
            raise RuntimeError("STARecModel metadata must be prepared before item text lookup.")
        return self._item_text_lookup[int(item_id)]

    def user_profile_text(self, user_id: int) -> str:
        if not self._user_profile_lookup:
            raise RuntimeError("STARecModel metadata must be prepared before user profile lookup.")
        return self._user_profile_lookup.get(int(user_id), f"User Profile:\n- User ID: {int(user_id)}")

    def record_timestamp(self, record: dict[str, Any]) -> int | None:
        value = record.get(TIMESTAMP)
        if value is None or pd.isna(value):
            return None
        return int(value)

    def record_feedback(self, record: dict[str, Any]) -> str:
        value = record.get(LABEL)
        if value is None or pd.isna(value) or float(value) > 0:
            return "Actually Liked"
        return "Actually Disliked"

    def _rank_candidates_once(self, *, memory: STARecUserMemory, candidate_item_ids: Sequence[int]) -> str:
        if self.config.backend == "deterministic":
            return "\n".join(
                f"{rank}. [ItemID: {item_id}] {self.item_text(item_id)}"
                for rank, item_id in enumerate(candidate_item_ids, start=1)
            )
        candidate_lines = [self.format_item_line(item_id) for item_id in candidate_item_ids]
        messages = build_ranking_messages(
            memory=memory,
            candidate_lines=candidate_lines,
            history_limit=None,
            item_domain_singular=self._item_domain_singular,
            item_domain_plural=self._item_domain_plural,
        )
        return self._complete_openai(messages)

    def _reflect_once(
        self,
        *,
        memory: STARecUserMemory,
        target_line: str,
        system_prediction: str,
        actual_feedback: str,
    ) -> str:
        if self.config.backend == "deterministic":
            return (
                "Updated User Description: "
                f"{memory.current_user_description} Recent evidence: {actual_feedback.lower()} {target_line}."
            )
        return self._complete_openai(
            build_reflection_messages(
                memory=memory,
                target_line=target_line,
                system_prediction=system_prediction,
                actual_feedback=actual_feedback,
                history_limit=None,
                item_domain_singular=self._item_domain_singular,
                item_domain_plural=self._item_domain_plural,
            )
        )

    def _complete_openai(self, messages: list[Message]) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"STARec openai backend requires environment variable {self.config.api_key_env}.")
        backoff = max(0.1, float(self.config.retry_backoff_sec))
        last_error: Exception | None = None
        for attempt in range(1, max(1, int(self.config.request_retries)) + 1):
            try:
                client = _build_openai_client(
                    api_key=api_key,
                    base_url=self.config.api_base_url,
                    timeout=float(self.config.request_timeout_sec),
                )
                response = client.chat.completions.create(
                    model=self.config.api_model_name,
                    messages=messages,
                    temperature=float(self.config.temperature),
                    top_p=float(self.config.top_p),
                    max_tokens=int(self.config.max_output_tokens),
                )
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("OpenAI SDK response did not include message content.")
                return str(content)
            except Exception as exc:
                last_error = exc
                if attempt >= max(1, int(self.config.request_retries)):
                    break
                time.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError(f"Failed to call the configured STARec openai backend after retries: {last_error}") from last_error

    def _build_item_text_lookup(self, prepared_data) -> list[str]:
        item_table = prepared_data.get_item_table()
        num_items = int(prepared_data.get_num_items())
        lookup = [f"item {item_id}" for item_id in range(num_items)]
        for record in item_table.to_dict(orient="records"):
            item_id = int(record[ITEM_ID])
            lookup[item_id] = self._resolve_item_text(record, item_id=item_id)
        return lookup

    def _resolve_item_text(self, item_record: dict[str, Any], *, item_id: int) -> str:
        for field_name in (self.config.item_text_field, self.config.fallback_item_text_field, "title", "metadata_text", ITEM_ID):
            if not field_name or field_name not in item_record:
                continue
            value = item_record[field_name]
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return f"item {item_id}"

    def _build_user_profile_lookup(self, prepared_data) -> dict[int, str]:
        user_table = prepared_data.get_user_table()
        profiles: dict[int, str] = {}
        for record in user_table.to_dict(orient="records"):
            user_id = int(record[USER_ID])
            lines = [f"User Profile:", f"- User ID: {user_id}"]
            for field_name in self.config.user_profile_fields:
                if field_name not in record:
                    continue
                value = record[field_name]
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    lines.append(f"- {field_name}: {text}")
            profiles[user_id] = "\n".join(lines)
        return profiles


def _build_openai_client(*, api_key: str, base_url: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "STARec openai backend requires the openai package. Install it with `uv run --extra starec ...`."
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=str(base_url).rstrip("/") or None,
        timeout=timeout,
    )


__all__ = [
    "STARecModel",
    "STARecPassthroughCollator",
]
