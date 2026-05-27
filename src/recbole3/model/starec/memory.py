from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class STARecMemoryInteraction:
    item_id: int
    item_text: str
    feedback: str
    timestamp: int | None = None
    label: float | None = None


@dataclass(slots=True)
class STARecReflectionRecord:
    target_item_id: int
    target_item_text: str
    system_prediction: str
    actual_feedback: str
    previous_user_description: str
    updated_user_description: str
    raw_reflection_output: str | None


@dataclass(slots=True)
class STARecUserMemory:
    user_id: int
    profile_text: str
    current_user_description: str
    interaction_history: list[STARecMemoryInteraction] = field(default_factory=list)
    reflection_history: list[STARecReflectionRecord] = field(default_factory=list)

    def append_interaction(
        self,
        *,
        item_id: int,
        item_text: str,
        feedback: str,
        timestamp: int | None = None,
        label: float | None = None,
    ) -> None:
        self.interaction_history.append(
            STARecMemoryInteraction(
                item_id=int(item_id),
                item_text=str(item_text),
                feedback=str(feedback),
                timestamp=timestamp,
                label=label,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "STARecUserMemory":
        memory = cls(
            user_id=int(record["user_id"]),
            profile_text=str(record.get("profile_text", "")),
            current_user_description=str(record.get("current_user_description", "")),
        )
        memory.interaction_history = [
            STARecMemoryInteraction(
                item_id=int(item["item_id"]),
                item_text=str(item.get("item_text", "")),
                feedback=str(item.get("feedback", "")),
                timestamp=_optional_int(item.get("timestamp")),
                label=_optional_float(item.get("label")),
            )
            for item in record.get("interaction_history", [])
        ]
        memory.reflection_history = [
            STARecReflectionRecord(
                target_item_id=int(item["target_item_id"]),
                target_item_text=str(item.get("target_item_text", "")),
                system_prediction=str(item.get("system_prediction", "")),
                actual_feedback=str(item.get("actual_feedback", "")),
                previous_user_description=str(item.get("previous_user_description", "")),
                updated_user_description=str(item.get("updated_user_description", "")),
                raw_reflection_output=item.get("raw_reflection_output"),
            )
            for item in record.get("reflection_history", [])
        ]
        return memory


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "STARecMemoryInteraction",
    "STARecReflectionRecord",
    "STARecUserMemory",
]
