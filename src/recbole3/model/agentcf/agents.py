from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserAgentState:
    """Maintains a user's evolving self-description and interaction history."""

    user_id: int
    description: str = ""
    memory_history: list[str] = field(default_factory=list)
    update_history: list[str] = field(default_factory=list)
    historical_interactions: dict[str, Any] = field(default_factory=dict)

    @property
    def current_description(self) -> str:
        if self.update_history:
            return self.update_history[-1]
        return self.description

    def update(self, new_description: str) -> None:
        self.update_history.append(new_description)

    def commit_memory(self) -> None:
        """Commit current description to memory_history (called after each batch)."""
        self.memory_history.append(self.current_description)


@dataclass
class ItemAgentState:
    """Maintains an item's evolving description."""

    item_id: int
    title: str = ""
    category: str = ""
    initial_description: str = ""
    update_history: list[str] = field(default_factory=list)
    description_embeddings: dict[str, Any] = field(default_factory=dict)

    @property
    def current_description(self) -> str:
        if self.update_history:
            return self.update_history[-1]
        return self.initial_description

    def update(self, new_description: str) -> None:
        self.update_history.append(new_description)


@dataclass
class RecAgentState:
    """Maintains the recommender agent's per-user interaction examples for RAG."""

    user_examples: dict[int, dict[tuple[Any, ...], Any]] = field(default_factory=dict)

    def add_example(
        self,
        user_id: int,
        user_desc: str,
        pos_title: str,
        neg_title: str,
        pos_desc: str,
        neg_desc: str,
        accuracy: int,
        reason: str,
        embedding: Any = None,
    ) -> None:
        if user_id not in self.user_examples:
            self.user_examples[user_id] = {}
        key = (user_desc, pos_title, neg_title, pos_desc, neg_desc, accuracy, reason)
        self.user_examples[user_id][key] = embedding
