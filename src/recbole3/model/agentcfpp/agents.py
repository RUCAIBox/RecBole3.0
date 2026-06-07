from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserAgentState:
    """A user's evolving dual-layer memory across domains.

    - private[domain]: single-domain self-introduction (preferences/dislikes).
    - cross[domain]:    cross-domain deduced preference for that domain.
    - long_memory:      snapshots of past descriptions (used by the B+R strategy).
    """

    user_id: int
    private: dict[str, str] = field(default_factory=dict)
    cross: dict[str, str] = field(default_factory=dict)
    long_memory: list[str] = field(default_factory=list)
    active_domains: set[str] = field(default_factory=set)

    def init_domain(self, domain: str, init_text: str) -> None:
        self.active_domains.add(domain)
        self.private.setdefault(domain, init_text)
        self.cross.setdefault(domain, init_text)

    def private_description(self, domain: str) -> str:
        return self.private.get(domain, "")

    def cross_description(self, domain: str) -> str:
        return self.cross.get(domain, self.private.get(domain, ""))

    def all_private_description(self) -> str:
        """Concatenate every active domain's private memory (for cross-domain deduction)."""
        parts = []
        for domain in sorted(self.active_domains):
            text = self.private.get(domain, "")
            if text:
                parts.append(f"--- preferences in {domain} ---\n{text}\n")
        return "\n".join(parts)

    def update_private(self, domain: str, new_text: str) -> None:
        self.active_domains.add(domain)
        self.private[domain] = new_text

    def update_cross(self, domain: str, new_text: str) -> None:
        self.active_domains.add(domain)
        self.cross[domain] = new_text

    def commit_memory(self, domain: str) -> None:
        """Snapshot the current cross-domain description into long memory."""
        text = self.cross_description(domain)
        if text:
            self.long_memory.append(text)


@dataclass
class ItemAgentState:
    """An item's evolving description, seeded from its metadata fields."""

    item_id: int
    title: str = ""
    domain: str = ""
    initial_description: str = ""
    update_history: list[str] = field(default_factory=list)

    @property
    def current_description(self) -> str:
        if self.update_history:
            return self.update_history[-1]
        return self.initial_description

    def update(self, new_description: str) -> None:
        if new_description:
            self.update_history.append(new_description)


@dataclass
class GroupState:
    """Shared group memory: group name -> memory text, plus user membership."""

    groups: dict[str, dict[str, str]] = field(default_factory=dict)
    """group_name -> {domain -> recent-interactions text}."""
    user_to_groups: dict[int, list[str]] = field(default_factory=dict)
    """framework user_id -> list of group names the user belongs to."""

    def render_group_mem(self, user_id: int, domain: str) -> str:
        """Build the group-memory snippet injected into the eval prompt for one user."""
        group_names = self.user_to_groups.get(int(user_id), [])
        if not group_names:
            return ""
        chunks = []
        for group_name in group_names:
            group = self.groups.get(group_name)
            if not group:
                continue
            header = (
                f"Users who have similar preferences to me in {group_name} "
                f"have interacted with the following items recently:\n"
            )
            domain_text = group.get(domain, "")
            if domain_text:
                chunks.append(header + domain_text)
        return "\n".join(chunks)


__all__ = [
    "GroupState",
    "ItemAgentState",
    "UserAgentState",
]
