from __future__ import annotations

import re
from typing import Sequence

from recbole3.model.starec.memory import STARecUserMemory


Message = dict[str, str]
DEFAULT_ITEM_DOMAIN = ("item", "items")
CDS_ITEM_DOMAIN = ("CD or music product", "CDs or music products")
MOVIE_ITEM_DOMAIN = ("movie", "movies")


def resolve_item_domain(
    *,
    dataset_name: str | None,
    category: str | None,
    override_singular: str | None = None,
    override_plural: str | None = None,
) -> tuple[str, str]:
    singular = _clean_optional(override_singular)
    plural = _clean_optional(override_plural)
    if bool(singular) != bool(plural):
        raise ValueError("item-domain singular and plural overrides must be set together.")
    if singular and plural:
        return singular, plural

    normalized_category = _normalize_domain_hint(category)
    if normalized_category == "cdsandvinyl":
        return CDS_ITEM_DOMAIN

    normalized_name = _normalize_domain_hint(dataset_name)
    if "ml1m" in normalized_name or "movielens" in normalized_name:
        return MOVIE_ITEM_DOMAIN

    return DEFAULT_ITEM_DOMAIN


def build_memory_init_messages(
    *,
    profile_text: str,
    history_lines: Sequence[str],
    item_domain_singular: str,
    item_domain_plural: str,
) -> list[Message]:
    history = "\n".join(history_lines) if history_lines else "- None"
    return [
        {
            "role": "system",
            "content": (
                "<role>\n"
                f"You are a recommendation user-modeling analyst for {item_domain_plural}.\n"
                "</role>"
            ),
        },
        {
            "role": "user",
            "content": (
                "<task>\n"
                f"Write the user's current preference description for future {item_domain_singular} ranking.\n"
                "</task>\n\n"
                "<context>\n"
                "This description will be used as long-term memory for a recommendation agent. "
                f"It should help rank future candidate {item_domain_plural}, not summarize every past interaction.\n"
                "</context>\n\n"
                f"<user_profile>\n{profile_text}\n</user_profile>\n\n"
                f"<interaction_history>\n{history}\n</interaction_history>\n\n"
                "<constraints>\n"
                "- Base the description only on the provided user profile and interaction history.\n"
                f"- Focus on stable {item_domain_singular} preferences and important exceptions when supported by evidence.\n"
                "- Mention dislikes only when the history supports them.\n"
                "- Do not copy or list the full interaction history.\n"
                "- Do not invent creators, categories, dates, ratings, or preferences not supported by the input.\n"
                "- Current User Description must be <= 120 words.\n"
                "- Do not include analysis, bullet points, or extra sections.\n"
                "- The example demonstrates output shape only; do not reuse its content unless supported by the input.\n"
                "</constraints>\n\n"
                "<output_format>\n"
                "Current User Description: [description]\n"
                "</output_format>\n\n"
                "<example>\n"
                "Current User Description: Prefers acoustic jazz collections and remastered albums; "
                "may avoid poorly recorded live releases when prior feedback supports that pattern.\n"
                "</example>"
            ),
        },
    ]


def build_ranking_messages(
    *,
    memory: STARecUserMemory,
    candidate_lines: Sequence[str],
    history_limit: int | None,
    item_domain_singular: str,
    item_domain_plural: str,
) -> list[Message]:
    history = _memory_history_text(memory, history_limit=history_limit)
    candidates = "\n".join(candidate_lines)
    return [
        {
            "role": "system",
            "content": (
                "<role>\n"
                f"You are an expert recommendation ranking assistant for {item_domain_plural}.\n"
                "</role>"
            ),
        },
        {
            "role": "user",
            "content": (
                "<task>\n"
                f"Rank all candidate {item_domain_plural} from most likely to least likely to be liked by the user.\n"
                "</task>\n\n"
                "<context>\n"
                "Use Current User Description as the distilled preference memory. "
                "Use Interaction History as supporting evidence when it is relevant.\n"
                "</context>\n\n"
                f"<user_profile>\n{memory.profile_text}\n</user_profile>\n\n"
                f"<current_user_description>\n{memory.current_user_description}\n</current_user_description>\n\n"
                f"<interaction_history>\n{history}\n</interaction_history>\n\n"
                f"<candidate_items>\n{candidates}\n</candidate_items>\n\n"
                "<constraints>\n"
                "- Return every candidate exactly once.\n"
                "- Use only candidate items listed in <candidate_items>.\n"
                "- Preserve the exact [ItemID: ...] for each candidate.\n"
                f"- Do not add extra {item_domain_plural}.\n"
                "- Do not include a separate analysis section, correction note, or alternate ranking.\n"
                "- Brief explanations should cite preference evidence or item metadata, not hidden reasoning.\n"
                "- If evidence is weak, still rank every candidate using the provided metadata and memory.\n"
                "- The example demonstrates output shape only; do not reuse its item ids or content.\n"
                "</constraints>\n\n"
                "<output_format>\n"
                "1. [ItemID: <id>] [Item Title] - [Brief evidence-grounded explanation]\n"
                "2. [ItemID: <id>] [Item Title] - [Brief evidence-grounded explanation]\n"
                "... continue until every candidate is ranked.\n"
                "</output_format>\n\n"
                "<example>\n"
                "1. [ItemID: 102] Midnight Quartet - Matches the user's interest in acoustic jazz collections.\n"
                "2. [ItemID: 205] Arena Lights - Less aligned with the user's quieter listening pattern.\n"
                "</example>"
            ),
        },
    ]


def build_reflection_messages(
    *,
    memory: STARecUserMemory,
    target_line: str,
    system_prediction: str,
    actual_feedback: str,
    history_limit: int | None,
    item_domain_singular: str,
    item_domain_plural: str,
) -> list[Message]:
    history = _memory_history_text(memory, history_limit=history_limit)
    return [
        {
            "role": "system",
            "content": (
                "<role>\n"
                f"You are an expert preference analyst for {item_domain_plural} in a recommendation agent.\n"
                "</role>"
            ),
        },
        {
            "role": "user",
            "content": (
                "<task>\n"
                f"Update the user's preference description using one new {item_domain_singular} feedback event.\n"
                "</task>\n\n"
                "<context>\n"
                f"This updated description will be used as memory for future {item_domain_plural} ranking prompts. "
                "It must stay compact and improve future recommendations without copying the full interaction history.\n"
                "</context>\n\n"
                f"<user_profile>\n{memory.profile_text}\n</user_profile>\n\n"
                f"<current_user_description>\n{memory.current_user_description}\n</current_user_description>\n\n"
                f"<interaction_history>\n{history}\n</interaction_history>\n\n"
                f"<target_item>\n{target_line}\n</target_item>\n\n"
                f"<system_prediction>\n{system_prediction}\n</system_prediction>\n\n"
                f"<actual_feedback>\n{actual_feedback}\n</actual_feedback>\n\n"
                "<constraints>\n"
                "- Updated User Description must be <= 120 words.\n"
                "- Do not list or copy the full interaction history.\n"
                "- Preserve stable preferences and important exceptions.\n"
                "- Revise an old assumption only when the new feedback contradicts it.\n"
                f"- Add a new {item_domain_singular} preference or dislike only when the target item and feedback provide clear evidence.\n"
                "- Do not mention the system prediction unless it changes the preference description.\n"
                "- Do not include analysis, bullet points, or extra sections.\n"
                "- The example demonstrates output shape only; do not reuse its content unless supported by the input.\n"
                "</constraints>\n\n"
                "<output_format>\n"
                "Updated User Description: [description]\n"
                "</output_format>\n\n"
                "<example>\n"
                "Updated User Description: Prefers acoustic jazz collections and remastered albums; "
                "may avoid noisy live recordings when feedback indicates poor audio quality.\n"
                "</example>"
            ),
        },
    ]


def _memory_history_text(memory: STARecUserMemory, *, history_limit: int | None) -> str:
    rows = memory.interaction_history
    if history_limit is not None and history_limit > 0:
        rows = rows[-int(history_limit) :]
    if not rows:
        return "- None"
    return "\n".join(
        f"- [ItemID: {item.item_id}] {item.item_text}; Feedback: {item.feedback}"
        for item in rows
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_domain_hint(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


__all__ = [
    "CDS_ITEM_DOMAIN",
    "DEFAULT_ITEM_DOMAIN",
    "MOVIE_ITEM_DOMAIN",
    "Message",
    "build_memory_init_messages",
    "build_ranking_messages",
    "build_reflection_messages",
    "resolve_item_domain",
]
