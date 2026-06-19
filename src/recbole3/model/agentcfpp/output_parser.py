from __future__ import annotations

import re


def parse_recommendation(text: str) -> tuple[str, str]:
    """Parse the forward pairwise-choice output into (choice_title, explanation)."""
    choice = ""
    explanation = ""

    choice_match = re.search(r"Choice:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if choice_match:
        choice = choice_match.group(1).strip().strip("'\"")

    explanation_match = re.search(r"Explanation:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    if not choice:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if lines:
            choice = lines[0].strip("'\"")

    return choice, explanation


def parse_user_update(text: str) -> str:
    """Parse user memory update output into the new self-description."""
    match = re.search(r"My updated self-introduction:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("'\"")
    return text.strip()


def parse_crossdomain_update(text: str) -> str:
    """Parse cross-domain deduction output into the new cross-domain preference."""
    match = re.search(r"My deduced preference:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("'\"")
    return text.strip()


def parse_item_update(text: str) -> tuple[str, str]:
    """Parse item update output into (neg_item_description, pos_item_description).

    The prompt asks for 'first item' (neg) and 'second item' (pos) descriptions.
    Returns (neg_description, pos_description) to match the prompt ordering.
    """
    first_match = re.search(
        r"(?:The )?updated description of the first item\s*(?:is)?:\s*(.+?)"
        r"(?=(?:The )?updated description of the second|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    second_match = re.search(
        r"(?:The )?updated description of the second item\s*(?:is)?:\s*(.+?)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    first_desc = first_match.group(1).strip().rstrip(".").strip() if first_match else ""
    second_desc = second_match.group(1).strip().rstrip(".").strip() if second_match else ""

    if not first_desc and not second_desc:
        parts = re.split(r"\n\s*\n|\n", text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            first_desc = parts[0]
            second_desc = parts[1]
        elif len(parts) == 1:
            first_desc = parts[0]
            second_desc = parts[0]

    return first_desc, second_desc


def parse_evaluation_ranking(text: str) -> list[str]:
    """Parse evaluation ranking output into an ordered list of item titles."""
    rank_match = re.search(r"Rank:\s*\{?\s*(.+?)\s*\}?\s*$", text, re.IGNORECASE | re.DOTALL)
    rank_text = rank_match.group(1) if rank_match else text

    lines = [line.strip() for line in rank_text.strip().split("\n") if line.strip()]

    titles = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            titles.append(cleaned)
    return titles


def fuzzy_match_title(query: str, candidates: list[str]) -> tuple[str, int]:
    """Find the best matching candidate title for a query string.

    Returns (matched_title, index_in_candidates). Uses substring containment first,
    then a longest-common-subsequence ratio.
    """
    if not candidates:
        return "", -1

    best_score = -1.0
    best_idx = 0
    query_lower = query.lower().strip()

    for idx, candidate in enumerate(candidates):
        candidate_lower = candidate.lower().strip()
        if candidate_lower and (candidate_lower in query_lower or query_lower in candidate_lower):
            score = len(candidate_lower) / max(len(query_lower), 1)
            if score > best_score:
                best_score = score
                best_idx = idx

    if best_score > 0:
        return candidates[best_idx], best_idx

    for idx, candidate in enumerate(candidates):
        score = _lcs_ratio(query_lower, candidate.lower().strip())
        if score > best_score:
            best_score = score
            best_idx = idx

    return candidates[best_idx], best_idx


def _lcs_ratio(a: str, b: str) -> float:
    """Compute an LCS-based similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    if len(a) > 500:
        a = a[:500]
    if len(b) > 500:
        b = b[:500]
    m, n = len(a), len(b)

    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    return (2.0 * lcs_len) / (m + n)


__all__ = [
    "parse_recommendation",
    "parse_user_update",
    "parse_crossdomain_update",
    "parse_item_update",
    "parse_evaluation_ranking",
    "fuzzy_match_title",
]
