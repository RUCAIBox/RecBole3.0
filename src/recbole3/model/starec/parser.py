from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


NUMBERED_LINE_RE = re.compile(r"^\s*\d+\s*[\).\:-]\s*(.+?)\s*$")
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
CURRENT_DESCRIPTION_RE = re.compile(r"Current User Description:\s*(.+)", re.IGNORECASE | re.DOTALL)
UPDATED_DESCRIPTION_RE = re.compile(r"Updated User Description:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class STARecRankingParseResult:
    ranked_item_ids: list[int]
    valid: bool
    missing_item_ids: list[int]
    duplicate_item_ids: list[int]
    unknown_lines: list[str]


def parse_ranking_output(
    output: str,
    candidate_item_ids: Sequence[int],
    *,
    id_label: str = "ItemID",
) -> STARecRankingParseResult:
    candidates = [int(item_id) for item_id in candidate_item_ids]
    candidate_set = set(candidates)
    ranked_item_ids: list[int] = []
    unknown_lines: list[str] = []
    for line in strip_think_blocks(output).splitlines():
        match = NUMBERED_LINE_RE.match(line)
        if not match:
            continue
        item_id = _extract_item_id(match.group(1), candidate_set, id_label=id_label)
        if item_id is None:
            unknown_lines.append(line)
            continue
        ranked_item_ids.append(item_id)

    seen: set[int] = set()
    duplicates: list[int] = []
    for item_id in ranked_item_ids:
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)

    missing = [item_id for item_id in candidates if item_id not in seen]
    valid = not missing and not duplicates and not unknown_lines and len(ranked_item_ids) == len(candidates)
    return STARecRankingParseResult(
        ranked_item_ids=ranked_item_ids,
        valid=valid,
        missing_item_ids=missing,
        duplicate_item_ids=duplicates,
        unknown_lines=unknown_lines,
    )


def complete_ranked_item_ids(parsed: STARecRankingParseResult, candidate_item_ids: Sequence[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for item_id in parsed.ranked_item_ids:
        normalized = int(item_id)
        if normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    for item_id in candidate_item_ids:
        normalized = int(item_id)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def strip_think_blocks(output: str) -> str:
    return THINK_BLOCK_RE.sub("", str(output))


def parse_current_description(output: str) -> str:
    cleaned_output = strip_think_blocks(output)
    match = CURRENT_DESCRIPTION_RE.search(cleaned_output)
    return _clean_description(match.group(1) if match else cleaned_output)


def parse_updated_description(output: str) -> str | None:
    match = UPDATED_DESCRIPTION_RE.search(strip_think_blocks(output))
    if not match:
        return None
    return _clean_description(match.group(1))


def _extract_item_id(text: str, candidate_ids: set[int], *, id_label: str) -> int | None:
    id_match = re.search(rf"{re.escape(id_label)}:\s*([^\]\s;,]+)", text, re.IGNORECASE)
    if not id_match:
        return None
    try:
        item_id = int(id_match.group(1))
    except ValueError:
        return None
    return item_id if item_id in candidate_ids else None


def _clean_description(value: str) -> str:
    return str(value).strip().strip("[]").strip()


__all__ = [
    "STARecRankingParseResult",
    "complete_ranked_item_ids",
    "parse_current_description",
    "parse_ranking_output",
    "parse_updated_description",
    "strip_think_blocks",
]
