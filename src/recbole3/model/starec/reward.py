from __future__ import annotations

import json
from typing import Any

from recbole3.model.starec.parser import parse_ranking_output, strip_think_blocks


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> float:
    """VeRL-compatible ranking reward for STARec.

    VeRL custom reward functions are loaded by name and called with
    (data_source, solution_str, ground_truth, extra_info). The data_source is not
    needed here but remains part of the public signature expected by VeRL.
    """

    del data_source
    payload = _ground_truth_payload(ground_truth, extra_info=extra_info)
    return starec_ranking_reward(
        solution_str,
        target_item_id=int(payload["target_item_id"]),
        candidate_item_ids=payload.get("candidate_item_ids"),
        topk=int(payload.get("topk", 20)),
    )


def starec_ranking_reward(
    output: str,
    *,
    target_item_id: int,
    candidate_item_ids: list[int] | tuple[int, ...] | None = None,
    topk: int = 20,
) -> float:
    """Map target-item rank to the STARec paper reward bins."""

    rank = target_rank(output, target_item_id=int(target_item_id), candidate_item_ids=candidate_item_ids)
    if rank is None or rank > int(topk):
        return -1.0
    if rank == 1:
        return 1.0
    if rank <= 5:
        return 0.5
    if rank <= 10:
        return 0.0
    if rank <= 20:
        return -0.5
    return -1.0


def target_rank(
    output: str,
    *,
    target_item_id: int,
    candidate_item_ids: list[int] | tuple[int, ...] | None = None,
) -> int | None:
    """Return the 1-based target rank in generated STARec ranking text."""

    if candidate_item_ids:
        candidates = [int(item_id) for item_id in candidate_item_ids]
        parsed = parse_ranking_output(output, candidates)
        if not parsed.valid:
            return None
        ranked_item_ids = parsed.ranked_item_ids
    else:
    else:
        ranked_item_ids = _extract_ranked_item_ids(output)
    try:
        return [int(item_id) for item_id in ranked_item_ids].index(int(target_item_id)) + 1
    except ValueError:
        return None


def _ground_truth_payload(
    ground_truth: str | dict[str, Any],
    *,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        payload = dict(ground_truth)
    else:
        text = str(ground_truth).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"target_item_id": text}
        payload = dict(parsed) if isinstance(parsed, dict) else {"target_item_id": parsed}
    if extra_info:
        for key in ("target_item_id", "candidate_item_ids", "topk"):
            if key not in payload and key in extra_info:
                payload[key] = extra_info[key]
    if "target_item_id" not in payload:
        raise ValueError("STARec reward ground_truth must include target_item_id.")
    return payload


def _extract_ranked_item_ids(output: str) -> list[int]:
    ranked_item_ids: list[int] = []
    for line in strip_think_blocks(output).splitlines():
        if "ItemID:" not in line:
            continue
        try:
            item_text = line.split("ItemID:", 1)[1].split("]", 1)[0]
            ranked_item_ids.append(int(item_text.strip()))
        except (IndexError, ValueError):
            continue
    return ranked_item_ids


__all__ = [
    "compute_score",
    "starec_ranking_reward",
    "target_rank",
]
