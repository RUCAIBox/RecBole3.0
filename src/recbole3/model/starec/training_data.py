from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LLAMAFACTORY_DATASET_INFO_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
}
SFT_REASONING_MODES = {"answer-only", "think-tags"}


def write_user_split_artifacts(
    eligible_user_ids: Iterable[int],
    *,
    teacher_user_count: int,
    heldout_eval_user_count: int,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write deterministic, disjoint teacher/eval user artifacts."""

    teacher_user_count = int(teacher_user_count)
    heldout_eval_user_count = int(heldout_eval_user_count)
    if teacher_user_count < 0 or heldout_eval_user_count < 0:
        raise ValueError("teacher_user_count and heldout_eval_user_count must be >= 0.")
    if teacher_user_count == 0 and heldout_eval_user_count == 0:
        raise ValueError("At least one of teacher_user_count or heldout_eval_user_count must be positive.")

    unique_user_ids = _dedupe_preserving_order(int(user_id) for user_id in eligible_user_ids)
    required = teacher_user_count + heldout_eval_user_count
    if len(unique_user_ids) < required:
        raise ValueError(
            f"Requested {required} users, but only {len(unique_user_ids)} eligible users are available."
        )

    shuffled_user_ids = list(unique_user_ids)
    random.Random(int(seed)).shuffle(shuffled_user_ids)
    teacher_user_ids = tuple(sorted(shuffled_user_ids[:teacher_user_count]))
    heldout_eval_user_ids = tuple(sorted(shuffled_user_ids[teacher_user_count:required]))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    teacher_path = output_root / "teacher_users.jsonl"
    heldout_path = output_root / "heldout_eval_users.jsonl"
    manifest_path = output_root / "user_split_manifest.json"
    _write_user_ids(teacher_path, teacher_user_ids)
    _write_user_ids(heldout_path, heldout_eval_user_ids)

    manifest = {
        "seed": int(seed),
        "eligible_user_count": len(unique_user_ids),
        "teacher_user_count": len(teacher_user_ids),
        "heldout_eval_user_count": len(heldout_eval_user_ids),
        "teacher_users_path": str(teacher_path),
        "heldout_eval_users_path": str(heldout_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        **manifest,
        "teacher_user_ids": list(teacher_user_ids),
        "heldout_eval_user_ids": list(heldout_eval_user_ids),
        "manifest_path": str(manifest_path),
    }


def export_sft_from_teacher_trace(
    trace_path: str | Path,
    output_path: str | Path,
    *,
    rejected_path: str | Path | None = None,
    dataset_info_path: str | Path | None = None,
    dataset_name: str = "starec_sft",
    rank_threshold: int = 5,
    max_description_words: int = 120,
    allow_noop_reflection: bool = False,
    sft_reasoning_mode: str = "answer-only",
) -> dict[str, Any]:
    """Export LlamaFactory OpenAI-format SFT records from STARec teacher traces."""

    sft_reasoning_mode = str(sft_reasoning_mode)
    if sft_reasoning_mode not in SFT_REASONING_MODES:
        raise ValueError(f"sft_reasoning_mode must be one of {sorted(SFT_REASONING_MODES)}.")

    trace_records = read_jsonl(trace_path)
    rankings_by_user = _rankings_by_user(trace_records)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for record in trace_records:
        turn_type = str(record.get("turn_type", ""))
        if turn_type == "ranking":
            reason = _ranking_rejection_reason(record, rank_threshold=rank_threshold)
            if reason is None:
                reason = _sft_reasoning_rejection_reason(record, sft_reasoning_mode=sft_reasoning_mode)
            if reason is None:
                accepted.append(_sft_record(record, sft_reasoning_mode=sft_reasoning_mode))
            else:
                rejected.append(_reject_record(record, reason=reason))
            continue

        if turn_type == "init_memory":
            reason = _memory_rejection_reason(
                record,
                rankings_by_user=rankings_by_user,
                rank_threshold=rank_threshold,
                max_description_words=max_description_words,
                description_key="current_user_description",
            )
            if reason is None:
                reason = _sft_reasoning_rejection_reason(record, sft_reasoning_mode=sft_reasoning_mode)
            if reason is None:
                accepted.append(
                    _sft_record(
                        record,
                        probe=_next_ranking_probe(record, rankings_by_user),
                        sft_reasoning_mode=sft_reasoning_mode,
                    )
                )
            else:
                rejected.append(_reject_record(record, reason=reason))
            continue

        if turn_type == "reflection":
            reason = _memory_rejection_reason(
                record,
                rankings_by_user=rankings_by_user,
                rank_threshold=rank_threshold,
                max_description_words=max_description_words,
                description_key="updated_user_description",
                previous_description_key="previous_user_description",
                allow_noop_reflection=allow_noop_reflection,
            )
            if reason is None:
                reason = _sft_reasoning_rejection_reason(record, sft_reasoning_mode=sft_reasoning_mode)
            if reason is None:
                accepted.append(
                    _sft_record(
                        record,
                        probe=_next_ranking_probe(record, rankings_by_user),
                        sft_reasoning_mode=sft_reasoning_mode,
                    )
                )
            else:
                rejected.append(_reject_record(record, reason=reason))

    output_path = Path(output_path)
    write_jsonl(output_path, accepted)
    if rejected_path is not None:
        write_jsonl(rejected_path, rejected)
    if dataset_info_path is not None:
        write_llamafactory_dataset_info(
            dataset_info_path,
            dataset_name=dataset_name,
            file_name=output_path.name,
        )

    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output_path": str(output_path),
        "rejected_path": str(rejected_path) if rejected_path is not None else None,
        "dataset_info_path": str(dataset_info_path) if dataset_info_path is not None else None,
        "sft_reasoning_mode": sft_reasoning_mode,
    }


def export_verl_ranking_from_teacher_trace(
    trace_path: str | Path,
    output_path: str | Path,
    *,
    rejected_path: str | Path | None = None,
    rank_threshold: int | None = None,
    data_source: str = "starec_ranking",
) -> dict[str, Any]:
    """Export ranking prompts with JSON ground-truth payloads for VeRL."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in read_jsonl(trace_path):
        if record.get("turn_type") != "ranking":
            continue
        reason = None
        if rank_threshold is not None:
            reason = _ranking_rejection_reason(record, rank_threshold=int(rank_threshold))
        elif not bool(record.get("parse_valid")):
            reason = "ranking_parse_invalid"
        if reason is not None:
            rejected.append(_reject_record(record, reason=reason))
            continue

        ground_truth = {
            "target_item_id": int(record["target_item_id"]),
            "candidate_item_ids": [int(item_id) for item_id in record.get("candidate_item_ids", [])],
            "topk": 20,
        }
        accepted.append(
            {
                "data_source": data_source,
                "prompt": list(record.get("messages", [])),
                "reward_model": {"ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
                "extra_info": {
                    "source_trace_id": record.get("trace_id"),
                    "user_id": int(record["user_id"]),
                    "sequence_index": int(record["sequence_index"]),
                },
            }
        )

    output_path = Path(output_path)
    write_jsonl(output_path, accepted)
    if rejected_path is not None:
        write_jsonl(rejected_path, rejected)
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output_path": str(output_path),
        "rejected_path": str(rejected_path) if rejected_path is not None else None,
        "rank_threshold": rank_threshold,
    }


def write_llamafactory_dataset_info(path: str | Path, *, dataset_name: str, file_name: str) -> None:
    payload = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": LLAMAFACTORY_DATASET_INFO_TAGS,
        }
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sft_record(
    record: dict[str, Any],
    probe: dict[str, Any] | None = None,
    *,
    sft_reasoning_mode: str = "answer-only",
) -> dict[str, Any]:
    result = {
        "messages": [
            *list(record.get("messages", [])),
            {"role": "assistant", "content": _assistant_content(record, sft_reasoning_mode=sft_reasoning_mode)},
        ],
        "source_trace_id": record.get("trace_id"),
        "turn_type": record.get("turn_type"),
    }
    if probe is not None:
        result["probe_trace_id"] = probe.get("trace_id")
        result["probe_target_rank"] = probe.get("target_rank")
    return result


def _assistant_content(record: dict[str, Any], *, sft_reasoning_mode: str) -> str:
    raw_output = str(record.get("raw_output") or "")
    if sft_reasoning_mode == "answer-only":
        return raw_output
    if sft_reasoning_mode == "think-tags":
        reasoning_content = _reasoning_content(record)
        if not reasoning_content:
            raise ValueError("think-tags SFT records require reasoning_content.")
        return f"<think>\n{reasoning_content}\n</think>\n\n{raw_output}"
    raise ValueError(f"sft_reasoning_mode must be one of {sorted(SFT_REASONING_MODES)}.")


def _rankings_by_user(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    rankings: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("turn_type") != "ranking":
            continue
        rankings[int(record["user_id"])].append(record)
    for user_rankings in rankings.values():
        user_rankings.sort(key=lambda item: int(item["sequence_index"]))
    return rankings


def _next_ranking_probe(
    record: dict[str, Any],
    rankings_by_user: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    user_rankings = rankings_by_user.get(int(record["user_id"]), [])
    sequence_index = int(record.get("sequence_index", -1))
    for ranking in user_rankings:
        if int(ranking["sequence_index"]) > sequence_index:
            return ranking
    return None


def _ranking_rejection_reason(record: dict[str, Any], *, rank_threshold: int) -> str | None:
    if not bool(record.get("parse_valid")):
        return "ranking_parse_invalid"
    target_rank = record.get("target_rank")
    if target_rank is None:
        return "target_rank_missing"
    if int(target_rank) > int(rank_threshold):
        return f"target_rank>{int(rank_threshold)}"
    return None


def _sft_reasoning_rejection_reason(record: dict[str, Any], *, sft_reasoning_mode: str) -> str | None:
    if sft_reasoning_mode != "think-tags":
        return None
    if not _reasoning_content(record):
        return "reasoning_content_missing"
    return None


def _reasoning_content(record: dict[str, Any]) -> str:
    return str(record.get("reasoning_content") or "").strip()


def _memory_rejection_reason(
    record: dict[str, Any],
    *,
    rankings_by_user: dict[int, list[dict[str, Any]]],
    rank_threshold: int,
    max_description_words: int,
    description_key: str,
    previous_description_key: str | None = None,
    allow_noop_reflection: bool = False,
) -> str | None:
    description = str(record.get(description_key) or "").strip()
    if not description:
        return "description_empty"
    if _word_count(description) > int(max_description_words):
        return f"description_words>{int(max_description_words)}"
    if previous_description_key is not None and not allow_noop_reflection:
        previous = str(record.get(previous_description_key) or "").strip()
        if previous and description == previous:
            return "reflection_noop"
    probe = _next_ranking_probe(record, rankings_by_user)
    if probe is None:
        return "next_ranking_probe_missing"
    probe_reason = _ranking_rejection_reason(probe, rank_threshold=rank_threshold)
    if probe_reason is not None:
        return f"next_ranking_probe_{probe_reason}"
    return None


def _reject_record(record: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "source_trace_id": record.get("trace_id"),
        "turn_type": record.get("turn_type"),
        "user_id": record.get("user_id"),
        "sequence_index": record.get("sequence_index"),
        "reason": reason,
    }


def _write_user_ids(path: Path, user_ids: Iterable[int]) -> None:
    write_jsonl(path, ({"user_id": int(user_id)} for user_id in user_ids))


def _dedupe_preserving_order(user_ids: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for user_id in user_ids:
        normalized = int(user_id)
        if normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


def _word_count(text: str) -> int:
    return len(str(text).split())


__all__ = [
    "SFT_REASONING_MODES",
    "export_sft_from_teacher_trace",
    "export_verl_ranking_from_teacher_trace",
    "read_jsonl",
    "write_jsonl",
    "write_llamafactory_dataset_info",
    "write_user_split_artifacts",
]
