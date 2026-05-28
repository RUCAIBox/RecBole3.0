from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.model import get_model_spec
from recbole3.model.starec.candidates import select_history_eligible_user_ids
from recbole3.model.starec.config import STARecConfig
from recbole3.model.starec.training_data import (
    SFT_REASONING_MODES,
    export_sft_from_teacher_trace,
    export_verl_ranking_from_teacher_trace,
    write_user_split_artifacts,
)
from recbole3.run import compose_config
from recbole3.utils import require_component_cfg, require_component_name


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STARec training-reproduction data utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split-users", help="Write disjoint teacher/eval user id artifacts.")
    split_parser.add_argument("--teacher-user-count", type=int, required=True)
    split_parser.add_argument("--heldout-eval-user-count", type=int, required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example dataset=amazon2014_retrieval")

    sft_parser = subparsers.add_parser("export-sft", help="Export LlamaFactory OpenAI-format SFT JSONL.")
    sft_parser.add_argument("--trace-path", required=True)
    sft_parser.add_argument("--output-path", required=True)
    sft_parser.add_argument("--rejected-path", default=None)
    sft_parser.add_argument("--dataset-info-path", default=None)
    sft_parser.add_argument("--dataset-name", default="starec_sft")
    sft_parser.add_argument("--rank-threshold", type=int, default=5)
    sft_parser.add_argument("--max-description-words", type=int, default=120)
    sft_parser.add_argument("--sft-reasoning-mode", choices=tuple(sorted(SFT_REASONING_MODES)), default="answer-only")

    rl_parser = subparsers.add_parser("export-rl", help="Export VeRL ranking prompt JSONL with reward ground truth.")
    rl_parser.add_argument("--trace-path", required=True)
    rl_parser.add_argument("--output-path", required=True)
    rl_parser.add_argument("--rejected-path", default=None)
    rl_parser.add_argument(
        "--rank-threshold",
        type=_optional_rank_threshold,
        default=None,
        metavar="N|none",
        help="Optional teacher target-rank filter for RL prompts. Defaults to none.",
    )
    rl_parser.add_argument("--data-source", default="starec_ranking")

    return parser.parse_args(list(argv) if argv is not None else None)


def _optional_rank_threshold(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "off", "false", "-1"}:
        return None
    try:
        threshold = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rank-threshold must be an integer or 'none'.") from exc
    if threshold < 1:
        raise argparse.ArgumentTypeError("--rank-threshold must be >= 1, or 'none'.")
    return threshold


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    if args.command == "split-users":
        result = _split_users(args)
    elif args.command == "export-sft":
        result = export_sft_from_teacher_trace(
            args.trace_path,
            args.output_path,
            rejected_path=args.rejected_path,
            dataset_info_path=args.dataset_info_path,
            dataset_name=args.dataset_name,
            rank_threshold=args.rank_threshold,
            max_description_words=args.max_description_words,
            sft_reasoning_mode=args.sft_reasoning_mode,
        )
    elif args.command == "export-rl":
        result = export_verl_ranking_from_teacher_trace(
            args.trace_path,
            args.output_path,
            rejected_path=args.rejected_path,
            rank_threshold=args.rank_threshold,
            data_source=args.data_source,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _split_users(args: argparse.Namespace) -> dict:
    overrides = ["model=starec", *list(args.overrides)]
    cfg = compose_config(overrides=overrides)
    runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
    dataset_cfg = require_component_cfg(cfg, "dataset")
    model_cfg = require_component_cfg(cfg, "model")
    trainer_cfg = require_component_cfg(cfg, "trainer")

    dataset_name = require_component_name(dataset_cfg, "dataset")
    dataset_spec = get_dataset_spec(dataset_name)
    model_spec = get_model_spec("starec")
    dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
    trainer = model_spec.trainer_cls(instantiate_dataclass(model_spec.trainer_config_cls, trainer_cfg))
    model_config = instantiate_dataclass(STARecConfig, model_cfg)
    model_config = replace(model_config, selected_user_count=-1, selected_user_ids_path=None)

    pipeline = model_spec.pipeline_cls(cfg=cfg, model_spec=model_spec)
    with pipeline._accelerate_runtime_device(runtime_cfg.device):
        stage_start = time.perf_counter()
        print("[starec:split-users] preparing task dataset", flush=True)
        task_data = dataset.prepare(eval_config=trainer.config.eval)
        print(f"[starec:split-users] task dataset ready in {time.perf_counter() - stage_start:.2f}s", flush=True)
        stage_start = time.perf_counter()
        print("[starec:split-users] selecting eligible users", flush=True)
        eligible_user_ids = select_history_eligible_user_ids(task_data, model_config=model_config)
        print(
            "[starec:split-users] eligible user selection ready "
            f"in {time.perf_counter() - stage_start:.2f}s users={len(eligible_user_ids)}",
            flush=True,
        )

    result = write_user_split_artifacts(
        eligible_user_ids,
        teacher_user_count=args.teacher_user_count,
        heldout_eval_user_count=args.heldout_eval_user_count,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(f"[starec:split-users] wrote user artifacts to {args.output_dir}", flush=True)
    return result


if __name__ == "__main__":
    main()
