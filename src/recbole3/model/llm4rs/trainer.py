from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MetricSpec
from recbole3.model.base import BaseModel
from recbole3.model.llm4rs.model import LLM4RSModel, LLM4RSOutcome, LLM4RS_RECORD_INDEX
from recbole3.trainer import Trainer
from recbole3.trainer_config import OptimizerConfig, TrainerConfig


@dataclass(slots=True)
class LLM4RSTrainerConfig(TrainerConfig):
    """Evaluation-only trainer settings for the original LLM4RS protocol."""

    batch_size: int = field(default=32, metadata={"help": "Number of candidate rows evaluated per batch."})
    shuffle: bool = field(default=False, metadata={"help": "LLM4RS evaluates in stable record order."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Prompt generation is kept in the main process."})
    max_epochs: int = field(default=0, metadata={"help": "LLM4RS is inference-only."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="full",
            metrics=(
                MetricSpec(name="ndcg", ks=(1, 3, 5)),
                MetricSpec(name="recall", ks=(1, 3, 5)),
                MetricSpec(name="mrr", ks=(1, 3, 5)),
                MetricSpec(name="precision", ks=(1, 3, 5)),
                MetricSpec(name="map", ks=(1, 3, 5)),
            ),
            neg_sampling_num=0,
            candidate_seed=2023,
        ),
        metadata={"help": "Candidate-ranking evaluation metrics; mrr, precision, and map are supported here."},
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig, metadata={"help": "Unused compatibility field."})


class LLM4RSTrainer(Trainer):
    """Runs official LLM4RS inference and preserves its tie-aware metric semantics."""

    config_cls = LLM4RSTrainerConfig

    def run(
        self,
        model: BaseModel,
        prepared_data: Any,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if not isinstance(model, LLM4RSModel):
            raise TypeError(f"LLM4RSTrainer requires LLM4RSModel, got {type(model).__name__}.")
        self._setup_logger(model, prepared_data, output_dir)
        total_start = time.perf_counter()
        try:
            valid_result = None
            if len(prepared_data.get_eval_dataset("valid")) > 0:
                valid_result = self.evaluate(model, prepared_data, split="valid")
                if (logger := getattr(self, "_logger", None)) is not None:
                    logger.log_validation(epoch=0, metrics=valid_result["metrics"])
            test_result = self.evaluate(model, prepared_data, split="test")
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                logger.log_summary(
                    stopped_early=False,
                    total_epochs=0,
                    best_epoch=None,
                    total_time=time.perf_counter() - total_start,
                )
            return {"valid": valid_result, "test": test_result}
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def evaluate(
        self,
        model: BaseModel,
        prepared_data: Any,
        split: Literal["valid", "test"] = "valid",
    ) -> dict[str, Any]:
        if not isinstance(model, LLM4RSModel):
            raise TypeError("LLM4RS evaluation requires LLM4RSModel.")
        source_dataset = prepared_data.get_eval_dataset(split)
        if not isinstance(source_dataset, FrameDataset):
            raise TypeError(f"LLM4RS evaluation requires FrameDataset, got {type(source_dataset).__name__}.")
        source_frame = source_dataset.frame.copy()
        if CANDIDATE_ITEM_IDS not in source_frame.columns:
            raise TypeError("LLM4RS evaluation requires candidate_item_ids in each evaluation row.")

        input_collator = model.build_eval_collator(prepared_data)
        model.configure_examples(source_frame)
        eval_frame = source_frame.iloc[int(model.config.begin_index) : model.config.end_index].reset_index(drop=True)
        eval_frame[LLM4RS_RECORD_INDEX] = list(range(len(eval_frame)))
        eval_dataset = FrameDataset(eval_frame)
        dataloader = self.build_dataloader(eval_dataset, input_collator, shuffle=False)
        total_records = len(eval_frame)
        total_batches = len(dataloader)
        print(
            f"[llm4rs:eval:{split}] {total_records} records, "
            f"{total_batches} batches (batch_size={int(self.config.batch_size)})"
        )

        outcomes: list[LLM4RSOutcome] = []
        target_item_ids: list[int] = []
        num_batches = 0
        progress_bar = self._create_progress_bar(dataloader, split=f"llm4rs:{split}")
        for model_inputs in progress_bar:
            candidate_batches = model_inputs["candidate_item_ids"]
            batch_target_item_ids = model_inputs["target_item_ids"]
            batch_record_indices = model_inputs["record_indices"]
            outcomes.extend(
                model.rank_candidate_batches(
                    model_inputs["history_texts"],
                    candidate_batches,
                    target_item_ids=batch_target_item_ids,
                    record_indices=batch_record_indices,
                )
            )
            target_item_ids.extend(batch_target_item_ids)
            num_batches += 1
            if hasattr(progress_bar, "set_postfix_str"):
                failed_so_far = sum(
                    1
                    for outcome, target in zip(outcomes, target_item_ids, strict=True)
                    if not outcome.target_ranks(target)
                )
                progress_bar.set_postfix_str(
                    f"records={len(target_item_ids)}/{total_records} failed={failed_so_far}"
                )
        if hasattr(progress_bar, "close"):
            progress_bar.close()

        target_rank_rows = [
            outcome.target_ranks(target_item_id)
            for outcome, target_item_id in zip(outcomes, target_item_ids, strict=True)
        ]
        failed_records = sum(1 for ranks in target_rank_rows if not ranks)
        metrics = self._compute_metrics(
            target_rank_rows,
            ranking_policy=str(model.config.ranking_policy),
            candidate_num=int(model.config.candidate_num),
        )
        result: dict[str, Any] = {
            "split": split,
            "protocol": "llm4rs_candidate",
            "loss": None,
            "metrics": metrics,
            "num_batches": num_batches,
            "evaluated_records": len(target_rank_rows),
            "failed_records": failed_records,
            "reserved_prefix_records": min(int(model.config.begin_index), len(source_frame)),
            "requested_prompts": sum(len(outcome.responses) for outcome in outcomes),
            "failed_subrequests": sum(int(outcome.failed_subrequests) for outcome in outcomes),
            "data_stats": self._build_result_data_stats(prepared_data),
        }
        if self.config.save_inference_results:
            result["inference_results"] = {
                "target_ranks": [list(ranks) for ranks in target_rank_rows],
                "pred_item_ids": [list(outcome.ordered_item_ids()) for outcome in outcomes],
            }
        return result

    def _compute_metrics(
        self,
        target_rank_rows: list[tuple[int, ...]],
        *,
        ranking_policy: str = "list",
        candidate_num: int | None = None,
    ) -> dict[str, float]:
        results: dict[str, float] = {}
        for metric_spec in self.config.eval.metrics:
            name = str(metric_spec.name).strip().lower()
            if name not in {"ndcg", "recall", "mrr", "precision", "map"}:
                raise ValueError(f"Unsupported LLM4RS metric '{metric_spec.name}'.")
            if name == "map" and str(ranking_policy).strip().lower() != "list":
                continue
            for k in metric_spec.ks:
                if candidate_num is not None and int(k) > int(candidate_num):
                    continue
                key = f"{name}@{int(k)}"
                if not target_rank_rows:
                    results[key] = 0.0
                    continue
                per_record = [
                    (
                        sum(_metric_for_target_rank(name, rank, int(k)) for rank in ranks) / len(ranks)
                        if ranks
                        else 0.0
                    )
                    for ranks in target_rank_rows
                ]
                results[key] = float(sum(per_record) / len(per_record))
        return results


def _metric_for_target_rank(name: str, rank: int, k: int) -> float:
    if int(rank) >= int(k):
        return 0.0
    if name == "recall":
        return 1.0
    if name == "precision":
        return 1.0 / float(k)
    if name in {"mrr", "map"}:
        return 1.0 / float(int(rank) + 1)
    return 1.0 / math.log2(float(int(rank) + 2))


__all__ = [
    "LLM4RSTrainer",
    "LLM4RSTrainerConfig",
]
