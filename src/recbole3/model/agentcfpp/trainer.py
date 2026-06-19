from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recbole3.dataset.base import BaseTaskDataset
from recbole3.dataset.utils import CANDIDATE_ITEM_IDS
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MetricSpec, RetrievalEvalData
from recbole3.model.base import BaseModel, BaseRetrievalModel
from recbole3.trainer import Trainer
from recbole3.trainer_config import OptimizerConfig, TrainerConfig


@dataclass(slots=True)
class AgentCFPPTrainerConfig(TrainerConfig):
    """Trainer config for AgentCF++'s LLM-based training loop."""

    batch_size: int = field(default=20, metadata={"help": "Batch size for training (small due to API calls)."})
    shuffle: bool = field(default=False, metadata={"help": "Whether to shuffle training data."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Keep single-process for API-heavy workload."})
    max_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="full",
            metrics=(
                MetricSpec(name="ndcg", ks=(1, 5, 10)),
                MetricSpec(name="recall", ks=(1, 5, 10)),
            ),
            neg_sampling_num=0,
            candidate_seed=42,
        ),
        metadata={"help": "Evaluation configuration (MRR is added on top by the trainer)."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig,
        metadata={"help": "Unused placeholder (AgentCF++ has no gradient training)."},
    )
    mrr_ks: tuple[int, ...] = field(
        default=(1, 5, 10),
        metadata={"help": "Cutoffs for the package-local MRR metric."},
    )


def _compute_mrr(eval_data: RetrievalEvalData, ks: tuple[int, ...]) -> dict[str, float]:
    """Mean reciprocal rank of the first relevant target within top-k.

    Computed inside the AgentCF++ package so the shared metric module is untouched.
    """
    pred = np.asarray(eval_data.pred_item_ids)
    targets = np.asarray(eval_data.target_item_ids)
    mask = np.asarray(eval_data.target_mask, dtype=bool)
    num_rows = pred.shape[0]
    if num_rows == 0:
        return {f"mrr@{k}": 0.0 for k in ks}

    results: dict[str, float] = {}
    for k in ks:
        topk = pred[:, :k]
        reciprocal = np.zeros(num_rows, dtype=np.float64)
        for row in range(num_rows):
            row_targets = set(int(t) for t, m in zip(targets[row], mask[row]) if m)
            if not row_targets:
                continue
            for rank, item in enumerate(topk[row], start=1):
                if int(item) in row_targets:
                    reciprocal[row] = 1.0 / rank
                    break
        results[f"mrr@{k}"] = float(reciprocal.mean())
    return results


def _build_agentcfpp_evaluation_method(metric_specs: tuple[MetricSpec, ...]):
    from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod

    class AgentCFPPEvaluationMethod(BaseRetrievalEvaluationMethod):
        """Evaluate AgentCF++ using LLM-based ranking on per-user candidate subsets."""

        protocol = "full"

        def _collect_retrieval_batch(
            self,
            model: BaseModel,
            model_inputs: Any,
            records: Any,
            max_k: int,
        ) -> RetrievalEvalData:
            if not isinstance(model, BaseRetrievalModel):
                raise TypeError("AgentCF++ evaluation requires BaseRetrievalModel.")

            if isinstance(records, pd.DataFrame):
                missing = CANDIDATE_ITEM_IDS not in records.columns or records[CANDIDATE_ITEM_IDS].isna().any()
            else:
                missing = any(self._record_value(r, CANDIDATE_ITEM_IDS) is None for r in records)
            if missing:
                raise TypeError("AgentCF++ evaluation requires candidate_item_ids in every eval row.")

            device = self._infer_device(model_inputs)
            target_item_ids, target_mask = self._single_target_tensors(records, device=device)

            if len(records) == 0:
                pred_item_ids = np.empty((0, max(0, max_k)), dtype=np.int64)
            elif max_k <= 0:
                pred_item_ids = np.empty((len(records), 0), dtype=np.int64)
            else:
                candidate_item_ids, _ = self._pad_int_lists(records, CANDIDATE_ITEM_IDS, device=device)
                pred = model.predict(model_inputs, k=max_k, candidate_item_ids=candidate_item_ids)
                if pred.ndim != 2 or tuple(pred.shape) != (len(records), max_k):
                    raise ValueError(
                        f"predict() must return [batch, k]. Got {tuple(pred.shape)} for {(len(records), max_k)}."
                    )
                pred_item_ids = self._to_numpy(pred.to(dtype=torch.long))

            return RetrievalEvalData(
                pred_item_ids=pred_item_ids,
                target_item_ids=self._to_numpy(target_item_ids),
                target_mask=self._to_numpy(target_mask),
            )

    return AgentCFPPEvaluationMethod(metric_specs=metric_specs)


class AgentCFPPTrainer(Trainer):
    """Custom trainer for AgentCF++'s non-gradient LLM-based training loop."""

    config_cls = AgentCFPPTrainerConfig

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        from recbole3.model.agentcfpp.model import AgentCFPPModel

        if not isinstance(model, AgentCFPPModel):
            raise TypeError("AgentCFPPTrainer requires an AgentCFPPModel.")

        model.ensure_initialized(prepared_data)
        collator = model.build_train_collator(prepared_data)
        train_dataset = prepared_data.get_train_dataset()
        train_dataloader = self.build_dataloader(train_dataset, collator, shuffle=self.config.shuffle)

        train_history: list[dict[str, Any]] = []
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.perf_counter()
            accuracies: list[float] = []
            print(f"[agentcfpp:train] epoch={epoch}/{self.config.max_epochs}")
            for batch_idx, batch in enumerate(train_dataloader):
                step_result = model.train_step(batch)
                accuracies.append(step_result.get("accuracy", 0.0))
                print(f"[agentcfpp:train] batch={batch_idx + 1} accuracy={step_result.get('accuracy', 0.0):.4f}")
            elapsed = time.perf_counter() - epoch_start
            avg_accuracy = sum(accuracies) / max(len(accuracies), 1)
            train_history.append(
                {"epoch": epoch, "avg_accuracy": avg_accuracy, "num_batches": len(accuracies), "elapsed_seconds": elapsed}
            )
            print(f"[agentcfpp:train] epoch={epoch} avg_accuracy={avg_accuracy:.4f} elapsed={elapsed:.1f}s")

        if model.config.save_agent_state and output_dir:
            save_path = Path(output_dir) / "agent_states"
            model.save_agent_states(save_path)
            print(f"[agentcfpp:train] agent states saved to {save_path}")

        return {
            "train_history": train_history,
            "valid_history": [],
            "data_stats": self._build_result_data_stats(prepared_data),
            "stopped_early": False,
            "best_epoch": None,
            "best_metric": None,
            "checkpoint_paths": {"best": None, "last": None},
        }

    def evaluate(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        split: str = "valid",
    ) -> dict[str, Any]:
        model.ensure_initialized(prepared_data)
        eval_method = _build_agentcfpp_evaluation_method(self.config.eval.metrics)
        collator = model.build_eval_collator(prepared_data)
        eval_dataset = prepared_data.get_eval_dataset(split)
        eval_dataloader = self.build_dataloader(eval_dataset, collator, shuffle=False)

        all_eval_data: list[RetrievalEvalData] = []
        max_k = max((k for spec in self.config.eval.metrics for k in spec.ks), default=0)
        max_k = max(max_k, max(self.config.mrr_ks, default=0))
        print(f"[agentcfpp:eval:{split}] starting evaluation (max_k={max_k})")

        for batch_idx, batch in enumerate(eval_dataloader):
            records = batch.get("records", pd.DataFrame())
            eval_data = eval_method._collect_retrieval_batch(model, batch, records, max_k)
            all_eval_data.append(eval_data)
            if (batch_idx + 1) % 5 == 0:
                print(f"[agentcfpp:eval:{split}] processed {batch_idx + 1} batches")

        if not all_eval_data:
            return {"metrics": {}}

        merged = RetrievalEvalData(
            pred_item_ids=np.concatenate([d.pred_item_ids for d in all_eval_data], axis=0),
            target_item_ids=np.concatenate([d.target_item_ids for d in all_eval_data], axis=0),
            target_mask=np.concatenate([d.target_mask for d in all_eval_data], axis=0),
        )

        metrics = eval_method.compute_metrics(all_eval_data)
        metrics.update(_compute_mrr(merged, self.config.mrr_ks))
        print(f"[agentcfpp:eval:{split}] metrics={metrics}")
        return {"metrics": metrics}

    def run(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        self._setup_logger(model, prepared_data, output_dir)

        total_start = time.perf_counter()
        fit_result = self.fit(model, prepared_data, output_dir=output_dir)

        # Build group memory between training and evaluation (depends on trained memories).
        if getattr(model.config, "use_group_memory", False):
            from recbole3.model.agentcfpp.group_memory import build_group_state

            print("[agentcfpp:trainer] building group memory")
            group_state = build_group_state(model, prepared_data, model.config)
            model.set_group_state(group_state)

        print("[agentcfpp:trainer] starting validation evaluation")
        valid_result = self.evaluate(model, prepared_data, split="valid")
        print("[agentcfpp:trainer] starting test evaluation")
        test_result = self.evaluate(model, prepared_data, split="test")

        total_elapsed = time.perf_counter() - total_start
        print(f"[agentcfpp:trainer] total time: {total_elapsed:.1f}s")

        if (logger := getattr(self, "_logger", None)) is not None:
            logger.log_test(test_result)
            logger.log_summary(
                stopped_early=False,
                total_epochs=self.config.max_epochs,
                best_epoch=None,
                total_time=total_elapsed,
            )

        return {"fit": fit_result, "valid": valid_result, "test": test_result}

    @staticmethod
    def _build_result_data_stats(prepared_data: BaseTaskDataset) -> dict[str, int]:
        return {
            "num_users": prepared_data.get_num_users(),
            "num_items": prepared_data.get_num_items(),
            "num_train": len(prepared_data.get_train_dataset()),
            "num_valid": len(prepared_data.get_eval_dataset("valid")),
            "num_test": len(prepared_data.get_eval_dataset("test")),
        }


__all__ = ["AgentCFPPTrainer", "AgentCFPPTrainerConfig"]
