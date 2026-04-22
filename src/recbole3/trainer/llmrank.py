from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recbole3.dataset import BaseTaskDataset
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import BaseModel
from recbole3.trainer.base import OptimizerConfig, Trainer, TrainerConfig


@dataclass(slots=True)
class LLMRankTrainerConfig(TrainerConfig):
    """Inference-only trainer config for prompt-based LLM reranking."""

    name: str = field(default="llmrank", metadata={"help": "Registered trainer name."})
    batch_size: int = field(default=8, metadata={"help": "Batch size used during evaluation-only reranking."})
    shuffle: bool = field(default=False, metadata={"help": "LLM reranking keeps evaluation batches deterministic."})
    max_epochs: int = field(default=0, metadata={"help": "LLM reranking does not perform gradient-based training."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="sampled",
            metrics=(MetricSpec(name="ndcg", ks=(10,)), MetricSpec(name="recall", ks=(10,))),
            neg_sampling_num=19,
            candidate_seed=42,
        ),
        metadata={"help": "Evaluation configuration used by the inference-only LLM reranker."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig,
        metadata={"help": "Unused placeholder optimizer config kept for compatibility with TrainerConfig."},
    )


class LLMRankTrainer(Trainer):
    """Trainer variant that skips optimization and only evaluates the reranker."""

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset[Any, Any],
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        accelerator = self.create_accelerator()
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        valid_history: list[dict[str, Any]] = []
        best_value = None
        monitor = None
        if len(prepared_data.get_eval_dataset("valid")) > 0:
            valid_result = self._run_evaluation(
                model,
                prepared_data,
                split="valid",
                accelerator=accelerator,
                model_is_prepared=False,
            )
            valid_result["epoch"] = 0
            valid_history.append(valid_result)
            monitor = self._resolve_monitor(prepared_data)
            best_value = None if monitor is None else self._extract_monitor_value(valid_result["metrics"], monitor.name)
        if checkpoint_paths["best"] is not None:
            self._save_model_checkpoint(model, accelerator, checkpoint_paths["best"])
        if checkpoint_paths["last"] is not None:
            self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])

        return {
            "train_history": [],
            "valid_history": valid_history,
            "data_stats": self._build_result_data_stats(prepared_data),
            "stopped_early": False,
            "best_epoch": 0 if best_value is not None else None,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {key: (str(path) if path is not None else None) for key, path in checkpoint_paths.items()},
        }


__all__ = [
    "LLMRankTrainer",
    "LLMRankTrainerConfig",
]
