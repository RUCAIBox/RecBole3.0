from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recbole3.dataset import BaseTaskDataset
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MetricSpec
from recbole3.model.base import BaseModel
from recbole3.trainer import Trainer
from recbole3.trainer_config import OptimizerConfig, TrainerConfig


@dataclass(slots=True)
class LLMRankTrainerConfig(TrainerConfig):
    """Inference-only trainer config for prompt-based LLM reranking."""

    batch_size: int = field(default=256, metadata={"help": "Batch size used during evaluation-only reranking."})
    shuffle: bool = field(default=False, metadata={"help": "LLM reranking keeps evaluation batches deterministic."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Keep dataloaders single-process; local text-heavy reranking is faster without worker fan-out."},
    )
    max_epochs: int = field(default=0, metadata={"help": "LLM reranking does not perform gradient-based training."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="sampled",
            metrics=(MetricSpec(name="ndcg", ks=(10,)), MetricSpec(name="recall", ks=(10,))),
            neg_sampling_num=99,
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

    config_cls = LLMRankTrainerConfig

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        accelerator = self.create_accelerator()
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        valid_history: list[dict[str, Any]] = []
        best_value = None
        monitor = None
        if len(prepared_data.get_eval_dataset("valid")) > 0:
            print("[llmrank] starting validation evaluation")
            valid_result = self._run_evaluation(
                model,
                prepared_data,
                split="valid",
                accelerator=accelerator,
                model_is_prepared=False,
            )
            print("[llmrank] finished validation evaluation")
            valid_result["epoch"] = 0
            valid_history.append(valid_result)
            monitor_name = str(self.config.monitor or "").strip()
            if monitor_name:
                monitor = self._resolve_monitor(prepared_data)
                best_value = self._extract_monitor_value(valid_result["metrics"], monitor.name)
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
