from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recbole3.dataset import BaseTaskDataset
from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MetricSpec, RetrievalEvalData
from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod
from recbole3.model.base import BaseModel, BaseModelDataset, BaseRetrievalModel
from recbole3.trainer import Trainer
from recbole3.trainer_config import OptimizerConfig, TrainerConfig


@dataclass(slots=True)
class LLMRankTrainerConfig(TrainerConfig):
    """Inference-only trainer config for prompt-based LLM reranking."""

    batch_size: int = field(default=100, metadata={"help": "Batch size used during evaluation-only reranking."})
    shuffle: bool = field(default=False, metadata={"help": "LLM reranking keeps evaluation batches deterministic."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Keep dataloaders single-process; local text-heavy reranking is faster without worker fan-out."},
    )
    max_epochs: int = field(default=0, metadata={"help": "LLM reranking does not perform gradient-based training."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="full",
            metrics=(MetricSpec(name="ndcg", ks=(10,)), MetricSpec(name="recall", ks=(10,))),
            neg_sampling_num=0,
            candidate_seed=42,
        ),
        metadata={"help": "Evaluation configuration used by the inference-only LLM reranker."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig,
        metadata={"help": "Unused placeholder optimizer config kept for compatibility with TrainerConfig."},
    )


class LLMRankEvaluationMethod(BaseRetrievalEvaluationMethod):
    """Evaluate one official candidate subset per request while keeping full-protocol semantics."""

    protocol = "full"

    def _collect_retrieval_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Any,
        max_k: int,
    ) -> RetrievalEvalData:
        if not isinstance(model, BaseRetrievalModel):
            raise TypeError("LLMRank evaluation requires BaseRetrievalModel.")
        if isinstance(records, pd.DataFrame):
            missing_candidates = CANDIDATE_ITEM_IDS not in records.columns or records[CANDIDATE_ITEM_IDS].isna().any()
        else:
            missing_candidates = any(self._record_value(record, CANDIDATE_ITEM_IDS) is None for record in records)
        if missing_candidates:
            raise TypeError("LLMRank evaluation requires candidate_item_ids in every eval row.")

        device = self._infer_device(model_inputs)
        target_item_ids, target_mask = self._single_target_tensors(records, device=device)
        if len(records) == 0:
            pred_item_ids = torch.empty((0, max(0, max_k)), dtype=torch.long, device=device)
        elif max_k <= 0:
            pred_item_ids = torch.empty((len(records), 0), dtype=torch.long, device=device)
        else:
            candidate_item_ids, candidate_mask = self._pad_int_lists(records, CANDIDATE_ITEM_IDS, device=device)
            valid_candidate_count = torch.sum(candidate_mask, dim=1)
            if torch.any(valid_candidate_count != valid_candidate_count[0]):
                raise ValueError("LLMRank evaluation requires equal candidate counts in every row.")
            candidate_count = int(valid_candidate_count[0].item()) if len(valid_candidate_count) > 0 else 0
            if candidate_count < max_k:
                raise ValueError(
                    "LLMRank evaluation requires at least k candidates per row. "
                    f"Got k={max_k} with candidate count {candidate_count}."
                )
            pred_item_ids = model.predict(
                model_inputs,
                k=max_k,
                candidate_item_ids=candidate_item_ids,
            )
            if pred_item_ids.ndim != 2 or tuple(pred_item_ids.shape) != (len(records), max_k):
                raise ValueError(
                    "Retrieval predict() must return top-k item ids with shape [batch, k]. "
                    f"Got {tuple(pred_item_ids.shape)} for expected {(len(records), max_k)}."
                )
            pred_item_ids = pred_item_ids.to(dtype=torch.long)

        return RetrievalEvalData(
            pred_item_ids=self._to_numpy(pred_item_ids),
            target_item_ids=self._to_numpy(target_item_ids),
            target_mask=self._to_numpy(target_mask),
        )


class LLMRankTrainer(Trainer):
    """Trainer variant that skips optimization and only evaluates the reranker."""

    config_cls = LLMRankTrainerConfig

    def create_evaluation_method(self, prepared_data: BaseTaskDataset | None = None):
        if str(self.config.eval.protocol).strip().lower() == "full":
            return LLMRankEvaluationMethod(metric_specs=tuple(self.config.eval.metrics))
        return super().create_evaluation_method(prepared_data)

    def evaluate(self, model: BaseModel, prepared_data: BaseTaskDataset, split: str = "valid") -> dict[str, Any]:
        return super().evaluate(model, self._build_llmrank_prepared_data(prepared_data, model=model), split=split)

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        valid_history: list[dict[str, Any]] = []
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        best_value = None
        monitor = None
        prepared_eval_data = self._build_llmrank_prepared_data(prepared_data, model=model)
        if len(prepared_eval_data.get_eval_dataset("valid")) > 0:
            print("[llmrank] starting validation evaluation")
            valid_result = super().evaluate(model, prepared_eval_data, split="valid")
            print("[llmrank] finished validation evaluation")
            valid_result["epoch"] = 0
            valid_history.append(valid_result)
            monitor_name = str(self.config.monitor or "").strip()
            if monitor_name:
                monitor = self._resolve_monitor(prepared_eval_data)
                best_value = self._extract_monitor_value(valid_result["metrics"], monitor.name)

        if checkpoint_paths["best"] is not None:
            accelerator = self.create_accelerator()
            self._save_model_checkpoint(model, accelerator, checkpoint_paths["best"])
        if checkpoint_paths["last"] is not None:
            accelerator = self.create_accelerator()
            self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])

        return {
            "train_history": [],
            "valid_history": valid_history,
            "data_stats": self._build_result_data_stats(prepared_eval_data),
            "stopped_early": False,
            "best_epoch": 0 if best_value is not None else None,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {key: (str(path) if path is not None else None) for key, path in checkpoint_paths.items()},
        }

    def _build_llmrank_prepared_data(self, prepared_data: BaseTaskDataset, *, model: Any) -> BaseTaskDataset:
        cloned = type(prepared_data).__new__(type(prepared_data))
        BaseModelDataset._copy_task_dataset_state(cloned, prepared_data)
        for split in ("valid", "test"):
            eval_dataset = prepared_data.get_eval_dataset(split)
            if not isinstance(eval_dataset, FrameDataset):
                raise TypeError(f"LLMRank evaluation requires FrameDataset, got {type(eval_dataset).__name__}.")
            frame = eval_dataset.frame.copy()
            if CANDIDATE_ITEM_IDS not in frame.columns:
                raise TypeError("LLMRank evaluation requires candidate_item_ids in the eval frame.")
            frame[CANDIDATE_ITEM_IDS] = [
                self._build_official_candidate_row(
                    candidate_item_ids=candidate_item_ids,
                    target_item_id=int(target_item_id),
                    split=split,
                    row_index=row_index,
                    model=model,
                )
                for row_index, (candidate_item_ids, target_item_id) in enumerate(
                    zip(frame[CANDIDATE_ITEM_IDS].tolist(), frame[ITEM_ID].tolist(), strict=True)
                )
            ]
            if split == "valid":
                cloned._valid_dataset = FrameDataset(frame)
            else:
                cloned._test_dataset = FrameDataset(frame)
        return cloned

    @staticmethod
    def _build_official_candidate_row(
        *,
        candidate_item_ids: Any,
        target_item_id: int,
        split: str,
        row_index: int,
        model: Any,
    ) -> tuple[int, ...]:
        backbone_candidates = [int(item_id) for item_id in (candidate_item_ids or ())]
        recall_budget = int(model.config.recall_budget)
        has_gt = bool(model.config.has_gt)
        fix_pos = int(model.config.fix_pos)
        required = recall_budget - 1 if has_gt else recall_budget

        if has_gt and target_item_id in backbone_candidates:
            backbone_candidates.remove(target_item_id)
        if len(backbone_candidates) < required:
            raise ValueError(
                f"Backbone candidate row {row_index} for split '{split}' only has {len(backbone_candidates)} items, "
                f"but official recall_budget={recall_budget} requires {required} non-ground-truth candidates."
            )

        final_candidates = backbone_candidates[:required]
        if has_gt:
            if fix_pos == -1 or fix_pos == recall_budget - 1:
                final_candidates.append(target_item_id)
            elif fix_pos == 0:
                final_candidates = [target_item_id, *final_candidates]
            else:
                if not 0 <= fix_pos < recall_budget:
                    raise ValueError(f"fix_pos={fix_pos} is out of range for recall_budget={recall_budget}.")
                final_candidates = [
                    *final_candidates[:fix_pos],
                    target_item_id,
                    *final_candidates[fix_pos:],
                ]

        if bool(model.config.shuffle):
            split_offset = 0 if split == "valid" else 10_000
            rng = np.random.default_rng(int(model.config.candidate_seed) + split_offset + int(row_index))
            rng.shuffle(final_candidates)

        return tuple(int(item_id) for item_id in final_candidates)


__all__ = [
    "LLMRankEvaluationMethod",
    "LLMRankTrainer",
    "LLMRankTrainerConfig",
]
