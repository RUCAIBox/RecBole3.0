from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from recbole3.dataset import BaseTaskDataset
from recbole3.model.base import BaseModel
from recbole3.trainer_config import TrainerConfig


class E4SRecTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def run(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.ensure_initialized(prepared_data)
        model_cfg = model.config

        stage = str(getattr(model_cfg, "pipeline_stage", "training") or "training")
        if stage == "evaluation":
            return self._run_evaluation_stage(model, prepared_data, model_cfg)

        return self._run_training_stage(model, prepared_data, output_dir, model_cfg)

    def _run_training_stage(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        output_dir: Path,
        model_cfg: Any,
    ) -> dict[str, Any]:
        # --- HF Trainer --------------------------------------------------
        from transformers import Trainer as HFTrainer
        hf_trainer = HFTrainer(
            model=model,
            args=_build_hf_training_args(self.config, model_cfg, output_dir),
            train_dataset=prepared_data.get_train_dataset(),
            data_collator=model.build_train_collator(prepared_data),
        )

        total_start = time.perf_counter()
        hf_trainer.train()

        # --- Test evaluation ---------------------------------------------
        test_result = self._evaluate(model, prepared_data, "test")

        return _make_result(test_result, total_start)

    def _run_evaluation_stage(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        model_cfg: Any,
    ) -> dict[str, Any]:
        checkpoint_path = str(getattr(model_cfg, "checkpoint_path", "") or "")
        if not checkpoint_path:
            raise ValueError(
                "checkpoint_path is required when pipeline_stage='evaluation'."
            )
        print(f"[trainer] loading checkpoint from {checkpoint_path}")
        _load_checkpoint(model, Path(checkpoint_path))

        test_result = self._evaluate(model, prepared_data, "test")
        return _make_result(test_result)

    def _evaluate(
        self, model: BaseModel, prepared_data: BaseTaskDataset, split: str
    ) -> dict[str, Any]:
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState
        from recbole3.evaluation.methods import create_evaluation_method
        from recbole3.evaluation.metric import RetrievalEvalData

        method = create_evaluation_method(self.config.eval)
        collate_fn = method.build_eval_collate_fn(model, prepared_data)
        loader = DataLoader(
            prepared_data.get_eval_dataset(split),
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=int(self.config.dataloader_num_workers),
            pin_memory=bool(self.config.pin_memory),
            collate_fn=collate_fn,
        )

        accelerator = (
            Accelerator()
            if AcceleratorState._shared_state
            else Accelerator(mixed_precision=self.config.mixed_precision)
        )
        model, loader = accelerator.prepare(model, loader)

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        model.eval()
        unwrap_model = accelerator.unwrap_model(model)
        with torch.no_grad():
            for model_inputs, records in loader:
                ed: RetrievalEvalData = method.collect_batch(unwrap_model, model_inputs, records)
                preds = torch.from_numpy(np.asarray(ed.pred_item_ids, dtype=np.int64))
                targets = torch.from_numpy(np.asarray(ed.target_item_ids, dtype=np.int64))
                masks = torch.from_numpy(np.asarray(ed.target_mask, dtype=bool))
                preds, targets, masks = accelerator.gather_for_metrics(
                    (preds.to(accelerator.device),
                     targets.to(accelerator.device),
                     masks.to(accelerator.device))
                )
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(masks.cpu())

        if all_preds:
            gathered = RetrievalEvalData(
                pred_item_ids=torch.cat(all_preds, dim=0).numpy(),
                target_item_ids=torch.cat(all_targets, dim=0).numpy(),
                target_mask=torch.cat(all_masks, dim=0).numpy(),
            )
        else:
            gathered = RetrievalEvalData(
                pred_item_ids=np.empty((0, 0), dtype=np.int64),
                target_item_ids=np.empty((0, 0), dtype=np.int64),
                target_mask=np.empty((0, 0), dtype=bool),
            )

        metrics = method.compute_metrics([gathered])
        return {
            "split": split,
            "protocol": str(method.protocol),
            "loss": None,
            "metrics": metrics,
            "data_stats": _data_stats(prepared_data),
        }


def _build_hf_training_args(trainer_cfg, model_cfg, output_dir):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=trainer_cfg.max_epochs,
        per_device_train_batch_size=trainer_cfg.batch_size,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        learning_rate=trainer_cfg.optimizer.kwargs['lr'],
        warmup_steps=model_cfg.warmup_steps,
        lr_scheduler_type=model_cfg.lr_scheduler_type,
        fp16=trainer_cfg.mixed_precision == "fp16",
        bf16=trainer_cfg.mixed_precision == "bf16",
        optim=model_cfg.optim,
        save_total_limit=3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )


def _load_checkpoint(model: BaseModel, checkpoint_dir: Path) -> None:
    """Load from HF Trainer's checkpoint (model.safetensors).

    Uses ``strict=False`` because ``E4SRecModel.state_dict()`` excludes the frozen
    backbone — the file only contains LoRA adapters + non-LLM params.
    """
    from safetensors.torch import load_file

    safetensors = checkpoint_dir / "model.safetensors"
    if safetensors.exists():
        state_dict = load_file(str(safetensors))
        model.load_state_dict(state_dict, strict=False)
        print(f"[trainer] loaded checkpoint from {safetensors}")
        return

    return


def _data_stats(prepared_data: BaseTaskDataset) -> dict[str, int]:
    return {
        "num_users": int(prepared_data.get_num_users()),
        "num_items": int(prepared_data.get_num_items()),
    }


def _make_result(
    test_result: dict[str, Any],
    start_time: float | None = None,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - start_time if start_time else None
    return {
        "fit": {
            "train_history": [],
            "valid_history": [],
            "data_stats": test_result.get("data_stats", {}),
            "stopped_early": False,
            "best_epoch": None,
            "best_metric": None,
            **(dict(total_elapsed=elapsed) if elapsed else {}),
        },
        "test": test_result,
    }


__all__ = ["E4SRecTrainer"]
