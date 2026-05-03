from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.base import BaseModel
from recbole3.model.rpg.model import RPGModel
from recbole3.trainer import Trainer
from recbole3.trainer_config import SchedulerConfig, TrainerConfig


@dataclass(slots=True)
class RPGTrainerConfig(TrainerConfig):
    """RPG training config aligned with the released implementation."""

    eval_batch_size: int = field(default=32, metadata={"help": "Batch size used by RPG validation/test dataloaders."})
    warmup_steps: int = field(default=10000, metadata={"help": "Warmup steps passed to HF cosine scheduler."})
    steps: int | None = field(
        default=None,
        metadata={"help": "Optional total training optimizer steps. If unset, RPG uses len(train_dataloader) * max_epochs."},
    )
    scheduler: SchedulerConfig | None = field(
        default_factory=lambda: SchedulerConfig(name="cosine", interval="step", kwargs={}),
        metadata={"help": "RPG uses HuggingFace get_scheduler(name='cosine') with warmup_steps."},
    )
    max_grad_norm: float | None = field(
        default=1.0,
        metadata={"help": "RPG defaults to L2 grad-norm clipping at 1.0, matching the released implementation."},
    )


class RPGTrainer(Trainer):
    """Trainer shim that mirrors RPG's validation/test graph-decoding schedule."""

    config_cls = RPGTrainerConfig

    def build_dataloader(
        self,
        dataset: Dataset[Any],
        collate_fn,
        *,
        shuffle: bool,
    ) -> DataLoader:
        del shuffle
        train_dataset_id = getattr(self, "_rpg_train_dataset_id", None)
        is_train_dataset = train_dataset_id is not None and id(dataset) == train_dataset_id
        batch_size = self.config.batch_size if is_train_dataset else int(self.config.eval_batch_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.config.shuffle if is_train_dataset else False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )

    def build_scheduler(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        *,
        steps_per_epoch: int | None = None,
    ) -> Any:
        del num_training_steps
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required when building RPG's scheduler.")
        from transformers.optimization import get_scheduler

        return get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=int(self.config.warmup_steps),
            num_training_steps=self._resolve_total_steps(steps_per_epoch),
        )

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        if isinstance(model, RPGModel):
            model.generate_w_decoding_graph = False
        self._rpg_train_dataset_id = id(prepared_data.get_train_dataset())
        original_max_epochs = self.config.max_epochs
        if self.config.steps is not None:
            probe_collator = model.build_train_collator(prepared_data)
            probe_dataloader = self.build_dataloader(prepared_data.get_train_dataset(), probe_collator, shuffle=self.config.shuffle)
            self.config.max_epochs = self._resolve_epoch_count(max(1, len(probe_dataloader)))
        try:
            return super().fit(model, prepared_data, output_dir=output_dir)
        finally:
            self.config.max_epochs = original_max_epochs

    def run(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        test_uses_graph = bool(getattr(model.config, "use_decoding_graph", False))
        if isinstance(model, RPGModel):
            model.generate_w_decoding_graph = False

        fit_result = self.fit(model, prepared_data, output_dir=output_dir)
        best_checkpoint = fit_result["checkpoint_paths"].get("best")
        if best_checkpoint:
            state_dict = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

        if isinstance(model, RPGModel):
            model.generate_w_decoding_graph = test_uses_graph
        test_result = self.evaluate(model, prepared_data, split="test")
        return {
            "fit": fit_result,
            "test": test_result,
        }

    def _resolve_total_steps(self, steps_per_epoch: int) -> int:
        if self.config.steps is not None:
            return max(1, int(self.config.steps))
        return max(1, int(steps_per_epoch) * int(self.config.max_epochs))

    def _resolve_epoch_count(self, steps_per_epoch: int) -> int:
        return max(1, int(math.ceil(self._resolve_total_steps(steps_per_epoch) / int(steps_per_epoch))))


__all__ = ["RPGTrainer", "RPGTrainerConfig"]
