from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from recbole3.dataset import BaseTaskDataset
from recbole3.model.base import BaseCollator, BaseModel
from recbole3.trainer_config import (
    CheckpointConfig,
    EarlyStoppingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)


EvalSplitName = Literal["valid", "test"]


@dataclass(frozen=True, slots=True)
class MonitorSpec:
    name: str
    higher_is_better: bool


class Trainer:
    """Align prepared data with train and evaluation flows and execute them through accelerate."""

    config_cls = TrainerConfig

    def __init__(self, config: TrainerConfig):
        self.config = config

    def create_accelerator(self) -> Any:
        from accelerate import Accelerator

        return Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

    def build_dataloader(
        self,
        dataset: Dataset[Any],
        collate_fn: Callable[[Sequence[Any]], Any] | BaseCollator,
        *,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )

    def create_evaluation_method(self, prepared_data: BaseTaskDataset[Any, Any] | None = None) -> BaseEvaluationMethod:
        from recbole3.evaluation.methods import create_evaluation_method as build_evaluation_method

        return build_evaluation_method(self.config.eval)

    def build_optimizer(self, model: BaseModel) -> Optimizer:
        optimizer_cls = self._resolve_optimizer_class(self.config.optimizer.name)
        return optimizer_cls(model.parameters(), **dict(self.config.optimizer.kwargs))

    def build_scheduler(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        *,
        steps_per_epoch: int | None = None,
    ) -> LRScheduler | ReduceLROnPlateau | None:
        scheduler_config = self.config.scheduler
        if scheduler_config is None:
            return None

        scheduler_cls = self._resolve_scheduler_class(scheduler_config.name)
        if scheduler_config.interval not in ("step", "epoch"):
            raise ValueError(f"Unknown scheduler interval '{scheduler_config.interval}'. Expected 'step' or 'epoch'.")
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required when building a scheduler.")
        if scheduler_cls is ReduceLROnPlateau:
            if scheduler_config.interval != "epoch":
                raise ValueError("ReduceLROnPlateau requires SchedulerConfig.interval='epoch'.")
            if not str(self.config.monitor or "").strip():
                raise ValueError("TrainerConfig.monitor is required when scheduler.name is 'ReduceLROnPlateau'.")

        scheduler_kwargs = self._build_scheduler_kwargs(
            scheduler_cls,
            num_training_steps=num_training_steps,
            steps_per_epoch=steps_per_epoch,
        )
        return scheduler_cls(optimizer, **scheduler_kwargs)

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset[Any, Any],
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        accelerator = self.create_accelerator()
        collator = model.build_train_collator(prepared_data)
        train_dataset = prepared_data.get_train_dataset()
        train_dataloader = self.build_dataloader(train_dataset, collator, shuffle=self.config.shuffle)
        optimizer = self.build_optimizer(model)
        monitor = self._resolve_monitor(prepared_data)
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        steps_per_epoch = max(1, len(train_dataloader))
        num_training_steps = max(1, steps_per_epoch * self.config.max_epochs)
        scheduler = self.build_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            steps_per_epoch=steps_per_epoch,
        )
        scheduler_interval = self.config.scheduler.interval if self.config.scheduler is not None else None

        if scheduler is None:
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        else:
            model, optimizer, train_dataloader, scheduler = accelerator.prepare(
                model,
                optimizer,
                train_dataloader,
                scheduler,
            )

        train_history: list[dict[str, Any]] = []
        valid_history: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_value: float | None = None
        bad_epoch_count = 0
        stopped_early = False

        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            losses: list[float] = []
            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model.forward(batch)
                    loss = model.compute_loss(batch, outputs)
                    accelerator.backward(loss)
                    optimizer.step()
                    if scheduler is not None and scheduler_interval == "step":
                        scheduler.step()
                losses.append(float(loss.detach().float().item()))

            train_history.append(
                {
                    "epoch": epoch,
                    "loss": self._mean_or_none(losses),
                    "losses": losses,
                    "num_batches": len(losses),
                }
            )

            valid_result = self._run_evaluation(
                model,
                prepared_data,
                split="valid",
                accelerator=accelerator,
                model_is_prepared=True,
            )
            valid_result["epoch"] = epoch
            valid_history.append(valid_result)

            current_value: float | None = None
            improved = False
            if monitor is not None:
                current_value = self._extract_monitor_value(valid_result["metrics"], monitor.name)
                improved = self._is_improvement(
                    current_value,
                    best_value,
                    higher_is_better=monitor.higher_is_better,
                    min_delta=float(self.config.early_stopping.min_delta),
                )
                if improved:
                    best_value = current_value
                    best_epoch = epoch
                    bad_epoch_count = 0
                    if checkpoint_paths["best"] is not None:
                        self._save_model_checkpoint(model, accelerator, checkpoint_paths["best"])
                elif self.config.early_stopping.enabled:
                    bad_epoch_count += 1
            if scheduler is not None and scheduler_interval == "epoch":
                self._step_epoch_scheduler(scheduler, current_value=current_value)
            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])
            if self.config.early_stopping.enabled and not improved and bad_epoch_count >= int(self.config.early_stopping.patience):
                stopped_early = True
                break

        return {
            "train_history": train_history,
            "valid_history": valid_history,
            "data_stats": self._build_result_data_stats(prepared_data),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {key: (str(path) if path is not None else None) for key, path in checkpoint_paths.items()},
        }

    def evaluate(self, model: BaseModel, prepared_data: BaseTaskDataset[Any, Any], split: EvalSplitName = "valid") -> dict[str, Any]:
        accelerator = self.create_accelerator()
        return self._run_evaluation(
            model,
            prepared_data,
            split=split,
            accelerator=accelerator,
            model_is_prepared=False,
        )

    def run(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset[Any, Any],
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        fit_result = self.fit(model, prepared_data, output_dir=output_dir)
        best_checkpoint = fit_result["checkpoint_paths"].get("best")
        if best_checkpoint:
            state_dict = torch.load(best_checkpoint, map_location="cpu")
            model.load_state_dict(state_dict)
        test_result = self.evaluate(model, prepared_data, split="test")
        return {
            "fit": fit_result,
            "test": test_result,
        }

    def _run_evaluation(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset[Any, Any],
        split: EvalSplitName,
        accelerator: Any,
        model_is_prepared: bool,
    ) -> dict[str, Any]:
        method = self.create_evaluation_method(prepared_data)
        collator_model = accelerator.unwrap_model(model)
        eval_dataset = prepared_data.get_eval_dataset(split)
        eval_collate_fn = method.build_eval_collate_fn(collator_model, prepared_data)
        eval_dataloader = self.build_dataloader(eval_dataset, eval_collate_fn, shuffle=False)
        if model_is_prepared:
            prepared_model = model
            eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            prepared_model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        prepared_model.eval()
        scoring_model = accelerator.unwrap_model(prepared_model)
        batch_eval_data: list[Any] = []
        num_batches = 0
        with torch.no_grad():
            for model_inputs, records in eval_dataloader:
                batch_eval_data.append(method.collect_batch(scoring_model, model_inputs, records))
                num_batches += 1

        return {
            "split": split,
            "protocol": method.protocol,
            "loss": None,
            "metrics": method.compute_metrics(batch_eval_data),
            "num_batches": num_batches,
            "data_stats": self._build_result_data_stats(prepared_data),
        }

    def _resolve_monitor(self, prepared_data: BaseTaskDataset[Any, Any]) -> MonitorSpec | None:
        monitor_name = str(self.config.monitor or "").strip()
        requires_monitor = bool(self.config.early_stopping.enabled or self.config.checkpoint.save_best)
        if self._scheduler_requires_monitor():
            requires_monitor = True
        if requires_monitor and not monitor_name:
            raise ValueError(
                "TrainerConfig.monitor is required when early stopping, best checkpointing, or a metric-driven scheduler is enabled."
            )
        if not monitor_name:
            return None

        directions = self.create_evaluation_method(prepared_data).metric_directions()
        if monitor_name not in directions:
            available = ", ".join(sorted(directions)) or "<empty>"
            raise ValueError(f"Unknown trainer monitor '{monitor_name}'. Available metrics: {available}.")
        return MonitorSpec(name=monitor_name, higher_is_better=bool(directions[monitor_name]))

    def _resolve_checkpoint_paths(self, output_dir: str | Path | None) -> dict[str, Path | None]:
        save_best = bool(self.config.checkpoint.save_best)
        save_last = bool(self.config.checkpoint.save_last)
        if not save_best and not save_last:
            return {"best": None, "last": None}
        if output_dir is None:
            raise ValueError("Checkpoint saving requires output_dir to be passed into fit() or run().")

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return {
            "best": checkpoint_dir / "best_model.pt" if save_best else None,
            "last": checkpoint_dir / "last_model.pt" if save_last else None,
        }

    def _save_model_checkpoint(self, model: BaseModel, accelerator: Any, path: Path) -> None:
        state_dict = accelerator.unwrap_model(model).state_dict()
        torch.save(state_dict, path)

    def _resolve_optimizer_class(self, name: str) -> type[Optimizer]:
        optimizer_cls = getattr(torch.optim, name, None)
        if optimizer_cls is None or not inspect.isclass(optimizer_cls) or not issubclass(optimizer_cls, Optimizer):
            raise ValueError(f"Unknown torch optimizer '{name}'.")
        return optimizer_cls

    def _resolve_scheduler_class(self, name: str) -> type[LRScheduler] | type[ReduceLROnPlateau]:
        scheduler_cls = getattr(torch.optim.lr_scheduler, name, None)
        valid_scheduler_class = (
            inspect.isclass(scheduler_cls)
            and (issubclass(scheduler_cls, LRScheduler) or scheduler_cls is ReduceLROnPlateau)
        )
        if not valid_scheduler_class:
            raise ValueError(f"Unknown torch scheduler '{name}'.")
        return scheduler_cls

    def _build_scheduler_kwargs(
        self,
        scheduler_cls: type[LRScheduler] | type[ReduceLROnPlateau],
        *,
        num_training_steps: int,
        steps_per_epoch: int,
    ) -> dict[str, Any]:
        scheduler_kwargs = dict(self.config.scheduler.kwargs) if self.config.scheduler is not None else {}
        parameters = inspect.signature(scheduler_cls).parameters
        if "total_steps" in parameters and "total_steps" not in scheduler_kwargs:
            if "epochs" not in scheduler_kwargs and "steps_per_epoch" not in scheduler_kwargs:
                scheduler_kwargs["total_steps"] = num_training_steps
        elif "total_steps" not in scheduler_kwargs:
            if "epochs" in parameters and "epochs" not in scheduler_kwargs:
                scheduler_kwargs["epochs"] = self.config.max_epochs
            if "steps_per_epoch" in parameters and "steps_per_epoch" not in scheduler_kwargs:
                scheduler_kwargs["steps_per_epoch"] = steps_per_epoch
        return scheduler_kwargs

    def _scheduler_requires_monitor(self) -> bool:
        scheduler_config = self.config.scheduler
        return scheduler_config is not None and scheduler_config.name == "ReduceLROnPlateau"

    @staticmethod
    def _step_epoch_scheduler(
        scheduler: LRScheduler | ReduceLROnPlateau | Any,
        *,
        current_value: float | None,
    ) -> None:
        inner_scheduler = Trainer._unwrap_scheduler(scheduler)
        if isinstance(inner_scheduler, ReduceLROnPlateau):
            if current_value is None:
                raise ValueError("ReduceLROnPlateau requires a validation monitor value at epoch end.")
            scheduler.step(current_value)
            return
        scheduler.step()

    @staticmethod
    def _unwrap_scheduler(scheduler: Any) -> Any:
        inner_scheduler = scheduler
        while hasattr(inner_scheduler, "scheduler"):
            inner_scheduler = inner_scheduler.scheduler
        return inner_scheduler

    @staticmethod
    def _extract_monitor_value(metrics: Mapping[str, Any], monitor_name: str) -> float:
        if monitor_name not in metrics:
            available = ", ".join(sorted(metrics)) or "<empty>"
            raise ValueError(f"Validation result does not contain monitor '{monitor_name}'. Available metrics: {available}.")
        return float(metrics[monitor_name])

    @staticmethod
    def _is_improvement(
        current_value: float,
        best_value: float | None,
        *,
        higher_is_better: bool,
        min_delta: float,
    ) -> bool:
        if best_value is None:
            return True
        if higher_is_better:
            return current_value > best_value + min_delta
        return current_value < best_value - min_delta

    @staticmethod
    def _build_best_metric_payload(monitor: MonitorSpec | None, best_value: float | None) -> dict[str, Any] | None:
        if monitor is None or best_value is None:
            return None
        return {
            "name": monitor.name,
            "value": float(best_value),
            "higher_is_better": bool(monitor.higher_is_better),
        }

    @staticmethod
    def _build_result_data_stats(prepared_data: BaseTaskDataset[Any, Any]) -> dict[str, int]:
        return {
            "num_users": int(prepared_data.get_num_users()),
            "num_items": int(prepared_data.get_num_items()),
        }

    @staticmethod
    def _mean_or_none(values: Sequence[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))


__all__ = [
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "MonitorSpec",
    "OptimizerConfig",
    "SchedulerConfig",
    "Trainer",
    "TrainerConfig",
]
