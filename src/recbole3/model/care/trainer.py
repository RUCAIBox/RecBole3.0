from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from recbole3.dataset import BaseTaskDataset
from recbole3.model.base import BaseModel
from recbole3.trainer import MonitorSpec, Trainer
from recbole3.trainer_config import TrainerConfig


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


@dataclass(slots=True)
class CARETrainerConfig(TrainerConfig):
    """RecBole-compatible trainer config aligned with CARE TrainingArguments."""

    warmup_steps: int = field(default=50, metadata={"help": "CARE TrainingArguments.warmup_steps."})
    warmup_ratio: float = field(default=0.0, metadata={"help": "Used when warmup_steps <= 0."})
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_lr_on_plateau",
    ] = field(default="cosine", metadata={"help": "CARE TrainingArguments.lr_scheduler_type."})
    logging_steps: int = field(default=10, metadata={"help": "CARE TrainingArguments.logging_steps."})
    save_and_eval_strategy: Literal["epoch", "steps"] = field(
        default="epoch",
        metadata={"help": "CARE evaluation_strategy/save_strategy."},
    )
    save_and_eval_steps: int = field(default=1000, metadata={"help": "CARE eval_steps/save_steps when strategy='steps'."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "CARE TrainingArguments.load_best_model_at_end."})


class CARETrainer(Trainer):
    """CARE-style trainer implemented inside RecBole's trainer protocol."""

    config_cls = CARETrainerConfig
    config: CARETrainerConfig

    def build_scheduler(self, optimizer: Optimizer, num_training_steps: int, *, steps_per_epoch: int | None = None) -> LRScheduler | ReduceLROnPlateau | None:
        if self.config.lr_scheduler_type == "reduce_lr_on_plateau":
            return super().build_scheduler(optimizer, num_training_steps=num_training_steps, steps_per_epoch=steps_per_epoch)
        from transformers import get_scheduler
        warmup_steps = int(self.config.warmup_steps)
        if warmup_steps <= 0 and float(self.config.warmup_ratio) > 0:
            warmup_steps = int(num_training_steps * float(self.config.warmup_ratio))
        return get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=max(0, min(warmup_steps, int(num_training_steps))),
            num_training_steps=int(num_training_steps),
        )

    def fit(self, model: BaseModel, prepared_data: BaseTaskDataset, *, output_dir: str | Path | None = None) -> Any:
        accelerator = self.create_accelerator()
        collator = model.build_train_collator(prepared_data)
        train_dataloader = self.build_dataloader(prepared_data.get_train_dataset(), collator, shuffle=self.config.shuffle)
        optimizer = self.build_optimizer(model)
        monitor = self._resolve_monitor(prepared_data)
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        steps_per_epoch = max(1, len(train_dataloader))
        num_training_steps = max(1, steps_per_epoch * int(self.config.max_epochs))
        scheduler = self.build_scheduler(optimizer, num_training_steps=num_training_steps, steps_per_epoch=steps_per_epoch)
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

        train_history: list[dict[str, Any]] = []
        valid_history: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_step: int | None = None
        best_value: float | None = None
        bad_eval_count = 0
        stopped_early = False
        global_step = 0
        losses: list[float] = []
        eval_every = max(1, int(self.config.save_and_eval_steps))
        log_every = max(1, int(self.config.logging_steps))

        for epoch in range(1, int(self.config.max_epochs) + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_losses: list[float] = []
            progress_bar = self._create_train_progress_bar(train_dataloader, epoch=epoch, max_epochs=int(self.config.max_epochs))
            for batch in progress_bar:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model.forward(batch)
                    loss = accelerator.unwrap_model(model).compute_loss(batch, outputs)
                    accelerator.backward(loss)
                    optimizer.step()
                    if scheduler is not None and self.config.lr_scheduler_type != "reduce_lr_on_plateau":
                        scheduler.step()
                global_step += 1
                loss_value = float(loss.detach().float().item())
                losses.append(loss_value)
                epoch_losses.append(loss_value)
                if hasattr(progress_bar, "set_postfix_str") and global_step % log_every == 0:
                    lr = optimizer.param_groups[0].get("lr", None)
                    progress_bar.set_postfix_str(f"step={global_step} loss={loss_value:.6f} lr={lr}")
                if self.config.save_and_eval_strategy == "steps" and global_step % eval_every == 0:
                    bad_eval_count, best_value, best_epoch, best_step, stopped_early = self._care_validate_and_save(
                        model, prepared_data, accelerator, monitor, checkpoint_paths, valid_history, epoch, global_step,
                        bad_eval_count, best_value, best_epoch, best_step,
                    )
                    if stopped_early:
                        break
            if hasattr(progress_bar, "close"):
                progress_bar.close()
            epoch_loss = self._mean_or_none(epoch_losses)
            train_history.append({"epoch": epoch, "step": global_step, "loss": epoch_loss, "losses": epoch_losses, "num_batches": len(epoch_losses), "elapsed_seconds": time.perf_counter() - epoch_start, "lr": optimizer.param_groups[0].get("lr", None)})
            print(f"[care-train] epoch={epoch}/{int(self.config.max_epochs)} step={global_step} avg_loss={(f'{epoch_loss:.6f}' if epoch_loss is not None else 'n/a')}")
            if stopped_early:
                break
            if self.config.save_and_eval_strategy == "epoch":
                bad_eval_count, best_value, best_epoch, best_step, stopped_early = self._care_validate_and_save(
                    model, prepared_data, accelerator, monitor, checkpoint_paths, valid_history, epoch, global_step,
                    bad_eval_count, best_value, best_epoch, best_step,
                )
                if scheduler is not None and self.config.lr_scheduler_type == "reduce_lr_on_plateau" and valid_history:
                    self._step_epoch_scheduler(scheduler, current_value=self._extract_monitor_value(valid_history[-1]["metrics"], monitor.name) if monitor else None)
                if stopped_early:
                    break
            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])
                accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()
        return {"train_history": train_history, "valid_history": valid_history, "data_stats": self._build_result_data_stats(prepared_data), "stopped_early": stopped_early, "best_epoch": best_epoch, "best_step": best_step, "best_metric": self._build_best_metric_payload(monitor, best_value), "checkpoint_paths": {k: (str(v) if v is not None else None) for k, v in checkpoint_paths.items()}}

    def run(self, model: BaseModel, prepared_data: BaseTaskDataset, *, output_dir: str | Path | None = None) -> dict[str, Any]:
        self._setup_logger(model, prepared_data, output_dir)
        total_start = time.perf_counter()
        try:
            fit_result = self.fit(model, prepared_data, output_dir=output_dir)
            best_metric = fit_result.get("best_metric")
            if best_metric and (logger := getattr(self, "_logger", None)) is not None:
                logger.log_best(
                    epoch=fit_result.get("best_epoch"),
                    monitor_name=best_metric["name"],
                    best_value=best_metric["value"],
                )

            best_checkpoint = fit_result["checkpoint_paths"].get("best")
            if self.config.load_best_model_at_end and best_checkpoint:
                _distributed_barrier()
                model.load_state_dict(torch.load(best_checkpoint, map_location="cpu", weights_only=True))
                _distributed_barrier()
                print(f"[care-trainer] loaded best checkpoint before test: {best_checkpoint}")

            print("[care-trainer] starting test evaluation via CAREModel.predict -> CARE.generate")
            test_result = self.evaluate(model, prepared_data, split="test")
            print("[care-trainer] finished test evaluation")

            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                logger.log_summary(
                    stopped_early=fit_result.get("stopped_early", False),
                    total_epochs=len(fit_result.get("train_history", [])),
                    best_epoch=fit_result.get("best_epoch"),
                    total_time=time.perf_counter() - total_start,
                )
            return {"fit": fit_result, "test": test_result}
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def _resolve_monitor(self, prepared_data: BaseTaskDataset) -> MonitorSpec:
        del prepared_data
        return MonitorSpec(name="eval_loss", higher_is_better=False)

    def _care_validate_and_save(self, model, prepared_data, accelerator, monitor, checkpoint_paths, valid_history, epoch: int, step: int, bad_eval_count: int, best_value: float | None, best_epoch: int | None, best_step: int | None):
        result = self._run_validation_loss(model, prepared_data, accelerator, epoch=epoch, step=step)
        valid_history.append(result)
        print(f"[care-valid-loss] epoch={epoch} step={step} eval_loss={result['loss']:.6f}")
        current = float(result["loss"])
        improved = self._is_improvement(current, best_value, higher_is_better=monitor.higher_is_better, min_delta=float(self.config.early_stopping.min_delta))
        if improved:
            best_value, best_epoch, best_step, bad_eval_count = current, epoch, step, 0
            if checkpoint_paths["best"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["best"])
                accelerator.wait_for_everyone()
        elif self.config.early_stopping.enabled:
            bad_eval_count += 1
        stopped = bool(self.config.early_stopping.enabled and not improved and bad_eval_count >= int(self.config.early_stopping.patience))
        return bad_eval_count, best_value, best_epoch, best_step, stopped

    def _run_validation_loss(self, model, prepared_data: BaseTaskDataset, accelerator, *, epoch: int, step: int) -> dict[str, Any]:
        collator_model = accelerator.unwrap_model(model)
        valid_dataloader = self.build_dataloader(
            prepared_data.get_eval_dataset("valid"),
            collator_model.build_train_collator(prepared_data),
            shuffle=False,
            batch_size=self.config.eval_batch_size,
        )
        valid_dataloader = accelerator.prepare(valid_dataloader)
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in valid_dataloader:
                outputs = model.forward(batch)
                loss = collator_model.compute_loss(batch, outputs)
                losses.append(float(loss.detach().float().item()))
        model.train()
        local_sum = torch.tensor(sum(losses), dtype=torch.float32, device=accelerator.device)
        local_count = torch.tensor(len(losses), dtype=torch.float32, device=accelerator.device)
        global_sum = accelerator.reduce(local_sum, reduction="sum")
        global_count = accelerator.reduce(local_count, reduction="sum")
        eval_loss = float((global_sum / global_count.clamp_min(1.0)).detach().cpu().item())
        return {
            "split": "valid",
            "epoch": epoch,
            "step": step,
            "loss": eval_loss,
            "metrics": {"eval_loss": eval_loss},
            "num_batches": len(losses),
            "data_stats": self._build_result_data_stats(prepared_data),
        }


__all__ = ["CARETrainer", "CARETrainerConfig"]