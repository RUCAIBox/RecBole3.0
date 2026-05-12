from __future__ import annotations

import math
import time
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
        metadata={
            "help": "Optional total training optimizer steps. If unset, RPG uses len(train_dataloader) * max_epochs."
        },
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
        if self.config.scheduler is None:
            return None
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

        accelerator = self.create_accelerator()
        collator = model.build_train_collator(prepared_data)
        train_dataset = prepared_data.get_train_dataset()
        train_dataloader = self.build_dataloader(train_dataset, collator, shuffle=self.config.shuffle)
        optimizer = self.build_optimizer(model)
        monitor = self._resolve_monitor(prepared_data)
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        steps_per_epoch = max(1, len(train_dataloader))
        max_optimizer_steps = max(1, int(self.config.steps)) if self.config.steps is not None else None
        total_epochs = (
            self._resolve_epoch_count(steps_per_epoch)
            if max_optimizer_steps is not None
            else int(self.config.max_epochs)
        )
        num_training_steps = self._resolve_total_steps(steps_per_epoch)
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
        eval_steps = max(1, int(self.config.eval_steps))
        optimizer_steps = 0

        for epoch in range(1, total_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            unwrapped_model = accelerator.unwrap_model(model)
            losses: list[float] = []
            progress_bar = self._create_train_progress_bar(
                train_dataloader,
                epoch=epoch,
                max_epochs=total_epochs,
            )
            for batch in progress_bar:
                if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                    break
                with accelerator.accumulate(model):
                    outputs = model.forward(batch)
                    loss = unwrapped_model.compute_loss(batch, outputs)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and self.config.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), float(self.config.max_grad_norm))
                    optimizer.step()
                    if accelerator.sync_gradients:
                        optimizer_steps += 1
                    if scheduler is not None and scheduler_interval == "step" and accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad()
                losses.append(float(loss.detach().float().item()))

            if hasattr(progress_bar, "close"):
                progress_bar.close()

            epoch_loss = self._mean_or_none(losses)
            elapsed = time.perf_counter() - epoch_start
            lr = optimizer.param_groups[0].get("lr", None)
            train_history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "losses": losses,
                    "num_batches": len(losses),
                    "elapsed_seconds": elapsed,
                    "lr": lr,
                    "optimizer_steps": optimizer_steps,
                }
            )
            print(
                f"[train] epoch={epoch}/{total_epochs} "
                f"avg_loss={(f'{epoch_loss:.6f}' if epoch_loss is not None else 'n/a')} "
                f"num_batches={len(losses)} optimizer_steps={optimizer_steps}"
            )
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_epoch(
                    epoch=epoch,
                    max_epochs=total_epochs,
                    loss=epoch_loss,
                    num_batches=len(losses),
                    elapsed_seconds=elapsed,
                    lr=lr,
                )

            reached_step_limit = max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps
            should_run_validation = (epoch % eval_steps == 0) or (epoch == total_epochs) or reached_step_limit
            valid_result: dict[str, Any] | None = None
            if should_run_validation:
                # IMPORTANT: run evaluation ONLY on the main process.
                # If every rank evaluates independently, the (slightly) different metrics can cause
                # rank divergence in checkpoint/early-stopping decisions and even deadlocks.
                if accelerator.is_main_process:
                    valid_result = self._run_evaluation(
                        model,
                        prepared_data,
                        split="valid",
                        accelerator=accelerator,
                        model_is_prepared=True,
                    )
                    valid_result["epoch"] = epoch
                    valid_history.append(valid_result)
                    print(
                        f"[eval:valid] epoch={epoch}/{total_epochs} "
                        f"metrics={self._format_metrics(valid_result['metrics'])}"
                    )
                    if (logger := getattr(self, "_logger", None)) is not None:
                        logger.log_validation(epoch=epoch, metrics=valid_result["metrics"])
                accelerator.wait_for_everyone()

            current_value: float | None = None
            improved = False
            if monitor is not None and should_run_validation:
                if accelerator.is_main_process:
                    if valid_result is None:
                        raise RuntimeError("Expected valid_result on main process when should_run_validation is True.")
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

                decision = torch.tensor(
                    [
                        1 if improved else 0,
                        int(bad_epoch_count),
                        int(best_epoch or 0),
                        float(best_value) if best_value is not None else float("nan"),
                        float(current_value) if current_value is not None else float("nan"),
                    ],
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                if accelerator.num_processes > 1:
                    import torch.distributed as dist

                    if not dist.is_available() or not dist.is_initialized():
                        raise RuntimeError(
                            "Expected torch.distributed to be initialized when running multi-process training."
                        )
                    dist.broadcast(decision, src=0)
                improved = bool(int(decision[0].item()))
                bad_epoch_count = int(decision[1].item())
                best_epoch = int(decision[2].item()) if int(decision[2].item()) > 0 else None
                best_value = float(decision[3].item()) if not torch.isnan(decision[3]) else None
                current_value = float(decision[4].item()) if not torch.isnan(decision[4]) else None

                # Keep all ranks aligned regardless of whether a checkpoint was saved.
                accelerator.wait_for_everyone()
            if scheduler is not None and scheduler_interval == "epoch":
                if self._scheduler_requires_monitor():
                    if valid_result is not None:
                        self._step_epoch_scheduler(scheduler, current_value=current_value)
                else:
                    self._step_epoch_scheduler(scheduler, current_value=current_value)
            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])
                accelerator.wait_for_everyone()
            if (
                valid_result is not None
                and self.config.early_stopping.enabled
                and not improved
                and bad_epoch_count >= int(self.config.early_stopping.patience)
            ):
                stopped_early = True
                if (logger := getattr(self, "_logger", None)) is not None:
                    logger.log_early_stopping(
                        stopped=True,
                        epoch=epoch,
                        patience=int(self.config.early_stopping.patience),
                    )
                break
            if reached_step_limit:
                break

        if not stopped_early:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_early_stopping(
                    stopped=False,
                    epoch=len(train_history),
                    patience=int(self.config.early_stopping.patience),
                )

        return {
            "train_history": train_history,
            "valid_history": valid_history,
            "data_stats": self._build_result_data_stats(prepared_data),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {
                key: (str(path) if path is not None else None) for key, path in checkpoint_paths.items()
            },
            "optimizer_steps": optimizer_steps,
        }

    def run(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        self._setup_logger(model, prepared_data, output_dir)
        total_start = time.perf_counter()
        try:
            test_uses_graph = bool(getattr(model.config, "use_decoding_graph", False))
            if isinstance(model, RPGModel):
                model.generate_w_decoding_graph = False

            fit_result = self.fit(model, prepared_data, output_dir=output_dir)
            if (logger := getattr(self, "_logger", None)) is not None:
                best_metric = fit_result.get("best_metric")
                if best_metric:
                    logger.log_best(
                        epoch=fit_result["best_epoch"],
                        monitor_name=best_metric["name"],
                        best_value=best_metric["value"],
                    )

            best_checkpoint = fit_result["checkpoint_paths"].get("best")
            if best_checkpoint:
                state_dict = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)

            if isinstance(model, RPGModel):
                model.generate_w_decoding_graph = test_uses_graph
            test_result = self.evaluate(model, prepared_data, split="test")
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                logger.log_summary(
                    stopped_early=fit_result.get("stopped_early", False),
                    total_epochs=len(fit_result.get("train_history", [])),
                    best_epoch=fit_result.get("best_epoch"),
                    total_time=time.perf_counter() - total_start,
                )
            return {
                "fit": fit_result,
                "test": test_result,
            }
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def _resolve_total_steps(self, steps_per_epoch: int) -> int:
        if self.config.steps is not None:
            return max(1, int(self.config.steps))
        return max(1, self._optimizer_steps_per_epoch(steps_per_epoch) * int(self.config.max_epochs))

    def _resolve_epoch_count(self, steps_per_epoch: int) -> int:
        return max(
            1,
            int(math.ceil(self._resolve_total_steps(steps_per_epoch) / self._optimizer_steps_per_epoch(steps_per_epoch))),
        )

    def _optimizer_steps_per_epoch(self, steps_per_epoch: int) -> int:
        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        return max(1, int(math.ceil(int(steps_per_epoch) / accumulation_steps)))


__all__ = ["RPGTrainer", "RPGTrainerConfig"]
