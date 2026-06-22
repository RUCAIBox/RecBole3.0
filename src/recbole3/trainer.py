from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from recbole3.dataset import BaseTaskDataset
from recbole3.evaluation.metric import RankingEvalData, RetrievalEvalData
from recbole3.logger import TrainingLogger
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

    def _setup_logger(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        output_dir: str | Path | None,
    ) -> None:
        """Create the training logger and write initial configuration / info sections."""
        if output_dir is None:
            output_dir = Path(".")
        model.ensure_initialized(prepared_data)
        model_name = str(getattr(model.config, "name", "") or "unknown")
        dataset_name = str(getattr(prepared_data.config, "name", "") or "unknown")
        category_name = str(getattr(prepared_data.config, "category", "") or "")
        self._logger = TrainingLogger(
            output_dir=Path(output_dir),
            model_name=model_name,
            dataset_name=dataset_name,
            category_name=category_name,
        )
        self._logger.log_config("Trainer", self.config)
        self._logger.log_config("Model", model.config)
        self._logger.log_config("Dataset", prepared_data.config)
        self._logger.log_dataset_info(prepared_data)
        self._logger.log_model_info(model)

    @staticmethod
    def _reset_accelerator_state() -> None:
        try:
            from accelerate.state import AcceleratorState
        except ModuleNotFoundError:
            return

        shared_state = getattr(AcceleratorState, "_shared_state", None)
        reset_state = getattr(AcceleratorState, "_reset_state", None)
        if shared_state and callable(reset_state):
            reset_state()

    def create_accelerator(self) -> Any:
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState

        # # GRPO 等流程在 accelerate launch 下已初始化 AcceleratorState；再 new Accelerator() 会报错。
        # if AcceleratorState._shared_state:
        #     return _ExistingAcceleratorEvalContext()

        return Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

    def build_dataloader(
        self,
        dataset: Dataset[Any],
        collate_fn: Callable[[Any], Any] | BaseCollator,
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

    def create_evaluation_method(self, prepared_data: BaseTaskDataset | None = None) -> BaseEvaluationMethod:
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
        prepared_data: BaseTaskDataset,
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
        eval_steps = max(1, int(self.config.eval_steps))

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            losses: list[float] = []
            progress_bar = self._create_train_progress_bar(train_dataloader, epoch=epoch, 
                                                           max_epochs=int(self.config.max_epochs), 
                                                           disable=not accelerator.is_main_process)
            for batch in progress_bar:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model.forward(batch)
                    unwrap_model = accelerator.unwrap_model(model)
                    loss = unwrap_model.compute_loss(batch, outputs)
                    accelerator.backward(loss)
                    optimizer.step()
                    if scheduler is not None and scheduler_interval == "step":
                        scheduler.step()
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
                }
            )
            print(
                f"[train] epoch={epoch}/{int(self.config.max_epochs)} "
                f"avg_loss={(f'{epoch_loss:.6f}' if epoch_loss is not None else 'n/a')} "
                f"num_batches={len(losses)}"
            )
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_epoch(
                    epoch=epoch,
                    max_epochs=int(self.config.max_epochs),
                    loss=epoch_loss,
                    num_batches=len(losses),
                    elapsed_seconds=elapsed,
                    lr=lr,
                )

            should_run_validation = (epoch % eval_steps == 0) or (epoch == int(self.config.max_epochs))
            valid_result: dict[str, Any] | None = None
            if should_run_validation:
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
                    f"[eval:valid] epoch={epoch}/{int(self.config.max_epochs)} "
                    f"metrics={self._format_metrics(valid_result['metrics'])}"
                )
                if (logger := getattr(self, "_logger", None)) is not None:
                    logger.log_validation(epoch=epoch, metrics=valid_result["metrics"])

            current_value: float | None = None
            improved = False
            if monitor is not None and valid_result is not None:
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
                if self._scheduler_requires_monitor():
                    if valid_result is not None:
                        self._step_epoch_scheduler(scheduler, current_value=current_value)
                else:
                    self._step_epoch_scheduler(scheduler, current_value=current_value)
            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])
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

        if not stopped_early:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_early_stopping(
                    stopped=False,
                    epoch=int(self.config.max_epochs),
                    patience=int(self.config.early_stopping.patience),
                )

        return {
            "train_history": train_history,
            "valid_history": valid_history,
            "data_stats": self._build_result_data_stats(prepared_data),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {key: (str(path) if path is not None else None) for key, path in checkpoint_paths.items()},
        }

    def evaluate(self, model: BaseModel, prepared_data: BaseTaskDataset, split: EvalSplitName = "valid") -> dict[str, Any]:
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
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        self._setup_logger(model, prepared_data, output_dir)
        total_start = time.perf_counter()
        try:
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
            # fit() leaves AcceleratorState initialized; post-training evaluate() must
            # create a fresh Accelerator so eval batches are placed on the model device.
            self._reset_accelerator_state()
            print("[trainer] starting test evaluation")
            test_result = self.evaluate(model, prepared_data, split="test")
            print("[trainer] finished test evaluation")

            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                total_elapsed = time.perf_counter() - total_start
                logger.log_summary(
                    stopped_early=fit_result.get("stopped_early", False),
                    total_epochs=len(fit_result.get("train_history", [])),
                    best_epoch=fit_result.get("best_epoch"),
                    total_time=total_elapsed,
                )

            return {
                "fit": fit_result,
                "test": test_result,
            }
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def _run_evaluation(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
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
        progress_bar = self._create_progress_bar(eval_dataloader, split=split, disable=not accelerator.is_main_process)
        with torch.no_grad():
            for model_inputs, records in progress_bar:
                batch_eval_data.append(self._collect_eval_batch(method, scoring_model, model_inputs, records))
                num_batches += 1
                if hasattr(progress_bar, "set_postfix_str"):
                    progress_bar.set_postfix_str(f"batches={num_batches}")

        if hasattr(progress_bar, "close"):
            progress_bar.close()

        result = {
            "split": split,
            "protocol": method.protocol,
            "loss": None,
            "metrics": method.compute_metrics(batch_eval_data),
            "num_batches": num_batches,
            "data_stats": self._build_result_data_stats(prepared_data),
        }
        if self.config.save_inference_results:
            result["inference_results"] = self._build_inference_results(batch_eval_data)
        return result

    def _collect_eval_batch(
        self,
        method: Any,
        model: BaseModel,
        model_inputs: Any,
        records: Any,
    ) -> RankingEvalData | RetrievalEvalData:
        inference_topk = self.config.inference_topk
        if inference_topk is None:
            return method.collect_batch(model, model_inputs, records)

        from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod

        if not isinstance(method, BaseRetrievalEvaluationMethod):
            raise ValueError("TrainerConfig.inference_topk is only supported for retrieval evaluation methods.")
        if int(inference_topk) < 0:
            raise ValueError("TrainerConfig.inference_topk must be non-negative when provided.")
        return method._collect_retrieval_batch(
            model=model,
            model_inputs=model_inputs,
            records=records,
            max_k=int(inference_topk),
        )

    @staticmethod
    def _build_inference_results(batch_eval_data: Sequence[RankingEvalData | RetrievalEvalData]) -> dict[str, Any]:
        if not batch_eval_data:
            return {}
        first_batch = batch_eval_data[0]
        if isinstance(first_batch, RetrievalEvalData):
            pred_item_ids: list[list[int]] = []
            target_item_ids: list[list[int]] = []
            target_mask: list[list[bool]] = []
            for batch_data in batch_eval_data:
                if not isinstance(batch_data, RetrievalEvalData):
                    raise TypeError("Mixed evaluation batch types are not supported when saving inference results.")
                pred_item_ids.extend([[int(item_id) for item_id in row] for row in batch_data.pred_item_ids.tolist()])
                target_item_ids.extend([[int(item_id) for item_id in row] for row in batch_data.target_item_ids.tolist()])
                target_mask.extend([[bool(flag) for flag in row] for row in batch_data.target_mask.tolist()])
            return {
                "pred_item_ids": pred_item_ids,
                "target_item_ids": target_item_ids,
                "target_mask": target_mask,
            }

        scores: list[float] = []
        labels: list[float] = []
        group_ids: list[int] = []
        for batch_data in batch_eval_data:
            if not isinstance(batch_data, RankingEvalData):
                raise TypeError("Mixed evaluation batch types are not supported when saving inference results.")
            scores.extend(float(value) for value in batch_data.scores.reshape(-1).tolist())
            labels.extend(float(value) for value in batch_data.labels.reshape(-1).tolist())
            group_ids.extend(int(value) for value in batch_data.group_ids.reshape(-1).tolist())
        return {
            "scores": scores,
            "labels": labels,
            "group_ids": group_ids,
        }

    def _resolve_monitor(self, prepared_data: BaseTaskDataset) -> MonitorSpec | None:
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
        if accelerator.is_main_process:
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
    def _build_result_data_stats(prepared_data: BaseTaskDataset) -> dict[str, int]:
        return {
            "num_users": int(prepared_data.get_num_users()),
            "num_items": int(prepared_data.get_num_items()),
        }

    @staticmethod
    def _format_metrics(metrics: Mapping[str, Any]) -> str:
        if not metrics:
            return "{}"
        parts: list[str] = []
        for name, value in metrics.items():
            try:
                parts.append(f"{name}={float(value):.6f}")
            except (TypeError, ValueError):
                parts.append(f"{name}={value}")
        return "{ " + ", ".join(parts) + " }"

    @staticmethod
    def _create_progress_bar(eval_dataloader: DataLoader, *, split: str, disable: bool = False) -> Any:
        description = f"[eval:{split}]"
        try:
            from tqdm.auto import tqdm

            return tqdm(eval_dataloader, desc=description, total=len(eval_dataloader), leave=True, disable=disable)
        except ModuleNotFoundError:
            print(f"{description} progress logging enabled without tqdm; total_batches={len(eval_dataloader)}")
            return eval_dataloader

    @staticmethod
    def _create_train_progress_bar(train_dataloader: DataLoader, *, epoch: int, max_epochs: int, disable: bool) -> Any:
        description = f"[train:{epoch}/{max_epochs}]"
        try:
            from tqdm.auto import tqdm

            return tqdm(train_dataloader, desc=description, total=len(train_dataloader), leave=True, disable=disable)
        except ModuleNotFoundError:
            print(f"{description} progress logging enabled without tqdm; total_batches={len(train_dataloader)}")
            return train_dataloader

    @staticmethod
    def _mean_or_none(values: Sequence[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))


class _ExistingAcceleratorEvalContext:
    """Evaluation helper when AcceleratorState is already initialized (e.g. post-GRPO on rank 0)."""

    @property
    def device(self) -> torch.device:
        from accelerate.state import AcceleratorState

        return AcceleratorState().device

    @property
    def is_main_process(self) -> bool:
        from accelerate import PartialState

        return PartialState().is_main_process

    def wait_for_everyone(self) -> None:
        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            distributed.barrier()

    def prepare(self, *args: Any) -> Any:
        if len(args) == 1:
            return args[0]
        model, dataloader = args
        return model, dataloader

    @staticmethod
    def unwrap_model(model: Any) -> Any:
        return model

    @contextmanager
    def accumulate(self, model: Any) -> Iterator[None]:
        yield

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:
        loss.backward(**kwargs)

    @staticmethod
    def print(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)


__all__ = [
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "MonitorSpec",
    "OptimizerConfig",
    "SchedulerConfig",
    "Trainer",
    "TrainerConfig",
]
