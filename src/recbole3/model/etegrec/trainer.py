from __future__ import annotations

import time
from collections.abc import Mapping
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from transformers.optimization import get_scheduler

from recbole3.dataset import BaseTaskDataset
from recbole3.model.base import BaseModel
from recbole3.model.etegrec.config import ETEGRecTrainerConfig
from recbole3.model.etegrec.model import ETEGRecModel
from recbole3.trainer import Trainer


class ETEGRecTrainer(Trainer):
    """Trainer for ETEGRec's alternating recommender/tokenizer optimization."""

    config: ETEGRecTrainerConfig
    config_cls = ETEGRecTrainerConfig

    def __init__(self, config: ETEGRecTrainerConfig):
        super().__init__(config)
        self._building_eval_dataloader = False

    def create_accelerator(self) -> Any:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        return Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

    def build_dataloader(self, dataset: Any, collate_fn: Any, *, shuffle: bool) -> Any:
        if not self._building_eval_dataloader or self.config.eval_batch_size is None:
            return super().build_dataloader(dataset, collate_fn, shuffle=shuffle)
        eval_batch_size = int(self.config.eval_batch_size)
        if eval_batch_size <= 0:
            raise ValueError("ETEGRec eval_batch_size must be positive.")

        original_batch_size = self.config.batch_size
        self.config.batch_size = eval_batch_size
        try:
            return super().build_dataloader(dataset, collate_fn, shuffle=shuffle)
        finally:
            self.config.batch_size = original_batch_size

    def build_optimizers(self, model: ETEGRecModel) -> tuple[Optimizer, Optimizer]:
        optimizer_cls = self._resolve_optimizer_class(self.config.optimizer.name)
        rec_optimizer = optimizer_cls(
            list(model.recommender_parameters()),
            lr=float(self.config.lr_rec),
            weight_decay=float(self.config.optimizer.kwargs.get("weight_decay", 0.0)),
        )
        id_optimizer = optimizer_cls(
            list(model.rqvae_parameters()),
            lr=float(self.config.lr_id),
            weight_decay=float(self.config.optimizer.kwargs.get("weight_decay", 0.0)),
        )
        return rec_optimizer, id_optimizer

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        if not isinstance(model, ETEGRecModel):
            raise TypeError("ETEGRecTrainer requires ETEGRecModel.")
        if int(self.config.cycle) <= 0:
            raise ValueError("ETEGRecTrainerConfig.cycle must be positive.")

        accelerator = self.create_accelerator()
        collator = model.build_train_collator(prepared_data)
        train_dataloader = self.build_dataloader(prepared_data.get_train_dataset(), collator, shuffle=self.config.shuffle)
        rec_optimizer, id_optimizer = self.build_optimizers(model)
        scheduler_steps = self._resolve_scheduler_steps(train_dataloader)
        rec_scheduler = self._build_scheduler(
            rec_optimizer,
            total_steps=scheduler_steps["rec_total_steps"],
            warmup_steps=scheduler_steps["rec_warmup_steps"],
        )
        id_scheduler = self._build_scheduler(
            id_optimizer,
            total_steps=scheduler_steps["id_total_steps"],
            warmup_steps=scheduler_steps["id_warmup_steps"],
        )
        monitor = self._resolve_monitor(prepared_data)
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)

        (
            model,
            rec_optimizer,
            id_optimizer,
            train_dataloader,
            rec_scheduler,
            id_scheduler,
        ) = self._prepare_joint_training_components(
            accelerator,
            model,
            rec_optimizer,
            id_optimizer,
            train_dataloader,
            rec_scheduler,
            id_scheduler,
        )
        accelerator.unwrap_model(model).refresh_item_codes()

        train_history: list[dict[str, Any]] = []
        valid_history: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_value: float | None = None
        bad_epoch_count = 0
        stopped_early = False
        eval_steps = max(1, int(self.config.eval_steps))

        for epoch in range(1, int(self.config.max_epochs) + 1):
            epoch_start = time.perf_counter()
            train_tokenizer = ((epoch - 1) % int(self.config.cycle)) == 0
            mode = "rqvae" if train_tokenizer else "rec"
            losses = self._train_epoch(
                model=model,
                dataloader=train_dataloader,
                rec_optimizer=rec_optimizer,
                id_optimizer=id_optimizer,
                rec_scheduler=rec_scheduler,
                id_scheduler=id_scheduler,
                accelerator=accelerator,
                train_tokenizer=train_tokenizer,
                epoch=epoch,
            )
            if train_tokenizer:
                accelerator.unwrap_model(model).refresh_item_codes()

            epoch_loss = self._mean_or_none(losses)
            elapsed = time.perf_counter() - epoch_start
            train_history.append(
                {
                    "epoch": epoch,
                    "mode": mode,
                    "loss": epoch_loss,
                    "losses": losses,
                    "num_batches": len(losses),
                    "elapsed_seconds": elapsed,
                    "lr_rec": rec_optimizer.param_groups[0].get("lr"),
                    "lr_id": id_optimizer.param_groups[0].get("lr"),
                }
            )
            print(
                f"[etegrec:{mode}] epoch={epoch}/{int(self.config.max_epochs)} "
                f"avg_loss={(f'{epoch_loss:.6f}' if epoch_loss is not None else 'n/a')} "
                f"num_batches={len(losses)}"
            )

            valid_result: dict[str, Any] | None = None
            if (epoch % eval_steps == 0) or (epoch == int(self.config.max_epochs)):
                accelerator.unwrap_model(model).refresh_item_codes()
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
                        self._save_model_checkpoint_and_wait(model, accelerator, checkpoint_paths["best"])
                elif self.config.early_stopping.enabled:
                    bad_epoch_count += 1

            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint_and_wait(model, accelerator, checkpoint_paths["last"])

            if (
                valid_result is not None
                and self.config.early_stopping.enabled
                and not improved
                and bad_epoch_count >= int(self.config.early_stopping.patience)
            ):
                stopped_early = True
                break

        result = {
            "train_history": train_history,
            "valid_history": valid_history,
            "data_stats": self._build_result_data_stats(prepared_data),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {key: (str(path) if path is not None else None) for key, path in checkpoint_paths.items()},
        }
        if not bool(self.config.finetune_enabled):
            return result

        finetune_result = self._finetune(
            model=model,
            prepared_data=prepared_data,
            train_dataloader=train_dataloader,
            accelerator=accelerator,
            monitor=monitor,
            joint_checkpoint_paths=checkpoint_paths,
            output_dir=output_dir,
        )
        return {
            **result,
            "joint_train_history": result["train_history"],
            "joint_valid_history": result["valid_history"],
            "joint_best_epoch": result["best_epoch"],
            "joint_best_metric": result["best_metric"],
            "joint_checkpoint_paths": result["checkpoint_paths"],
            "finetune_train_history": finetune_result["train_history"],
            "finetune_valid_history": finetune_result["valid_history"],
            "finetune_best_epoch": finetune_result["best_epoch"],
            "finetune_best_metric": finetune_result["best_metric"],
            "finetune_checkpoint_paths": finetune_result["checkpoint_paths"],
            "finetune_stopped_early": finetune_result["stopped_early"],
            "train_history": [*result["train_history"], *finetune_result["train_history"]],
            "valid_history": [*result["valid_history"], *finetune_result["valid_history"]],
            "stopped_early": finetune_result["stopped_early"],
            "best_epoch": finetune_result["best_epoch"],
            "best_metric": finetune_result["best_metric"],
            "checkpoint_paths": finetune_result["checkpoint_paths"],
        }

    def evaluate(self, model: BaseModel, prepared_data: BaseTaskDataset, split: str = "valid") -> dict[str, Any]:
        if not isinstance(model, ETEGRecModel):
            raise TypeError("ETEGRecTrainer requires ETEGRecModel.")
        model.ensure_initialized(prepared_data)
        model.refresh_item_codes()
        return super().evaluate(model, prepared_data, split=split)  # type: ignore[arg-type]

    def _run_evaluation(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        split: Any,
        accelerator: Any,
        model_is_prepared: bool,
    ) -> dict[str, Any]:
        if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
            self._wait_for_everyone(accelerator)
            result = None
            if bool(getattr(accelerator, "is_main_process", True)):
                eval_model = accelerator.unwrap_model(model)
                result = self._run_unsharded_evaluation_on_main_process(
                    eval_model,
                    prepared_data,
                    split=split,
                )
            result = self._broadcast_evaluation_result(result, accelerator)
            self._wait_for_everyone(accelerator)
            return result

        self._building_eval_dataloader = True
        try:
            return super()._run_evaluation(
                model,
                prepared_data,
                split=split,
                accelerator=accelerator,
                model_is_prepared=model_is_prepared,
            )
        finally:
            self._building_eval_dataloader = False

    def _run_unsharded_evaluation_on_main_process(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        split: Any,
    ) -> dict[str, Any]:
        method = self.create_evaluation_method(prepared_data)
        eval_dataset = prepared_data.get_eval_dataset(split)
        eval_collate_fn = method.build_eval_collate_fn(model, prepared_data)
        self._building_eval_dataloader = True
        try:
            eval_dataloader = self.build_dataloader(eval_dataset, eval_collate_fn, shuffle=False)
        finally:
            self._building_eval_dataloader = False

        model.eval()
        device = self._model_device(model)
        batch_eval_data: list[Any] = []
        num_batches = 0
        progress_bar = self._create_progress_bar(eval_dataloader, split=split)
        with torch.no_grad():
            for model_inputs, records in progress_bar:
                model_inputs = self._move_to_device(model_inputs, device)
                batch_eval_data.append(self._collect_eval_batch(method, model, model_inputs, records))
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

    def _train_epoch(
        self,
        *,
        model: ETEGRecModel,
        dataloader: Any,
        rec_optimizer: Optimizer,
        id_optimizer: Optimizer,
        rec_scheduler: Any | None,
        id_scheduler: Any | None,
        accelerator: Any,
        train_tokenizer: bool,
        epoch: int,
    ) -> list[float]:
        model.train()
        losses: list[float] = []
        progress_bar = self._create_train_progress_bar(dataloader, epoch=epoch, max_epochs=int(self.config.max_epochs))
        for batch in progress_bar:
            if train_tokenizer:
                loss = self._train_tokenizer_step(model, batch, id_optimizer, id_scheduler, accelerator, epoch=epoch)
            else:
                loss = self._train_recommender_step(model, batch, rec_optimizer, rec_scheduler, accelerator, epoch=epoch)
            losses.append(float(loss.detach().float().item()))
        if hasattr(progress_bar, "close"):
            progress_bar.close()
        return losses

    def _train_recommender_step(
        self,
        model: ETEGRecModel,
        batch: dict[str, torch.Tensor],
        optimizer: Optimizer,
        scheduler: Any | None,
        accelerator: Any,
        *,
        epoch: int,
    ) -> torch.Tensor:
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            unwrapped_model = accelerator.unwrap_model(model)
            self._set_train_mode_for_recommender_step(unwrapped_model)
            loss_parts = model(
                batch,
                mode="rec_loss",
                use_alignment=self._use_alignment(epoch),
                rec_code_loss=float(self.config.rec_code_loss),
                rec_kl_loss=float(self.config.rec_kl_loss),
                rec_dec_cl_loss=float(self.config.rec_dec_cl_loss),
            )
            loss = loss_parts["loss"]
            accelerator.backward(loss)
            self._clip_grad_norm(accelerator, unwrapped_model.recommender_parameters())
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return loss

    def _train_tokenizer_step(
        self,
        model: ETEGRecModel,
        batch: dict[str, torch.Tensor],
        optimizer: Optimizer,
        scheduler: Any | None,
        accelerator: Any,
        *,
        epoch: int,
    ) -> torch.Tensor:
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            unwrapped_model = accelerator.unwrap_model(model)
            self._set_train_mode_for_tokenizer_step(unwrapped_model)
            loss_parts = model(
                batch,
                mode="tokenizer_loss",
                use_alignment=self._use_alignment(epoch),
                id_vq_loss=float(self.config.id_vq_loss),
                id_code_loss=float(self.config.id_code_loss),
                id_kl_loss=float(self.config.id_kl_loss),
                id_dec_cl_loss=float(self.config.id_dec_cl_loss),
            )
            loss = loss_parts["loss"]
            accelerator.backward(loss)
            self._clip_grad_norm(accelerator, unwrapped_model.rqvae_parameters())
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return loss

    def _get_num_update_steps_per_epoch(self, train_dataloader: Any) -> int:
        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        return max(len(train_dataloader) // accumulation_steps, 1)

    def _get_total_training_steps(self, train_dataloader: Any) -> int:
        update_steps_per_epoch = self._get_num_update_steps_per_epoch(train_dataloader)
        return max(ceil(int(self.config.max_epochs) * update_steps_per_epoch), 1)

    def _resolve_scheduler_steps(self, train_dataloader: Any) -> dict[str, int]:
        max_steps = self._get_total_training_steps(train_dataloader)
        cycle = max(1, int(self.config.cycle))
        warmup_steps = max(0, int(self.config.warmup_steps))
        rec_warmup_steps = (
            int(self.config.rec_warmup_steps)
            if self.config.rec_warmup_steps is not None
            else warmup_steps
        )
        id_warmup_steps = (
            int(self.config.id_warmup_steps)
            if self.config.id_warmup_steps is not None
            else warmup_steps // cycle
        )
        return {
            "rec_total_steps": max_steps,
            "id_total_steps": max(max_steps // cycle, 1),
            "rec_warmup_steps": max(0, rec_warmup_steps),
            "id_warmup_steps": max(0, id_warmup_steps),
        }

    def _build_scheduler(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        scheduler_type: str | None = None,
        use_config_scheduler: bool = True,
    ) -> Any | None:
        if scheduler_type is None and use_config_scheduler:
            scheduler_type = self.config.lr_scheduler_type
        if scheduler_type is None or str(scheduler_type).strip().lower() in {"", "none", "null"}:
            return None
        scheduler_name = str(scheduler_type).strip().lower()
        if scheduler_name not in {"cosine", "linear", "constant"}:
            raise ValueError(f"Unsupported ETEGRec lr_scheduler_type: {self.config.lr_scheduler_type!r}")
        return get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=max(0, int(warmup_steps)),
            num_training_steps=max(1, int(total_steps)),
        )

    def _finetune(
        self,
        *,
        model: ETEGRecModel,
        prepared_data: BaseTaskDataset,
        train_dataloader: Any,
        accelerator: Any,
        monitor: Any,
        joint_checkpoint_paths: dict[str, Path | None],
        output_dir: str | Path | None,
    ) -> dict[str, Any]:
        joint_best_path = joint_checkpoint_paths["best"]
        if joint_best_path is None or not joint_best_path.exists():
            raise ValueError("ETEGRec finetune requires an existing joint best checkpoint.")

        unwrapped_model = accelerator.unwrap_model(model)
        self._load_model_checkpoint_with_barrier(unwrapped_model, joint_best_path, accelerator)
        unwrapped_model.refresh_item_codes()

        finetune_paths = self._resolve_finetune_checkpoint_paths(output_dir)
        optimizer = self._build_finetune_optimizer(unwrapped_model)
        scheduler = self._build_scheduler(
            optimizer,
            total_steps=self._get_finetune_total_training_steps(train_dataloader),
            warmup_steps=int(self.config.finetune_warmup_steps),
            scheduler_type=self.config.finetune_lr_scheduler_type,
            use_config_scheduler=False,
        )
        optimizer, scheduler = self._prepare_finetune_components(accelerator, optimizer, scheduler)

        train_history: list[dict[str, Any]] = []
        valid_history: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_value: float | None = 0.0 if monitor is not None and monitor.higher_is_better else None
        bad_epoch_count = 0
        stopped_early = False
        eval_steps = max(1, int(self.config.finetune_eval_steps))

        for epoch in range(1, int(self.config.finetune_epochs) + 1):
            epoch_start = time.perf_counter()
            losses = self._finetune_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                epoch=epoch,
            )
            epoch_loss = self._mean_or_none(losses)
            elapsed = time.perf_counter() - epoch_start
            train_history.append(
                {
                    "epoch": epoch,
                    "mode": "finetune",
                    "loss": epoch_loss,
                    "losses": losses,
                    "num_batches": len(losses),
                    "elapsed_seconds": elapsed,
                    "lr_rec": optimizer.param_groups[0].get("lr"),
                }
            )
            print(
                f"[etegrec:finetune] epoch={epoch}/{int(self.config.finetune_epochs)} "
                f"avg_loss={(f'{epoch_loss:.6f}' if epoch_loss is not None else 'n/a')} "
                f"num_batches={len(losses)}"
            )

            valid_result: dict[str, Any] | None = None
            if (epoch % eval_steps == 0) or (epoch == int(self.config.finetune_epochs)):
                accelerator.unwrap_model(model).refresh_item_codes()
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
                    f"[eval:finetune-valid] epoch={epoch}/{int(self.config.finetune_epochs)} "
                    f"metrics={self._format_metrics(valid_result['metrics'])}"
                )

            improved = False
            if monitor is not None and valid_result is not None:
                current_value = self._extract_monitor_value(valid_result["metrics"], monitor.name)
                improved = best_epoch is None or self._is_improvement(
                    current_value,
                    best_value,
                    higher_is_better=monitor.higher_is_better,
                    min_delta=float(self.config.early_stopping.min_delta),
                )
                if improved:
                    best_value = current_value
                    best_epoch = epoch
                    bad_epoch_count = 0
                    if finetune_paths["best"] is not None:
                        self._save_model_checkpoint_and_wait(model, accelerator, finetune_paths["best"])
                else:
                    bad_epoch_count += 1

            if finetune_paths["last"] is not None:
                self._save_model_checkpoint_and_wait(model, accelerator, finetune_paths["last"])

            if valid_result is not None and not improved and bad_epoch_count >= int(self.config.finetune_patience):
                stopped_early = True
                break

        best_path = finetune_paths["best"]
        if best_path is None or not best_path.exists():
            raise ValueError("ETEGRec finetune did not produce a best checkpoint.")
        return {
            "train_history": train_history,
            "valid_history": valid_history,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric": self._build_best_metric_payload(monitor, best_value),
            "checkpoint_paths": {key: (str(path) if path is not None else None) for key, path in finetune_paths.items()},
        }

    def _finetune_epoch(
        self,
        *,
        model: ETEGRecModel,
        dataloader: Any,
        optimizer: Optimizer,
        scheduler: Any | None,
        accelerator: Any,
        epoch: int,
    ) -> list[float]:
        model.train()
        losses: list[float] = []
        progress_bar = self._create_train_progress_bar(
            dataloader,
            epoch=epoch,
            max_epochs=int(self.config.finetune_epochs),
        )
        for batch in progress_bar:
            loss = self._finetune_step(model, batch, optimizer, scheduler, accelerator)
            losses.append(float(loss.detach().float().item()))
        if hasattr(progress_bar, "close"):
            progress_bar.close()
        return losses

    def _finetune_step(
        self,
        model: ETEGRecModel,
        batch: dict[str, torch.Tensor],
        optimizer: Optimizer,
        scheduler: Any | None,
        accelerator: Any,
    ) -> torch.Tensor:
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            unwrapped_model = accelerator.unwrap_model(model)
            self._set_train_mode_for_recommender_step(unwrapped_model)
            outputs = model(batch, mode="finetune")
            loss = outputs["loss"]
            accelerator.backward(loss)
            self._clip_grad_norm(accelerator, unwrapped_model.recommender_parameters())
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return loss

    def _build_finetune_optimizer(self, model: ETEGRecModel) -> Optimizer:
        optimizer_cls = self._resolve_optimizer_class(self.config.optimizer.name)
        return optimizer_cls(
            list(model.recommender_parameters()),
            lr=float(self.config.finetune_lr),
            weight_decay=float(self.config.optimizer.kwargs.get("weight_decay", 0.0)),
        )

    def _get_finetune_total_training_steps(self, train_dataloader: Any) -> int:
        update_steps_per_epoch = self._get_num_update_steps_per_epoch(train_dataloader)
        return max(ceil(int(self.config.finetune_epochs) * update_steps_per_epoch), 1)

    def _resolve_finetune_checkpoint_paths(self, output_dir: str | Path | None) -> dict[str, Path]:
        if output_dir is None:
            raise ValueError("ETEGRec finetune checkpoint saving requires output_dir.")
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return {
            "best": checkpoint_dir / "finetune_best_model.pt",
            "last": checkpoint_dir / "finetune_last_model.pt",
        }

    def _load_model_checkpoint(self, model: ETEGRecModel, path: Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    def _save_model_checkpoint_and_wait(self, model: ETEGRecModel, accelerator: Any, path: Path) -> None:
        self._save_model_checkpoint(model, accelerator, path)
        self._wait_for_everyone(accelerator)

    def _load_model_checkpoint_with_barrier(self, model: ETEGRecModel, path: Path, accelerator: Any) -> None:
        self._wait_for_everyone(accelerator)
        self._load_model_checkpoint(model, path)
        self._wait_for_everyone(accelerator)

    def _prepare_joint_training_components(
        self,
        accelerator: Any,
        model: ETEGRecModel,
        rec_optimizer: Optimizer,
        id_optimizer: Optimizer,
        train_dataloader: Any,
        rec_scheduler: Any | None,
        id_scheduler: Any | None,
    ) -> tuple[Any, Optimizer, Optimizer, Any, Any | None, Any | None]:
        components: list[Any] = [model, rec_optimizer, id_optimizer, train_dataloader]
        names = ["model", "rec_optimizer", "id_optimizer", "train_dataloader"]
        if rec_scheduler is not None:
            components.append(rec_scheduler)
            names.append("rec_scheduler")
        if id_scheduler is not None:
            components.append(id_scheduler)
            names.append("id_scheduler")
        prepared_values = accelerator.prepare(*components)
        if len(components) == 1:
            prepared_values = (prepared_values,)
        prepared = dict(zip(names, prepared_values, strict=True))
        return (
            prepared["model"],
            prepared["rec_optimizer"],
            prepared["id_optimizer"],
            prepared["train_dataloader"],
            prepared.get("rec_scheduler", rec_scheduler),
            prepared.get("id_scheduler", id_scheduler),
        )

    def _prepare_finetune_components(
        self,
        accelerator: Any,
        optimizer: Optimizer,
        scheduler: Any | None,
    ) -> tuple[Optimizer, Any | None]:
        if scheduler is None:
            return accelerator.prepare(optimizer), None
        prepared_optimizer, prepared_scheduler = accelerator.prepare(optimizer, scheduler)
        return prepared_optimizer, prepared_scheduler

    @staticmethod
    def _wait_for_everyone(accelerator: Any) -> None:
        wait = getattr(accelerator, "wait_for_everyone", None)
        if wait is not None:
            wait()

    @staticmethod
    def _model_device(model: BaseModel) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_to_device(self, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        return value

    @staticmethod
    def _broadcast_evaluation_result(result: dict[str, Any] | None, accelerator: Any) -> dict[str, Any]:
        if accelerator is None or int(getattr(accelerator, "num_processes", 1)) <= 1:
            if result is None:
                raise RuntimeError("ETEGRec evaluation result is missing.")
            return result

        object_list = [result if bool(getattr(accelerator, "is_main_process", True)) else None]
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("ETEGRec multi-GPU evaluation requires an initialized torch.distributed process group.")
        torch.distributed.broadcast_object_list(object_list, src=0)
        if object_list[0] is None:
            raise RuntimeError("ETEGRec multi-GPU evaluation broadcast returned no result.")
        return object_list[0]

    def _build_recommender_loss_parts(
        self,
        model: ETEGRecModel,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        code_loss = model.compute_loss(batch, outputs)
        zero = code_loss * 0
        use_alignment = self._use_alignment(epoch)
        kl_loss = zero
        dec_cl_loss = zero
        if use_alignment:
            alignment_parts = model.compute_alignment_loss_parts(batch, outputs)
            kl_loss = alignment_parts["kl_loss"]
            dec_cl_loss = alignment_parts["dec_cl_loss"]
        loss = (
            float(self.config.rec_code_loss) * code_loss
            + float(self.config.rec_kl_loss) * kl_loss
            + float(self.config.rec_dec_cl_loss) * dec_cl_loss
        )
        return {
            "loss": loss,
            "code_loss": code_loss,
            "kl_loss": kl_loss,
            "dec_cl_loss": dec_cl_loss,
        }

    def _build_tokenizer_loss_parts(
        self,
        model: ETEGRecModel,
        batch: dict[str, torch.Tensor],
        rqvae_parts: dict[str, torch.Tensor],
        *,
        outputs: dict[str, Any] | None = None,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        vq_loss = rqvae_parts["loss"]
        zero = vq_loss * 0
        use_alignment = self._use_alignment(epoch)
        code_loss = zero
        kl_loss = zero
        dec_cl_loss = zero
        if use_alignment:
            if outputs is None:
                outputs = model(batch)
            alignment_parts = model.compute_alignment_loss_parts(batch, outputs)
            code_loss = alignment_parts["code_loss"]
            kl_loss = alignment_parts["kl_loss"]
            dec_cl_loss = alignment_parts["dec_cl_loss"]
        loss = (
            float(self.config.id_vq_loss) * vq_loss
            + float(self.config.id_code_loss) * code_loss
            + float(self.config.id_kl_loss) * kl_loss
            + float(self.config.id_dec_cl_loss) * dec_cl_loss
        )
        return {
            "loss": loss,
            "vq_loss": vq_loss,
            "code_loss": code_loss,
            "kl_loss": kl_loss,
            "dec_cl_loss": dec_cl_loss,
            "recon_loss": rqvae_parts["recon_loss"],
            "rq_loss": rqvae_parts["rq_loss"],
        }

    def _use_alignment(self, epoch: int) -> bool:
        zero_based_epoch = int(epoch) - 1
        return zero_based_epoch >= int(self.config.warm_epoch)

    def _set_train_mode_for_recommender_step(self, model: ETEGRecModel) -> None:
        self._set_requires_grad(model.recommender_parameters(), True)
        self._set_requires_grad(model.rqvae_parameters(), False)
        if model.semantic_embedding is not None:
            model.semantic_embedding.requires_grad_(False)

    def _set_train_mode_for_tokenizer_step(self, model: ETEGRecModel) -> None:
        self._set_requires_grad(model.recommender_parameters(), False)
        self._set_requires_grad(model.rqvae_parameters(), True)
        if model.semantic_embedding is not None:
            model.semantic_embedding.requires_grad_(False)

    @staticmethod
    def _set_requires_grad(parameters: Any, enabled: bool) -> None:
        for parameter in parameters:
            parameter.requires_grad_(enabled)

    def _clip_grad_norm(self, accelerator: Any, parameters: Any) -> None:
        max_norm = float(self.config.gradient_clip_norm)
        if max_norm > 0:
            accelerator.clip_grad_norm_(list(parameters), max_norm)


__all__ = ["ETEGRecTrainer"]
