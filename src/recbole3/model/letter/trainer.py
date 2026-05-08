from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from recbole3.model.rqvae.trainer import RQVAETrainer


class LETTERTrainer(RQVAETrainer):
    """LETTER trainer built on top of RQ-VAE trainer with external diversity labels."""

    @staticmethod
    def _constrained_km(data: Any, n_clusters: int = 10) -> tuple[Any, list[int]]:
        from k_means_constrained import KMeansConstrained
        import torch

        x = data
        num_points = len(x)
        if num_points < int(n_clusters):
            raise ValueError(
                f"KMeansConstrained requires at least n_clusters points, got {num_points} < {int(n_clusters)}."
            )
        size_min = max(1, min(num_points // (int(n_clusters) * 2), 10))
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=n_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        return t_centers, t_labels

    def _build_diversity_labels(self, model: Any) -> dict[str, list[int]]:
        n_clusters = 10
        embs = [layer.get_code_embs().detach().cpu().numpy() for layer in model._rq_layer.vq_layers]
        labels: dict[str, list[int]] = {}
        for idx, emb in enumerate(embs):
            _, label = self._constrained_km(emb, n_clusters=n_clusters)
            labels[str(idx)] = label
        return labels

    def _sync_diversity_labels(self, model: Any, accelerator: Any) -> dict[str, list[int]]:
        labels = self._build_diversity_labels(model) if accelerator.is_main_process else None
        label_payload = [labels]

        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            distributed.broadcast_object_list(label_payload, src=0)
        synced_labels = label_payload[0]
        if synced_labels is None:
            raise RuntimeError("Failed to synchronize LETTER diversity labels from the main process.")
        return synced_labels

    @staticmethod
    def _reduce_sum(tensor: Any, accelerator: Any) -> Any:
        if hasattr(accelerator, "reduce"):
            return accelerator.reduce(tensor, reduction="sum")

        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        return tensor

    @staticmethod
    def _gather_for_metrics(tensor: Any, accelerator: Any) -> Any:
        if hasattr(accelerator, "gather_for_metrics"):
            return accelerator.gather_for_metrics(tensor)
        if hasattr(accelerator, "gather"):
            return accelerator.gather(tensor)
        return tensor

    def fit(
        self,
        model: Any,
        prepared_data: Any,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        accelerator = self.create_accelerator()
        if accelerator.is_main_process:
            print(model)

        collator = model.build_train_collator(prepared_data)
        train_dataset = prepared_data.get_train_dataset()
        train_dataloader = self.build_dataloader(train_dataset, collator, shuffle=self.config.shuffle)
        monitor_name = str(self.config.monitor or "").strip()
        monitor = None
        if monitor_name:
            from recbole3.trainer import MonitorSpec
            monitor = MonitorSpec(name=monitor_name, higher_is_better=False)
        checkpoint_paths = self._resolve_checkpoint_paths(output_dir)
        steps_per_epoch = max(1, len(train_dataloader))
        num_training_steps = max(1, steps_per_epoch * self.config.max_epochs)
        scheduler_interval = self.config.scheduler.interval if self.config.scheduler is not None else None

        # Initialize codebook before DDP wrapping so parameters are consistent across ranks.
        from tqdm import tqdm
        import torch

        if getattr(model, "_initted", False) is False:
            if accelerator.is_main_process:
                all_embeddings = []
                with torch.no_grad():
                    for batch in tqdm(
                        train_dataloader,
                        desc="Collecting embeddings for codebook init",
                        disable=not accelerator.is_main_process,
                    ):
                        all_embeddings.append(batch["item_embeddings"])
                all_embeddings = torch.cat(all_embeddings, dim=0)
                model.init_codebook(all_embeddings)
            accelerator.wait_for_everyone()

        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            steps_per_epoch=steps_per_epoch,
        )

        if scheduler is None:
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        else:
            model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

        train_history: list[dict[str, Any]] = []
        valid_history: list[dict[str, Any]] = []
        self._collision_rates = {"valid": [], "test": []}
        best_epoch: int | None = None
        best_value: float | None = None
        bad_epoch_count = 0
        stopped_early = False

        eval_steps = max(1, int(getattr(self.config, "eval_steps", 1) or 1))
        total_epochs = int(self.config.max_epochs)
        self._cached_eval_dataloader: Any | None = None
        self._cached_eval_split: str | None = None

        for epoch in tqdm(range(1, total_epochs + 1), desc="Training"):
            epoch_start = time.perf_counter()
            model.train()

            unwrapped_model = accelerator.unwrap_model(model)
            diversity_labels = self._sync_diversity_labels(unwrapped_model, accelerator)
            unwrapped_model.set_diversity_labels(diversity_labels)

            total_losses: list[float] = []
            recon_losses: list[float] = []
            quant_losses: list[float] = []
            cf_losses: list[float] = []
            unused_codes: list[int] = []

            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model.forward(batch)
                    loss_dict = unwrapped_model.compute_loss(batch, outputs)

                    accelerator.backward(loss_dict["loss"])
                    optimizer.step()
                    if scheduler is not None and scheduler_interval == "step":
                        scheduler.step()

                total_losses.append(float(loss_dict["loss"].detach().float().item()))
                recon_losses.append(float(loss_dict["recon_loss"].detach().float().item()))
                quant_losses.append(float(loss_dict["quant_loss"].detach().float().item()))
                cf_losses.append(float(loss_dict["cf_loss"].detach().float().item()))
                unused_codes.append(int(outputs["unused_codes"]))

            elapsed = time.perf_counter() - epoch_start
            lr = optimizer.param_groups[0].get("lr", None)
            avg_total_loss = self._mean_or_none(total_losses)
            avg_recon_loss = self._mean_or_none(recon_losses)
            avg_quant_loss = self._mean_or_none(quant_losses)
            avg_cf_loss = self._mean_or_none(cf_losses)
            avg_unused_codes = self._mean_or_none(unused_codes)

            train_history.append(
                {
                    "epoch": epoch,
                    "loss": avg_total_loss,
                    "losses": total_losses,
                    "recon_loss": avg_recon_loss,
                    "quant_loss": avg_quant_loss,
                    "cf_loss": avg_cf_loss,
                    "unused_codes": avg_unused_codes,
                    "num_batches": len(total_losses),
                    "elapsed_seconds": elapsed,
                    "lr": lr,
                }
            )

            print(
                f"[train] epoch={epoch}/{total_epochs} "
                f"avg_loss={(f'{avg_total_loss:.6f}' if avg_total_loss is not None else 'n/a')} "
                f"recon_loss={(f'{avg_recon_loss:.6f}' if avg_recon_loss is not None else 'n/a')} "
                f"num_batches={len(total_losses)}"
            )
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_epoch(
                    epoch=epoch,
                    max_epochs=total_epochs,
                    loss=avg_total_loss,
                    num_batches=len(total_losses),
                    elapsed_seconds=elapsed,
                    lr=lr,
                    recon_loss=avg_recon_loss,
                    quant_loss=avg_quant_loss,
                    cf_loss=avg_cf_loss,
                    unused_codes=avg_unused_codes,
                )

            should_evaluate = (epoch % eval_steps == 0) or (epoch == total_epochs)
            current_value: float | None = None
            improved = False
            if should_evaluate:
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
                            accelerator.wait_for_everyone()
                    elif self.config.early_stopping.enabled:
                        bad_epoch_count += 1

            if scheduler is not None and scheduler_interval == "epoch":
                if self._scheduler_requires_monitor():
                    if current_value is not None:
                        self._step_epoch_scheduler(scheduler, current_value=current_value)
                else:
                    self._step_epoch_scheduler(scheduler, current_value=current_value)
            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])
                accelerator.wait_for_everyone()
            if (
                should_evaluate
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
                    epoch=total_epochs,
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

    def _run_evaluation(
        self,
        model: Any,
        prepared_data: Any,
        split: str,
        accelerator: Any,
        model_is_prepared: bool,
    ) -> dict[str, Any]:
        unwrapped_for_labels = accelerator.unwrap_model(model)
        diversity_labels = self._sync_diversity_labels(unwrapped_for_labels, accelerator)
        unwrapped_for_labels.set_diversity_labels(diversity_labels)

        collator_model = accelerator.unwrap_model(model)
        cached = getattr(self, "_cached_eval_dataloader", None)
        cached_split = getattr(self, "_cached_eval_split", None)
        if cached is not None and cached_split == split and model_is_prepared:
            prepared_model = model
            eval_dataloader = cached
        else:
            eval_dataset = prepared_data.get_eval_dataset(split)
            eval_collate_fn = collator_model.build_eval_collator(prepared_data)
            eval_dataloader = self.build_dataloader(eval_dataset, eval_collate_fn, shuffle=False)
            if model_is_prepared:
                prepared_model = model
                eval_dataloader = accelerator.prepare(eval_dataloader)
            else:
                prepared_model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
            if model_is_prepared:
                self._cached_eval_dataloader = eval_dataloader
                self._cached_eval_split = split

        import torch
        import torch.nn.functional as F

        prepared_model.eval()
        all_tokens: list[torch.Tensor] = []
        per_example_metrics: list[torch.Tensor] = []
        unused_codes_sum = torch.zeros((), dtype=torch.float64, device=accelerator.device)
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = prepared_model.forward(batch)

                all_tokens.append(self._gather_for_metrics(outputs["tokens"].detach(), accelerator).cpu())
                per_example_metrics.append(
                    self._gather_for_metrics(
                        self._compute_eval_example_metrics(unwrapped_for_labels.config, batch, outputs, F),
                        accelerator,
                    ).double().cpu()
                )
                unused_codes_sum += float(outputs["unused_codes"])
                num_batches += 1

        if all_tokens:
            all_tokens_tensor = torch.cat(all_tokens, dim=0)
        else:
            all_tokens_tensor = torch.empty(
                (0, int(unwrapped_for_labels.config.codebook_num)),
                dtype=torch.long,
            )
        collision_rate = self._compute_collision_rate(all_tokens_tensor)
        self._collision_rates[split].append(collision_rate)

        reduced_num_batches = int(
            self._reduce_sum(torch.tensor(num_batches, dtype=torch.long, device=accelerator.device), accelerator)
            .cpu()
            .item()
        )
        all_example_metrics = torch.cat(per_example_metrics, dim=0) if per_example_metrics else torch.empty((0, 4))
        loss_means = [float(value.item()) for value in all_example_metrics.mean(dim=0)] if all_example_metrics.numel() else [None, None, None, None]
        reduced_unused_codes_sum = self._reduce_sum(unused_codes_sum, accelerator).cpu()
        unused_codes_mean = (
            float(reduced_unused_codes_sum.item()) / reduced_num_batches
            if reduced_num_batches > 0
            else None
        )

        metrics = {
            "collision_rate": collision_rate,
            "loss": loss_means[0],
            "recon_loss": loss_means[1],
            "quant_loss": loss_means[2],
            "cf_loss": loss_means[3],
            "unused_codes": unused_codes_mean,
        }
        return {
            "split": split,
            "protocol": "full",
            "loss": loss_means[0],
            "metrics": metrics,
            "num_batches": reduced_num_batches,
            "data_stats": self._build_result_data_stats(prepared_data),
        }

    def _compute_eval_example_metrics(self, config: Any, batch: dict[str, Any], outputs: dict[str, Any], functional: Any) -> Any:
        import torch

        if config.loss_type == "mse":
            recon_loss = functional.mse_loss(outputs["reconstruction"], batch["item_embeddings"], reduction="none").mean(dim=1)
        elif config.loss_type == "l1":
            recon_loss = functional.l1_loss(outputs["reconstruction"], batch["item_embeddings"], reduction="none").mean(dim=1)
        else:
            raise ValueError(f"Unsupported loss_type: {config.loss_type}. Expected 'mse' or 'l1'.")

        quant_loss = outputs["quant_loss"].detach().expand_as(recon_loss) * config.quant_loss_weight
        similarities = torch.matmul(outputs["quantized"], batch["cf_embeddings"].transpose(0, 1))
        labels = torch.arange(outputs["quantized"].size(0), dtype=torch.long, device=outputs["quantized"].device)
        cf_loss = functional.cross_entropy(similarities, labels, reduction="none")
        loss = recon_loss + quant_loss + config.cf_loss_weight * cf_loss
        return torch.stack([loss, recon_loss, quant_loss, cf_loss], dim=1).detach()

    def generate_sids(
        self,
        model: Any,
        prepared_data: Any,
        *,
        output_dir: str | Path,
    ) -> None:
        accelerator = self.create_accelerator()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            diversity_labels = self._build_diversity_labels(unwrapped_model)
            unwrapped_model.set_diversity_labels(diversity_labels)
        super().generate_sids(model, prepared_data, output_dir=output_dir)


__all__ = ["LETTERTrainer"]
