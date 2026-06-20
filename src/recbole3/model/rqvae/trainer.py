from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tqdm import tqdm
from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.base import BaseModel
from recbole3.trainer import Trainer, EvalSplitName, TrainerConfig, MonitorSpec


class RQVAETrainer(Trainer):
    """Custom trainer for RQ-VAE with collision rate tracking and RQ-VAE specific metrics.

    This trainer extends the base Trainer with:
    1. Tracking of collision rate (duplicate token assignments across items)
    2. Tracking of unused codebook entries
    3. Separate monitoring of reconstruction and quantization losses
    4. Custom evaluation method for RQ-VAE specific metrics
    """

    config_cls = TrainerConfig

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self._collision_rates: dict[Literal["valid", "test"], list[float]] = {"valid": [], "test": []}

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

            from accelerate.utils import extract_model_from_parallel

            model = extract_model_from_parallel(model)
            self._cached_eval_dataloader = None
            self._cached_eval_split = None

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
            test_result = self.evaluate(model, prepared_data, split="test")

            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                total_elapsed = time.perf_counter() - total_start
                logger.log_summary(
                    stopped_early=fit_result.get("stopped_early", False),
                    total_epochs=len(fit_result.get("train_history", [])),
                    best_epoch=fit_result.get("best_epoch"),
                    total_time=total_elapsed,
                )

            self.generate_sids(model, prepared_data, output_dir=output_dir)

            return {
                "fit": fit_result,
                "test": test_result,
            }
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> Any:
        """Fit the RQ-VAE model with extended logging.

        Args:
            model: The RQVAEModel to train.
            prepared_data: Prepared task data.
            output_dir: Optional output directory for checkpoints.

        Returns:
            Training result with extended RQ-VAE metrics.
        """
        accelerator = self.create_accelerator()
        if accelerator.is_main_process:
            print(model)

        collator = model.build_train_collator(prepared_data)
        train_dataset = prepared_data.get_train_dataset()
        train_dataloader = self.build_dataloader(train_dataset, collator, shuffle=self.config.shuffle)

        optimizer = self.build_optimizer(model)
        monitor_name = str(self.config.monitor or "").strip()
        monitor = MonitorSpec(name=monitor_name, higher_is_better=False) if monitor_name else None
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

        # Initialize codebook using K-means clustering on entire training data
        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Collecting embeddings for codebook init"):
                all_embeddings.append(batch["item_embeddings"])
        all_embeddings = torch.cat(all_embeddings, dim=0)
        model.init_codebook(all_embeddings)

        train_history: list[dict[str, Any]] = []
        valid_history: list[dict[str, Any]] = []
        self._collision_rates = {"valid": [], "test": []}
        eval_steps = max(1, int(self.config.eval_steps))
        best_epoch: int | None = None
        best_value: float | None = None
        bad_epoch_count = 0
        stopped_early = False

        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Training"):
            epoch_start = time.perf_counter()
            model.train()
            total_losses: list[float] = []
            recon_losses: list[float] = []
            quant_losses: list[float] = []
            unused_codes: list[int] = []

            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model.forward(batch)
                    loss_dict = model.compute_loss(batch, outputs)

                    accelerator.backward(loss_dict["loss"])
                    optimizer.step()
                    if scheduler is not None and scheduler_interval == "step":
                        scheduler.step()

                total_losses.append(float(loss_dict["loss"].detach().float().item()))
                recon_losses.append(float(loss_dict["recon_loss"].detach().float().item()))
                quant_losses.append(float(loss_dict["quant_loss"].detach().float().item()))
                unused_codes.append(int(outputs["unused_codes"]))

            elapsed = time.perf_counter() - epoch_start
            lr = optimizer.param_groups[0].get("lr", None)
            avg_total_loss = self._mean_or_none(total_losses)
            avg_recon_loss = self._mean_or_none(recon_losses)
            avg_quant_loss = self._mean_or_none(quant_losses)
            avg_unused_codes = self._mean_or_none(unused_codes)

            train_history.append(
                {
                    "epoch": epoch,
                    "loss": avg_total_loss,
                    "losses": total_losses,
                    "recon_loss": avg_recon_loss,
                    "quant_loss": avg_quant_loss,
                    "unused_codes": avg_unused_codes,
                    "num_batches": len(total_losses),
                    "elapsed_seconds": elapsed,
                    "lr": lr,
                }
            )

            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_epoch(
                    epoch=epoch,
                    max_epochs=int(self.config.max_epochs),
                    loss=avg_total_loss,
                    num_batches=len(total_losses),
                    elapsed_seconds=elapsed,
                    lr=lr,
                    recon_loss=avg_recon_loss,
                    quant_loss=avg_quant_loss,
                    unused_codes=avg_unused_codes,
                )

            current_value: float | None = None
            improved = False
            should_run_validation = (epoch % eval_steps == 0) or (epoch == int(self.config.max_epochs))
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
                    elif self.config.early_stopping.enabled:
                        bad_epoch_count += 1
            if scheduler is not None and scheduler_interval == "epoch":
                self._step_epoch_scheduler(scheduler, current_value=current_value)
            if checkpoint_paths["last"] is not None:
                self._save_model_checkpoint(model, accelerator, checkpoint_paths["last"])
            if should_run_validation and self.config.early_stopping.enabled and not improved and bad_epoch_count >= int(self.config.early_stopping.patience):
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

    def _run_evaluation(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        split: EvalSplitName,
        accelerator: Any,
        model_is_prepared: bool,
    ) -> dict[str, Any]:
        """Run evaluation with RQ-VAE specific metrics.

        Args:
            model: The RQVAEModel to evaluate.
            prepared_data: Prepared task data.
            split: Evaluation split ('valid' or 'test').
            accelerator: Accelerate accelerator instance.
            model_is_prepared: Whether model is already prepared by accelerator.

        Returns:
            Evaluation result with RQ-VAE metrics.
        """
        collator_model = accelerator.unwrap_model(model)
        eval_dataset = prepared_data.get_eval_dataset(split)
        eval_collate_fn = collator_model.build_eval_collator(prepared_data)
        eval_dataloader = self.build_dataloader(eval_dataset, eval_collate_fn, shuffle=False)
        if model_is_prepared:
            prepared_model = model
            eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            prepared_model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        prepared_model.eval()
        all_tokens: list[torch.Tensor] = []
        total_losses: list[float] = []
        total_recon_losses: list[float] = []
        total_quant_losses: list[float] = []
        total_unused_codes: list[int] = []
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = prepared_model.forward(batch)
                loss_dict = prepared_model.compute_loss(batch, outputs)

                all_tokens.append(outputs["tokens"].cpu())
                total_losses.append(float(loss_dict["loss"].item()))
                total_recon_losses.append(float(loss_dict["recon_loss"].item()))
                total_quant_losses.append(float(loss_dict["quant_loss"].item()))
                total_unused_codes.append(int(outputs["unused_codes"]))
                num_batches += 1

        # Compute RQ-VAE specific metrics
        all_tokens_tensor = torch.cat(all_tokens, dim=0)
        collision_rate = self._compute_collision_rate(all_tokens_tensor)

        self._collision_rates[split].append(collision_rate)

        # RQ-VAE uses reconstruction loss as the primary metric (lower is better)
        metrics = {
            "collision_rate": collision_rate,
            "loss": self._mean_or_none(total_losses),
            "recon_loss": self._mean_or_none(total_recon_losses),
            "quant_loss": self._mean_or_none(total_quant_losses),
            "unused_codes": self._mean_or_none(total_unused_codes),
        }

        return {
            "split": split,
            "protocol": "full",
            "loss": self._mean_or_none(total_losses),
            "metrics": metrics,
            "num_batches": num_batches,
            "data_stats": self._build_result_data_stats(prepared_data),
        }


    def generate_sids(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path,
    ) -> None:
        """Generate semantic IDs (SIDs) for all items.

        Args:
            model: The RQVAEModel to use for encoding.
            prepared_data: Prepared task data.
            split: Data split to use for SID generation ('test' uses all items).
            output_dir: Optional output directory for saving SIDs.
        """
        import json

        accelerator = self.create_accelerator()

        # Only generate SIDs on rank 0 to avoid duplicate work
        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            return

        # Get the unwrapped model to access config
        unwrapped_model = accelerator.unwrap_model(model)
        config = unwrapped_model.config

        # Create dataloader with all items (no shuffle to maintain order)
        eval_dataset = prepared_data.get_eval_dataset("test")
        eval_collate_fn = unwrapped_model.build_eval_collator(prepared_data)
        eval_dataloader = self.build_dataloader(eval_dataset, eval_collate_fn, shuffle=False)

        # Collect all tokens and item IDs
        unwrapped_model.eval()
        all_tokens: list[torch.Tensor] = []
        all_sem_embs: list[torch.Tensor] = []
        device = next(unwrapped_model.parameters()).device

        with torch.no_grad():
            for batch in eval_dataloader:
                # Use model directly (not prepared) to ensure consistent results
                sem_embs = batch["item_embeddings"].to(device)
                tokens = unwrapped_model.predict(sem_embs)
                all_tokens.append(tokens.cpu())
                all_sem_embs.append(sem_embs)

        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens, dim=0).cpu().numpy()  # (num_items, codebook_num)
        all_sem_embs = torch.cat(all_sem_embs, dim=0)  # (num_items, sem_emb_dim)

        # Handle collisions based on the configured method
        if config.sid_collision_handling == "sinkhorn":
            item2sids = self._generate_sids_sinkhorn(
                unwrapped_model,
                all_tokens,
                all_sem_embs,
            )
        elif config.sid_collision_handling == "extend":
            item2sids = self._extend_tokens(
                all_tokens,
                config,
            )
        else:
            raise ValueError(
                f"Unknown sid_collision_handling: {config.sid_collision_handling}. "
                f"Must be 'sinkhorn' or 'extend'."
            )

        # Save SIDs to file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sid_path = output_dir / config.sid_output_file
        output_format = getattr(config, "sid_output_format", "int")
        if str(config.sid_output_file).endswith(".index.json"):
            output_format = "minionerec_index"

        payload: Any
        if output_format == "minionerec_index":
            payload = self._to_minionerec_index_json(
                item2sids,
                token_prefixes=tuple(getattr(config, "minionerec_token_prefixes", ("a", "b", "c", "d", "e"))),
                token_offset=int(getattr(config, "minionerec_token_offset", 0)),
            )
        else:
            payload = item2sids

        with open(sid_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        accelerator.print(f"[SID] Generated {len(item2sids)} semantic IDs saved to {sid_path}")

        # Wait for all processes
        accelerator.wait_for_everyone()

    @staticmethod
    def _to_minionerec_index_json(
        item2sids: dict[Any, Any],
        *,
        token_prefixes: tuple[str, ...],
        token_offset: int,
    ) -> dict[str, list[str]]:
        """Convert integer code tuples into MiniOneRec `item.index.json` token lists.

        Notes:
        - Requires that each item's SID is a fixed-width integer tuple (typically codebook_num).
        - Enforces uniqueness by minimally disambiguating remaining duplicates via one extra token
          using the next available prefix (same idea as RecBole's MiniOneRec adapter).
        """

        def _format(level: int, code: int) -> str:
            prefix = token_prefixes[level] if level < len(token_prefixes) else chr(ord("a") + level)
            return f"<{prefix}_{int(code) + int(token_offset)}>"

        formatted: dict[str, list[str]] = {}
        for raw_item_id, raw_codes in item2sids.items():
            codes = list(raw_codes) if isinstance(raw_codes, (list, tuple)) else None
            if not codes:
                raise ValueError(f"RQVAE SID for item {raw_item_id!r} must be a non-empty list/tuple of ints.")
            if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in codes):
                raise ValueError(f"RQVAE SID for item {raw_item_id!r} must contain only integers, got {raw_codes!r}.")
            formatted[str(raw_item_id)] = [_format(level, int(code)) for level, code in enumerate(codes)]

        # Disambiguate duplicates to guarantee 1-1 decoding for item-level evaluation.
        sid_to_items: dict[str, list[str]] = {}
        for item_id, tokens in formatted.items():
            sid_to_items.setdefault("".join(tokens), []).append(item_id)
        duplicates = {sid: items for sid, items in sid_to_items.items() if len(items) > 1}
        if duplicates:
            for sid, items in sorted(duplicates.items()):
                current_len = len(formatted[items[0]])
                if current_len >= len(token_prefixes):
                    raise ValueError(
                        "RQVAE minionerec_index formatting produced duplicate SIDs but no remaining token_prefixes are available "
                        f"to disambiguate (need index {current_len}, have {len(token_prefixes)}). Example: {sid!r} -> {items}."
                    )
                prefix = token_prefixes[current_len]
                for offset, item_id in enumerate(sorted(items), start=1):
                    formatted[item_id] = [*formatted[item_id], f"<{prefix}_{int(token_offset) + offset}>"]

        return formatted

    def _generate_sids_sinkhorn(
        self,
        model: BaseModel,
        initial_tokens: np.ndarray,
        all_sem_embs: torch.Tensor,
    ) -> dict[str, tuple[int, ...]]:
        """Generate SIDs using iterative sinkhorn algorithm to resolve collisions.

        Args:
            model: The RQVAEModel (unwrapped).
            initial_tokens: Initial token assignments (numpy array).
            config: Model config.

        Returns:
            Dictionary mapping item IDs to their semantic ID tuples.
        """

        def _str_sids(sem_ids: np.ndarray):
            str_sem_ids = [ str(ids.tolist()) for ids in sem_ids]
            return np.array(str_sem_ids)

        def _check_collision(str_sem_ids):
            tot_item = len(str_sem_ids)
            tot_ids = len(set(str_sem_ids.tolist()))
            print(f'[TOKENIZER] Collision rate: {(tot_item - tot_ids) / tot_item}')
            return tot_item == tot_ids

        def _get_collision_items(str_sem_ids):
            sem_id2item = defaultdict(list)
            for i, str_sem_id in enumerate(str_sem_ids):
                sem_id2item[str_sem_id].append(i)

            collision_item_groups = []
            for str_sem_id in sem_id2item:
                if len(sem_id2item[str_sem_id]) > 1:
                    collision_item_groups.append(sem_id2item[str_sem_id])

            return collision_item_groups

        def _convert_to_dict(sem_ids):
            item2sem_ids = {}
            for i, ids in enumerate(sem_ids):
                item2sem_ids[i] = tuple(ids.tolist())

            return item2sem_ids
        # Convert to string representation for collision detection
        str_sem_ids = _str_sids(initial_tokens)
        # Iteratively resolve collisions using sinkhorn
        max_iters = 20
        self._enable_sinkhorn_inference(model)
        for iteration in range(max_iters):
            if _check_collision(str_sem_ids):
                break
            collision_item_groups = _get_collision_items(str_sem_ids)
            # Re-encode colliding items using sinkhorn
            model.eval()
            with torch.no_grad():
                for collision_items in collision_item_groups:
                    # Get embeddings for colliding items
                    colliding_item_embs = all_sem_embs[collision_items]

                    # Re-encode with sinkhorn enabled
                    # Note: We need to reconstruct the encoding process with use_sk=True
                    new_tokens = model.predict(colliding_item_embs, infer_use_sk=True)
                    # Update the tokens and string representations
                    new_tokens = new_tokens.cpu().numpy()
                    for idx, sids in zip(collision_items, new_tokens):
                        initial_tokens[idx] = sids
                        str_sem_ids[idx] = str(sids.tolist())

        # Final collision check
        unique_ids = set(str_sem_ids)
        final_collision_rate = (len(str_sem_ids) - len(unique_ids)) / len(str_sem_ids) if len(str_sem_ids) else 0
        print(f"[SID] Final collision rate: {final_collision_rate:.4f}")

        # Convert to item2sids dictionary (index as item_id)
        item2sids = _convert_to_dict(initial_tokens)

        return item2sids

    @staticmethod
    def _enable_sinkhorn_inference(model: BaseModel) -> None:
        """Enable Sinkhorn inference on the last RQ level only.

        This is used by the `sid_collision_handling="sinkhorn"` SID generation path to
        iteratively re-encode colliding items.
        """

        rq_layer = getattr(model, "_rq_layer", None)
        vq_layers = getattr(rq_layer, "vq_layers", ())
        if not vq_layers:
            return
        last_layer = vq_layers[-1]
        if getattr(last_layer, "sk_epsilon", -1) <= 0:
            last_layer.sk_epsilon = 0.003
        if getattr(last_layer, "sk_iters", -1) <= 0:
            last_layer.sk_iters = 50
        last_layer.use_sk = True

    def _extend_tokens(
        self,
        all_item_tokens: np.ndarray,
        config: Any,
    ) -> dict[str, tuple[int, ...]]:
        """Handle collisions by extending tokens with a conflict counter.

        This method adds an extra dimension to the token to distinguish colliding items.

        Args:
            all_item_tokens: Token assignments (numpy array of shape [num_items, codebook_num]).
            config: Model config.

        Returns:
            Dictionary mapping item IDs to their extended semantic ID tuples.
        """
        # Get codebook sizes
        if isinstance(config.codebook_size, int):
            codebook_sizes = [config.codebook_size] * config.codebook_num
        else:
            codebook_sizes = list(config.codebook_size)

        # Build mapping from token string to items that have it
        item2sids = {}
        max_conflict = 0
        tokens2item: dict[str, list[int]] = defaultdict(list)
        offset = np.cumsum([0] + codebook_sizes)
        for i in range(all_item_tokens.shape[0]):
            str_id = " ".join(map(str, all_item_tokens[i].tolist()))
            tokens2item[str_id].append(i)
            tokens = offset + np.array( all_item_tokens[i].tolist() + [len(tokens2item[str_id])])
            item2sids[i] = tuple(tokens.astype(int).tolist())
            max_conflict = max(max_conflict, len(tokens2item[str_id]))

        print(f'[TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        print(f'Conflict rate: {(all_item_tokens.shape[0] - len(tokens2item)) / all_item_tokens.shape[0]}')

        return item2sids

    def _compute_collision_rate(self, tokens: torch.Tensor) -> float:
        """Compute collision rate (proportion of duplicate token assignments).

        Args:
            tokens: Tensor of shape (num_items, codebook_num) containing token indices.

        Returns:
            Collision rate between 0 and 1, where 0 means all items have unique tokens.
        """
        num_items, codebook_num = tokens.shape
        if num_items == 0:
            return 0.0

        # Convert each multi-dimensional token to a unique tuple
        token_tuples = [tuple(tokens[i].tolist()) for i in range(num_items)]
        unique_tokens = set(token_tuples)

        # Collision rate = 1 - (unique_tokens / total_tokens)
        collision_rate = 1.0 - (len(unique_tokens) / num_items)
        return collision_rate


__all__ = ["RQVAETrainer"]
