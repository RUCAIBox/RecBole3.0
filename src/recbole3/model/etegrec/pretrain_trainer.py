from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import optim
from torch.utils.data import DataLoader

from recbole3.model.etegrec.pretrain_rqvae import RQVAE


@dataclass(slots=True)
class ETEGRecRQVAEPretrainTrainerConfig:
    lr: float = 1e-3
    epochs: int = 10000
    batch_size: int = 1024
    num_workers: int = 2
    eval_step: int = 50
    learner: str = "AdamW"
    weight_decay: float = 1e-4
    lr_scheduler_type: Literal["linear", "constant"] = "linear"
    warmup_epochs: int = 50
    save_limit: int = 3
    gradient_clip_norm: float = 1.0
    device: str = "auto"


class ETEGRecRQVAEPretrainTrainer:
    """Standalone RQVAE pretrainer adapted from the original ETEGRec RQVAE trainer."""

    def __init__(self, config: ETEGRecRQVAEPretrainTrainerConfig, model: RQVAE, *, output_dir: str | Path):
        self.config = config
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.best_loss = float("inf")
        self.best_collision_rate = float("inf")
        self.best_save_heap: list[tuple[float, Path]] = []
        self.newest_save_queue: list[tuple[float, Path]] = []

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        normalized = str(device or "auto").lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(normalized)

    def _build_optimizer(self) -> optim.Optimizer:
        learner = self.config.learner.lower()
        params = self.model.parameters()
        if learner == "adam":
            return optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        if learner == "adagrad":
            return optim.Adagrad(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        if learner == "rmsprop":
            return optim.RMSprop(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        if learner == "adamw":
            return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optim.Adam(params, lr=self.config.lr)

    def _build_scheduler(self, *, steps_per_epoch: int) -> object:
        from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

        warmup_steps = int(self.config.warmup_epochs) * int(steps_per_epoch)
        max_steps = int(self.config.epochs) * int(steps_per_epoch)
        if self.config.lr_scheduler_type.lower() == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        return get_constant_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
        )

    def fit(self, dataloader: DataLoader[torch.Tensor]) -> dict[str, object]:
        self.scheduler = self._build_scheduler(steps_per_epoch=max(1, len(dataloader)))
        train_history = []
        eval_step = min(max(1, int(self.config.eval_step)), int(self.config.epochs))
        best_loss_path = self.output_dir / "best_loss_model.pth"
        best_collision_path = self.output_dir / "best_collision_model.pth"

        for epoch in range(int(self.config.epochs)):
            start = time.perf_counter()
            total_loss, total_recon_loss = self._train_epoch(dataloader)
            elapsed = time.perf_counter() - start
            epoch_result = {
                "epoch": epoch,
                "loss": total_loss,
                "recon_loss": total_recon_loss,
                "elapsed_seconds": elapsed,
            }
            train_history.append(epoch_result)
            print(
                f"[etegrec-rqvae] epoch={epoch + 1}/{int(self.config.epochs)} "
                f"loss={total_loss:.6f} recon_loss={total_recon_loss:.6f}"
            )

            if (epoch + 1) % eval_step == 0:
                collision_rate = self.evaluate_collision_rate(dataloader)
                epoch_result["collision_rate"] = collision_rate
                print(f"[etegrec-rqvae] epoch={epoch + 1} collision_rate={collision_rate:.6f}")
                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    self._save_checkpoint(epoch, best_loss_path)
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    self._save_checkpoint(epoch, best_collision_path)
                checkpoint_path = self.output_dir / f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth"
                self._save_checkpoint(epoch, checkpoint_path, collision_rate=collision_rate)
                self._prune_checkpoints(checkpoint_path, collision_rate)

        return {
            "train_history": train_history,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "best_loss_path": str(best_loss_path),
            "best_collision_path": str(best_collision_path),
        }

    def _train_epoch(self, dataloader: DataLoader[torch.Tensor]) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            data = batch.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, _ = self.model(data)
            loss, recon_loss = self.model.compute_loss(out, rq_loss, xs=data)
            if torch.isnan(loss):
                raise ValueError("ETEGRec RQVAE pretraining loss is NaN.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.gradient_clip_norm))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += float(loss.detach().item())
            total_recon_loss += float(recon_loss.detach().item())
            num_batches += 1
        denom = max(1, num_batches)
        return total_loss / denom, total_recon_loss / denom

    @torch.no_grad()
    def evaluate_collision_rate(self, dataloader: DataLoader[torch.Tensor]) -> float:
        self.model.eval()
        codes: set[str] = set()
        num_samples = 0
        for batch in dataloader:
            data = batch.to(self.device)
            indices = self.model.get_indices(data).view(-1, len(self.model.config.num_emb_list)).cpu().tolist()
            num_samples += len(indices)
            for index in indices:
                codes.add("-".join(str(int(value)) for value in index))
        if num_samples == 0:
            return 0.0
        return float(num_samples - len(codes)) / float(num_samples)

    def _save_checkpoint(self, epoch: int, path: Path, *, collision_rate: float | None = None) -> None:
        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "collision_rate": collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path, pickle_protocol=4)

    def _prune_checkpoints(self, checkpoint_path: Path, collision_rate: float) -> None:
        if int(self.config.save_limit) <= 0:
            return
        current = (-float(collision_rate), checkpoint_path)
        if len(self.newest_save_queue) < int(self.config.save_limit):
            self.newest_save_queue.append(current)
            heapq.heappush(self.best_save_heap, current)
            return
        old = self.newest_save_queue.pop(0)
        self.newest_save_queue.append(current)
        if collision_rate < -self.best_save_heap[0][0]:
            bad = heapq.heappop(self.best_save_heap)
            heapq.heappush(self.best_save_heap, current)
            if bad not in self.newest_save_queue:
                bad[1].unlink(missing_ok=True)
        if old not in self.best_save_heap:
            old[1].unlink(missing_ok=True)


__all__ = ["ETEGRecRQVAEPretrainTrainer", "ETEGRecRQVAEPretrainTrainerConfig"]
