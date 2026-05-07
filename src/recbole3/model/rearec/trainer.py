"""ReaRec-specific Trainer extension.

Responsibilities beyond the base ``Trainer``:

1. **on_epoch_begin hook** – calls ``model.on_epoch_begin(epoch)`` at the start
   of each training epoch by overriding ``_create_train_progress_bar``.  This
   ensures PRL warmup gating (``warmup_epochs > 0``) works without any changes
   to the base training loop.

2. **Clean fit() result** – the base ``fit()`` stores per-batch loss values in
   ``train_history[i]["losses"]`` (a ``list[float]`` with one entry per batch).
   These are verbose and unhelpful in the printed YAML summary; they are removed
   before the result is returned.

3. **Trim valid_history** – the base result includes one entry per validated
   epoch.  For concise terminal output the list is replaced with only the
   best-epoch record (or the last record when no best epoch is tracked).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from recbole3.dataset import BaseTaskDataset
from recbole3.model.base import BaseModel
from recbole3.trainer import Trainer


class ReaRecTrainer(Trainer):
    """Trainer variant for ReaRec with epoch-begin hooks and clean result output."""

    def fit(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        # Store the original (pre-accelerate-wrap) model so that
        # _create_train_progress_bar can call on_epoch_begin on it.
        self._rearec_model: BaseModel | None = model
        try:
            result = super().fit(model, prepared_data, output_dir=output_dir)
        finally:
            self._rearec_model = None

        # --- strip per-batch loss lists -----------------------------------
        # train_history[i]["losses"] is a list[float] with one float per
        # training batch.  With ~80 batches/epoch × 20 epochs this becomes
        # ~1600 lines of YAML noise.  The epoch-average "loss" field is kept.
        for ep in result.get("train_history", []):
            ep.pop("losses", None)

        # --- trim valid_history to best-epoch record ----------------------
        # The base Trainer accumulates one entry per validated epoch.  Keep
        # only the best-epoch record (or the final record when unknown).
        valid_hist: list[dict[str, Any]] = result.get("valid_history", [])
        if valid_hist:
            best_epoch: int | None = result.get("best_epoch")
            if best_epoch is not None:
                best_records = [r for r in valid_hist if r.get("epoch") == best_epoch]
                result["valid_history"] = best_records if best_records else valid_hist[-1:]
            else:
                result["valid_history"] = valid_hist[-1:]

        return result

    # ------------------------------------------------------------------
    # Override static method to inject on_epoch_begin hook.
    # The base Trainer calls self._create_train_progress_bar(...) inside
    # the epoch loop; overriding as an instance method here intercepts that
    # call without duplicating any of the training-loop logic.
    # ------------------------------------------------------------------

    def _create_train_progress_bar(  # type: ignore[override]
        self,
        train_dataloader: DataLoader,
        *,
        epoch: int,
        max_epochs: int,
    ) -> Any:
        model = getattr(self, "_rearec_model", None)
        if model is not None and hasattr(model, "on_epoch_begin"):
            model.on_epoch_begin(epoch)
        return Trainer._create_train_progress_bar(
            train_dataloader, epoch=epoch, max_epochs=max_epochs
        )


__all__ = ["ReaRecTrainer"]
