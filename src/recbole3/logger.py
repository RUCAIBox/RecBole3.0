from __future__ import annotations

import dataclasses
import os
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _is_main_process() -> bool:
    """Return True on global rank 0 or when not running under DDP."""
    rank = os.environ.get("RANK", "")
    if rank == "":
        return True
    try:
        return int(rank) in (-1, 0)
    except ValueError:
        return True


class TrainingLogger:
    """File-based training logger for RecBole3.0.

    Creates a timestamped log file with configuration, model/dataset info,
    per-epoch training statistics, validation metrics, and final test results.
    """

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        dataset_name: str,
        category_name: str,
    ) -> None:
        self._is_main = _is_main_process()
        self._file = None
        self._epoch_header_printed = False
        self._epoch_extra_keys: tuple[str, ...] = ()

        if not self._is_main:
            return

        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_name}_{dataset_name}_{category_name}_{timestamp}.log"

        self._file = open(str(log_path), "w", encoding="utf-8")
        self._start_time = time_module.perf_counter()
        self._start_dt = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

        self._write_rule()
        self._writeln("  RecBole3.0 Training Log")
        self._write_rule()
        self._writeln(f"  Model: {model_name}  |  Dataset: {dataset_name}  |  Started: {self._start_dt}")
        self._write_rule()
        self._writeln()

    # ── public write helpers ──────────────────────────────────────────

    def log_config(self, title: str, config: Any) -> None:
        """Write a dataclass-based config section."""
        if not self._file:
            return
        self._writeln(f"--- Config: {title} " + "-" * (70 - len(title)))
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            self._write_config_tree(dataclasses.asdict(config), indent=0)
        elif isinstance(config, Mapping):
            self._write_config_tree(dict(config), indent=0)
        else:
            self._writeln(f"  {config}")
        self._writeln()

    def log_model_info(self, model: Any) -> None:
        """Write model name, class, parameter counts."""
        if not self._file:
            return
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config = getattr(model, "config", None)
        name = getattr(config, "name", "") or "unknown"
        cls_name = type(model).__name__

        self._writeln("--- Model " + "-" * 69)
        self._writeln(f"  Name:         {name}")
        self._writeln(f"  Class:        {cls_name}")
        self._writeln(f"  Params:       {total:,}")
        self._writeln(f"  Trainable:    {trainable:,}")
        self._writeln()

    def log_dataset_info(self, prepared_data: Any) -> None:
        """Write dataset name, task, and split sizes."""
        if not self._file:
            return
        config = getattr(prepared_data, "config", None)
        ds_name = getattr(config, "name", "") or "unknown"
        task = getattr(prepared_data, "task", "unknown")
        num_users = prepared_data.get_num_users()
        num_items = prepared_data.get_num_items()
        train_n = len(prepared_data.get_train_dataset())
        valid_n = len(prepared_data.get_eval_dataset("valid"))
        test_n = len(prepared_data.get_eval_dataset("test"))

        self._writeln("--- Dataset " + "-" * 67)
        self._writeln(f"  Name:         {ds_name}")
        self._writeln(f"  Task:         {task}")
        self._writeln(f"  Users:        {num_users:,}")
        self._writeln(f"  Items:        {num_items:,}")
        self._writeln(f"  Train:        {train_n:,} records")
        self._writeln(f"  Valid:        {valid_n:,} records")
        self._writeln(f"  Test:         {test_n:,} records")
        self._writeln()

    def log_epoch(
        self,
        epoch: int,
        max_epochs: int,
        loss: float | None,
        num_batches: int,
        elapsed_seconds: float,
        lr: float | None = None,
        **extra: float | None,
    ) -> None:
        """Write one row to the epoch table."""
        if not self._file:
            return

        extra_keys = tuple(extra.keys())
        if not self._epoch_header_printed:
            self._epoch_header_printed = True
            self._epoch_extra_keys = extra_keys
            self._write_epoch_header(extra_keys)

        loss_str = f"{loss:.6f}" if loss is not None else "-"
        lr_str = f"{lr:.6f}" if lr is not None else "-"
        time_str = f"{elapsed_seconds:.2f}"

        row = (
            f"  {epoch:>5d}/{max_epochs:<5d}  "
            f"{loss_str:<12s}  "
            f"{time_str:<10s}  "
            f"{num_batches:<8d}  "
            f"{lr_str:<10s}"
        )
        for key in self._epoch_extra_keys:
            val = extra.get(key)
            row += f"  {val:<14}" if val is not None else f"  {'-':<14}"
        self._writeln(row)

    def log_validation(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Write validation metrics for an epoch."""
        if not self._file:
            return
        formatted = self._format_metrics_line(metrics)
        self._writeln(f"    Valid epoch {epoch}: {formatted}")

    def log_best(self, epoch: int, monitor_name: str, best_value: float) -> None:
        """Record the best-epoch marker."""
        if not self._file:
            return
        self._writeln(f"\n  Best: epoch {epoch}  {monitor_name}={best_value:.6f}")

    def log_early_stopping(self, stopped: bool, epoch: int, patience: int) -> None:
        if not self._file:
            return
        if stopped:
            self._writeln(f"\n  Early stopping triggered at epoch {epoch} (patience={patience}).")
        else:
            self._writeln(f"\n  Training completed (max_epochs reached).")

    def log_test(self, test_result: Mapping[str, Any]) -> None:
        """Write final test results."""
        if not self._file:
            return
        self._writeln()
        self._write_rule()
        self._writeln("  TEST RESULTS")
        self._write_rule()
        protocol = test_result.get("protocol", "unknown")
        metrics = test_result.get("metrics", {})
        self._writeln(f"  Protocol:     {protocol}")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                try:
                    self._writeln(f"  {key}:{' ' * (13 - len(key))}{float(value):.6f}")
                except (TypeError, ValueError):
                    self._writeln(f"  {key}:{' ' * (13 - len(key))}{value}")
        self._writeln()

    def log_summary(
        self,
        stopped_early: bool,
        total_epochs: int,
        best_epoch: int | None,
        total_time: float,
    ) -> None:
        """Write closing summary."""
        if not self._file:
            return
        self._writeln(f"  Total epochs:   {total_epochs}")
        self._writeln(f"  Best epoch:     {best_epoch if best_epoch is not None else 'N/A'}")
        self._writeln(f"  Stopped early:  {stopped_early}")

    def close(self) -> None:
        """Write footer and close the log file."""
        if not self._file:
            return
        elapsed = time_module.perf_counter() - self._start_time
        end_dt = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self._writeln()
        self._write_rule()
        self._writeln(f"  Run completed: {end_dt}  (total: {elapsed:.2f} s)")
        self._write_rule()
        self._file.close()
        self._file = None

    # ── internal helpers ─────────────────────────────────────────────

    def _writeln(self, text: str = "") -> None:
        if self._file:
            self._file.write(text + "\n")
            self._file.flush()

    def _write_rule(self) -> None:
        self._writeln("=" * 80)

    def _write_epoch_header(self, extra_keys: tuple[str, ...]) -> None:
        self._writeln()
        self._writeln("--- Epochs " + "-" * 69)
        header = "  Epoch      Avg Loss      Time (s)    Steps      LR        "
        for key in extra_keys:
            header += f"  {key:<14}"
        self._writeln(header)
        sep = "  " + "-" * (len(header) - 2)
        self._writeln(sep)

    def _write_config_tree(self, obj: Any, indent: int) -> None:
        prefix = "  " * (indent + 1)
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                if _is_nested_container(value):
                    self._writeln(f"{prefix}{key}:")
                    self._write_config_tree(value, indent + 1)
                else:
                    self._writeln(f"{prefix}{key}: {_format_scalar(value)}")
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            for item in obj:
                if isinstance(item, Mapping):
                    self._writeln(f"{prefix}-")
                    self._write_config_tree(item, indent + 1)
                else:
                    self._writeln(f"{prefix}- {_format_scalar(item)}")
        else:
            self._writeln(f"{prefix}{_format_scalar(obj)}")

    @staticmethod
    def _format_metrics_line(metrics: Mapping[str, Any]) -> str:
        if not metrics:
            return "{}"
        parts: list[str] = []
        for name, value in metrics.items():
            try:
                parts.append(f"{name}={float(value):.6f}")
            except (TypeError, ValueError):
                parts.append(f"{name}={value}")
        return "  ".join(parts)


def _is_nested_container(obj: Any) -> bool:
    return isinstance(obj, Mapping) or (isinstance(obj, Sequence) and not isinstance(obj, str))


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, type):
        return value.__name__
    return str(value)
