from __future__ import annotations

from pathlib import Path
from typing import Any

from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig

import time
import torch


class LARESTrainer(Trainer):
    """Trainer with test-time recurrence scaling evaluation for LARES."""

    config_cls = TrainerConfig

    def run(
        self,
        model: Any,
        prepared_data: Any,
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
            print("[trainer] starting test evaluation")
            test_recurrence_ratios = getattr(model.config, "test_recurrence_ratios", None)

            mean_T = model.config.mean_recurrence
            recurrence_list: list[int] = [mean_T]
            for ratio in test_recurrence_ratios:
                r = max(1, int(mean_T * float(ratio)))
                recurrence_list.append(r)
            recurrence_list = sorted(set(recurrence_list))

            accelerator = self.create_accelerator()
            scaling_results: dict[str, dict[str, float]] = {}
            for recurrence in recurrence_list:
                model._eval_recurrence_override = recurrence
                try:
                    eval_result = self._run_evaluation(
                        model,
                        prepared_data,
                        split="test",
                        accelerator=accelerator,
                        model_is_prepared=False,
                    )
                    scaling_results[str(recurrence)] = eval_result["metrics"]
                finally:
                    model._eval_recurrence_override = None
            print("[trainer] finished test evaluation")

            if (logger := getattr(self, "_logger", None)) is not None:
                flat: dict[str, float] = {}
                for r, metrics in scaling_results.items():
                    for name, val in metrics.items():
                        flat[f"{name} (r={r})"] = val
                logger.log_test({"protocol": "recurrence_scaling", "metrics": flat})
                
                total_elapsed = time.perf_counter() - total_start
                logger.log_summary(
                    stopped_early=fit_result.get("stopped_early", False),
                    total_epochs=len(fit_result.get("train_history", [])),
                    best_epoch=fit_result.get("best_epoch"),
                    total_time=total_elapsed,
                )

            return {
                "fit": fit_result,
                "test": scaling_results.get(f"{mean_T}", {}),
                "test_scaling": scaling_results,
            }
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()
