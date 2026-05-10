from __future__ import annotations

from pathlib import Path
from typing import Any

from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig


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
        result = super().run(model, prepared_data, output_dir=output_dir)

        test_recurrence_ratios = getattr(model.config, "test_recurrence_ratios", None)
        if not test_recurrence_ratios:
            return result

        recurrence_list: list[int] = [1]
        mean_T = model.config.mean_recurrence
        for ratio in test_recurrence_ratios:
            r = max(1, int(mean_T * float(ratio)))
            recurrence_list.append(r)
        recurrence_list = sorted(set(recurrence_list))

        if len(recurrence_list) <= 1:
            return result

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

        result["test_scaling"] = scaling_results

        if (logger := getattr(self, "_logger", None)) is not None:
            logger.log_validation(
                recurrence_scaling={
                    f"r={r}": metrics for r, metrics in scaling_results.items()
                }
            )

        return result
