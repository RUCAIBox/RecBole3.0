"""BIGRec pipeline: orchestrates dataset preparation, LoRA training, and evaluation."""

from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import DictConfig

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.evaluation import EvalConfig
from recbole3.model.bigrec.config import BIGRecConfig
from recbole3.model.bigrec.data import BIGRecModelDataset
from recbole3.model.bigrec.trainer import BIGRecTrainer
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_cfg, require_component_name

logger = logging.getLogger(__name__)


class BIGRecPipeline(Pipeline):
    """End-to-end pipeline for BIGRec experiments.

    Orchestrates the following steps:
      1. Parse Hydra configuration (runtime / dataset / model blocks).
      2. Prepare the task dataset (splitting, item table, eval candidates).
      3. Build :class:`~recbole3.model.bigrec.data.BIGRecModelDataset` to inject
         ``history_item_ids`` into every split.
      4. Depending on ``config.pipeline_stage``:

         * ``'training'``: LoRA fine-tune on the training split via
           :meth:`BIGRecTrainer.fit`, then evaluate on the test split.
         * ``'evaluation'``: Load an existing checkpoint and evaluate.
    """

    def _parse_config(
        self,
        cfg: DictConfig,
    ) -> tuple[RuntimeConfig, DictConfig, DictConfig]:
        """Extract and return the three top-level config blocks.

        Args:
            cfg: Composed Hydra DictConfig.

        Returns:
            Tuple of ``(runtime_cfg, dataset_cfg, model_cfg)``.
        """
        runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
        dataset_cfg = require_component_cfg(cfg, "dataset")
        model_cfg = require_component_cfg(cfg, "model")
        return runtime_cfg, dataset_cfg, model_cfg

    def run(self) -> dict[str, Any]:
        """Execute the BIGRec experiment pipeline.

        Returns:
            Dict of metric scores (``"recall@K"``, ``"ndcg@K"``, …) or
            ``{"checkpoint_path": ...}`` when only training is requested.

        Raises:
            ValueError: If ``pipeline_stage`` is unknown or required config
                        fields are missing.
        """
        runtime_cfg, dataset_cfg, model_cfg = self._parse_config(self.cfg)

        # ── Dataset ────────────────────────────────────────────────────────────
        dataset_name = require_component_name(dataset_cfg, "dataset")
        dataset_spec = get_dataset_spec(dataset_name)

        bigrec_config: BIGRecConfig = instantiate_dataclass(BIGRecConfig, model_cfg)

        # Prepare the base task dataset (interactions, item table, eval splits).
        # Use the protocol from BIGRecConfig so sampled / full candidates are
        # built correctly.
        eval_config = EvalConfig(protocol=bigrec_config.eval_protocol)  # type: ignore[arg-type]
        task_data = dataset_spec.dataset_cls(
            instantiate_dataclass(dataset_spec.config_cls, dataset_cfg)
        ).prepare(eval_config=eval_config)

        # Wrap with BIGRecModelDataset to inject history_item_ids into all splits.
        bigrec_data = BIGRecModelDataset.from_task_dataset(
            task_data, model_config=bigrec_config
        )

        # ── Trainer ────────────────────────────────────────────────────────────
        trainer = BIGRecTrainer(bigrec_config)
        output_dir: str = runtime_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

        stage = bigrec_config.pipeline_stage.strip().lower()

        if stage == "training":
            logger.info("BIGRec pipeline: starting training stage …")
            fit_result = trainer.fit(bigrec_data, output_dir=output_dir)
            checkpoint_path: str = fit_result["checkpoint_path"]
            logger.info("BIGRec pipeline: evaluating on test split …")
            results = trainer.evaluate(
                bigrec_data, checkpoint_path=checkpoint_path, split="test"
            )
            results["checkpoint_path"] = checkpoint_path
            return results

        elif stage == "evaluation":
            if not bigrec_config.checkpoint_path:
                raise ValueError(
                    "BIGRecConfig.checkpoint_path must be set when "
                    "pipeline_stage='evaluation'."
                )
            logger.info(
                "BIGRec pipeline: evaluation-only stage (checkpoint=%s) …",
                bigrec_config.checkpoint_path,
            )
            return trainer.evaluate(
                bigrec_data,
                checkpoint_path=bigrec_config.checkpoint_path,
                split="test",
            )

        else:
            raise ValueError(
                f"Unknown pipeline_stage '{bigrec_config.pipeline_stage}'. "
                "Supported: 'training', 'evaluation'."
            )


__all__ = ["BIGRecPipeline"]
