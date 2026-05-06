from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import DictConfig

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.evaluation import EvalConfig
from recbole3.model.lcrec.config import LCRecConfig
from recbole3.model.lcrec.trainer import LCRecTrainer
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_cfg, require_component_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class LCRecPipeline(Pipeline):
    """Pipeline for running LCRec model experiments.

    This class implements the typical experiment flow:
    1. Parse configuration
    2. Load dataset and model specs
    3. Instantiate components
    4. Prepare data
    5. Run trainer
    """

    def _parse_config(
        self, 
        cfg: DictConfig,
    ) -> tuple[RuntimeConfig, DictConfig, DictConfig]:
        """Parse configuration components.

        Args:
            cfg: The composed Hydra configuration.

        Returns:
            Tuple of (runtime_cfg, dataset_cfg, model_cfg, trainer_cfg).
        """
        runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
        dataset_cfg = require_component_cfg(cfg, "dataset")
        model_cfg = require_component_cfg(cfg, "model")

        return runtime_cfg, dataset_cfg, model_cfg

    def run(self) -> dict[str, Any]:
        """Execute the experiment pipeline.

        Returns:
            Dictionary containing experiment results.
        """
        runtime_cfg, dataset_cfg, model_cfg = self._parse_config(self.cfg)

        # Get component names
        dataset_name = require_component_name(dataset_cfg, "dataset")
        # Get specs
        dataset_spec = get_dataset_spec(dataset_name)

        # Instantiate components
        dataset = dataset_spec.dataset_cls(
            instantiate_dataclass(dataset_spec.config_cls, dataset_cfg)
        )
        lcrec_config = instantiate_dataclass(LCRecConfig, model_cfg)

        # Prepare data
        eval_config = EvalConfig(protocol="full")
        task_data = dataset.prepare(eval_config=eval_config)

        # Run trainer
        trainer = LCRecTrainer(lcrec_config)
        output_dir = runtime_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if lcrec_config.pipeline_stage == "training":
            result = trainer.run(task_data, output_dir=output_dir)
        elif lcrec_config.pipeline_stage == "evaluation":
            if not lcrec_config.model_checkpoint_path:
                raise ValueError("model_checkpoint_path must be set for evaluation stage.")
            result = trainer.evaluate(
                task_data,
                checkpoint_path=lcrec_config.model_checkpoint_path
            )
        else:
            raise ValueError(f"Unknown pipeline_stage: {lcrec_config.pipeline_stage}")

        logger.info("LCRec run completed. Output dir: %s", output_dir)
        return result

    
