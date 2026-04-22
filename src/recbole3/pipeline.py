from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING

from omegaconf import OmegaConf, DictConfig

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, get_dataset_spec
from recbole3.model import BaseModelDataset
from recbole3.utils import require_component_cfg, require_component_name

if TYPE_CHECKING:
    from recbole3.model import ModelSpec


class Pipeline:
    """Pipeline for running experiments.

    This class implements the typical experiment flow:
    1. Parse configuration
    2. Load dataset and model specs
    3. Instantiate components
    4. Prepare data
    5. Run trainer
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_spec: ModelSpec,
    ):
        """Initialize pipeline with pre-instantiated components.

        Args:
            cfg: The composed Hydra configuration.
        """
        self.cfg = cfg
        self.model_spec = model_spec

    def _parse_config(
        self, 
        cfg: DictConfig,
    ) -> tuple[RuntimeConfig, DictConfig, DictConfig, DictConfig]:
        """Parse configuration components.

        Args:
            cfg: The composed Hydra configuration.

        Returns:
            Tuple of (runtime_cfg, dataset_cfg, model_cfg, trainer_cfg).
        """
        runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
        dataset_cfg = require_component_cfg(cfg, "dataset")
        model_cfg = require_component_cfg(cfg, "model")
        trainer_cfg = require_component_cfg(cfg, "trainer")

        return runtime_cfg, dataset_cfg, model_cfg, trainer_cfg

    def run(self) -> dict[str, Any]:
        """Execute the experiment pipeline.

        Returns:
            Dictionary containing experiment results.
        """
        runtime_cfg, dataset_cfg, model_cfg, trainer_cfg = self._parse_config(self.cfg)

        # Get component names
        dataset_name = require_component_name(dataset_cfg, "dataset")
        model_name = require_component_name(model_cfg, "model")

        # Get specs
        dataset_spec = get_dataset_spec(dataset_name)

        # Instantiate components
        dataset = dataset_spec.dataset_cls(
            instantiate_dataclass(dataset_spec.config_cls, dataset_cfg)
        )
        model = self.model_spec.model_cls(
            instantiate_dataclass(self.model_spec.config_cls, model_cfg)
        )
        trainer = self.model_spec.trainer_cls(
            instantiate_dataclass(self.model_spec.trainer_config_cls, trainer_cfg)
        )
        # Prepare data
        task_data = dataset.prepare(eval_config=trainer.config.eval)
        prepared_data = self._build_model_data(task_data, self.model_spec.model_data_cls, model_cfg)

        # Run trainer
        with self._accelerate_runtime_device(runtime_cfg.device):
            run_result = trainer.run(
                model, prepared_data, output_dir=runtime_cfg.output_dir
            )

        # Print results
        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "fit": run_result["fit"],
            "test": run_result["test"],
        }
        print(OmegaConf.to_yaml(OmegaConf.create(printable), resolve=True))

        return {
            "prepared_data": prepared_data,
            **run_result,
        }

    def _build_model_data(
        self,
        task_data: BaseTaskDataset,
        model_data_cls: BaseModelDataset | None,
        model_config: DictConfig,
    ) -> BaseTaskDataset:
        """Build model-specific data from task dataset."""
        if model_data_cls is None:
            return task_data

        model_data = model_data_cls.from_task_dataset(task_data, model_config=model_config)
        if not isinstance(model_data, BaseTaskDataset):
            raise TypeError(
                f"{model_data_cls.__name__}.from_task_dataset(...) must return a BaseTaskDataset."
            )
        if model_data.task != task_data.task:
            raise TypeError(
                f"{model_data_cls.__name__} changed task '{task_data.task}' to '{model_data.task}', "
                "which is not allowed."
            )
        return model_data

    @contextmanager
    def _accelerate_runtime_device(self, device: str | None) -> Iterator[None]:
        """Context manager for setting runtime device."""
        normalized_device = self._normalize_runtime_device(device)

        if normalized_device == "auto":
            yield
            return

        if int(os.environ.get("LOCAL_RANK", "-1")) != -1:
            raise ValueError(
                "`runtime.device` cannot override per-process device assignment in distributed launches. "
                "Use `runtime.device=auto` with `accelerate launch` or `torchrun`."
            )

        previous_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
        os.environ["ACCELERATE_TORCH_DEVICE"] = normalized_device
        try:
            yield
        finally:
            if previous_device is None:
                os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
            else:
                os.environ["ACCELERATE_TORCH_DEVICE"] = previous_device

    @staticmethod
    def _normalize_runtime_device(device: str | None) -> str:
        """Normalize runtime device string."""
        normalized = str(device or "auto").strip().lower()
        return normalized or "auto"

    @staticmethod
    def serialize_prepared_data(prepared_data: BaseTaskDataset) -> dict[str, Any]:
        """Render prepared dataset into a printable summary."""
        return {
            "type": type(prepared_data).__name__,
            "name": prepared_data.config.name,
            "num_users": int(prepared_data.get_num_users()),
            "num_items": int(prepared_data.get_num_items()),
            "num_train_records": len(prepared_data.get_train_dataset()),
            "num_valid_records": len(prepared_data.get_eval_dataset("valid")),
            "num_test_records": len(prepared_data.get_eval_dataset("test")),
        }
    

__all__ = [
    "Pipeline",
]