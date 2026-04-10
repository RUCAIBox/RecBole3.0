from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from omegaconf import DictConfig, OmegaConf

from recbole3.config import RuntimeConfig, configs_dir, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, get_dataset_spec
from recbole3.model import BaseModelDataset, ModelSpec, get_model_spec


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    """Instantiate configured components and execute one trainer run."""

    runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
    dataset_cfg = _require_component_cfg(cfg, "dataset")
    model_cfg = _require_component_cfg(cfg, "model")
    trainer_cfg = _require_component_cfg(cfg, "trainer")

    dataset_name = _require_component_name(dataset_cfg, "dataset")
    model_name = _require_component_name(model_cfg, "model")

    dataset_spec = get_dataset_spec(dataset_name)
    model_spec = get_model_spec(model_name)

    dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
    model = model_spec.model_cls(instantiate_dataclass(model_spec.config_cls, model_cfg))
    trainer = model_spec.trainer_cls(instantiate_dataclass(model_spec.trainer_config_cls, trainer_cfg))

    task_data = dataset.prepare(eval_config=trainer.config.eval)
    prepared_data = _build_model_data(task_data, model_spec, model.config)
    with _accelerate_runtime_device(runtime_cfg.device):
        run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)
    return {
        "prepared_data": prepared_data,
        **run_result,
    }


def compose_config(overrides: Sequence[str] | None = None, config_dir: str | Path | None = None) -> DictConfig:
    """Compose the root Hydra config from a config directory and override list."""

    import hydra

    config_root = Path(config_dir).resolve() if config_dir is not None else configs_dir().resolve()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        return hydra.compose(config_name="config", overrides=list(overrides or []))


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint that composes config, runs the experiment, and prints a summary."""

    cfg = compose_config(overrides=list(argv if argv is not None else sys.argv[1:]))
    result = run_experiment(cfg)
    printable = {
        "prepared_data": _serialize_prepared_data(result["prepared_data"]),
        "fit": result["fit"],
        "test": result["test"],
    }
    print(OmegaConf.to_yaml(OmegaConf.create(printable), resolve=True))
    return result


@contextmanager
def _accelerate_runtime_device(device: str | None) -> Iterator[None]:
    """Translate one runtime device hint into the accelerate environment contract."""

    normalized_device = _normalize_runtime_device(device)
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


def _normalize_runtime_device(device: str | None) -> str:
    """Normalize one runtime device string while preserving accelerate's device syntax."""

    normalized = str(device or "auto").strip().lower()
    return normalized or "auto"


def _build_model_data(
    prepared_data: BaseTaskDataset[Any, Any],
    model_spec: ModelSpec,
    model_config: Any,
) -> BaseTaskDataset[Any, Any]:
    model_data_cls = model_spec.model_data_cls
    if model_data_cls is None:
        return prepared_data
    if not issubclass(model_data_cls, BaseModelDataset):
        raise TypeError(f"{model_data_cls!r} must inherit BaseModelDataset.")

    model_data = model_data_cls.from_task_dataset(prepared_data, model_config=model_config)
    if not isinstance(model_data, BaseTaskDataset):
        raise TypeError(f"{model_data_cls.__name__}.from_task_dataset(...) must return a BaseTaskDataset.")
    if model_data.task != prepared_data.task:
        raise TypeError(
            f"{model_data_cls.__name__} changed task '{prepared_data.task}' to '{model_data.task}', which is not allowed."
        )
    return model_data


def _serialize_prepared_data(prepared_data: BaseTaskDataset[Any, Any]) -> dict[str, Any]:
    """Render one prepared dataset into a printable summary."""

    return {
        "type": type(prepared_data).__name__,
        "name": prepared_data.config.name,
        "num_users": int(prepared_data.get_num_users()),
        "num_items": int(prepared_data.get_num_items()),
        "num_train_records": len(prepared_data.get_train_dataset()),
        "num_valid_records": len(prepared_data.get_eval_dataset("valid")),
        "num_test_records": len(prepared_data.get_eval_dataset("test")),
    }


def _require_component_cfg(cfg: DictConfig, component: str) -> DictConfig:
    """Return one component config node and fail fast if it is missing."""

    value = cfg.get(component)
    if value is None:
        raise ValueError(
            f"Missing `{component}` configuration. Add a config group override such as `{component}=your_component`."
        )
    if not isinstance(value, DictConfig):
        raise TypeError(f"`{component}` config must be a DictConfig, got {type(value)!r}.")
    return value


def _require_component_name(component_cfg: DictConfig, component: str) -> str:
    """Return the registered component name from a component config node."""

    name = component_cfg.get("name")
    if not name:
        raise ValueError(f"`{component}.name` must be set to a known component name.")
    return str(name)


if __name__ == "__main__":
    main()
