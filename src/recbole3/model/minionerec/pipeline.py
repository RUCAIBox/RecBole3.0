from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model.minionerec.config import MiniOneRecConfig
from recbole3.model.minionerec.trainer import MiniOneRecTrainer
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_cfg, require_component_name


class MiniOneRecPipeline(Pipeline):
    """Pipeline for MiniOneRec SFT/evaluation using RecBole prepared datasets."""

    def _parse_config(self, cfg: DictConfig) -> tuple[RuntimeConfig, DictConfig, DictConfig]:
        runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
        dataset_cfg = require_component_cfg(cfg, "dataset")
        model_cfg = require_component_cfg(cfg, "model")
        return runtime_cfg, dataset_cfg, model_cfg

    def run(self) -> dict[str, Any]:
        runtime_cfg, dataset_cfg, model_cfg = self._parse_config(self.cfg)
        dataset_name = require_component_name(dataset_cfg, "dataset")
        dataset_spec = get_dataset_spec(dataset_name)

        dataset = dataset_spec.dataset_cls(
            instantiate_dataclass(dataset_spec.config_cls, dataset_cfg)
        )
        minionerec_config = instantiate_dataclass(MiniOneRecConfig, model_cfg)
        task_data = dataset.prepare(eval_config=self._build_eval_config(minionerec_config))

        trainer = MiniOneRecTrainer(minionerec_config)
        result = trainer.run(task_data, output_dir=runtime_cfg.output_dir)
        self._write_result_json(runtime_cfg.output_dir, result)
        return result

    @staticmethod
    def _build_eval_config(config: MiniOneRecConfig) -> EvalConfig:
        metric_specs = tuple(MetricSpec(name=str(metric), ks=tuple(int(k) for k in config.topk)) for metric in config.metrics)
        return EvalConfig(
            protocol="full",
            metrics=metric_specs,
            neg_sampling_num=0,
            candidate_seed=42,
            exclude_history=bool(config.exclude_history),
        )

    @staticmethod
    def _write_result_json(output_dir: str | Path, result: dict[str, Any]) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_path = output_path / "result.json"
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(result), file, indent=2, ensure_ascii=False)
        return result_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


__all__ = [
    "MiniOneRecPipeline",
]
