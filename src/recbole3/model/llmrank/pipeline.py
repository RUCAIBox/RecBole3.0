from __future__ import annotations

import time
from typing import Any

from omegaconf import DictConfig, OmegaConf

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, FrameDataset, get_dataset_spec
from recbole3.dataset.candidates import build_candidate_frames
from recbole3.model.llmrank.config import LLMRankConfig
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_cfg, require_component_name


class LLMRankPipeline(Pipeline):
    """Pipeline that injects configurable candidate sets before LLM reranking."""

    def run(self) -> dict[str, Any]:
        runtime_cfg, dataset_cfg, model_cfg, trainer_cfg = self._parse_config(self.cfg)
        dataset_name = require_component_name(dataset_cfg, "dataset")
        dataset_spec = get_dataset_spec(dataset_name)
        dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
        model_config = instantiate_dataclass(self.model_spec.config_cls, model_cfg)
        trainer = self.model_spec.trainer_cls(instantiate_dataclass(self.model_spec.trainer_config_cls, trainer_cfg))
        model = self.model_spec.model_cls(model_config)

        with self._accelerate_runtime_device(runtime_cfg.device):
            stage_start = time.perf_counter()
            print("[llmrank:pipeline] preparing task dataset")
            task_data = dataset.prepare(eval_config=trainer.config.eval)
            print(f"[llmrank:pipeline] task dataset ready in {time.perf_counter() - stage_start:.2f}s")
            stage_start = time.perf_counter()
            print("[llmrank:pipeline] building candidate frames")
            task_data = self._inject_candidates(
                task_data,
                model_config=model_config,
                runtime_cfg=runtime_cfg,
                dataset_cfg=dataset_cfg,
                trainer_cfg=trainer_cfg,
            )
            print(f"[llmrank:pipeline] candidate frames ready in {time.perf_counter() - stage_start:.2f}s")
            stage_start = time.perf_counter()
            print("[llmrank:pipeline] building model-side prepared data")
            prepared_data = self._build_model_data(task_data, self.model_spec.model_data_cls, model_cfg)
            print(f"[llmrank:pipeline] model-side prepared data ready in {time.perf_counter() - stage_start:.2f}s")
            print("[llmrank:pipeline] starting trainer run")
            run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)

        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "valid": self._sanitize_result_for_print(run_result["valid"]),
            "test": self._sanitize_result_for_print(run_result["test"]),
        }
        print(OmegaConf.to_yaml(OmegaConf.create(printable), resolve=True))
        return {
            "prepared_data": prepared_data,
            **run_result,
        }

    def _inject_candidates(
        self,
        task_data: BaseTaskDataset,
        *,
        model_config: LLMRankConfig,
        runtime_cfg: RuntimeConfig,
        dataset_cfg: DictConfig,
        trainer_cfg: DictConfig,
    ) -> BaseTaskDataset:
        valid_frame, test_frame = build_candidate_frames(
            task_data,
            model_config=model_config,
            runtime_cfg=runtime_cfg,
            dataset_cfg=dataset_cfg,
            trainer_cfg=trainer_cfg,
        )
        task_data._valid_dataset = FrameDataset(valid_frame)
        task_data._test_dataset = FrameDataset(test_frame)
        return task_data

    @staticmethod
    def _sanitize_result_for_print(result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            return None
        sanitized = dict(result)
        sanitized.pop("inference_results", None)
        return sanitized


__all__ = [
    "LLMRankPipeline",
]
