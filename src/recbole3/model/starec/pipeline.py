from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, FrameDataset, get_dataset_spec
from recbole3.model.llmrank.candidates import build_candidate_frames
from recbole3.model.starec.candidates import build_history_limited_frames, build_train_candidate_frame
from recbole3.model.starec.config import STARecConfig
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_name


class STARecPipeline(Pipeline):
    """Pipeline that injects random+GT candidates before STARec memory evaluation."""

    def run(self) -> dict[str, Any]:
        runtime_cfg, dataset_cfg, model_cfg, trainer_cfg = self._parse_config(self.cfg)
        dataset_name = require_component_name(dataset_cfg, "dataset")
        dataset_spec = get_dataset_spec(dataset_name)
        dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
        model_config = instantiate_dataclass(self.model_spec.config_cls, model_cfg)
        if not isinstance(model_config, STARecConfig):
            raise TypeError(f"STARecPipeline expected STARecConfig, got {type(model_config).__name__}.")
        trainer = self.model_spec.trainer_cls(instantiate_dataclass(self.model_spec.trainer_config_cls, trainer_cfg))
        model = self.model_spec.model_cls(model_config)

        with self._accelerate_runtime_device(runtime_cfg.device):
            stage_start = time.perf_counter()
            print("[starec:pipeline] preparing task dataset")
            task_data = dataset.prepare(eval_config=trainer.config.eval)
            print(f"[starec:pipeline] task dataset ready in {time.perf_counter() - stage_start:.2f}s")
            stage_start = time.perf_counter()
            print("[starec:pipeline] building random candidate frames")
            task_data = self._inject_candidates(
                task_data,
                model_config=model_config,
                runtime_cfg=runtime_cfg,
                dataset_cfg=dataset_cfg,
                trainer_cfg=trainer_cfg,
            )
            print(f"[starec:pipeline] candidate frames ready in {time.perf_counter() - stage_start:.2f}s")
            stage_start = time.perf_counter()
            print("[starec:pipeline] building model-side prepared data")
            prepared_data = self._build_model_data(task_data, self.model_spec.model_data_cls, model_cfg)
            print(f"[starec:pipeline] model-side prepared data ready in {time.perf_counter() - stage_start:.2f}s")
            print("[starec:pipeline] starting trainer run")
            run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)

        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "train_warmup": self._sanitize_result_for_print(run_result.get("train_warmup")),
            "valid": self._sanitize_result_for_print(run_result.get("valid")),
            "test": self._sanitize_result_for_print(run_result["test"]),
            "memory_path": run_result.get("memory_path"),
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
        model_config: STARecConfig,
        runtime_cfg: RuntimeConfig,
        dataset_cfg,
        trainer_cfg,
    ) -> BaseTaskDataset:
        history_train_frame, history_valid_frame, history_test_frame, selected_user_ids = build_history_limited_frames(
            task_data,
            model_config=model_config,
        )
        task_data._train_dataset = FrameDataset(history_train_frame)
        task_data._valid_dataset = FrameDataset(history_valid_frame)
        task_data._test_dataset = FrameDataset(history_test_frame)

        candidate_model_config = _candidate_model_config(model_config, selected_user_ids=selected_user_ids)
        valid_frame, test_frame = build_candidate_frames(
            task_data,
            model_config=candidate_model_config,  # type: ignore[arg-type]
            runtime_cfg=runtime_cfg,
            dataset_cfg=dataset_cfg,
            trainer_cfg=trainer_cfg,
        )
        train_frame = build_train_candidate_frame(
            task_data,
            model_config=model_config,
        )
        task_data._train_dataset = FrameDataset(train_frame)
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
    "STARecPipeline",
]


def _candidate_model_config(model_config: STARecConfig, *, selected_user_ids: tuple[int, ...]) -> STARecConfig:
    selection_signature = _selection_signature(model_config, selected_user_ids=selected_user_ids)
    return replace(
        model_config,
        selected_user_count=-1,
        candidate_cache_dir=str(Path(model_config.candidate_cache_dir) / "starec" / selection_signature),
        candidate_file_dir=str(Path(model_config.candidate_file_dir) / "starec" / selection_signature),
    )


def _selection_signature(model_config: STARecConfig, *, selected_user_ids: tuple[int, ...]) -> str:
    payload = {
        "history_max_length": model_config.history_max_length,
        "history_min_length": model_config.history_min_length,
        "train_init_interactions": model_config.train_init_interactions,
        "selected_user_count": model_config.selected_user_count,
        "selected_user_ids": [int(user_id) for user_id in selected_user_ids],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:12]
