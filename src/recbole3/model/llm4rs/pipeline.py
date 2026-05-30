from __future__ import annotations

import time
from random import Random
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID, USER_ID, get_dataset_spec
from recbole3.model.llm4rs.config import LLM4RSConfig
from recbole3.model.llmrank.candidates import build_candidate_frames
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_name


class LLM4RSPipeline(Pipeline):
    """Construct candidate sets and run LLM4RS as an evaluation-only model."""

    def run(self) -> dict[str, Any]:
        runtime_cfg, dataset_cfg, model_cfg, trainer_cfg = self._parse_config(self.cfg)
        dataset_spec = get_dataset_spec(require_component_name(dataset_cfg, "dataset"))
        dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
        model_config = instantiate_dataclass(self.model_spec.config_cls, model_cfg)
        trainer = self.model_spec.trainer_cls(instantiate_dataclass(self.model_spec.trainer_config_cls, trainer_cfg))
        model = self.model_spec.model_cls(model_config)

        with self._accelerate_runtime_device(runtime_cfg.device):
            stage_start = time.perf_counter()
            print("[llm4rs:pipeline] preparing task dataset")
            task_data = dataset.prepare(eval_config=trainer.config.eval)
            print(f"[llm4rs:pipeline] task dataset ready in {time.perf_counter() - stage_start:.2f}s")
            task_data = self._inject_candidates(
                task_data,
                model_config=model_config,
                runtime_cfg=runtime_cfg,
                dataset_cfg=dataset_cfg,
                trainer_cfg=trainer_cfg,
            )
            prepared_data = self._build_model_data(task_data, self.model_spec.model_data_cls, model_cfg)
            run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)

        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "valid": self._sanitize_result(run_result["valid"]),
            "test": self._sanitize_result(run_result["test"]),
        }
        print(OmegaConf.to_yaml(OmegaConf.create(printable), resolve=True))
        return {"prepared_data": prepared_data, **run_result}

    def _inject_candidates(
        self,
        task_data: BaseTaskDataset,
        *,
        model_config: LLM4RSConfig,
        runtime_cfg: RuntimeConfig,
        dataset_cfg: DictConfig,
        trainer_cfg: DictConfig,
    ) -> BaseTaskDataset:
        source_name = str(model_config.candidate_source).strip().lower()
        if source_name in {"random", "prepared"}:
            selected_user_ids = self._select_user_ids(task_data, config=model_config)
            frame_builder = self._random_candidate_frame if source_name == "random" else self._prepared_candidate_frame
            valid_frame = frame_builder(task_data, split="valid", config=model_config, selected_user_ids=selected_user_ids)
            test_frame = frame_builder(task_data, split="test", config=model_config, selected_user_ids=selected_user_ids)
        else:
            valid_frame, test_frame = build_candidate_frames(
                task_data,
                model_config=model_config,
                runtime_cfg=runtime_cfg,
                dataset_cfg=dataset_cfg,
                trainer_cfg=trainer_cfg,
            )
            valid_frame = self._official_candidate_frame(valid_frame, split="valid", config=model_config)
            test_frame = self._official_candidate_frame(test_frame, split="test", config=model_config)
        task_data._valid_dataset = FrameDataset(valid_frame)
        task_data._test_dataset = FrameDataset(test_frame)
        return task_data

    @staticmethod
    def _random_candidate_frame(
        task_data: BaseTaskDataset,
        *,
        split: str,
        config: LLM4RSConfig,
        selected_user_ids: tuple[int, ...],
    ) -> pd.DataFrame:
        source_dataset = task_data.get_eval_dataset(split)
        if not isinstance(source_dataset, FrameDataset):
            raise TypeError(f"LLM4RS random candidate construction requires FrameDataset, got {type(source_dataset).__name__}.")
        result = LLM4RSPipeline._filter_selected_users(source_dataset.frame.copy(), selected_user_ids)
        candidate_num = int(config.candidate_num)
        num_items = int(task_data.get_num_items())
        if candidate_num > num_items:
            raise ValueError(f"LLM4RS candidate_num={candidate_num} exceeds num_items={num_items}.")
        all_item_ids = np.arange(num_items, dtype=np.int64)
        split_offset = 0 if split == "valid" else 10_000
        candidate_rows: list[tuple[int, ...]] = []
        for row_index, target_item_id in enumerate(result[ITEM_ID].tolist()):
            target = int(target_item_id)
            negative_pool = all_item_ids[all_item_ids != target]
            rng = np.random.default_rng(int(config.candidate_seed) + split_offset + row_index)
            negatives = rng.choice(negative_pool, size=candidate_num - 1, replace=False).astype(np.int64).tolist()
            candidate_row = [target, *[int(item_id) for item_id in negatives]]
            if bool(config.shuffle_candidates) and len(candidate_row) > 1:
                rng.shuffle(candidate_row)
            candidate_rows.append(tuple(candidate_row))
        result[CANDIDATE_ITEM_IDS] = candidate_rows
        return result

    @staticmethod
    def _prepared_candidate_frame(
        task_data: BaseTaskDataset,
        *,
        split: str,
        config: LLM4RSConfig,
        selected_user_ids: tuple[int, ...],
    ) -> pd.DataFrame:
        source_dataset = task_data.get_eval_dataset(split)
        if not isinstance(source_dataset, FrameDataset):
            raise TypeError(f"LLM4RS prepared candidates require FrameDataset, got {type(source_dataset).__name__}.")
        result = LLM4RSPipeline._filter_selected_users(source_dataset.frame.copy(), selected_user_ids)
        if CANDIDATE_ITEM_IDS not in result.columns:
            raise TypeError("model.candidate_source=prepared requires candidate_item_ids already present in eval records.")
        candidate_num = int(config.candidate_num)
        rows: list[tuple[int, ...]] = []
        for row_index, (candidate_item_ids, target_item_id) in enumerate(
            zip(result[CANDIDATE_ITEM_IDS].tolist(), result[ITEM_ID].tolist(), strict=True)
        ):
            candidates = tuple(int(item_id) for item_id in candidate_item_ids)
            if len(candidates) != candidate_num or int(target_item_id) not in candidates:
                raise ValueError(
                    f"Prepared LLM4RS split '{split}' row {row_index} must have exactly {candidate_num} candidates "
                    "including its target item."
                )
            rows.append(candidates)
        result[CANDIDATE_ITEM_IDS] = rows
        return result

    @staticmethod
    def _official_candidate_frame(frame: pd.DataFrame, *, split: str, config: LLM4RSConfig) -> pd.DataFrame:
        result = frame.copy()
        candidate_num = int(config.candidate_num)
        split_offset = 0 if split == "valid" else 10_000
        rows: list[tuple[int, ...]] = []
        for row_index, (candidate_item_ids, target_item_id) in enumerate(
            zip(result[CANDIDATE_ITEM_IDS].tolist(), result[ITEM_ID].tolist(), strict=True)
        ):
            target = int(target_item_id)
            negatives = [int(item_id) for item_id in candidate_item_ids if int(item_id) != target]
            if len(negatives) < candidate_num - 1:
                raise ValueError(
                    f"LLM4RS split '{split}' row {row_index} requires {candidate_num - 1} negative candidates, "
                    f"but candidate generation supplied {len(negatives)}."
                )
            official_row = [target, *negatives[: candidate_num - 1]]
            if bool(config.shuffle_candidates) and len(official_row) > 1:
                rng = np.random.default_rng(int(config.candidate_seed) + split_offset + row_index)
                rng.shuffle(official_row)
            rows.append(tuple(official_row))
        result[CANDIDATE_ITEM_IDS] = rows
        return result

    @staticmethod
    def _select_user_ids(task_data: BaseTaskDataset, *, config: LLM4RSConfig) -> tuple[int, ...]:
        test_dataset = task_data.get_eval_dataset("test")
        if not isinstance(test_dataset, FrameDataset):
            raise TypeError(f"LLM4RS user selection requires FrameDataset, got {type(test_dataset).__name__}.")
        ordered_user_ids = tuple(dict.fromkeys(int(user_id) for user_id in test_dataset.frame[USER_ID].tolist()))
        count = int(config.selected_user_count)
        if count == -1 or count >= len(ordered_user_ids):
            return ordered_user_ids
        if count <= 0:
            raise ValueError("selected_user_count must be -1 or a positive integer.")
        sampled = set(Random(int(config.candidate_seed)).sample(list(ordered_user_ids), count))
        return tuple(user_id for user_id in ordered_user_ids if user_id in sampled)

    @staticmethod
    def _filter_selected_users(frame: pd.DataFrame, selected_user_ids: tuple[int, ...]) -> pd.DataFrame:
        selected = set(selected_user_ids)
        return frame.loc[frame[USER_ID].map(lambda value: int(value) in selected)].reset_index(drop=True).copy()

    @staticmethod
    def _sanitize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            return None
        sanitized = dict(result)
        sanitized.pop("inference_results", None)
        return sanitized


__all__ = [
    "LLM4RSPipeline",
]
