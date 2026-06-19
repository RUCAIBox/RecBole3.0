from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, FrameDataset, get_dataset_spec
from recbole3.dataset.utils import CANDIDATE_ITEM_IDS, ITEM_ID, USER_ID
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_cfg, require_component_name


class AgentCFPipeline(Pipeline):
    """Pipeline that injects candidate sets and runs AgentCF training + evaluation."""

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
            print("[agentcf:pipeline] preparing task dataset")
            task_data = dataset.prepare(eval_config=trainer.config.eval)
            print(f"[agentcf:pipeline] task dataset ready in {time.perf_counter() - stage_start:.2f}s")

            # Inject candidate sets from .random file
            stage_start = time.perf_counter()
            print("[agentcf:pipeline] injecting candidate sets")
            task_data = self._inject_candidates(task_data, model_config=model_config, dataset_cfg=dataset_cfg)
            print(f"[agentcf:pipeline] candidates injected in {time.perf_counter() - stage_start:.2f}s")

            # Build model-side data (adds history_item_ids)
            stage_start = time.perf_counter()
            print("[agentcf:pipeline] building model-side prepared data")
            prepared_data = self._build_model_data(task_data, self.model_spec.model_data_cls, model_cfg)
            print(f"[agentcf:pipeline] model-side data ready in {time.perf_counter() - stage_start:.2f}s")

            # Run trainer
            print("[agentcf:pipeline] starting trainer run")
            run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)

        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "valid": self._sanitize_result(run_result.get("valid")),
            "test": self._sanitize_result(run_result.get("test")),
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
        model_config: Any,
        dataset_cfg: DictConfig,
    ) -> BaseTaskDataset:
        """Load .random file and inject candidate_item_ids into eval frames."""
        candidate_suffix = getattr(model_config, "candidate_file_suffix", "random")
        recall_budget = getattr(model_config, "recall_budget", 20)
        has_gt = getattr(model_config, "has_gt", True)
        fix_pos = getattr(model_config, "fix_pos", -1)
        shuffle_candidates = getattr(model_config, "shuffle_candidates", True)

        # Try to find the candidate file
        data_dir = getattr(dataset_cfg, "data_dir", "") or ""
        dataset_name = getattr(dataset_cfg, "dataset_name", "") or ""

        from pathlib import Path
        candidate_file = None
        if data_dir and dataset_name:
            candidate_path = Path(data_dir) / dataset_name / f"{dataset_name}.{candidate_suffix}"
            if candidate_path.exists():
                candidate_file = candidate_path

        if candidate_file is None:
            print("[agentcf:pipeline] no candidate file found, using sampled negatives")
            return task_data

        # Load candidate items per user
        user2candidates = self._load_candidate_file(candidate_file, task_data)

        # Inject into valid and test frames
        for split in ("valid", "test"):
            eval_dataset = task_data.get_eval_dataset(split)
            if not isinstance(eval_dataset, FrameDataset):
                continue
            frame = eval_dataset.frame.copy()
            if frame.empty:
                continue

            candidate_lists = []
            for record_index, row in frame.iterrows():
                user_id = int(row[USER_ID])
                target_item = int(row[ITEM_ID])
                candidates = user2candidates.get(user_id, [])

                budget = recall_budget - (1 if has_gt else 0)
                candidate_set: list[int] = [int(c) for c in candidates if int(c) != target_item][:budget]

                # If the candidate file doesn't provide enough negatives, fill deterministically to avoid
                # downstream padding introducing item_id=0 as a fake candidate.
                if len(candidate_set) < budget:
                    num_items = int(task_data.get_num_items())
                    seed_fn = getattr(task_data, "_sample_seed", None)
                    seed = (
                        seed_fn(user_id=user_id, split=split, record_index=int(record_index))
                        if callable(seed_fn)
                        else user_id + (0 if split == "valid" else 10_000) + int(record_index)
                    )
                    rng = np.random.default_rng(seed)
                    while len(candidate_set) < budget and num_items > 1:
                        cand = int(rng.integers(0, num_items))
                        if cand != target_item and cand not in candidate_set:
                            candidate_set.append(cand)
                if has_gt:
                    if fix_pos == 0:
                        candidate_set = [target_item] + candidate_set
                    elif fix_pos == -1 or fix_pos >= len(candidate_set):
                        candidate_set = candidate_set + [target_item]
                    else:
                        candidate_set = candidate_set[:fix_pos] + [target_item] + candidate_set[fix_pos:]

                if shuffle_candidates and fix_pos == -1:
                    rng = np.random.default_rng(user_id + (0 if split == "valid" else 10000))
                    candidate_set = list(rng.permutation(candidate_set))

                candidate_lists.append(tuple(int(c) for c in candidate_set))

            frame[CANDIDATE_ITEM_IDS] = candidate_lists

            if split == "valid":
                task_data._valid_dataset = FrameDataset(frame)
            else:
                task_data._test_dataset = FrameDataset(frame)

        return task_data

    def _load_candidate_file(
        self,
        path: Any,
        task_data: BaseTaskDataset,
    ) -> dict[int, list[int]]:
        """Load pre-sampled candidate items from .random file.

        Format: user_id<TAB>item1 item2 item3 ...
        """
        from pathlib import Path

        user2candidates: dict[int, list[int]] = {}

        # We need to map raw IDs to framework IDs
        # Access the internal ID maps if available
        user_id_map = getattr(task_data, "_user_id_map", None)
        item_id_map = getattr(task_data, "_item_id_map", None)

        with open(Path(path), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                raw_uid = parts[0]
                raw_items = parts[1].split(" ")

                # Map to framework IDs
                if user_id_map and raw_uid in user_id_map:
                    uid = user_id_map[raw_uid]
                else:
                    try:
                        uid = int(raw_uid)
                    except ValueError:
                        continue

                items = []
                for raw_iid in raw_items:
                    if item_id_map and raw_iid in item_id_map:
                        items.append(item_id_map[raw_iid])
                    else:
                        try:
                            items.append(int(raw_iid))
                        except ValueError:
                            continue

                user2candidates[uid] = items

        print(f"[agentcf:pipeline] loaded candidates for {len(user2candidates)} users")
        return user2candidates

    @staticmethod
    def _sanitize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return dict(result)


__all__ = [
    "AgentCFPipeline",
]
