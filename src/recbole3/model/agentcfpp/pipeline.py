from __future__ import annotations

import time
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from recbole3.config import instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, FrameDataset, get_dataset_spec
from recbole3.dataset.utils import CANDIDATE_ITEM_IDS, ITEM_ID, USER_ID
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_name


class AgentCFPPPipeline(Pipeline):
    """Pipeline that injects per-domain candidate sets and runs AgentCF++ training + eval."""

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
            print("[agentcfpp:pipeline] preparing task dataset")
            task_data = dataset.prepare(eval_config=trainer.config.eval)
            print(f"[agentcfpp:pipeline] task dataset ready in {time.perf_counter() - stage_start:.2f}s")

            # Cross-domain context: item->domain map and per-domain candidate pools.
            item_domains = self._get_item_domains(task_data)
            candidate_pools = self._get_candidate_pools(task_data)
            model.set_cross_domain_context(item_domains=item_domains, domain_candidate_pools=candidate_pools)

            # Inject per-domain candidate sets into eval frames.
            stage_start = time.perf_counter()
            print("[agentcfpp:pipeline] injecting per-domain candidate sets")
            task_data = self._inject_candidates(task_data, model_config=model_config, candidate_pools=candidate_pools, item_domains=item_domains)
            print(f"[agentcfpp:pipeline] candidates injected in {time.perf_counter() - stage_start:.2f}s")

            stage_start = time.perf_counter()
            print("[agentcfpp:pipeline] building model-side prepared data")
            prepared_data = self._build_model_data(task_data, self.model_spec.model_data_cls, model_cfg)
            print(f"[agentcfpp:pipeline] model-side data ready in {time.perf_counter() - stage_start:.2f}s")

            print("[agentcfpp:pipeline] starting trainer run")
            run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)

        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "valid": self._sanitize_result(run_result.get("valid")),
            "test": self._sanitize_result(run_result.get("test")),
        }
        print(OmegaConf.to_yaml(OmegaConf.create(printable), resolve=True))
        return {"prepared_data": prepared_data, **run_result}

    @staticmethod
    def _get_item_domains(task_data: BaseTaskDataset) -> dict[int, str]:
        getter = getattr(task_data, "get_item_domains", None)
        return getter() if callable(getter) else {}

    @staticmethod
    def _get_candidate_pools(task_data: BaseTaskDataset) -> dict[str, dict[int, list[int]]]:
        getter = getattr(task_data, "get_domain_candidate_pools", None)
        return getter() if callable(getter) else {}

    def _inject_candidates(
        self,
        task_data: BaseTaskDataset,
        *,
        model_config: Any,
        candidate_pools: dict[str, dict[int, list[int]]],
        item_domains: dict[int, str],
    ) -> BaseTaskDataset:
        candidate_num = getattr(model_config, "candidate_num", 10)
        has_gt = getattr(model_config, "has_gt", True)
        shuffle_candidates = getattr(model_config, "shuffle_candidates", True)

        if not candidate_pools:
            print("[agentcfpp:pipeline] no candidate pools found; skipping injection")
            return task_data

        for split in ("valid", "test"):
            eval_dataset = task_data.get_eval_dataset(split)
            if not isinstance(eval_dataset, FrameDataset):
                continue
            frame = eval_dataset.frame.copy()
            if frame.empty:
                continue

            candidate_lists = []
            for _, row in frame.iterrows():
                user_id = int(row[USER_ID])
                target_item = int(row[ITEM_ID])
                domain = item_domains.get(target_item, "")
                pool = candidate_pools.get(domain, {}).get(user_id, [])

                budget = candidate_num - (1 if has_gt else 0)
                negatives = [c for c in pool if c != target_item][:budget]
                candidate_set = list(negatives)
                if has_gt:
                    candidate_set.append(target_item)

                if shuffle_candidates:
                    rng = np.random.default_rng(user_id + (0 if split == "valid" else 10000))
                    candidate_set = list(rng.permutation(candidate_set))

                candidate_lists.append(tuple(int(c) for c in candidate_set))

            frame[CANDIDATE_ITEM_IDS] = candidate_lists
            if split == "valid":
                task_data._valid_dataset = FrameDataset(frame)
            else:
                task_data._test_dataset = FrameDataset(frame)

        return task_data

    @staticmethod
    def _sanitize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return dict(result)


__all__ = ["AgentCFPPPipeline"]
