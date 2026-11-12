"""BIGRec generation utilities.

This module owns evaluation-time prompt generation and the offline vLLM
beam-search subprocess.  Keeping it separate from ``trainer.py`` leaves the
trainer focused on LoRA SFT.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from recbole3.dataset.utils import CANDIDATE_ITEM_IDS, ITEM_ID, SEEN_ITEM_IDS
from recbole3.model.bigrec.config import BIGRecConfig
from recbole3.model.bigrec.data import build_eval_prompts


logger = logging.getLogger(__name__)

_VLLM_OFFLINE_SCRIPT: str = os.path.join(os.path.dirname(__file__), "vllm_offline.py")


class BIGRecGenerator:
    """Build BIGRec prompts and run offline vLLM beam-search generation."""

    def __init__(
        self,
        config: BIGRecConfig,
        *,
        is_main_process: Callable[[], bool],
        log: Callable[..., None],
    ) -> None:
        self.config = config
        self._is_main_process = is_main_process
        self._log = log

    def resolve_vllm_python(self) -> str:
        """Return the Python executable used for the vLLM subprocess."""
        if not self.config.vllm_conda_env:
            return sys.executable

        try:
            result = subprocess.run(
                ["conda", "info", "--base"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            conda_base = result.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError(
                "conda not found on PATH. "
                "Add conda to PATH or set vllm_conda_env='' to use the current interpreter."
            ) from None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"'conda info --base' failed: {exc.stderr.strip()}") from exc

        python_rel = (
            os.path.join("Scripts", "python.exe")
            if sys.platform == "win32"
            else os.path.join("bin", "python")
        )
        python_path = os.path.join(conda_base, "envs", self.config.vllm_conda_env, python_rel)
        if not os.path.isfile(python_path):
            raise FileNotFoundError(
                f"Python not found at {python_path!r}. "
                f"Check that conda env '{self.config.vllm_conda_env}' exists."
            )
        return python_path

    @staticmethod
    def collect_eval_targets(
        eval_frame: pd.DataFrame,
    ) -> tuple[list[int], list[list[int] | None]]:
        """Extract target item ids and per-row candidate lists from an eval frame."""
        target_ids: list[int] = []
        cand_lists: list[list[int] | None] = []
        for row in eval_frame.itertuples(index=False):
            target_ids.append(int(getattr(row, ITEM_ID)))
            cand_val = getattr(row, CANDIDATE_ITEM_IDS, None)
            if cand_val is None or (
                not hasattr(cand_val, "__len__")
                and isinstance(cand_val, float)
                and np.isnan(cand_val)
            ):
                cand_lists.append(None)
            else:
                cand_lists.append(list(cand_val))
        return target_ids, cand_lists

    @staticmethod
    def collect_eval_exclusions(eval_frame: pd.DataFrame) -> list[list[int]]:
        """Return per-request histories used by full-ranking exclusion."""
        if SEEN_ITEM_IDS not in eval_frame.columns:
            return [[] for _ in range(len(eval_frame))]
        exclusions: list[list[int]] = []
        for seen_item_ids in eval_frame[SEEN_ITEM_IDS].tolist():
            if seen_item_ids is None or (
                isinstance(seen_item_ids, float) and np.isnan(seen_item_ids)
            ):
                exclusions.append([])
            else:
                exclusions.append([int(item_id) for item_id in seen_item_ids])
        return exclusions

    def run_vllm_offline(
        self,
        prompts: list[str],
        checkpoint_path: str,
        work_dir: str,
    ) -> list[str]:
        """Run vLLM ``LLM.beam_search`` in a one-shot subprocess."""
        python_exe = self.resolve_vllm_python()
        max_model_len = int(self.config.max_input_length + self.config.max_new_tokens)
        tp = max(1, int(self.config.vllm_tensor_parallel_size))
        vllm_gpu_ids = ",".join(str(self.config.vllm_device_id + i) for i in range(tp))

        os.makedirs(work_dir, exist_ok=True)
        prompts_path = os.path.join(work_dir, "vllm_prompts.json")
        output_path = os.path.join(work_dir, "vllm_outputs.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False)

        cmd: list[str] = [
            python_exe,
            _VLLM_OFFLINE_SCRIPT,
            "--prompts",
            prompts_path,
            "--output",
            output_path,
            "--model",
            self.config.llm_path,
            "--dtype",
            self.config.torch_dtype,
            "--beam-width",
            str(max(1, int(self.config.num_beams))),
            "--max-tokens",
            str(self.config.max_new_tokens),
            "--max-model-len",
            str(max_model_len),
            "--tp",
            str(tp),
            "--gpu-memory-utilization",
            str(self.config.vllm_gpu_memory_utilization),
        ]
        if self.config.use_lora:
            cmd += ["--lora", checkpoint_path, "--lora-rank", str(self.config.lora_r)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = vllm_gpu_ids

        self._log(
            "Launching vLLM offline beam-search: env=%s, CUDA_VISIBLE_DEVICES=%s, "
            "tp=%d, beam_width=%d, n_prompts=%d%s",
            self.config.vllm_conda_env or "(current)",
            vllm_gpu_ids,
            tp,
            max(1, int(self.config.num_beams)),
            len(prompts),
            f", lora={checkpoint_path}" if self.config.use_lora else "",
        )

        completed = subprocess.run(cmd, env=env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"vLLM offline subprocess exited with code {completed.returncode}. "
                f"Inspect the output above and the prompts file at {prompts_path}."
            )

        with open(output_path, "r", encoding="utf-8") as f:
            raw_outputs: list[str] = json.load(f)

        if len(raw_outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM offline returned {len(raw_outputs)} outputs but "
                f"{len(prompts)} prompts were sent."
            )

        clean_texts: list[str] = []
        for idx, text in enumerate(raw_outputs):
            cleaned = text.strip().strip('"').strip()
            clean_texts.append(cleaned or f"[empty_{idx}]")
        return clean_texts

    def generate_all_titles_vllm(
        self,
        eval_frame: pd.DataFrame,
        item_text_lookup: list[str],
        checkpoint_path: str,
        work_dir: str,
    ) -> tuple[list[str], list[int], list[list[int] | None]]:
        """Generate item titles for an eval split with offline vLLM."""
        prompts = build_eval_prompts(eval_frame, item_text_lookup, self.config)
        target_ids, cand_lists = self.collect_eval_targets(eval_frame)
        clean_texts = self.run_vllm_offline(prompts, checkpoint_path, work_dir)

        if self._is_main_process():
            for i in range(min(3, len(clean_texts))):
                logger.info(
                    "[vLLM sample %d]  generated: %r  |  target: %r",
                    i,
                    clean_texts[i],
                    item_text_lookup[target_ids[i]],
                )

        return clean_texts, target_ids, cand_lists


__all__ = ["BIGRecGenerator"]
