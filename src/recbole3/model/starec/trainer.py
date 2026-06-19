from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from recbole3.config import project_root
from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID, TIMESTAMP, USER_ID
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MetricSpec, RetrievalEvalData
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.model.starec.candidates import finalize_starec_candidate_row
from recbole3.model.starec.feedback import is_positive_or_unlabeled_record
from recbole3.model.starec.memory import STARecUserMemory
from recbole3.model.starec.parser import complete_ranked_item_ids
from recbole3.trainer import Trainer
from recbole3.trainer_config import OptimizerConfig, TrainerConfig


@dataclass(slots=True)
class _STARecStepResult:
    sequence_index: int
    pred_item_ids: list[int]
    target_item_id: int
    ranked_item_ids: list[int]
    invalid_ranking: bool
    reflection_failure: bool
    sample_log_record: dict[str, Any]
    teacher_trace_records: list[dict[str, Any]]


@dataclass(slots=True)
class _STARecUserSequenceResult:
    user_id: int
    memory: STARecUserMemory
    steps: list[_STARecStepResult]


@dataclass(slots=True)
class STARecTrainerConfig(TrainerConfig):
    """Inference-only trainer config for STARec sequential memory evaluation."""

    batch_size: int = field(default=1, metadata={"help": "Unused placeholder; STARec runs user sequences directly."})
    shuffle: bool = field(default=False, metadata={"help": "STARec sequence evaluation must be deterministic."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Unused placeholder for TrainerConfig compatibility."})
    max_epochs: int = field(default=0, metadata={"help": "STARec does not perform gradient-based training."})
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="full",
            metrics=(
                MetricSpec(name="ndcg", ks=(1, 5, 10, 20)),
                MetricSpec(name="recall", ks=(1, 5, 10, 20)),
            ),
            neg_sampling_num=0,
            candidate_seed=42,
        ),
        metadata={"help": "Retrieval metrics computed on STARec-ranked candidate lists."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig,
        metadata={"help": "Unused placeholder optimizer config kept for TrainerConfig compatibility."},
    )


class STARecTrainer(Trainer):
    """Trainer that runs train warmup, validation reflection, then final test evaluation."""

    config_cls = STARecTrainerConfig

    def run(
        self,
        model,
        prepared_data,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        self._setup_logger(model, prepared_data, output_dir)
        total_start = time.perf_counter()
        try:
            model.prepare_metadata(prepared_data)
            memories, loaded_memory_count = self._load_memories(model, prepared_data)
            should_run_train_warmup = bool(model.config.run_warmup)
            if loaded_memory_count and bool(model.config.skip_warmup_when_memory_loaded):
                should_run_train_warmup = False

            train_warmup_result = None
            if should_run_train_warmup:
                print("[starec] starting train memory warmup")
                train_warmup_result = self._run_train_warmup(
                    model,
                    prepared_data,
                    memories=memories,
                    output_dir=output_dir,
                )
                print("[starec] finished train memory warmup")

            valid_result = None
            print("[starec] starting validation evaluation")
            valid_result = self._run_sequence(
                model,
                prepared_data,
                split="valid",
                memories=memories,
                output_dir=output_dir,
                compute_metrics=True,
                allow_reflection=True,
                record_interactions=True,
            )
            print("[starec] finished validation evaluation")
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_validation(epoch=0, metrics=valid_result["metrics"])

            print("[starec] starting test evaluation")
            test_result = self._run_sequence(
                model,
                prepared_data,
                split="test",
                memories=memories,
                output_dir=output_dir,
                compute_metrics=True,
                allow_reflection=False,
                record_interactions=False,
            )
            print("[starec] finished test evaluation")

            memory_path = self._save_memories(model, memories, output_dir=output_dir)
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                logger.log_summary(
                    stopped_early=False,
                    total_epochs=0,
                    best_epoch=None,
                    total_time=time.perf_counter() - total_start,
                )
            return {
                "train_warmup": train_warmup_result,
                "valid": valid_result,
                "test": test_result,
                "memory_path": str(memory_path) if memory_path is not None else None,
                "loaded_memory_count": loaded_memory_count,
            }
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def _run_sequence(
        self,
        model,
        prepared_data,
        *,
        split: str,
        memories: dict[int, STARecUserMemory],
        output_dir: str | Path | None,
        compute_metrics: bool,
        allow_reflection: bool,
        record_interactions: bool,
        frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        max_k = self._required_max_k()
        if compute_metrics:
            method = self.create_evaluation_method(prepared_data)
            from recbole3.evaluation.methods.base import BaseRetrievalEvaluationMethod

            if not isinstance(method, BaseRetrievalEvaluationMethod):
                raise TypeError("STARec requires retrieval evaluation metrics.")
        else:
            method = None
        if frame is None:
            frame = self._sequence_frame(prepared_data, split=split)
        sample_log_path = self._resolve_output_path(model.config.sample_log_path, output_dir=output_dir)
        teacher_trace_path = self._teacher_trace_path(model, split=split, output_dir=output_dir)

        indexed_records = list(enumerate(frame.to_dict(orient="records")))
        step_results = self._run_user_sequences(
            model,
            split=split,
            indexed_records=indexed_records,
            memories=memories,
            max_k=max_k,
            allow_reflection=allow_reflection,
            record_interactions=record_interactions,
            allow_memory_only_rows=not compute_metrics,
        )
        step_results.sort(key=lambda step: step.sequence_index)

        pred_item_ids = [step.pred_item_ids for step in step_results]
        target_item_ids = [[step.target_item_id] for step in step_results]
        target_mask = [[True] for _ in step_results]
        inference_pred_item_ids = [step.ranked_item_ids for step in step_results]
        invalid_rankings = sum(1 for step in step_results if step.invalid_ranking)
        reflection_failures = sum(1 for step in step_results if step.reflection_failure)
        for step in step_results:
            self._write_sample_log(sample_log_path, step.sample_log_record)
            for trace_record in step.teacher_trace_records:
                self._write_jsonl(teacher_trace_path, trace_record)

        if compute_metrics:
            eval_data = RetrievalEvalData(
                pred_item_ids=np.asarray(pred_item_ids, dtype=np.int64).reshape(len(pred_item_ids), max_k),
                target_item_ids=np.asarray(target_item_ids, dtype=np.int64).reshape(len(target_item_ids), 1),
                target_mask=np.asarray(target_mask, dtype=bool).reshape(len(target_mask), 1),
            )
            metrics = method.compute_metrics([eval_data])
            protocol = method.protocol
        else:
            metrics = {}
            protocol = "warmup"
        result: dict[str, Any] = {
            "split": split,
            "protocol": protocol,
            "loss": None,
            "metrics": metrics,
            "num_batches": len(pred_item_ids),
            "data_stats": self._build_result_data_stats(prepared_data),
            "starec": {
                "invalid_rankings": invalid_rankings,
                "reflection_failures": reflection_failures,
                "memory_count": len(memories),
            },
        }
        if self.config.save_inference_results:
            result["inference_results"] = {
                "pred_item_ids": inference_pred_item_ids,
                "target_item_ids": target_item_ids,
                "target_mask": target_mask,
            }
        return result

    def _run_train_warmup(
        self,
        model,
        prepared_data,
        *,
        memories: dict[int, STARecUserMemory],
        output_dir: str | Path | None,
    ) -> dict[str, Any]:
        train_frame = self._sequence_frame(prepared_data, split="train")
        init_frame, warmup_frame = self._split_train_init_frame(
            train_frame,
            init_count=int(model.config.train_init_interactions),
        )
        self._initialize_train_memories(model, memories, init_frame, output_dir=output_dir)
        return self._run_sequence(
            model,
            prepared_data,
            split="train",
            memories=memories,
            output_dir=output_dir,
            compute_metrics=False,
            allow_reflection=True,
            record_interactions=True,
            frame=warmup_frame,
        )

    def _run_user_sequences(
        self,
        model,
        *,
        split: str,
        indexed_records: list[tuple[int, dict[str, Any]]],
        memories: dict[int, STARecUserMemory],
        max_k: int,
        allow_reflection: bool,
        record_interactions: bool,
        allow_memory_only_rows: bool,
    ) -> list[_STARecStepResult]:
        grouped_records = self._group_indexed_records_by_user(indexed_records)
        if not grouped_records:
            return []

        user_results: list[_STARecUserSequenceResult] = []
        with _STARecProgressBar(
            desc=f"[starec:{split}]",
            total_rows=len(indexed_records),
            total_users=len(grouped_records),
        ) as progress:
            if self._use_async_dispatch(model):
                max_workers = max(1, int(model.config.api_batch))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_one_user_sequence,
                            model,
                            split=split,
                            user_id=user_id,
                            indexed_records=user_records,
                            memory=memories.get(user_id),
                            max_k=max_k,
                            allow_reflection=allow_reflection,
                            record_interactions=record_interactions,
                            allow_memory_only_rows=allow_memory_only_rows,
                            progress=progress,
                        )
                        for user_id, user_records in grouped_records.items()
                    ]
                    for future in as_completed(futures):
                        user_results.append(future.result())
                        progress.update_user_done()
            else:
                for user_id, user_records in grouped_records.items():
                    user_results.append(
                        self._run_one_user_sequence(
                            model,
                            split=split,
                            user_id=user_id,
                            indexed_records=user_records,
                            memory=memories.get(user_id),
                            max_k=max_k,
                            allow_reflection=allow_reflection,
                            record_interactions=record_interactions,
                            allow_memory_only_rows=allow_memory_only_rows,
                            progress=progress,
                        )
                    )
                    progress.update_user_done()

        step_results: list[_STARecStepResult] = []
        for user_result in user_results:
            memories[user_result.user_id] = user_result.memory
            step_results.extend(user_result.steps)
        return step_results

    def _run_one_user_sequence(
        self,
        model,
        *,
        split: str,
        user_id: int,
        indexed_records: list[tuple[int, dict[str, Any]]],
        memory: STARecUserMemory | None,
        max_k: int,
        allow_reflection: bool,
        record_interactions: bool,
        allow_memory_only_rows: bool,
        progress: _STARecProgressBar,
    ) -> _STARecUserSequenceResult:
        steps: list[_STARecStepResult] = []
        for sequence_index, record in indexed_records:
            if memory is None:
                memory = model.build_initial_memory(
                    user_id=user_id,
                    history_item_ids=record.get(HISTORY_ITEM_IDS, ()),
                )
            if not is_positive_or_unlabeled_record(record, model_config=model.config):
                if not allow_memory_only_rows:
                    raise ValueError(f"STARec split '{split}' received a negative target row for user_id={user_id}.")
                if record_interactions:
                    _append_memory_interaction(model, memory, record)
                progress.update_rows(1)
                continue
            steps.append(
                self._run_sequence_step(
                    model,
                    split=split,
                    sequence_index=sequence_index,
                    record=record,
                    memory=memory,
                    max_k=max_k,
                    allow_reflection=allow_reflection,
                    record_interaction=record_interactions,
                )
            )
            progress.update_rows(1)
        if memory is None:
            raise ValueError(f"STARec split '{split}' did not receive records for user_id={user_id}.")
        return _STARecUserSequenceResult(user_id=user_id, memory=memory, steps=steps)

    def _run_sequence_step(
        self,
        model,
        *,
        split: str,
        sequence_index: int,
        record: dict[str, Any],
        memory: STARecUserMemory,
        max_k: int,
        allow_reflection: bool,
        record_interaction: bool,
    ) -> _STARecStepResult:
        user_id = int(record[USER_ID])
        target_item_id = int(record[ITEM_ID])
        row_index = int(record["_starec_row_index"])
        final_candidates = finalize_starec_candidate_row(
            candidate_item_ids=record.get(CANDIDATE_ITEM_IDS, ()),
            target_item_id=target_item_id,
            split=split,
            row_index=row_index,
            recall_budget=int(model.config.recall_budget),
            has_gt=bool(model.config.has_gt),
            fix_pos=int(model.config.fix_pos),
            shuffle=bool(model.config.shuffle),
            candidate_seed=int(model.config.candidate_seed),
            source_name="STARec random",
        )
        if len(final_candidates) < max_k:
            raise ValueError(
                f"STARec split '{split}' needs at least max_k={max_k} candidates, got {len(final_candidates)}."
            )

        memory_before = memory.snapshot()
        ranking_trace_id = _trace_id(split=split, turn_type="ranking", user_id=user_id, sequence_index=sequence_index)
        raw_ranking_output, parsed, ranking_prompt_trace = model.rank_candidates_with_trace(
            memory=memory,
            candidate_item_ids=final_candidates,
        )
        ranked_item_ids = complete_ranked_item_ids(parsed, final_candidates)
        rank_position = _target_rank(ranked_item_ids, target_item_id)
        system_prediction = _system_prediction(rank_position, int(model.config.prediction_liked_threshold))
        actual_feedback = model.record_feedback(record)
        should_reflect = allow_reflection and self._should_reflect(
            mode=str(model.config.reflection_mode),
            system_prediction=system_prediction,
            actual_feedback=actual_feedback,
        )
        raw_reflection_output = None
        reflection_valid = None
        reflection_error = None
        reflection_trace_record = None
        if should_reflect and parsed.valid:
            raw_reflection_output, reflection_valid, reflection_error, reflection_prompt_trace = model.reflect_with_trace(
                memory=memory,
                target_item_id=target_item_id,
                system_prediction=system_prediction,
                actual_feedback=actual_feedback,
            )
            reflection_trace_record = {
                "trace_id": _trace_id(
                    split=split,
                    turn_type="reflection",
                    user_id=user_id,
                    sequence_index=sequence_index,
                ),
                "turn_type": "reflection",
                "split": split,
                "sequence_index": sequence_index,
                "user_id": user_id,
                "target_item_id": target_item_id,
                "previous_ranking_trace_id": ranking_trace_id,
                "system_prediction": system_prediction,
                "actual_feedback": actual_feedback,
                "previous_user_description": memory_before["current_user_description"],
                "raw_output": raw_reflection_output,
                "reasoning_content": reflection_prompt_trace.reasoning_content,
                "messages": reflection_prompt_trace.messages,
                "parse_valid": bool(reflection_valid),
                "reflection_error": reflection_error,
                "updated_user_description": memory.current_user_description if reflection_valid else None,
            }

        if record_interaction:
            _append_memory_interaction(model, memory, record)
        memory_after = memory.snapshot()
        ranking_trace_record = {
            "trace_id": ranking_trace_id,
            "turn_type": "ranking",
            "split": split,
            "sequence_index": sequence_index,
            "user_id": user_id,
            "target_item_id": target_item_id,
            "candidate_item_ids": [int(item_id) for item_id in final_candidates],
            "memory_before_ranking": memory_before,
            "raw_output": raw_ranking_output,
            "reasoning_content": ranking_prompt_trace.reasoning_content,
            "messages": ranking_prompt_trace.messages,
            "parsed_ranking": [int(item_id) for item_id in parsed.ranked_item_ids],
            "parse_valid": bool(parsed.valid),
            "missing_item_ids": [int(item_id) for item_id in parsed.missing_item_ids],
            "duplicate_item_ids": [int(item_id) for item_id in parsed.duplicate_item_ids],
            "unknown_lines": list(parsed.unknown_lines),
            "target_rank": rank_position,
            "system_prediction": system_prediction,
            "actual_feedback": actual_feedback,
        }
        teacher_trace_records = [ranking_trace_record]
        if reflection_trace_record is not None:
            teacher_trace_records.append(reflection_trace_record)

        return _STARecStepResult(
            sequence_index=sequence_index,
            pred_item_ids=[int(item_id) for item_id in ranked_item_ids[:max_k]],
            target_item_id=target_item_id,
            ranked_item_ids=[int(item_id) for item_id in ranked_item_ids],
            invalid_ranking=not bool(parsed.valid),
            reflection_failure=bool(should_reflect and parsed.valid and not reflection_valid),
            teacher_trace_records=teacher_trace_records,
            sample_log_record={
                "split": split,
                "sequence_index": sequence_index,
                "user_id": user_id,
                "target_item_id": target_item_id,
                "candidate_item_ids": [int(item_id) for item_id in final_candidates],
                "memory_before_ranking": memory_before,
                "raw_ranking_output": raw_ranking_output,
                "parsed_ranking": [int(item_id) for item_id in parsed.ranked_item_ids],
                "parse_valid": bool(parsed.valid),
                "missing_item_ids": [int(item_id) for item_id in parsed.missing_item_ids],
                "duplicate_item_ids": [int(item_id) for item_id in parsed.duplicate_item_ids],
                "unknown_lines": list(parsed.unknown_lines),
                "target_rank": rank_position,
                "system_prediction": system_prediction,
                "actual_feedback": actual_feedback,
                "reflection_triggered": bool(should_reflect and parsed.valid),
                "raw_reflection_output": raw_reflection_output,
                "reflection_valid": reflection_valid,
                "reflection_error": reflection_error,
                "memory_after_step": memory_after,
            },
        )

    @staticmethod
    def _group_indexed_records_by_user(
        indexed_records: list[tuple[int, dict[str, Any]]],
    ) -> dict[int, list[tuple[int, dict[str, Any]]]]:
        grouped_records: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for sequence_index, record in indexed_records:
            grouped_records.setdefault(int(record[USER_ID]), []).append((sequence_index, record))
        return grouped_records

    @staticmethod
    def _use_async_dispatch(model) -> bool:
        return bool(getattr(model.config, "async_dispatch", False)) and int(getattr(model.config, "api_batch", 1)) > 1

    def _sequence_frame(self, prepared_data, *, split: str) -> pd.DataFrame:
        dataset = prepared_data.get_train_dataset() if split == "train" else prepared_data.get_eval_dataset(split)
        if not isinstance(dataset, FrameDataset):
            raise TypeError(f"STARec requires FrameDataset splits, got {type(dataset).__name__}.")
        frame = dataset.frame.copy().reset_index(drop=True)
        if CANDIDATE_ITEM_IDS not in frame.columns:
            raise TypeError("STARec requires candidate_item_ids in the eval frame.")
        frame["_starec_row_index"] = range(len(frame))
        sort_columns = [USER_ID]
        if TIMESTAMP in frame.columns and frame[TIMESTAMP].notna().all():
            sort_columns.append(TIMESTAMP)
        sort_columns.append("_starec_row_index")
        return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _split_train_init_frame(frame: pd.DataFrame, *, init_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if init_count < 0:
            raise ValueError("train_init_interactions must be >= 0.")
        init_parts: list[pd.DataFrame] = []
        warmup_parts: list[pd.DataFrame] = []
        for _, user_records in frame.groupby(USER_ID, sort=False):
            init_parts.append(user_records.iloc[:init_count].copy())
            warmup_parts.append(user_records.iloc[init_count:].copy())
        return _concat_like(init_parts, frame), _concat_like(warmup_parts, frame)

    def _initialize_train_memories(
        self,
        model,
        memories: dict[int, STARecUserMemory],
        init_frame: pd.DataFrame,
        *,
        output_dir: str | Path | None,
    ) -> None:
        grouped_records = list(init_frame.groupby(USER_ID, sort=False))
        if not grouped_records:
            return
        teacher_trace_path = self._teacher_trace_path(model, split="train", output_dir=output_dir)
        with _STARecProgressBar(
            desc="[starec:train:init]",
            total_rows=len(init_frame),
            total_users=len(grouped_records),
        ) as progress:
            for user_id, user_records in grouped_records:
                normalized_user_id = int(user_id)
                if normalized_user_id not in memories:
                    memory, trace_record = _build_initial_memory_from_records(
                        model,
                        user_id=normalized_user_id,
                        records=user_records.to_dict(orient="records"),
                    )
                    memories[normalized_user_id] = memory
                    self._write_jsonl(teacher_trace_path, trace_record)
                progress.update_rows(len(user_records))
                progress.update_user_done()

    def _load_memories(self, model, prepared_data) -> tuple[dict[int, STARecUserMemory], int]:
        path = self._resolve_input_path(model.config.memory_load_path)
        if path is None:
            return {}, 0
        if not path.exists():
            raise FileNotFoundError(f"STARec memory_load_path does not exist: {path}")
        memories: dict[int, STARecUserMemory] = {}
        num_users = int(prepared_data.get_num_users())
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                memory = STARecUserMemory.from_record(record)
                if not 0 <= int(memory.user_id) < num_users:
                    raise ValueError(
                        f"STARec memory line {line_number} has user_id={memory.user_id}, "
                        f"outside prepared user range [0, {num_users - 1}]."
                    )
                if not memory.profile_text:
                    memory.profile_text = model.user_profile_text(memory.user_id)
                memories[int(memory.user_id)] = memory
        return memories, len(memories)

    def _save_memories(
        self,
        model,
        memories: dict[int, STARecUserMemory],
        *,
        output_dir: str | Path | None,
    ) -> Path | None:
        path = self._resolve_output_path(model.config.memory_save_path, output_dir=output_dir)
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for user_id in sorted(memories):
                handle.write(json.dumps(memories[user_id].snapshot(), ensure_ascii=False) + "\n")
        return path

    def _required_max_k(self) -> int:
        metric_ks = [int(k) for metric in self.config.eval.metrics for k in metric.ks]
        if not metric_ks:
            raise ValueError("STARec retrieval metrics require at least one top-k value.")
        return max(metric_ks)

    @staticmethod
    def _should_reflect(*, mode: str, system_prediction: str, actual_feedback: str) -> bool:
        if mode == "none":
            return False
        if mode == "always":
            return True
        return (
            (system_prediction == "Predicted Liked" and actual_feedback == "Actually Disliked")
            or (system_prediction == "Predicted Disliked" and actual_feedback == "Actually Liked")
        )

    @staticmethod
    def _resolve_input_path(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = project_root() / path
        return path

    @staticmethod
    def _resolve_output_path(value: str | None, *, output_dir: str | Path | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        base = Path(output_dir) if output_dir is not None else project_root()
        return base / path

    @staticmethod
    def _write_sample_log(path: Path | None, record: dict[str, Any]) -> None:
        STARecTrainer._write_jsonl(path, record)

    @staticmethod
    def _write_jsonl(path: Path | None, record: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _teacher_trace_path(self, model, *, split: str, output_dir: str | Path | None) -> Path | None:
        if split != "train":
            return None
        return self._resolve_output_path(model.config.teacher_trace_path, output_dir=output_dir)


def _system_prediction(rank_position: int | None, threshold: int) -> str:
    if rank_position is None:
        return "Unknown"
    if rank_position <= int(threshold):
        return "Predicted Liked"
    return "Predicted Disliked"


def _target_rank(ranked_item_ids: list[int], target_item_id: int) -> int | None:
    try:
        return ranked_item_ids.index(int(target_item_id)) + 1
    except ValueError:
        return None


def _build_initial_memory_from_records(
    model,
    *,
    user_id: int,
    records: list[dict[str, Any]],
) -> tuple[STARecUserMemory, dict[str, Any]]:
    profile_text = model.user_profile_text(user_id)
    history_lines = [
        f"{model.format_item_line(int(record[ITEM_ID]))}; Feedback: {model.record_feedback_label(record)}"
        for record in records
    ]
    description, prompt_trace = model.initialize_user_description_with_trace(
        profile_text=profile_text,
        history_lines=history_lines,
    )
    memory = STARecUserMemory(
        user_id=int(user_id),
        profile_text=profile_text,
        current_user_description=description,
    )
    for record in records:
        _append_memory_interaction(model, memory, record)
    trace_record = {
        "trace_id": _trace_id(split="train", turn_type="init_memory", user_id=user_id, sequence_index=-1),
        "turn_type": "init_memory",
        "split": "train",
        "sequence_index": -1,
        "user_id": int(user_id),
        "history_item_ids": [int(record[ITEM_ID]) for record in records],
        "history_lines": history_lines,
        "profile_text": profile_text,
        "messages": prompt_trace.messages,
        "raw_output": prompt_trace.raw_output,
        "reasoning_content": prompt_trace.reasoning_content,
        "current_user_description": description,
    }
    return memory, trace_record


def _append_memory_interaction(model, memory: STARecUserMemory, record: dict[str, Any]) -> None:
    item_id = int(record[ITEM_ID])
    memory.append_interaction(
        item_id=item_id,
        item_text=model.item_text(item_id),
        feedback=model.record_feedback_label(record),
        timestamp=model.record_timestamp(record),
        label=model.record_feedback_value(record),
    )


def _trace_id(*, split: str, turn_type: str, user_id: int, sequence_index: int) -> str:
    return f"{split}:{int(user_id)}:{int(sequence_index)}:{turn_type}"


def _concat_like(frames: list[pd.DataFrame], template: pd.DataFrame) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return template.iloc[0:0].copy()
    return pd.concat(non_empty_frames, ignore_index=True, sort=False)


class _STARecProgressBar:
    def __init__(self, *, desc: str, total_rows: int, total_users: int):
        self._bar = _build_progress_bar(desc=desc, total=total_rows)
        self._lock = Lock()
        self._total_users = int(total_users)
        self._users_done = 0

    def __enter__(self) -> "_STARecProgressBar":
        self._set_postfix()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._bar is not None:
            self._bar.close()

    def update_rows(self, count: int) -> None:
        if self._bar is None:
            return
        with self._lock:
            self._bar.update(int(count))
            self._set_postfix_locked()

    def update_user_done(self) -> None:
        if self._bar is None:
            return
        with self._lock:
            self._users_done += 1
            self._set_postfix_locked()

    def _set_postfix(self) -> None:
        if self._bar is None:
            return
        with self._lock:
            self._set_postfix_locked()

    def _set_postfix_locked(self) -> None:
        self._bar.set_postfix_str(f"users={self._users_done}/{self._total_users}")


def _build_progress_bar(*, desc: str, total: int):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=int(total), desc=desc, leave=True)


__all__ = [
    "STARecTrainer",
    "STARecTrainerConfig",
]
