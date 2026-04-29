from __future__ import annotations

import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from recbole3.config import configs_dir, instantiate_dataclass, project_root
from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID, SEEN_ITEM_IDS, USER_ID
from recbole3.dataset.base import BaseTaskDataset
from recbole3.dataset.cache import DatasetCache
from recbole3.model.base import BaseRetrievalModel, ModelConfig
from recbole3.model.llmrank.config import LLMRankConfig
from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig


def build_candidate_frames(
    task_data: BaseTaskDataset,
    *,
    model_config: LLMRankConfig,
    runtime_cfg: Any,
    dataset_cfg: DictConfig,
    trainer_cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = _create_candidate_generator(
        task_data,
        model_config=model_config,
        runtime_cfg=runtime_cfg,
        dataset_cfg=dataset_cfg,
        trainer_cfg=trainer_cfg,
    )
    valid_frame = generator.build_split_frame("valid")
    test_frame = generator.build_split_frame("test")
    return valid_frame, test_frame


class BaseCandidateGenerator(ABC):
    source_name: str

    def __init__(
        self,
        task_data: BaseTaskDataset,
        *,
        model_config: LLMRankConfig,
        runtime_cfg: Any,
        dataset_cfg: DictConfig,
        trainer_cfg: DictConfig,
    ) -> None:
        self.task_data = task_data
        self.model_config = model_config
        self.runtime_cfg = runtime_cfg
        self.dataset_cfg = dataset_cfg
        self.trainer_cfg = trainer_cfg
        self.cache = DatasetCache(self._cache_root())
        self._selected_user_ids = self._select_user_ids()
        self._selected_eval_frames: dict[str, pd.DataFrame] = {}

    def build_split_frame(self, split: str) -> pd.DataFrame:
        source_frame = self._selected_eval_frame(split)
        if source_frame.empty:
            result = source_frame.copy()
            result[CANDIDATE_ITEM_IDS] = pd.Series(dtype=object)
            return result

        external_path = self._external_candidate_path(split)
        if bool(self.model_config.use_candidate_file) and not self.model_config.refresh_candidate_cache and external_path.exists():
            print(f"[llmrank:candidates] split={split} source={self.source_name} file=hit path={external_path}")
            external_frame = self._read_external_candidate_frame(external_path)
            return self._apply_cached_candidates(source_frame, external_frame, split=split)

        cache_path = self._cache_relative_path(split)
        if not self.model_config.refresh_candidate_cache and self.cache.exists(cache_path):
            print(f"[llmrank:candidates] split={split} source={self.source_name} cache=hit path={self.cache.path(cache_path)}")
            cached_frame = self.cache.read_frame(cache_path, required=True, description=f"{self.source_name} candidate cache")
            self._write_external_candidate_frame(external_path, cached_frame)
            return self._apply_cached_candidates(source_frame, cached_frame, split=split)

        stage_start = time.perf_counter()
        print(
            f"[llmrank:candidates] split={split} source={self.source_name} cache=miss rows={len(source_frame)} "
            f"path={self.cache.path(cache_path)}"
        )
        candidate_rows = self._generate_candidates(source_frame, split=split)
        candidate_frame = pd.DataFrame(
            {
                USER_ID: source_frame[USER_ID].to_numpy(),
                ITEM_ID: source_frame[ITEM_ID].to_numpy(),
                CANDIDATE_ITEM_IDS: candidate_rows,
            }
        )
        self.cache.write_frame(cache_path, candidate_frame)
        self._write_external_candidate_frame(external_path, candidate_frame)
        print(
            f"[llmrank:candidates] split={split} source={self.source_name} cache=written rows={len(candidate_frame)} "
            f"elapsed={time.perf_counter() - stage_start:.2f}s"
        )
        return self._apply_cached_candidates(source_frame, candidate_frame, split=split)

    @abstractmethod
    def _generate_candidates(self, eval_frame: pd.DataFrame, *, split: str) -> list[tuple[int, ...]]:
        ...

    def _apply_cached_candidates(self, source_frame: pd.DataFrame, cached_frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
        if len(cached_frame) != len(source_frame):
            raise ValueError(
                f"Cached {self.source_name} candidates for split '{split}' have {len(cached_frame)} rows, "
                f"but the current eval frame has {len(source_frame)} rows."
            )
        if not cached_frame[USER_ID].equals(source_frame[USER_ID].reset_index(drop=True)):
            raise ValueError(f"Cached {self.source_name} candidates for split '{split}' do not match user order.")
        if not cached_frame[ITEM_ID].equals(source_frame[ITEM_ID].reset_index(drop=True)):
            raise ValueError(f"Cached {self.source_name} candidates for split '{split}' do not match target item order.")
        result = source_frame.copy()
        result[CANDIDATE_ITEM_IDS] = [
            tuple(int(item_id) for item_id in candidate_item_ids)
            for candidate_item_ids in cached_frame[CANDIDATE_ITEM_IDS].tolist()
        ]
        return result

    def _cache_root(self) -> Path:
        root = Path(self.model_config.candidate_cache_dir)
        if not root.is_absolute():
            root = project_root() / root
        return root / self.task_data.config.name / self.source_name / self._cache_signature()

    def _cache_signature(self) -> str:
        payload = {
            "implementation_version": "llmrank-candidates-v4",
            "dataset": _normalize_value(self.dataset_cfg),
            "model": _normalize_value(
                {
                    "candidate_source": self.model_config.candidate_source,
                    "backbone_topk": self._backbone_topk(),
                    "recall_budget": self._recall_budget(),
                    "candidate_seed": self.model_config.candidate_seed,
                    "selected_user_count": self.model_config.selected_user_count,
                    "bm25_item_text_field": self.model_config.bm25_item_text_field,
                    "bm25_fallback_text_field": self.model_config.bm25_fallback_text_field,
                    "backbone_checkpoint_path": self.model_config.backbone_checkpoint_path,
                    "backbone_model": self.model_config.backbone_model,
                    "backbone_trainer": self.model_config.backbone_trainer,
                }
            ),
        }
        payload_text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(payload_text.encode("utf-8")).hexdigest()[:12]

    def _cache_relative_path(self, split: str) -> str:
        return f"{split}.jsonl"

    def _external_candidate_root(self) -> Path:
        root = Path(self.model_config.candidate_file_dir)
        if not root.is_absolute():
            root = project_root() / root
        return root / self.task_data.config.name / self.source_name / self._cache_signature()

    def _external_candidate_path(self, split: str) -> Path:
        return self._external_candidate_root() / f"{split}.jsonl"

    def _eval_frame(self, split: str) -> pd.DataFrame:
        dataset = self.task_data.get_eval_dataset(split)
        if not isinstance(dataset, FrameDataset):
            raise TypeError(f"LLMRank candidate generation requires FrameDataset, got {type(dataset).__name__}.")
        return dataset.frame.copy()

    def _selected_eval_frame(self, split: str) -> pd.DataFrame:
        cached_frame = self._selected_eval_frames.get(split)
        if cached_frame is not None:
            return cached_frame.copy()
        selected_frame = self._filter_selected_users(self._eval_frame(split))
        self._selected_eval_frames[split] = selected_frame.copy()
        return selected_frame

    def _select_user_ids(self) -> tuple[int, ...]:
        selected_user_count = int(self.model_config.selected_user_count)
        test_frame = self._eval_frame("test")
        ordered_user_ids: list[int] = []
        seen_user_ids: set[int] = set()
        for user_id in test_frame[USER_ID].tolist():
            normalized_user_id = int(user_id)
            if normalized_user_id in seen_user_ids:
                continue
            ordered_user_ids.append(normalized_user_id)
            seen_user_ids.add(normalized_user_id)
        if selected_user_count == -1 or selected_user_count >= len(ordered_user_ids):
            return tuple(ordered_user_ids)
        if selected_user_count <= 0:
            raise ValueError("selected_user_count must be -1 or a positive integer.")
        randomizer = random.Random(int(self.model_config.candidate_seed))
        sampled_user_ids = randomizer.sample(ordered_user_ids, selected_user_count)
        sampled_user_set = set(sampled_user_ids)
        return tuple(user_id for user_id in ordered_user_ids if user_id in sampled_user_set)

    def _filter_selected_users(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected_user_ids = set(self._selected_user_ids)
        if not selected_user_ids:
            return frame.iloc[0:0].copy()
        filtered = frame.loc[frame[USER_ID].map(lambda value: int(value) in selected_user_ids)].copy()
        return filtered.reset_index(drop=True)

    def _align_prepared_eval_frame(self, prepared_frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
        expected_frame = self._selected_eval_frame(split)
        filtered_frame = self._filter_selected_users(prepared_frame)
        if len(filtered_frame) != len(expected_frame):
            raise ValueError(
                f"Prepared {self.source_name} eval frame for split '{split}' has {len(filtered_frame)} selected-user rows, "
                f"but the LLMRank eval subset expects {len(expected_frame)} rows."
            )
        filtered_users = filtered_frame[USER_ID].reset_index(drop=True)
        expected_users = expected_frame[USER_ID].reset_index(drop=True)
        if not filtered_users.equals(expected_users):
            raise ValueError(f"Prepared {self.source_name} eval frame for split '{split}' does not match selected user order.")
        filtered_items = filtered_frame[ITEM_ID].reset_index(drop=True)
        expected_items = expected_frame[ITEM_ID].reset_index(drop=True)
        if not filtered_items.equals(expected_items):
            raise ValueError(f"Prepared {self.source_name} eval frame for split '{split}' does not match selected target-item order.")
        return filtered_frame.reset_index(drop=True)

    def _write_external_candidate_frame(self, path: Path, candidate_frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in candidate_frame.to_dict(orient="records"):
                payload = {
                    USER_ID: int(record[USER_ID]),
                    ITEM_ID: int(record[ITEM_ID]),
                    CANDIDATE_ITEM_IDS: [int(item_id) for item_id in record[CANDIDATE_ITEM_IDS]],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_external_candidate_frame(self, path: Path) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append(
                    {
                        USER_ID: int(payload[USER_ID]),
                        ITEM_ID: int(payload[ITEM_ID]),
                        CANDIDATE_ITEM_IDS: tuple(int(item_id) for item_id in payload[CANDIDATE_ITEM_IDS]),
                    }
                )
        return pd.DataFrame(rows, columns=[USER_ID, ITEM_ID, CANDIDATE_ITEM_IDS])

    def _backbone_topk(self) -> int:
        configured = int(self.model_config.backbone_topk)
        if configured <= 0:
            raise ValueError("backbone_topk must be a positive integer.")
        return configured

    def _recall_budget(self) -> int:
        configured = int(self.model_config.recall_budget)
        if configured <= 0:
            raise ValueError("recall_budget must be a positive integer.")
        return configured

    def _num_required_backbone_items(self) -> int:
        return self._backbone_topk()

    def _finalize_candidates(self, candidate_item_ids: list[int]) -> tuple[int, ...]:
        required = self._num_required_backbone_items()
        if len(candidate_item_ids) < required:
            raise ValueError(
                f"{self.source_name} produced only {len(candidate_item_ids)} candidates, "
                f"but backbone_topk={required} requires at least {required} items."
            )
        return tuple(int(item_id) for item_id in candidate_item_ids[:required])


class RandomCandidateGenerator(BaseCandidateGenerator):
    source_name = "random"

    def _generate_candidates(self, eval_frame: pd.DataFrame, *, split: str) -> list[tuple[int, ...]]:
        num_items = int(self.task_data.get_num_items())
        all_item_ids = np.arange(num_items, dtype=np.int64)
        candidate_rows: list[tuple[int, ...]] = []
        split_offset = 0 if split == "valid" else 10_000
        print(f"[llmrank:candidates] generating random candidates for split={split} rows={len(eval_frame)}")
        for row_index, record in enumerate(_progress_records(eval_frame, desc=f"[random:{split}]")):
            user_id = int(record[USER_ID])
            masked_item_ids = set(record.get(SEEN_ITEM_IDS, ()))
            available_item_ids = [int(item_id) for item_id in all_item_ids.tolist() if int(item_id) not in masked_item_ids]
            if len(available_item_ids) < self._num_required_backbone_items():
                raise ValueError(
                    f"random candidate generation only has {len(available_item_ids)} unmasked items for user {user_id}, "
                    f"but backbone_topk={self._num_required_backbone_items()} is required."
                )
            rng = np.random.default_rng(int(self.model_config.candidate_seed) + user_id + split_offset + int(row_index))
            sampled = rng.choice(
                np.asarray(available_item_ids, dtype=np.int64),
                size=self._num_required_backbone_items(),
                replace=False,
            ).tolist()
            candidate_rows.append(self._finalize_candidates(sampled))
        return candidate_rows


class BM25CandidateGenerator(BaseCandidateGenerator):
    source_name = "bm25"

    def __init__(
        self,
        task_data: BaseTaskDataset,
        *,
        model_config: LLMRankConfig,
        runtime_cfg: Any,
        dataset_cfg: DictConfig,
        trainer_cfg: DictConfig,
    ) -> None:
        super().__init__(
            task_data,
            model_config=model_config,
            runtime_cfg=runtime_cfg,
            dataset_cfg=dataset_cfg,
            trainer_cfg=trainer_cfg,
        )
        self._item_text_lookup = _build_item_text_lookup(
            task_data,
            primary_field=model_config.bm25_item_text_field,
            fallback_field=model_config.bm25_fallback_text_field,
        )
        print(f"[llmrank:candidates] building bm25 token corpus for {len(self._item_text_lookup)} items")
        stage_start = time.perf_counter()
        self._encoded_item_text = self._load_segment_text(self._item_text_lookup)
        print(f"[llmrank:candidates] tokenized bm25 corpus in {time.perf_counter() - stage_start:.2f}s")
        stage_start = time.perf_counter()
        self._bm25_model = BM25Model(self._encoded_item_text)
        print(f"[llmrank:candidates] built bm25 index in {time.perf_counter() - stage_start:.2f}s")
        self._all_item_ids = np.arange(int(self.task_data.get_num_items()), dtype=np.int64)

    def _generate_candidates(self, eval_frame: pd.DataFrame, *, split: str) -> list[tuple[int, ...]]:
        candidate_rows: list[tuple[int, ...]] = []
        print(f"[llmrank:candidates] generating bm25 candidates rows={len(eval_frame)}")
        for record in _progress_records(eval_frame, desc=f"[bm25:{split}]"):
            seen_item_ids = set(record.get(SEEN_ITEM_IDS, ()))
            query_tokens: list[str] = []
            for item_id in seen_item_ids:
                query_tokens.extend(self._encoded_item_text[int(item_id)])
            forbidden_mask = np.zeros(len(self._all_item_ids), dtype=bool)
            for item_id in seen_item_ids:
                if 0 <= int(item_id) < len(forbidden_mask):
                    forbidden_mask[int(item_id)] = True
            filtered_item_ids = self._bm25_model.get_topk_item_ids(
                query_tokens,
                forbidden_mask=forbidden_mask,
                limit=self._num_required_backbone_items(),
            )
            candidate_rows.append(self._finalize_candidates(filtered_item_ids))
        return candidate_rows

    @staticmethod
    def _load_segment_text(item_text_lookup: list[str]) -> list[list[str]]:
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "BM25 candidate generation requires nltk. Install nltk and the punkt/stopwords resources."
            ) from exc
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError as exc:
            raise LookupError("BM25 candidate generation requires nltk stopwords data. Run nltk.download('stopwords').") from exc
        segmented_text: list[list[str]] = []
        for text in item_text_lookup:
            try:
                tokens = word_tokenize(text)
            except LookupError as exc:
                raise LookupError("BM25 candidate generation requires nltk punkt data. Run nltk.download('punkt').") from exc
            segmented_text.append([token for token in tokens if token not in stop_words])
        return segmented_text

class ModelBackboneCandidateGenerator(BaseCandidateGenerator):
    source_name = "backbone"

    def __init__(
        self,
        task_data: BaseTaskDataset,
        *,
        model_config: LLMRankConfig,
        runtime_cfg: Any,
        dataset_cfg: DictConfig,
        trainer_cfg: DictConfig,
    ) -> None:
        self.backbone_name = str(model_config.candidate_source).strip().lower()
        self.source_name = self.backbone_name
        super().__init__(
            task_data,
            model_config=model_config,
            runtime_cfg=runtime_cfg,
            dataset_cfg=dataset_cfg,
            trainer_cfg=trainer_cfg,
        )
        self._model_spec = _load_backbone_model_spec(self.backbone_name)
        if self._model_spec.model_data_cls is None:
            raise TypeError(
                f"Candidate source '{self.backbone_name}' does not provide one model-side dataset adapter and cannot be used as one LLMRank backbone."
            )
        self._backbone_model_config, self._backbone_trainer_config = _load_backbone_defaults(
            backbone_name=self.backbone_name,
            model_config_cls=self._model_spec.config_cls,
            trainer_config_cls=self._model_spec.trainer_config_cls,
            model_overrides=self.model_config.backbone_model,
            trainer_overrides=self.model_config.backbone_trainer,
        )
        self._prepared_data = self._model_spec.model_data_cls.from_task_dataset(
            task_data,
            model_config=self._backbone_model_config,
        )
        self._backbone_model = self._model_spec.model_cls(self._backbone_model_config)
        if not isinstance(self._backbone_model, BaseRetrievalModel):
            raise TypeError(
                f"Candidate source '{self.backbone_name}' must instantiate BaseRetrievalModel, got {type(self._backbone_model).__name__}."
            )
        self._backbone_trainer = self._model_spec.trainer_cls(self._backbone_trainer_config)
        self._checkpoint_path = self._resolve_checkpoint_path()

    def _generate_candidates(self, eval_frame: pd.DataFrame, *, split: str) -> list[tuple[int, ...]]:
        from recbole3.evaluation.methods.base import BaseEvaluationMethod
        from recbole3.evaluation.methods.full import FullEvaluationMethod

        model = self._load_trained_backbone_model()
        prepared_split = self._prepared_data.get_eval_dataset(split)
        if not isinstance(prepared_split, FrameDataset):
            raise TypeError(
                f"{self.backbone_name} candidate generation requires FrameDataset, got {type(prepared_split).__name__}."
            )
        records = self._align_prepared_eval_frame(prepared_split.frame, split=split)
        eval_method = FullEvaluationMethod(metric_specs=tuple(), exclude_history=True)
        eval_collate_fn = eval_method.build_eval_collate_fn(model, self._prepared_data)
        eval_dataloader = self._backbone_trainer.build_dataloader(
            FrameDataset(records.reset_index(drop=True)),
            eval_collate_fn,
            shuffle=False,
        )
        accelerator = self._backbone_trainer.create_accelerator()
        prepared_model, prepared_dataloader = accelerator.prepare(model, eval_dataloader)
        scoring_model = accelerator.unwrap_model(prepared_model)
        candidate_rows: list[tuple[int, ...]] = []
        print(
            f"[llmrank:candidates] generating {self.backbone_name} candidates for split={split} rows={len(records)} batch_size={int(self._backbone_trainer_config.batch_size)}"
        )
        prepared_model.eval()
        with torch.no_grad():
            for model_inputs, batch_records in _progress_iterable(prepared_dataloader, desc=f"[{self.backbone_name}:{split}]"):
                device = BaseEvaluationMethod._infer_device(model_inputs)
                exclude_item_ids, exclude_mask = BaseEvaluationMethod._pad_int_lists(batch_records, SEEN_ITEM_IDS, device=device)
                pred_item_ids = scoring_model.predict(
                    model_inputs,
                    k=self._num_required_backbone_items(),
                    candidate_item_ids=None,
                    exclude_item_ids=exclude_item_ids,
                    exclude_mask=exclude_mask,
                )
                for ranked_item_ids in pred_item_ids.detach().cpu().tolist():
                    candidate_rows.append(self._finalize_candidates([int(item_id) for item_id in ranked_item_ids]))
        if len(candidate_rows) != len(eval_frame):
            raise ValueError(
                f"{self.backbone_name} candidate generation produced {len(candidate_rows)} rows, expected {len(eval_frame)} rows."
            )
        return candidate_rows

    def _resolve_checkpoint_path(self) -> Path | None:
        if self.model_config.backbone_checkpoint_path:
            checkpoint_path = Path(self.model_config.backbone_checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root() / checkpoint_path
            return checkpoint_path
        auto_train_dir = self.cache.path("backbone_auto_train")
        checkpoint_path = auto_train_dir / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists() and not self.model_config.refresh_candidate_cache:
            return checkpoint_path
        return None

    def _load_trained_backbone_model(self) -> BaseRetrievalModel:
        self._ensure_initialized_model()
        checkpoint_path = self._checkpoint_path
        if checkpoint_path is None:
            print(f"[llmrank:candidates] no {self.backbone_name} checkpoint provided; auto-training backbone")
            checkpoint_path = self._auto_train_backbone()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{self.backbone_name} checkpoint not found at {checkpoint_path}.")
        print(f"[llmrank:candidates] loading {self.backbone_name} checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._backbone_model.load_state_dict(state_dict)
        return self._backbone_model

    def _auto_train_backbone(self) -> Path:
        auto_train_dir = self.cache.path("backbone_auto_train")
        fit_result = self._backbone_trainer.fit(self._backbone_model, self._prepared_data, output_dir=auto_train_dir)
        checkpoint_path = fit_result["checkpoint_paths"].get("best") or fit_result["checkpoint_paths"].get("last")
        if not checkpoint_path:
            raise RuntimeError(f"Automatic {self.backbone_name} training did not produce one checkpoint.")
        self._checkpoint_path = Path(checkpoint_path)
        return self._checkpoint_path

    def _ensure_initialized_model(self) -> None:
        self._backbone_model.build_eval_collator(self._prepared_data)


class HSTUCandidateGenerator(ModelBackboneCandidateGenerator):
    """Backward-compatible alias for the generic model-backed backbone generator."""

    pass


class BM25Model:
    param_k1 = 1.5
    param_b = 0.75
    epsilon = 0.25

    def __init__(self, corpus: list[list[str]]):
        self.corpus_size = len(corpus)
        self.corpus = corpus
        self.doc_lengths = np.asarray([len(document) for document in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.corpus_size > 0 else 0.0
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._initialize()

    def _initialize(self) -> None:
        posting_tf: dict[str, list[tuple[int, int]]] = {}
        for document_index, document in enumerate(_progress_iterable(self.corpus, desc="[bm25:index]")):
            frequencies: dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            for word in frequencies:
                self.df[word] = self.df.get(word, 0) + 1
                posting_tf.setdefault(word, []).append((document_index, frequencies[word]))
        for word, freq in self.df.items():
            self.idf[word] = float(np.log(self.corpus_size - freq + 0.5) - np.log(freq + 0.5))
        average_idf = sum(float(value) for value in self.idf.values()) / max(len(self.idf), 1)
        length_norm = self.param_k1 * (
            1.0 - self.param_b + self.param_b * np.divide(self.doc_lengths, self.avgdl, out=np.zeros_like(self.doc_lengths), where=self.avgdl > 0)
        )
        for word, entries in posting_tf.items():
            indices = np.asarray([document_index for document_index, _ in entries], dtype=np.int32)
            term_frequencies = np.asarray([term_frequency for _, term_frequency in entries], dtype=np.float32)
            idf = self.idf[word] if self.idf[word] >= 0 else self.epsilon * average_idf
            contribution = (
                idf * term_frequencies * (self.param_k1 + 1.0) / (term_frequencies + length_norm[indices])
            ).astype(np.float32, copy=False)
            self._postings[word] = (indices, contribution)

    def get_topk_item_ids(
        self,
        query_tokens: list[str],
        *,
        forbidden_mask: np.ndarray,
        limit: int,
    ) -> list[int]:
        if limit <= 0 or self.corpus_size == 0:
            return []
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        query_term_counts = Counter(query_tokens)
        for word, query_count in query_term_counts.items():
            posting = self._postings.get(word)
            if posting is None:
                continue
            indices, contribution = posting
            scores[indices] += contribution * float(query_count)

        positive_indices = np.flatnonzero((scores > 0) & (~forbidden_mask))
        ranked_item_ids: list[int] = []
        if len(positive_indices) > 0:
            positive_scores = scores[positive_indices]
            order = np.lexsort((positive_indices, -positive_scores))
            ranked_item_ids = positive_indices[order].astype(np.int64).tolist()
            if len(ranked_item_ids) >= limit:
                return ranked_item_ids[:limit]

        selected_mask = forbidden_mask.copy()
        if ranked_item_ids:
            selected_mask[np.asarray(ranked_item_ids, dtype=np.int64)] = True
        fallback_item_ids = np.flatnonzero(~selected_mask).astype(np.int64).tolist()
        ranked_item_ids.extend(fallback_item_ids[: max(0, limit - len(ranked_item_ids))])
        return ranked_item_ids[:limit]


def _create_candidate_generator(
    task_data: BaseTaskDataset,
    *,
    model_config: LLMRankConfig,
    runtime_cfg: Any,
    dataset_cfg: DictConfig,
    trainer_cfg: DictConfig,
) -> BaseCandidateGenerator:
    source_name = str(model_config.candidate_source).strip().lower()
    if source_name == "random":
        return RandomCandidateGenerator(task_data, model_config=model_config, runtime_cfg=runtime_cfg, dataset_cfg=dataset_cfg, trainer_cfg=trainer_cfg)
    if source_name == "bm25":
        return BM25CandidateGenerator(task_data, model_config=model_config, runtime_cfg=runtime_cfg, dataset_cfg=dataset_cfg, trainer_cfg=trainer_cfg)
    return ModelBackboneCandidateGenerator(
        task_data,
        model_config=model_config,
        runtime_cfg=runtime_cfg,
        dataset_cfg=dataset_cfg,
        trainer_cfg=trainer_cfg,
    )


def _build_item_text_lookup(
    task_data: BaseTaskDataset,
    *,
    primary_field: str,
    fallback_field: str | None,
) -> list[str]:
    item_table = task_data.get_item_table().copy()
    num_items = int(task_data.get_num_items())
    lookup = [f"item {item_id}" for item_id in range(num_items)]
    for record in item_table.to_dict(orient="records"):
        item_id = int(record[ITEM_ID])
        for field_name in (primary_field, fallback_field, "title", "metadata_text", ITEM_ID):
            if not field_name or field_name not in record:
                continue
            value = record[field_name]
            if value is None:
                continue
            text = str(value).strip()
            if text:
                lookup[item_id] = text
                break
    return lookup


def _load_backbone_defaults(
    *,
    backbone_name: str,
    model_config_cls: type[ModelConfig],
    trainer_config_cls: type[TrainerConfig],
    model_overrides: Mapping[str, Any],
    trainer_overrides: Mapping[str, Any],
) -> tuple[ModelConfig, TrainerConfig]:
    config_path = configs_dir() / "model" / f"{backbone_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Candidate source '{backbone_name}' requires one config file at {config_path}, but none was found."
        )
    config = OmegaConf.load(config_path)
    merged_model_cfg = OmegaConf.merge(config.get("model"), OmegaConf.create(dict(model_overrides)))
    merged_trainer_cfg = OmegaConf.merge(config.get("trainer"), OmegaConf.create(dict(trainer_overrides)))
    stopping_step = merged_trainer_cfg.pop("stopping_step", None)
    if stopping_step is not None:
        merged_trainer_cfg["early_stopping"] = OmegaConf.merge(
            merged_trainer_cfg.get("early_stopping", {}),
            {"enabled": True, "patience": int(stopping_step)},
        )
    if "monitor" not in merged_trainer_cfg or merged_trainer_cfg.get("monitor") in (None, ""):
        monitor_name = _default_backbone_monitor_name(merged_trainer_cfg)
        if monitor_name:
            merged_trainer_cfg["monitor"] = monitor_name
    early_stopping_cfg = merged_trainer_cfg.get("early_stopping")
    if early_stopping_cfg is None:
        merged_trainer_cfg["early_stopping"] = {"enabled": True, "patience": 10, "min_delta": 0.0}
    elif "enabled" not in early_stopping_cfg:
        merged_trainer_cfg["early_stopping"] = OmegaConf.merge(early_stopping_cfg, {"enabled": True})
    merged_trainer_cfg["checkpoint"] = {"save_best": True, "save_last": True}
    return (
        instantiate_dataclass(model_config_cls, merged_model_cfg),
        instantiate_dataclass(trainer_config_cls, merged_trainer_cfg),
    )


def _default_backbone_monitor_name(trainer_cfg: Mapping[str, Any] | Any) -> str | None:
    eval_cfg = trainer_cfg.get("eval") if isinstance(trainer_cfg, Mapping) else None
    if not isinstance(eval_cfg, Mapping):
        return None
    metrics = eval_cfg.get("metrics")
    if not metrics:
        return None
    first_metric = metrics[0]
    if not isinstance(first_metric, Mapping):
        return None
    metric_name = str(first_metric.get("name", "")).strip()
    ks = first_metric.get("ks") or ()
    if not metric_name or not ks:
        return None
    return f"{metric_name}@{int(ks[0])}"


def _load_backbone_model_spec(backbone_name: str) -> Any:
    from recbole3.model import get_model_spec

    return get_model_spec(backbone_name)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _progress_records(frame: pd.DataFrame, *, desc: str) -> Any:
    records = frame.to_dict("records")
    try:
        from tqdm.auto import tqdm

        return tqdm(records, desc=desc, total=len(records), leave=True)
    except ModuleNotFoundError:
        return records


def _progress_iterable(values: Any, *, desc: str) -> Any:
    try:
        total = len(values)
    except TypeError:
        total = None
    try:
        from tqdm.auto import tqdm

        return tqdm(values, desc=desc, total=total, leave=True)
    except ModuleNotFoundError:
        return values


__all__ = [
    "BaseCandidateGenerator",
    "BM25CandidateGenerator",
    "HSTUCandidateGenerator",
    "ModelBackboneCandidateGenerator",
    "RandomCandidateGenerator",
    "build_candidate_frames",
]
