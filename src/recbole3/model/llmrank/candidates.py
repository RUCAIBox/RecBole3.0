from __future__ import annotations

import hashlib
import json
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
from recbole3.model.hstu import HSTUConfig, HSTUModel, HSTUModelDataset
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

    def build_split_frame(self, split: str) -> pd.DataFrame:
        source_frame = self._eval_frame(split)
        if source_frame.empty:
            result = source_frame.copy()
            result[CANDIDATE_ITEM_IDS] = pd.Series(dtype=object)
            return result
        cache_path = self._cache_relative_path(split)
        if not self.model_config.refresh_candidate_cache and self.cache.exists(cache_path):
            print(f"[llmrank:candidates] split={split} source={self.source_name} cache=hit path={self.cache.path(cache_path)}")
            cached_frame = self.cache.read_frame(cache_path, required=True, description=f"{self.source_name} candidate cache")
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
            "dataset": _normalize_value(self.dataset_cfg),
            "model": _normalize_value(
                {
                    "candidate_source": self.model_config.candidate_source,
                    "candidate_topk": self.model_config.candidate_topk,
                    "candidate_seed": self.model_config.candidate_seed,
                    "bm25_item_text_field": self.model_config.bm25_item_text_field,
                    "bm25_fallback_text_field": self.model_config.bm25_fallback_text_field,
                    "hstu_checkpoint_path": self.model_config.hstu_checkpoint_path,
                    "hstu_model_overrides": self.model_config.hstu_model_overrides,
                    "hstu_trainer_overrides": self.model_config.hstu_trainer_overrides,
                }
            ),
        }
        payload_text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(payload_text.encode("utf-8")).hexdigest()[:12]

    def _cache_relative_path(self, split: str) -> str:
        return f"{split}.jsonl"

    def _eval_frame(self, split: str) -> pd.DataFrame:
        dataset = self.task_data.get_eval_dataset(split)
        if not isinstance(dataset, FrameDataset):
            raise TypeError(f"LLMRank candidate generation requires FrameDataset, got {type(dataset).__name__}.")
        return dataset.frame.copy()

    def _num_negatives(self) -> int:
        if int(self.model_config.candidate_topk) <= 0:
            raise ValueError("candidate_topk must be a positive integer.")
        return int(self.model_config.candidate_topk) - 1

    def _finalize_candidates(
        self,
        *,
        target_item_id: int,
        negative_item_ids: list[int],
    ) -> tuple[int, ...]:
        negatives = [int(item_id) for item_id in negative_item_ids if int(item_id) != int(target_item_id)]
        required = self._num_negatives()
        if len(negatives) < required:
            raise ValueError(
                f"{self.source_name} produced only {len(negatives)} negatives for target {target_item_id}, "
                f"but candidate_topk={self.model_config.candidate_topk} requires {required} negatives."
            )
        return (int(target_item_id), *negatives[:required])


class RandomCandidateGenerator(BaseCandidateGenerator):
    source_name = "random"

    def _generate_candidates(self, eval_frame: pd.DataFrame, *, split: str) -> list[tuple[int, ...]]:
        num_items = int(self.task_data.get_num_items())
        all_item_ids = np.arange(num_items, dtype=np.int64)
        candidate_rows: list[tuple[int, ...]] = []
        split_offset = 0 if split == "valid" else 10_000
        print(f"[llmrank:candidates] generating random candidates for split={split} rows={len(eval_frame)}")
        for row_index, record in enumerate(_progress_records(eval_frame, desc=f"[random:{split}]")):
            target_item_id = int(record[ITEM_ID])
            seen_item_ids = {int(item_id) for item_id in record.get(SEEN_ITEM_IDS, ())}
            available_mask = np.ones(num_items, dtype=bool)
            if seen_item_ids:
                available_mask[list(seen_item_ids)] = False
            available_mask[target_item_id] = False
            available_item_ids = all_item_ids[available_mask]
            rng = np.random.default_rng(
                int(self.model_config.candidate_seed) + int(record[USER_ID]) + split_offset + int(row_index)
            )
            sampled = rng.choice(available_item_ids, size=self._num_negatives(), replace=False).tolist()
            candidate_rows.append(self._finalize_candidates(target_item_id=target_item_id, negative_item_ids=sampled))
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
        del split
        candidate_rows: list[tuple[int, ...]] = []
        print(f"[llmrank:candidates] generating bm25 candidates rows={len(eval_frame)}")
        for record in _progress_records(eval_frame, desc="[bm25]"):
            target_item_id = int(record[ITEM_ID])
            seen_item_ids = tuple(int(item_id) for item_id in record.get(SEEN_ITEM_IDS, ()))
            query_tokens: list[str] = []
            for item_id in seen_item_ids:
                query_tokens.extend(self._encoded_item_text[int(item_id)])
            forbidden_mask = np.zeros(len(self._all_item_ids), dtype=bool)
            if seen_item_ids:
                forbidden_mask[list(seen_item_ids)] = True
            forbidden_mask[target_item_id] = True
            filtered_item_ids = self._bm25_model.get_topk_item_ids(
                query_tokens,
                forbidden_mask=forbidden_mask,
                limit=self._num_negatives(),
            )
            candidate_rows.append(self._finalize_candidates(target_item_id=target_item_id, negative_item_ids=filtered_item_ids))
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


class HSTUCandidateGenerator(BaseCandidateGenerator):
    source_name = "hstu"

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
        self._hstu_model_config, self._hstu_trainer_config = _load_hstu_defaults(
            model_overrides=model_config.hstu_model_overrides,
            trainer_overrides=model_config.hstu_trainer_overrides,
        )
        self._hstu_prepared_data = HSTUModelDataset.from_task_dataset(task_data, model_config=self._hstu_model_config)
        self._hstu_model = HSTUModel(self._hstu_model_config)
        self._checkpoint_path = self._resolve_checkpoint_path()

    def _generate_candidates(self, eval_frame: pd.DataFrame, *, split: str) -> list[tuple[int, ...]]:
        model = self._load_trained_hstu_model()
        prepared_split = self._hstu_prepared_data.get_eval_dataset(split)
        if not isinstance(prepared_split, FrameDataset):
            raise TypeError(f"HSTU candidate generation requires FrameDataset, got {type(prepared_split).__name__}.")
        records = prepared_split.frame.reset_index(drop=True)
        collator = model.build_eval_collator(self._hstu_prepared_data)
        batch_size = int(self._hstu_trainer_config.batch_size)
        candidate_rows: list[tuple[int, ...]] = []
        print(f"[llmrank:candidates] generating hstu candidates for split={split} rows={len(records)} batch_size={batch_size}")
        model.eval()
        with torch.no_grad():
            for start in _progress_range(0, len(records), batch_size, desc=f"[hstu:{split}]"):
                batch_records = records.iloc[start : start + batch_size].reset_index(drop=True)
                model_inputs = collator(batch_records)
                exclude_item_ids, exclude_mask = _pad_int_lists(batch_records[SEEN_ITEM_IDS].tolist())
                pred_item_ids = model.predict(
                    model_inputs,
                    k=int(self.model_config.candidate_topk),
                    candidate_item_ids=None,
                    exclude_item_ids=exclude_item_ids,
                    exclude_mask=exclude_mask,
                )
                for row_index, ranked_item_ids in enumerate(pred_item_ids.detach().cpu().tolist()):
                    record = batch_records.iloc[row_index]
                    target_item_id = int(record[ITEM_ID])
                    filtered_item_ids = [int(item_id) for item_id in ranked_item_ids if int(item_id) != target_item_id]
                    candidate_rows.append(
                        self._finalize_candidates(target_item_id=target_item_id, negative_item_ids=filtered_item_ids)
                    )
        if len(candidate_rows) != len(eval_frame):
            raise ValueError(
                f"HSTU candidate generation produced {len(candidate_rows)} rows, expected {len(eval_frame)} rows."
            )
        return candidate_rows

    def _resolve_checkpoint_path(self) -> Path | None:
        if self.model_config.hstu_checkpoint_path:
            checkpoint_path = Path(self.model_config.hstu_checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root() / checkpoint_path
            return checkpoint_path
        auto_train_dir = self.cache.path("hstu_auto_train")
        checkpoint_path = auto_train_dir / "checkpoints" / "last_model.pt"
        if checkpoint_path.exists() and not self.model_config.refresh_candidate_cache:
            return checkpoint_path
        return None

    def _load_trained_hstu_model(self) -> HSTUModel:
        self._ensure_initialized_model()
        checkpoint_path = self._checkpoint_path
        if checkpoint_path is None:
            print("[llmrank:candidates] no HSTU checkpoint provided; auto-training backbone")
            checkpoint_path = self._auto_train_hstu()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"HSTU checkpoint not found at {checkpoint_path}.")
        print(f"[llmrank:candidates] loading HSTU checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._hstu_model.load_state_dict(state_dict)
        return self._hstu_model

    def _auto_train_hstu(self) -> Path:
        auto_train_dir = self.cache.path("hstu_auto_train")
        trainer = Trainer(self._hstu_trainer_config)
        fit_result = trainer.fit(self._hstu_model, self._hstu_prepared_data, output_dir=auto_train_dir)
        checkpoint_path = fit_result["checkpoint_paths"].get("last")
        if not checkpoint_path:
            raise RuntimeError("Automatic HSTU training did not produce a last checkpoint.")
        self._checkpoint_path = Path(checkpoint_path)
        return self._checkpoint_path

    def _ensure_initialized_model(self) -> None:
        self._hstu_model.build_eval_collator(self._hstu_prepared_data)


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
                idf
                * term_frequencies
                * (self.param_k1 + 1.0)
                / (term_frequencies + length_norm[indices])
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
    if model_config.candidate_source == "random":
        return RandomCandidateGenerator(task_data, model_config=model_config, runtime_cfg=runtime_cfg, dataset_cfg=dataset_cfg, trainer_cfg=trainer_cfg)
    if model_config.candidate_source == "bm25":
        return BM25CandidateGenerator(task_data, model_config=model_config, runtime_cfg=runtime_cfg, dataset_cfg=dataset_cfg, trainer_cfg=trainer_cfg)
    if model_config.candidate_source == "hstu":
        return HSTUCandidateGenerator(task_data, model_config=model_config, runtime_cfg=runtime_cfg, dataset_cfg=dataset_cfg, trainer_cfg=trainer_cfg)
    raise ValueError(f"Unsupported llmrank candidate_source '{model_config.candidate_source}'.")


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


def _load_hstu_defaults(
    *,
    model_overrides: Mapping[str, Any],
    trainer_overrides: Mapping[str, Any],
) -> tuple[HSTUConfig, TrainerConfig]:
    config = OmegaConf.load(configs_dir() / "model" / "hstu.yaml")
    merged_model_cfg = OmegaConf.merge(config.get("model"), OmegaConf.create(dict(model_overrides)))
    merged_trainer_cfg = OmegaConf.merge(config.get("trainer"), OmegaConf.create(dict(trainer_overrides)))
    merged_trainer_cfg["checkpoint"] = {"save_best": False, "save_last": True}
    return (
        instantiate_dataclass(HSTUConfig, merged_model_cfg),
        instantiate_dataclass(TrainerConfig, merged_trainer_cfg),
    )


def _pad_int_lists(rows: list[tuple[int, ...] | list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(row) for row in rows), default=0)
    values = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        if not row:
            continue
        row_tensor = torch.as_tensor(row, dtype=torch.long).reshape(-1)
        values[row_index, : len(row)] = row_tensor
        mask[row_index, : len(row)] = True
    return values, mask


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


def _progress_range(start: int, stop: int, step: int, *, desc: str) -> Any:
    values = range(start, stop, step)
    try:
        from tqdm.auto import tqdm

        return tqdm(values, desc=desc, total=len(values), leave=True)
    except ModuleNotFoundError:
        return values


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
    "RandomCandidateGenerator",
    "build_candidate_frames",
]
