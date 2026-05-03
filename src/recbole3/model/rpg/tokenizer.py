from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from recbole3.config import instantiate_dataclass
from recbole3.dataset import ITEM_ID
from recbole3.dataset.base import BaseTaskDataset
from recbole3.model.rpg.config import RPGConfig


RPG_PADDING_ITEM_ID = 0
RPG_ITEM_ID_OFFSET = 1
RPG_IGNORED_LABEL = -100


def ensure_rpg_config(config: RPGConfig | Mapping[str, Any]) -> RPGConfig:
    if isinstance(config, RPGConfig):
        return config
    return instantiate_dataclass(RPGConfig, config)


class RPGSemanticTokenizer:
    """Build RPG item-token mappings and sequence examples from RecBole data.

    RecBole stores real items as 0..N-1. The original RPG implementation stores
    real items as 1..N and reserves 0 for padding, so this adapter keeps that
    convention at the model boundary.
    """

    def __init__(self, config: RPGConfig, prepared_data: BaseTaskDataset):
        self.config = config
        self.num_items = int(prepared_data.get_num_items())
        self.n_digit = int(config.n_codebook)
        self.codebook_size = int(config.codebook_size)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = RPG_IGNORED_LABEL
        self.item_id2tokens = self._build_item_id2tokens(prepared_data)

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        return int(self.config.max_item_seq_len)

    def tokenize_train_sequence(self, item_seq: list[int]) -> list[dict[str, Any]]:
        """Tokenize one user's training sequence with RPG's sliding-window rule."""

        max_len = self.max_token_seq_len
        if len(item_seq) < 2:
            return []

        n_return_examples = max(len(item_seq) - max_len, 1)
        examples = [self._tokenize_first_n_items(item_seq[: min(len(item_seq), max_len + 1)])]
        for start in range(1, n_return_examples):
            examples.append(self._tokenize_later_items(item_seq[start : start + max_len + 1], pad_labels=True))
        return examples

    def tokenize_eval_sequence(self, item_seq: list[int]) -> dict[str, Any]:
        max_len = self.max_token_seq_len
        return self._tokenize_later_items(item_seq[-(max_len + 1) :], pad_labels=False)

    def _tokenize_first_n_items(self, item_seq: list[int]) -> dict[str, Any]:
        model_seq = self._to_model_item_seq(item_seq)
        input_ids = model_seq[:-1]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([RPG_PADDING_ITEM_ID] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = model_seq[1:]
        labels.extend([self.ignored_label] * pad_lens)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seq_lens": seq_lens,
        }

    def _tokenize_later_items(self, item_seq: list[int], *, pad_labels: bool) -> dict[str, Any]:

        model_seq = self._to_model_item_seq(item_seq)
        input_ids = model_seq[:-1]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        if labels:
            labels[-1] = model_seq[-1]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([RPG_PADDING_ITEM_ID] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)
        else:
            labels = [model_seq[-1]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seq_lens": seq_lens,
        }

    @staticmethod
    def _to_model_item_seq(item_seq: list[int]) -> list[int]:
        return [int(item_id) + RPG_ITEM_ID_OFFSET for item_id in item_seq]

    def _build_item_id2tokens(self, prepared_data: BaseTaskDataset) -> torch.Tensor:
        item2semantic_ids = self._load_or_generate_semantic_ids(prepared_data)
        item_id2tokens = torch.zeros((self.num_items + RPG_ITEM_ID_OFFSET, self.n_digit), dtype=torch.long)
        for recbole_item_id in range(self.num_items):
            semantic_ids = item2semantic_ids[recbole_item_id]
            item_id2tokens[recbole_item_id + RPG_ITEM_ID_OFFSET] = torch.tensor(
                self._semantic_ids_to_token_ids(semantic_ids),
                dtype=torch.long,
            )
        return item_id2tokens

    def _load_or_generate_semantic_ids(self, prepared_data: BaseTaskDataset) -> dict[int, tuple[int, ...]]:
        semantic_id_path = self._semantic_id_path(prepared_data)
        if semantic_id_path.exists():
            return self._load_semantic_ids(semantic_id_path)

        semantic_id_path.parent.mkdir(parents=True, exist_ok=True)
        generated = self._generate_semantic_ids(prepared_data)
        with semantic_id_path.open("w", encoding="utf-8") as f:
            json.dump({str(k): list(v) for k, v in generated.items()}, f)
        return generated

    def _semantic_id_path(self, prepared_data: BaseTaskDataset) -> Path:
        if self.config.semantic_id_file:
            return Path(self.config.semantic_id_file).expanduser()
        cache_file_name = self.config.semantic_id_cache_file or self._default_cache_file_name()
        return self._cache_base_dir(prepared_data) / cache_file_name

    def _default_cache_file_name(self) -> str:
        if self.config.semantic_embedding_file:
            embedding_token = Path(self.config.semantic_embedding_file).expanduser().stem or "embeddings"
        else:
            embedding_token = os.path.basename(str(self.config.sent_emb_model)) or "rpg"
        bits = self._get_codebook_bits(self.codebook_size)
        index_factory = f"OPQ{self.n_digit},IVF1,PQ{self.n_digit}x{bits}"
        pca_dim = int(self.config.sent_emb_pca)
        pca_token = f"_pca{pca_dim}" if pca_dim > 0 else ""
        return f"{embedding_token}_{index_factory}{pca_token}.semantic_ids.json"

    def _cache_base_dir(self, prepared_data: BaseTaskDataset) -> Path:
        if self.config.cache_dir:
            return Path(self.config.cache_dir).expanduser()
        return Path(prepared_data._parser.data_dir).expanduser() / "rpg"

    def _load_semantic_ids(self, path: Path) -> dict[int, tuple[int, ...]]:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        values = {int(key): value for key, value in raw.items()}
        return {item_id: tuple(int(value) for value in values[item_id]) for item_id in values}

    def _semantic_ids_to_token_ids(self, semantic_ids: tuple[int, ...]) -> tuple[int, ...]:
        token_ids: list[int] = []
        for digit, value in enumerate(semantic_ids):
            value = int(value)
            token_ids.append(value + digit * self.codebook_size + 1)
        return tuple(token_ids)

    def _generate_semantic_ids(self, prepared_data: BaseTaskDataset) -> dict[int, tuple[int, ...]]:
        embeddings = self._load_or_generate_semantic_embeddings(prepared_data)
        if self.config.sent_emb_pca > 0:
            pca = PCA(n_components=int(self.config.sent_emb_pca), whiten=True)
            embeddings = pca.fit_transform(embeddings)
        train_mask = self._get_training_item_mask(prepared_data)
        return self._generate_semantic_id_opq(embeddings, train_mask)

    def _load_or_generate_semantic_embeddings(self, prepared_data: BaseTaskDataset) -> np.ndarray:
        if self.config.semantic_embedding_file:
            path = Path(self.config.semantic_embedding_file).expanduser()
            if path.suffix == ".npy":
                return np.load(path)
            return np.fromfile(path, dtype=np.float32).reshape(-1, int(self.config.sent_emb_dim))

        texts = self._item_metadata_texts(prepared_data.get_item_table())
        if "text-embedding-3" in self.config.sent_emb_model:
            return self._encode_with_openai(texts)
        return self._encode_with_sentence_transformers(texts)

    def _item_metadata_texts(self, item_table: pd.DataFrame) -> list[str]:
        texts = [""] * self.num_items
        for row in item_table.to_dict("records"):
            item_id = int(row[ITEM_ID])
            text = row.get(self.config.metadata_text_field)
            texts[item_id] = str(text)
        return texts

    def _encode_with_sentence_transformers(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(self.config.sent_emb_model).to(device)
        return model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=int(self.config.sent_emb_batch_size),
            show_progress_bar=True,
            device=device,
        )

    def _encode_with_openai(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)
        embeddings: list[list[float]] = []
        batch_size = int(self.config.sent_emb_batch_size)
        for start in range(0, len(texts), batch_size):
            response = client.embeddings.create(
                input=texts[start : start + batch_size],
                model=self.config.sent_emb_model,
            )
            embeddings.extend(item.embedding for item in response.data)
        return np.asarray(embeddings, dtype=np.float32)

    def _get_training_item_mask(self, prepared_data: BaseTaskDataset) -> np.ndarray:
        train_items: set[int] = set()
        train_dataset = prepared_data.get_train_dataset()

        train_items.update(int(item_id) for item_id in train_dataset.frame[ITEM_ID].tolist())
        mask = np.zeros(self.num_items, dtype=bool)
        for item_id in train_items:
            mask[item_id] = True
        return mask

    def _generate_semantic_id_opq(self, embeddings: np.ndarray, train_mask: np.ndarray) -> dict[int, tuple[int, ...]]:
        import faiss

        n_codebook_bits = self._get_codebook_bits(self.codebook_size)
        index_factory = f"OPQ{self.n_digit},IVF1,PQ{self.n_digit}x{n_codebook_bits}"
        faiss.omp_set_num_threads(int(self.config.faiss_omp_num_threads))
        index = faiss.index_factory(int(embeddings.shape[1]), index_factory, faiss.METRIC_INNER_PRODUCT)

        if self.config.opq_use_gpu:
            resources = faiss.StandardGpuResources()
            resources.setTempMemory(1024 * 1024 * 512)
            clone_options = faiss.GpuClonerOptions()
            clone_options.useFloat16 = self.n_digit >= 56
            index = faiss.index_cpu_to_gpu(resources, int(self.config.opq_gpu_id), index, clone_options)

        index.train(np.asarray(embeddings[train_mask], dtype=np.float32))
        index.add(np.asarray(embeddings, dtype=np.float32))
        if self.config.opq_use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        list_size = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), list_size * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        item_ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size)

        item2semantic_ids: dict[int, tuple[int, ...]] = {}
        for item_id, u8code in zip(item_ids.tolist(), pq_codes, strict=True):
            reader = faiss.BitstringReader(faiss.swig_ptr(u8code), pq_codes.shape[1])
            code = tuple(int(reader.read(n_codebook_bits)) for _ in range(self.n_digit))
            item2semantic_ids[int(item_id)] = code
        return item2semantic_ids

    @staticmethod
    def _get_codebook_bits(codebook_size: int) -> int:
        bits = math.log2(int(codebook_size))
        return int(bits)


__all__ = [
    "RPG_IGNORED_LABEL",
    "RPG_ITEM_ID_OFFSET",
    "RPG_PADDING_ITEM_ID",
    "RPGSemanticTokenizer",
    "ensure_rpg_config",
]
