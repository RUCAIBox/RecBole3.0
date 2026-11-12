"""BIGRec embedding-grounding utilities.

This module owns BIGRec's second step: turn generated text into an oracle
embedding, rank actual items by L2 distance, and optionally inject statistical
signals into the distances.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from recbole3.dataset.utils import ITEM_ID
from recbole3.evaluation.metric import NDCGMetric, RecallMetric, RetrievalEvalData
from recbole3.model.bigrec.config import BIGRecConfig
from recbole3.model.bigrec.data import batchify


MetricFn = Callable[[RetrievalEvalData], dict[str, float]]
ExtractEmbeddingsFn = Callable[[Any, AutoTokenizer, list[str], int, torch.device], torch.Tensor]


class BIGRecGrounder:
    """Embedding extraction, L2 ranking, and grounding weight injection."""

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

    def extract_embeddings(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        texts: list[str],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode text strings as last-token hidden states from the final layer."""
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        base_transformer = getattr(model, "model", model)

        all_embs: list[torch.Tensor] = []
        for batch_texts in batchify(texts, batch_size):
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_length,
            ).to(device)

            with torch.no_grad():
                outputs = base_transformer(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                )

            last_layer: torch.Tensor = outputs.last_hidden_state
            all_embs.append(last_layer[:, -1, :].float().cpu())

        tokenizer.padding_side = orig_padding_side
        return torch.cat(all_embs, dim=0)

    def precompute_item_embeddings(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        item_texts: list[str],
        cache_path: str,
        device: torch.device,
        *,
        extract_embeddings: ExtractEmbeddingsFn | None = None,
    ) -> torch.Tensor:
        """Return item embeddings, loading from disk cache when possible."""
        if not self.config.refresh_embedding_cache and os.path.isfile(cache_path):
            self._log("Loading item embeddings from cache: %s", cache_path)
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        self._log("Pre-computing embeddings for %d items.", len(item_texts))
        extract = extract_embeddings or self.extract_embeddings
        embeddings = extract(
            model,
            tokenizer,
            item_texts,
            self.config.embedding_batch_size,
            device,
        )

        if self._is_main_process():
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(embeddings, cache_path)
            self._log("Item embeddings saved to %s", cache_path)

        return embeddings

    def compute_popularity_weights(self, task_data: Any, num_items: int) -> torch.Tensor:
        """Compute min-max normalised item popularity from training interactions."""
        train_frame: pd.DataFrame = task_data.get_train_dataset().frame  # type: ignore[attr-defined]
        counts = torch.zeros(num_items, dtype=torch.float32)

        for item_id, n in train_frame[ITEM_ID].value_counts().items():
            idx = int(item_id)
            if 0 <= idx < num_items:
                counts[idx] = float(n)

        total = counts.sum()
        if total <= 0.0:
            self._log("Popularity: all interaction counts are zero; weights default to 0.", level="warning")
            return counts

        ci = counts / total
        c_min, c_max = ci.min(), ci.max()
        pi = (ci - c_min) / (c_max - c_min) if c_max > c_min else torch.zeros_like(ci)

        self._log(
            "Popularity weights: mean=%.4f, max=%.4f, min=%.4f",
            pi.mean().item(),
            pi.max().item(),
            pi.min().item(),
        )
        return pi

    def load_cf_weights(self, num_items: int) -> torch.Tensor:
        """Load and min-max normalise pre-computed CF scores."""
        path = self.config.cf_score_path
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"CF score file not found: '{path}'. "
                "Set BIGRecConfig.cf_score_path to a .pt file with shape [num_items]."
            )

        scores: torch.Tensor = torch.load(path, weights_only=True, map_location="cpu").float()
        if scores.ndim != 1 or scores.shape[0] != num_items:
            raise ValueError(
                f"CF score tensor shape {tuple(scores.shape)} does not match "
                f"num_items={num_items}. Expected a 1-D tensor of length {num_items}."
            )

        s_min, s_max = scores.min(), scores.max()
        scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else torch.zeros_like(scores)

        self._log(
            "CF weights: mean=%.4f, max=%.4f, min=%.4f",
            scores.mean().item(),
            scores.max().item(),
            scores.min().item(),
        )
        return scores

    def build_grounding_weights(self, task_data: Any, num_items: int) -> torch.Tensor | None:
        """Build the optional grounding weight vector for Eq. 3."""
        mode = self.config.grounding_mode.strip().lower()
        if mode == "none":
            return None

        weights = torch.zeros(num_items, dtype=torch.float32)
        if "popularity" in mode:
            weights = weights + self.compute_popularity_weights(task_data, num_items)
        if "cf" in mode:
            weights = weights + self.load_cf_weights(num_items)

        if "popularity" in mode and "cf" in mode:
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                weights = (weights - w_min) / (w_max - w_min)
            self._log("Combined popularity+CF weights built (re-normalised).")

        return weights

    @staticmethod
    def apply_grounding_weights(
        dist: torch.Tensor,
        weights: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Apply BIGRec Eq. 3 distance reweighting."""
        dist_min = dist.min(dim=1, keepdim=True)[0]
        dist_max = dist.max(dim=1, keepdim=True)[0]
        dist_hat = (dist - dist_min) / (dist_max - dist_min + 1e-8)
        multiplier = torch.pow(1.0 + weights.unsqueeze(0), -gamma)
        return dist_hat * multiplier

    @staticmethod
    def default_gamma_search_values() -> tuple[float, ...]:
        """Return the official BIGRec 199-value gamma grid."""
        fine = tuple(round(i * 0.01, 2) for i in range(100))
        coarse = tuple(float(i) for i in range(1, 100))
        return fine + coarse

    @staticmethod
    def _batch_topk_full(dist: torch.Tensor, k: int) -> np.ndarray:
        """Return smallest-distance item ids for every row without full sorting."""
        actual_k = min(k, dist.shape[1])
        top_values, top_indices = torch.topk(dist, k=actual_k, dim=1, largest=False)
        top_indices = top_indices.masked_fill(~torch.isfinite(top_values), -1)
        top_k = top_indices.cpu().numpy()
        if actual_k < k:
            padding = np.full((top_k.shape[0], k - actual_k), -1, dtype=np.int64)
            top_k = np.concatenate([top_k, padding], axis=1)
        return top_k.astype(np.int64, copy=False)

    def mask_history_items(
        self,
        dist: torch.Tensor,
        excluded_item_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        """Set seen-item distances to infinity for full-ranking evaluation."""
        if (
            not self.config.exclude_history
            or self.config.eval_protocol != "full"
            or excluded_item_ids is None
        ):
            return dist
        if len(excluded_item_ids) != dist.shape[0]:
            raise ValueError(
                "BIGRec history exclusion row count does not match distance matrix: "
                f"{len(excluded_item_ids)} != {dist.shape[0]}."
            )

        row_indices: list[int] = []
        item_indices: list[int] = []
        num_items = int(dist.shape[1])
        for row_index, item_ids in enumerate(excluded_item_ids):
            for item_id in item_ids:
                normalized_item_id = int(item_id)
                if 0 <= normalized_item_id < num_items:
                    row_indices.append(row_index)
                    item_indices.append(normalized_item_id)
        if not row_indices:
            return dist

        masked = dist.clone()
        masked[
            torch.tensor(row_indices, dtype=torch.long, device=dist.device),
            torch.tensor(item_indices, dtype=torch.long, device=dist.device),
        ] = float("inf")
        return masked

    @staticmethod
    def _sampled_topk_rows(
        dist: torch.Tensor,
        cand_lists: list[list[int] | None],
        k: int,
        device: torch.device,
    ) -> np.ndarray:
        """Return top-k item ids when each row may have a different candidate set."""
        rows: list[np.ndarray] = []
        num_items = dist.shape[1]

        for i, cand in enumerate(cand_lists):
            if cand is None:
                actual_k = min(k, num_items)
                top_k = torch.topk(dist[i], k=actual_k, largest=False).indices.cpu().numpy()
            else:
                cand_t = torch.tensor(cand, dtype=torch.long, device=device)
                actual_k = min(k, cand_t.numel())
                if actual_k == 0:
                    top_k = np.empty(0, dtype=np.int64)
                else:
                    cand_dists = dist[i, cand_t]
                    top_idx = torch.topk(cand_dists, k=actual_k, largest=False).indices
                    top_k = cand_t[top_idx].cpu().numpy()

            if len(top_k) < k:
                top_k = np.concatenate([top_k, np.full(k - len(top_k), -1, dtype=np.int64)])
            rows.append(top_k.reshape(1, k))

        return np.concatenate(rows, axis=0)

    def run_gamma_search(
        self,
        dist: torch.Tensor,
        grounding_weights: torch.Tensor,
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        device: torch.device,
        gamma_values: tuple[float, ...],
        excluded_item_ids: list[list[int]] | None = None,
        *,
        compute_metrics: MetricFn | None = None,
    ) -> dict[str, float]:
        """Grid-search the best gamma per metric@K on a validation split."""
        maxk = max(self.config.eval_topk)
        is_sampled = self.config.eval_protocol == "sampled"
        n = len(target_ids)
        target_arr = np.array(target_ids, dtype=np.int64).reshape(n, 1)
        mask_arr = np.ones((n, 1), dtype=bool)
        metric_fn = compute_metrics or self.compute_metrics

        metric_keys = [
            f"{m.lower()}@{k}"
            for m in self.config.eval_metrics
            for k in self.config.eval_topk
        ]
        best_scores = {key: -1.0 for key in metric_keys}
        best_gammas = {key: gamma_values[0] for key in metric_keys}

        for gamma in tqdm(gamma_values, desc="Gamma search", disable=not self._is_main_process()):
            eff_dist = self.apply_grounding_weights(dist, grounding_weights, gamma)
            eff_dist = self.mask_history_items(eff_dist, excluded_item_ids)
            pred_item_ids = (
                self._sampled_topk_rows(eff_dist, cand_lists, maxk, device)
                if is_sampled
                else self._batch_topk_full(eff_dist, maxk)
            )

            scores = metric_fn(
                RetrievalEvalData(
                    pred_item_ids=pred_item_ids,
                    target_item_ids=target_arr,
                    target_mask=mask_arr,
                )
            )
            for key, val in scores.items():
                if val > best_scores.get(key, -1.0):
                    best_scores[key] = val
                    best_gammas[key] = gamma

        self._log("Gamma search complete: best gamma per metric@K:")
        for key in metric_keys:
            self._log("  %s -> gamma=%.3f  (val=%.4f)", key, best_gammas[key], best_scores[key])
        return best_gammas

    def evaluate_from_dist_per_k_gammas(
        self,
        dist: torch.Tensor,
        grounding_weights: torch.Tensor | None,
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        best_gammas: dict[str, float],
        device: torch.device,
        excluded_item_ids: list[list[int]] | None = None,
    ) -> dict[str, float]:
        """Evaluate with independently selected gamma values per metric@K."""
        maxk = max(self.config.eval_topk)
        is_sampled = self.config.eval_protocol == "sampled"
        n = len(target_ids)
        target_arr = np.array(target_ids, dtype=np.int64).reshape(n, 1)
        mask_arr = np.ones((n, 1), dtype=bool)
        results: dict[str, float] = {}

        for metric_name in self.config.eval_metrics:
            name = metric_name.strip().lower()
            if name not in ("recall", "ndcg"):
                self._log("Unknown eval metric '%s'; skipping.", metric_name, level="warning")
                continue
            for k in self.config.eval_topk:
                key = f"{name}@{k}"
                gamma = best_gammas.get(key, self.config.grounding_gamma)
                eff_dist = (
                    self.apply_grounding_weights(dist, grounding_weights, gamma)
                    if grounding_weights is not None
                    else dist
                )
                eff_dist = self.mask_history_items(eff_dist, excluded_item_ids)

                pred_item_ids = (
                    self._sampled_topk_rows(eff_dist, cand_lists, maxk, device)
                    if is_sampled
                    else self._batch_topk_full(eff_dist, maxk)
                )

                eval_data = RetrievalEvalData(
                    pred_item_ids=pred_item_ids,
                    target_item_ids=target_arr,
                    target_mask=mask_arr,
                )
                scores = RecallMetric((k,)).compute(eval_data) if name == "recall" else NDCGMetric((k,)).compute(eval_data)
                if key in scores:
                    results[key] = scores[key]
                    self._log("  %s (gamma=%.3f) = %.4f", key, gamma, scores[key])
        return results


    def rank_from_texts(
        self,
        emb_model: Any,
        tokenizer: AutoTokenizer,
        item_emb_device: torch.Tensor,
        generated_texts: list[str],
        target_ids: list[int],
        cand_lists: list[list[int] | None],
        grounding_weights: torch.Tensor | None,
        device: torch.device,
        excluded_item_ids: list[list[int]] | None = None,
        *,
        extract_embeddings: ExtractEmbeddingsFn | None = None,
        compute_metrics: MetricFn | None = None,
    ) -> dict[str, float]:
        """Extract oracle embeddings, rank items by L2 distance, and score metrics."""
        weights_device = grounding_weights.to(device) if grounding_weights is not None else None
        maxk = max(self.config.eval_topk)
        is_sampled = self.config.eval_protocol == "sampled"
        batch_size = self.config.embedding_batch_size
        n = len(generated_texts)
        extract = extract_embeddings or self.extract_embeddings
        metric_fn = compute_metrics or self.compute_metrics

        all_pred_item_ids: list[np.ndarray] = []
        all_target_item_ids: list[np.ndarray] = []
        all_target_masks: list[np.ndarray] = []

        pbar = tqdm(
            range(0, n, batch_size),
            desc="BIGRec oracle embed+rank",
            disable=not self._is_main_process(),
        )

        for start in pbar:
            batch_texts = generated_texts[start : start + batch_size]
            batch_target_ids = target_ids[start : start + batch_size]
            batch_cand_lists = cand_lists[start : start + batch_size]
            actual_bs = len(batch_texts)
            oracle_embs = extract(emb_model, tokenizer, batch_texts, actual_bs, device)
            distances = torch.cdist(oracle_embs.to(device), item_emb_device, p=2.0)
            effective_dist = (
                self.apply_grounding_weights(distances, weights_device, self.config.grounding_gamma)
                if weights_device is not None
                else distances
            )
            batch_excluded_item_ids = (
                excluded_item_ids[start : start + actual_bs]
                if excluded_item_ids is not None
                else None
            )
            effective_dist = self.mask_history_items(
                effective_dist,
                batch_excluded_item_ids,
            )

            batch_pred_item_ids = (
                self._sampled_topk_rows(effective_dist, batch_cand_lists, maxk, device)
                if is_sampled
                else self._batch_topk_full(effective_dist, maxk)
            )

            for i in range(actual_bs):
                target_id = batch_target_ids[i]
                top_k_ids = batch_pred_item_ids[i]
                all_pred_item_ids.append(top_k_ids.reshape(1, maxk))
                all_target_item_ids.append(np.array([[target_id]], dtype=np.int64))
                all_target_masks.append(np.array([[True]], dtype=bool))

        return metric_fn(
            RetrievalEvalData(
                pred_item_ids=np.concatenate(all_pred_item_ids, axis=0),
                target_item_ids=np.concatenate(all_target_item_ids, axis=0),
                target_mask=np.concatenate(all_target_masks, axis=0),
            )
        )

    def compute_metrics(self, eval_data: RetrievalEvalData) -> dict[str, float]:
        """Compute configured BIGRec retrieval metrics."""
        results: dict[str, float] = {}
        ks = tuple(self.config.eval_topk)
        for metric_name in self.config.eval_metrics:
            name = metric_name.strip().lower()
            if name == "recall":
                scores = RecallMetric(ks).compute(eval_data)
            elif name == "ndcg":
                scores = NDCGMetric(ks).compute(eval_data)
            else:
                self._log("Unknown eval metric '%s'; skipping.", metric_name, level="warning")
                continue
            results.update(scores)

        for key, val in results.items():
            self._log("  %s = %.4f", key, val)
        return results


__all__ = ["BIGRecGrounder"]
