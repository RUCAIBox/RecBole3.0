from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Sequence

import numpy as np
import torch

from recbole3.dataset.base import BaseTaskDataset
from recbole3.evaluation.metric import (
    BaseRankingMetric,
    BaseRetrievalMetric,
    EvalProtocol,
    MetricSpec,
    RankingEvalData,
    RetrievalEvalData,
    create_builtin_metrics,
)
from recbole3.model import BaseModel


class BaseEvaluationMethod(ABC):
    protocol: EvalProtocol

    def __init__(self, *, metric_specs: Sequence[MetricSpec]) -> None:
        self._metrics = tuple(create_builtin_metrics(tuple(metric_specs), self.protocol))

    @abstractmethod
    def collect_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Sequence[Any],
    ) -> RankingEvalData | RetrievalEvalData:
        ...

    @abstractmethod
    def compute_metrics(
        self,
        batch_eval_data: Sequence[RankingEvalData | RetrievalEvalData],
    ) -> dict[str, float]:
        ...

    def metric_directions(self) -> dict[str, bool]:
        directions: dict[str, bool] = {}
        for metric in self._metrics:
            directions.update(metric.result_directions())
        return directions

    def build_eval_collate_fn(
        self,
        model: BaseModel,
        prepared_data: BaseTaskDataset[Any, Any],
    ) -> Callable[[Sequence[Any]], tuple[Any, list[Any]]]:
        input_collator = model.build_eval_collator(prepared_data)

        def collate_fn(samples: Sequence[Any]) -> tuple[Any, list[Any]]:
            eval_records = list(samples)
            return input_collator(eval_records), eval_records

        return collate_fn

    @staticmethod
    def _record_value(record: Any, key: str) -> Any:
        if isinstance(record, Mapping):
            return record[key]
        return getattr(record, key)

    @classmethod
    def _record_list(cls, record: Any, key: str) -> Sequence[Any]:
        values = cls._record_value(record, key)
        if values is None:
            return ()
        return values

    @classmethod
    def _tensor_1d(
        cls,
        records: Sequence[Any],
        key: str,
        dtype: torch.dtype,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.tensor([cls._record_value(record, key) for record in records], dtype=dtype, device=device)

    @classmethod
    def _pad_int_lists(
        cls,
        records: Sequence[Any],
        key: str,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [cls._record_list(record, key) for record in records]
        width = max((len(row) for row in rows), default=0)
        values = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        for row_index, row in enumerate(rows):
            row_width = len(row)
            if row_width == 0:
                continue
            row_tensor = torch.as_tensor(row, dtype=torch.long, device=device).reshape(-1)
            values[row_index, :row_width] = row_tensor
            mask[row_index, :row_width] = True
        return values, mask

    @classmethod
    def _single_target_tensors(
        cls,
        records: Sequence[Any],
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_item_ids = cls._tensor_1d(records, "item_id", torch.long, device=device).reshape(-1, 1)
        target_mask = torch.ones(target_item_ids.shape, dtype=torch.bool, device=device)
        return target_item_ids, target_mask

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> Any:
        return value.detach().cpu().numpy()

    @staticmethod
    def _infer_device(value: Any) -> torch.device:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, Mapping):
            for item in value.values():
                device = BaseEvaluationMethod._try_infer_device(item)
                if device is not None:
                    return device
        if isinstance(value, (list, tuple)):
            for item in value:
                device = BaseEvaluationMethod._try_infer_device(item)
                if device is not None:
                    return device
        return torch.device("cpu")

    @staticmethod
    def _try_infer_device(value: Any) -> torch.device | None:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, Mapping):
            for item in value.values():
                device = BaseEvaluationMethod._try_infer_device(item)
                if device is not None:
                    return device
        if isinstance(value, (list, tuple)):
            for item in value:
                device = BaseEvaluationMethod._try_infer_device(item)
                if device is not None:
                    return device
        return None

    @staticmethod
    def _concat_1d(values: Sequence[np.ndarray], *, dtype: Any) -> np.ndarray:
        if not values:
            return np.empty(0, dtype=dtype)
        arrays = [np.asarray(value, dtype=dtype).reshape(-1) for value in values]
        total_size = sum(array.size for array in arrays)
        result = np.empty(total_size, dtype=dtype)
        offset = 0
        for array in arrays:
            next_offset = offset + array.size
            result[offset:next_offset] = array
            offset = next_offset
        return result

    @staticmethod
    def _concat_padded_2d(values: Sequence[np.ndarray], *, fill_value: Any, dtype: Any) -> np.ndarray:
        if not values:
            return np.empty((0, 0), dtype=dtype)
        arrays = [np.asarray(value, dtype=dtype) for value in values]
        width = max((array.shape[1] if array.ndim == 2 else 0) for array in arrays)
        total_rows = 0
        for array in arrays:
            if array.ndim != 2:
                raise ValueError(f"Expected one 2D batch array, got shape {array.shape}.")
            total_rows += int(array.shape[0])
        result = np.full((total_rows, width), fill_value, dtype=dtype)
        offset = 0
        for array in arrays:
            row_count = int(array.shape[0])
            next_offset = offset + row_count
            result[offset:next_offset, : array.shape[1]] = array
            offset = next_offset
        return result


class BaseRankingEvaluationMethod(BaseEvaluationMethod):
    def compute_metrics(
        self,
        batch_eval_data: Sequence[RankingEvalData | RetrievalEvalData],
    ) -> dict[str, float]:
        ranking_metrics = self._require_ranking_metrics()
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        group_ids: list[np.ndarray] = []
        for batch_data in batch_eval_data:
            if not isinstance(batch_data, RankingEvalData):
                raise TypeError("Expected ranking evaluation batches.")
            scores.append(np.asarray(batch_data.scores))
            labels.append(np.asarray(batch_data.labels))
            group_ids.append(np.asarray(batch_data.group_ids))

        eval_data = RankingEvalData(
            scores=self._concat_1d(scores, dtype=np.float64),
            labels=self._concat_1d(labels, dtype=np.float64),
            group_ids=self._concat_1d(group_ids, dtype=np.int64),
        )
        results: dict[str, float] = {}
        for metric in ranking_metrics:
            results.update(metric.compute(eval_data))
        return results

    def _require_ranking_metrics(self) -> list[BaseRankingMetric]:
        ranking_metrics: list[BaseRankingMetric] = []
        for metric in self._metrics:
            if not isinstance(metric, BaseRankingMetric):
                raise TypeError(f"Metric '{metric.name}' is incompatible with labeled evaluation data.")
            ranking_metrics.append(metric)
        return ranking_metrics


class BaseRetrievalEvaluationMethod(BaseEvaluationMethod):
    def collect_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Sequence[Any],
    ) -> RetrievalEvalData:
        return self._collect_retrieval_batch(
            model=model,
            model_inputs=model_inputs,
            records=records,
            max_k=self._required_max_k(),
        )

    @abstractmethod
    def _collect_retrieval_batch(
        self,
        model: BaseModel,
        model_inputs: Any,
        records: Sequence[Any],
        max_k: int,
    ) -> RetrievalEvalData:
        ...

    def compute_metrics(
        self,
        batch_eval_data: Sequence[RankingEvalData | RetrievalEvalData],
    ) -> dict[str, float]:
        retrieval_metrics = self._require_retrieval_metrics()
        pred_item_ids: list[np.ndarray] = []
        target_item_ids: list[np.ndarray] = []
        target_mask: list[np.ndarray] = []
        for batch_data in batch_eval_data:
            if not isinstance(batch_data, RetrievalEvalData):
                raise TypeError("Expected retrieval evaluation batches.")
            pred_item_ids.append(np.asarray(batch_data.pred_item_ids))
            target_item_ids.append(np.asarray(batch_data.target_item_ids))
            target_mask.append(np.asarray(batch_data.target_mask))

        eval_data = RetrievalEvalData(
            pred_item_ids=self._concat_padded_2d(pred_item_ids, fill_value=-1, dtype=np.int64),
            target_item_ids=self._concat_padded_2d(target_item_ids, fill_value=0, dtype=np.int64),
            target_mask=self._concat_padded_2d(target_mask, fill_value=False, dtype=bool),
        )
        results: dict[str, float] = {}
        for metric in retrieval_metrics:
            results.update(metric.compute(eval_data))
        return results

    def _required_max_k(self) -> int:
        retrieval_metrics = self._require_retrieval_metrics()
        return max((max(metric.ks) for metric in retrieval_metrics), default=0)

    def _require_retrieval_metrics(self) -> list[BaseRetrievalMetric]:
        retrieval_metrics: list[BaseRetrievalMetric] = []
        for metric in self._metrics:
            if not isinstance(metric, BaseRetrievalMetric):
                raise TypeError(f"Metric '{metric.name}' is incompatible with retrieval evaluation data.")
            retrieval_metrics.append(metric)
        return retrieval_metrics


