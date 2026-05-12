"""ReaRec model-side dataset and collators.

SASRec backbone
---------------
  BaseSequentialModelDataset builds ``history_item_ids`` (tuple of raw item IDs,
  chronological) for every split. The collators then LEFT-PAD these histories to
  ``history_max_length``.  Left-padding convention matches the official ReaRec
  implementation::

      [PADDING_ID, ..., PADDING_ID, item_{t-N}, ..., item_{t-1}]

  where PADDING_ID = num_items (one beyond the valid range 0..num_items-1).

HSTU backbone
-------------
  Delegates dataset construction to ``HSTUModelDataset`` which additionally
  builds ``history_timestamps`` for every split (timestamps are required by
  HSTU's relative time-bucketed attention bias).  The collators produce
  RIGHT-PADDED sequences of length ``history_max_length + 1``::

      [item_0, ..., item_{L-1}, 0, ..., 0, 0]   (0 = HSTU_PADDING_ITEM_ID)
       ← L real history slots →               ← 1 virtual query-timestamp slot

  The extra slot at position ``history_lengths[b]`` carries the target item's
  timestamp (but NOT the target item ID).  This mirrors the standalone
  ``HSTUEvalCollator`` behaviour and is required so that
  ``HSTUModel._encode_sequence_embeddings`` computes correct relative-time
  attention bias for the virtual "query" position.

  The target item is kept as a separate ``ITEM_ID`` key and is NOT appended to
  the history sequence (unlike ``HSTUTrainCollator``).  ReaRec uses CE loss
  against the target item independently, so the target must stay out of the
  history that is encoded by the backbone.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from recbole3.dataset import FrameDataset, ITEM_ID, TIMESTAMP
from recbole3.model.base import BaseCollator, ModelConfig
from recbole3.model.hstu.config import HSTU_PADDING_ITEM_ID
from recbole3.model.hstu.data import HISTORY_TIMESTAMPS, HistoryState, HSTUModelDataset
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS


class ReaRecModelDataset(HSTUModelDataset, BaseSequentialModelDataset):
    """ReaRec model-side dataset supporting SASRec and HSTU backbones.

    Inherits from both ``HSTUModelDataset`` and ``BaseSequentialModelDataset``
    (diamond inheritance via the shared ``BaseModelDataset`` base) so that all
    internal helpers — ``_build_sequential_frame`` for the SASRec path and
    ``_build_hstu_train_frame`` / ``_build_hstu_frame`` for the HSTU path — are
    available on ``self`` regardless of which branch is taken.

    * backbone='sasrec': delegates to ``BaseSequentialModelDataset`` which adds
      only ``history_item_ids`` (no timestamps required).
    * backbone='hstu':  delegates to ``HSTUModelDataset`` which additionally
      adds ``history_timestamps``.
    """

    def _build_model_datasets(self, *, model_config: ModelConfig):  # type: ignore[override]
        backbone = str(getattr(model_config, "backbone", "sasrec")).lower()
        if backbone == "hstu":
            # Full HSTU data prep: history_item_ids + history_timestamps
            return HSTUModelDataset._build_model_datasets(self, model_config=model_config)
        # SASRec / default: history_item_ids only (no timestamps)
        return BaseSequentialModelDataset._build_model_datasets(
            self, model_config=model_config  # type: ignore[arg-type]
        )

    def _build_hstu_train_frame(
        self,
        records: pd.DataFrame,
        *,
        history_max_length: int,
    ) -> tuple[pd.DataFrame, HistoryState]:
        """Use prefix splitting for HSTU training: one record per interaction, not one per user.

        Standalone HSTU compensates for its one-per-user scheme with all-position loss
        (O(T) signals per record). ReaRec applies last-position CE only, so that scheme
        yields O(1) signals per user per epoch. Prefix splitting restores O(T) signals,
        matching the SASRec backbone's training density.
        """
        return self._build_hstu_frame(records, history_max_length=history_max_length)


# ---------------------------------------------------------------------------
# SASRec collators (left-padded, no timestamps)
# ---------------------------------------------------------------------------

class ReaRecTrainCollator(BaseCollator):
    """Collate training records into left-padded history tensors + target item IDs.

    Args:
        config: Model configuration (ModelConfig subclass).
        prepared_data: Prepared task dataset (used by base class only).
        num_items: Total number of items; padding token = num_items.
        history_max_length: Fixed padded sequence length L.
    """

    def __init__(
        self,
        config: ModelConfig,
        prepared_data: Any,
        *,
        num_items: int,
        history_max_length: int,
    ) -> None:
        super().__init__(config, prepared_data)
        self._padding_id = int(num_items)
        self._history_max_length = int(history_max_length)

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        batch = _build_rearec_history_batch(
            feature_records,
            padding_id=self._padding_id,
            history_max_length=self._history_max_length,
        )
        batch[ITEM_ID] = torch.as_tensor(
            feature_records[ITEM_ID].to_numpy(), dtype=torch.long
        )
        return batch


class ReaRecEvalCollator(BaseCollator):
    """Collate evaluation records into left-padded history tensors (no target ID)."""

    def __init__(
        self,
        config: ModelConfig,
        prepared_data: Any,
        *,
        num_items: int,
        history_max_length: int,
    ) -> None:
        super().__init__(config, prepared_data)
        self._padding_id = int(num_items)
        self._history_max_length = int(history_max_length)

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _build_rearec_history_batch(
            feature_records,
            padding_id=self._padding_id,
            history_max_length=self._history_max_length,
        )


def _build_rearec_history_batch(
    records: pd.DataFrame,
    *,
    padding_id: int,
    history_max_length: int,
) -> dict[str, torch.Tensor]:
    """Build a left-padded history batch from a DataFrame of model records.

    Returns:
        history_item_ids: [B, history_max_length] left-padded with padding_id.
        history_lengths:  [B] actual (non-padded) history length per sample.
    """
    histories: list[tuple[int, ...]] = [
        tuple(int(x) for x in v) for v in records[HISTORY_ITEM_IDS].tolist()
    ]
    B = len(records)
    history_lengths = torch.tensor(
        [min(len(h), history_max_length) for h in histories], dtype=torch.long
    )

    history_item_ids = torch.full((B, history_max_length), padding_id, dtype=torch.long)
    for i, hist in enumerate(histories):
        n = min(len(hist), history_max_length)
        if n > 0:
            items = hist[-history_max_length:]
            history_item_ids[i, history_max_length - n:] = torch.tensor(
                items, dtype=torch.long
            )

    return {
        HISTORY_ITEM_IDS: history_item_ids,   # [B, L]
        "history_lengths": history_lengths,    # [B]
    }


# ---------------------------------------------------------------------------
# HSTU collators (right-padded, with timestamps, target NOT in sequence)
# ---------------------------------------------------------------------------

class ReaRecHSTUTrainCollator(BaseCollator):
    """Train collator for HSTU backbone: right-padded history + timestamps.

    Produces fixed-length sequences of length ``history_max_length``:
      - items at positions 0..history_lengths[b]-1  (right-padding with 0)
      - ``history_timestamps`` aligned with item positions
      - ``ITEM_ID`` as separate target key (NOT included in history sequence)

    The last distinction is what separates this collator from
    ``HSTUTrainCollator``, which appends the target item to the sequence for
    autoregressive training.  ReaRec needs the target *outside* the sequence.
    """

    def __init__(
        self,
        config: ModelConfig,
        prepared_data: Any,
        *,
        history_max_length: int,
    ) -> None:
        super().__init__(config, prepared_data)
        self._history_max_length = int(history_max_length)

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        batch = _build_hstu_rearec_history_batch(feature_records, self._history_max_length)
        batch[ITEM_ID] = torch.as_tensor(
            feature_records[ITEM_ID].to_numpy(), dtype=torch.long
        )
        return batch


class ReaRecHSTUEvalCollator(BaseCollator):
    """Eval collator for HSTU backbone: right-padded history + timestamps, no target."""

    def __init__(
        self,
        config: ModelConfig,
        prepared_data: Any,
        *,
        history_max_length: int,
    ) -> None:
        super().__init__(config, prepared_data)
        self._history_max_length = int(history_max_length)

    def __call__(self, feature_records: pd.DataFrame) -> dict[str, torch.Tensor]:
        return _build_hstu_rearec_history_batch(feature_records, self._history_max_length)


def _build_hstu_rearec_history_batch(
    records: pd.DataFrame,
    history_max_length: int,
) -> dict[str, torch.Tensor]:
    """Right-padded history batch with timestamps and a query-timestamp slot.

    Allocates ``history_max_length + 1`` columns so that the virtual query slot
    at position ``history_lengths[b]`` is always within bounds.  The target item
    timestamp is written into that slot — mirroring what the standalone
    ``HSTUEvalCollator`` (``include_target_item=False``) does — without appending
    the target item ID to the sequence.  This is required because
    ``_encode_sequence_embeddings`` uses
    ``sequence_lengths = min(history_lengths + 1, tensor_width)``; without the
    extra column, full-history users would have their virtual slot clipped and
    the relative-time bias would be computed against a zero timestamp.

    Returns:
        history_item_ids:   [B, L+1]  right-padded with HSTU_PADDING_ITEM_ID (0).
        history_timestamps: [B, L+1]  item timestamps + query timestamp at slot L_b.
        history_lengths:    [B]       actual (non-padded) history length per sample.
    """
    history_items = [tuple(v) for v in records[HISTORY_ITEM_IDS].tolist()]
    history_times = [tuple(v) for v in records[HISTORY_TIMESTAMPS].tolist()]
    target_timestamps = records[TIMESTAMP].tolist()
    B = len(records)
    L = history_max_length
    W = L + 1  # tensor width: L real slots + 1 virtual query slot

    history_lengths = torch.tensor(
        [min(len(h), L) for h in history_items], dtype=torch.long
    )
    history_item_ids = torch.full((B, W), HSTU_PADDING_ITEM_ID, dtype=torch.long)
    history_timestamps = torch.zeros((B, W), dtype=torch.float32)

    for i, (items, times) in enumerate(zip(history_items, history_times)):
        n = min(len(items), L)
        if n > 0:
            items_t = list(items)[-L:]
            times_t = list(times)[-L:]
            history_item_ids[i, :n] = torch.tensor(items_t, dtype=torch.long)
            history_timestamps[i, :n] = torch.tensor(times_t, dtype=torch.float32)
        # Write query timestamp into the virtual slot (position = actual history length).
        # This is required for correct relative-time bias in HSTU attention.
        history_timestamps[i, n] = float(target_timestamps[i])

    return {
        HISTORY_ITEM_IDS: history_item_ids,     # [B, L+1]
        HISTORY_TIMESTAMPS: history_timestamps,  # [B, L+1]
        "history_lengths": history_lengths,      # [B]
    }


__all__ = [
    "HISTORY_TIMESTAMPS",
    "ReaRecModelDataset",
    "ReaRecTrainCollator",
    "ReaRecEvalCollator",
    "ReaRecHSTUTrainCollator",
    "ReaRecHSTUEvalCollator",
]
