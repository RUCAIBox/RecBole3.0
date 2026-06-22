from __future__ import annotations

from typing import Any, Mapping

import torch

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.model.tiger.config import TIGERConfig
from recbole3.model.tiger.data import TIGEREvalCollator, TIGERModelDataset, TIGERSIDCodec, TIGERTrainCollator


class TIGERModel(BaseRetrievalModel):
    """TIGER generative retrieval model backed by a small T5 decoder target space."""

    config: TIGERConfig

    def __init__(self, config: TIGERConfig):
        super().__init__(config)
        self._t5: torch.nn.Module | None = None
        self._codec: TIGERSIDCodec | None = None
        self._pad_token = 0
        self._base_user_token: int | None = None
        self._eos_token: int | None = None
        self._vocab_size: int | None = None
        self._max_observed_history_items: int | None = None

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return TIGERTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return TIGEREvalCollator(self.config, prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        t5 = self._t5_module()
        outputs = t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )
        return {"loss": outputs.loss, "logits": outputs.logits, "outputs": outputs}

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, Any]) -> torch.Tensor:
        del batch
        loss = outputs["loss"]
        if loss is None:
            raise ValueError("TIGER forward outputs did not include a loss. Training batches must include labels.")
        return loss

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_item_ids is not None:
            raise NotImplementedError("TIGER phase 1 supports only full evaluation, not sampled evaluation.")

        t5 = self._t5_module()
        device = next(t5.parameters()).device
        input_ids = model_inputs["input_ids"].to(device)
        batch_size = int(input_ids.shape[0])
        if k <= 0:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        if int(self.config.num_beams) < int(k):
            raise ValueError(
                f"TIGERConfig.num_beams ({self.config.num_beams}) must be >= requested top-k ({k}). "
                "Increase model.num_beams or lower eval top-k."
            )

        codec = self._sid_codec()
        attention_mask = model_inputs["attention_mask"].to(device)
        beam_width = int(self.config.num_beams)

        generated = t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=codec.n_digit + 1,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        sequences = generated.sequences.reshape(batch_size, beam_width, -1)
        if int(sequences.shape[-1]) < codec.n_digit + 1:
            raise ValueError(
                "TIGER generation returned sequences shorter than one SID tuple. "
                f"Got width {int(sequences.shape[-1])}, expected at least {codec.n_digit + 1}."
            )
        token_tuples = sequences[:, :, 1 : 1 + codec.n_digit]
        token_tuples_cpu = token_tuples.detach().cpu()
        excluded = self._excluded_item_sets(exclude_item_ids, exclude_mask, batch_size=batch_size)

        predictions: list[list[int]] = []
        for row_index in range(batch_size):
            selected: list[int] = []
            selected_set: set[int] = set()
            for beam_index in range(beam_width):
                item_id = codec.token_tuple_to_item(token_tuples_cpu[row_index, beam_index].tolist())
                if item_id is None or item_id in selected_set or item_id in excluded[row_index]:
                    continue
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) == k:
                    break
            if len(selected) < k:
                for item_id in codec.fallback_item_ids:
                    if item_id in selected_set or item_id in excluded[row_index]:
                        continue
                    selected.append(item_id)
                    selected_set.add(item_id)
                    if len(selected) == k:
                        break
            if len(selected) < k:
                raise ValueError(f"TIGER could not produce {k} predictions from {len(codec.fallback_item_ids)} items.")
            predictions.append(selected)

        return torch.tensor(predictions, dtype=torch.long, device=input_ids.device)

    def _ensure_initialized(self, prepared_data: TIGERModelDataset) -> None:
        if not hasattr(prepared_data, "tiger_codec"):
            raise RuntimeError("TIGERModel requires TIGERModelDataset prepared data.")
        codec = prepared_data.tiger_codec
        if self._codec is not None:
            if self._codec.n_digit != codec.n_digit or self._codec.semantic_vocab_size != codec.semantic_vocab_size:
                raise ValueError("TIGERModel was already initialized with an incompatible SID codec.")
            return

        self._codec = codec
        self._base_user_token = codec.semantic_vocab_size + 1
        self._eos_token = self._base_user_token + int(self.config.n_user_tokens)
        self._vocab_size = self._eos_token + 1
        self._max_observed_history_items = _max_observed_history_items(prepared_data)
        self._t5 = self._build_t5()

    def _build_t5(self) -> torch.nn.Module:
        try:
            from transformers import T5Config, T5ForConditionalGeneration
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("TIGER requires `transformers`. Install it before using model.name=tiger.") from exc

        max_positions = self._max_positions()
        t5_config = T5Config(
            num_layers=self.config.num_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_heads=self.config.num_heads,
            d_kv=self.config.d_kv,
            dropout_rate=self.config.dropout_rate,
            activation_function=self.config.activation_function,
            vocab_size=self._require_vocab_size(),
            pad_token_id=self._pad_token,
            eos_token_id=self._require_eos_token(),
            decoder_start_token_id=self._pad_token,
            feed_forward_proj=self.config.feed_forward_proj,
            n_positions=max_positions,
        )
        return T5ForConditionalGeneration(config=t5_config)

    def _max_positions(self) -> int:
        history_max_length = self.config.history_max_length
        if history_max_length is None:
            # None means unbounded history in SequentialModelConfig. The model dataset
            # has already materialized histories, so use the observed maximum instead
            # of collapsing to an empty-history T5 position budget.
            max_history_items = max(int(self._max_observed_history_items or 0), 1)
        else:
            max_history_items = int(history_max_length)
        return max_history_items * self._sid_codec().n_digit + 2

    def _t5_module(self) -> torch.nn.Module:
        if self._t5 is None:
            raise RuntimeError("TIGERModel must be initialized via build_train_collator/build_eval_collator before use.")
        return self._t5

    def _sid_codec(self) -> TIGERSIDCodec:
        if self._codec is None:
            raise RuntimeError("TIGERModel has no SID codec. Build a collator with prepared data first.")
        return self._codec

    def _require_eos_token(self) -> int:
        if self._eos_token is None:
            raise RuntimeError("TIGERModel eos token is not initialized.")
        return self._eos_token

    def _require_vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("TIGERModel vocab size is not initialized.")
        return self._vocab_size

    @staticmethod
    def _excluded_item_sets(
        exclude_item_ids: torch.Tensor | None,
        exclude_mask: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> list[set[int]]:
        excluded = [set() for _ in range(batch_size)]
        if exclude_item_ids is None or exclude_mask is None or exclude_item_ids.numel() == 0:
            return excluded
        ids = exclude_item_ids.detach().cpu()
        mask = exclude_mask.detach().cpu().to(dtype=torch.bool)
        for row_index in range(min(batch_size, int(ids.shape[0]))):
            excluded[row_index] = {
                int(item_id)
                for item_id, keep in zip(ids[row_index].tolist(), mask[row_index].tolist(), strict=False)
                if keep
            }
        return excluded


def _max_observed_history_items(prepared_data: TIGERModelDataset) -> int:
    max_history = 0
    for split in ("train", "valid", "test"):
        dataset = prepared_data.get_train_dataset() if split == "train" else prepared_data.get_eval_dataset(split)
        frame = getattr(dataset, "frame", None)
        if frame is None or HISTORY_ITEM_IDS not in frame:
            continue
        for history in frame[HISTORY_ITEM_IDS].tolist():
            max_history = max(max_history, len(history or ()))
    return max_history


__all__ = ["TIGERModel"]
