from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.lsrm.config import LSRMConfig
from recbole3.model.lsrm.data import ITEM_ID_OFFSET, LABEL_IGNORE, LSRMEvalCollator, LSRMTrainCollator


class LSRMModel(BaseRetrievalModel):
    """LSRM retrieval model based on SASRec, using GPT-2 as backbone."""

    def __init__(self, config: LSRMConfig):
        super().__init__(config)
        self._num_items: int = 0
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE)

    def ensure_initialized(self, prepared_data) -> None:
        num_items = int(prepared_data.get_num_items())
        if self._num_items == num_items:
            return
        if self._num_items != 0:
            raise ValueError(
                f"LSRMModel was initialized for num_items={self._num_items}, got {num_items}."
            )
        self._num_items = num_items
        vocab_size = num_items + 2  # 0=padding, 1~num_items=items, num_items+1=eos
        gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=self.config.history_max_length + 1,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=self.config.n_inner,
            activation_function=self.config.activation_function,
            resid_pdrop=self.config.resid_pdrop,
            embd_pdrop=self.config.embd_pdrop,
            attn_pdrop=self.config.attn_pdrop,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            initializer_range=self.config.initializer_range,
        )
        self.gpt2 = GPT2LMHeadModel(gpt2_config)

    def build_train_collator(self, prepared_data) -> BaseCollator:
        return LSRMTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        return LSRMEvalCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.gpt2(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return {"logits": outputs.logits}

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = outputs["logits"].view(-1, outputs["logits"].shape[-1])
        labels = batch["labels"].view(-1)
        return self.loss_fct(logits, labels)

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.forward(model_inputs)
        logits = outputs["logits"]
        history_lengths = model_inputs["history_lengths"].to(dtype=torch.long, device=logits.device)

        # gather logits at the last valid position of each sequence
        # history_lengths is the length of input_ids (for eval, this is just history length)
        # we need the logits at position history_lengths - 1 (last real token)
        gather_index = (history_lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        last_logits = logits.gather(dim=1, index=gather_index).squeeze(1)

        if candidate_item_ids is not None:
            candidate_item_ids = candidate_item_ids.to(device=logits.device, dtype=torch.long)
            candidate_token_ids = candidate_item_ids + ITEM_ID_OFFSET
            candidate_logits = last_logits.gather(1, candidate_token_ids)
            topk_indices = torch.topk(candidate_logits, k=k, dim=1).indices
            return torch.gather(candidate_item_ids, 1, topk_indices)

        # mask excluded items
        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
            exclude_token_ids = (exclude_item_ids + ITEM_ID_OFFSET).to(
                device=logits.device, dtype=torch.long
            )
            mask = exclude_mask.to(device=logits.device, dtype=torch.bool)
            last_logits.scatter_(1, exclude_token_ids, float("-inf") * mask.float())

        # mask padding token (0) and EOS token (num_items + 1)
        vocab_size = self._num_items + 2
        invalid_tokens = torch.tensor([0, self._num_items + 1], device=logits.device)
        last_logits[:, invalid_tokens] = float("-inf")

        topk_token_ids = torch.topk(last_logits, k=k, dim=1).indices
        return topk_token_ids - ITEM_ID_OFFSET


__all__ = ["LSRMModel"]
