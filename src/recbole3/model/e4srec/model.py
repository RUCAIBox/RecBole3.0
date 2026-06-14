from __future__ import annotations

import os
import pickle
from collections.abc import Mapping

import torch
from torch import nn
import numpy as np

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.e4srec.config import E4SRecConfig
from recbole3.model.e4srec.data import (
    E4SRecCollator,
    ITEM_ID_OFFSET,
)


class E4SRecModel(BaseRetrievalModel):
    """
    E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation
    https://arxiv.org/pdf/2312.02443
    Reference to the official codebase: https://github.com/HestiaSky/E4SRec
    """

    def __init__(self, config: E4SRecConfig):
        super().__init__(config)
        self._num_items: int = 0
        self.loss_fct = nn.CrossEntropyLoss()

    def ensure_initialized(self, prepared_data) -> None:
        num_items = int(prepared_data.get_num_items())
        if self._num_items == num_items:
            return
        if self._num_items != 0:
            raise ValueError(
                f"E4SRecModel already initialized for num_items={self._num_items}, "
                f"cannot re-initialize for {num_items}."
            )
        self._num_items = num_items

        # -- Resolve dtype -------------------------------------------------
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        device_map = self._resolve_device_map(self.config.device_map)

        from transformers import AutoModel, AutoTokenizer

        load_kwargs: dict = {
            "dtype": torch_dtype,
            "device_map": device_map,
        }
        if self.config.cache_dir:
            load_kwargs["cache_dir"] = self.config.cache_dir
        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        self.llm_model = AutoModel.from_pretrained(
            self.config.base_model,
            **load_kwargs,
        )

        # -- Prepare for int8 training (before LoRA) -----------------------
        if self.config.load_in_8bit:
            from peft import prepare_model_for_int8_training

            self.llm_model = prepare_model_for_int8_training(self.llm_model)

        # -- Apply LoRA ----------------------------------------------------
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
        )
        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model.config.use_cache = False

        # -- Tokenizer (used only for the fixed prompt template) -----------
        tokenizer_kwargs: dict = {"use_fast": False}
        if self.config.cache_dir:
            tokenizer_kwargs["cache_dir"] = self.config.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            **tokenizer_kwargs,
        )
        self.tokenizer.padding_side = "right"

        # Tokenize the instruction and response templates once
        instruction = self.config.prompt_template.format(
            instruction=self.config.instruction_text
        )
        response = self.config.response_split
        self._instruct_ids, self._instruct_mask = self.tokenizer(
            instruction,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).values()
        self._response_ids, self._response_mask = self.tokenizer(
            response,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).values()

        # -- Pre-trained item embeddings -----------------------------------
        pretrained_embeds = np.load(self.config.item_embed_path)
        pretrained_embeds = torch.from_numpy(pretrained_embeds).to(torch.float32)

        if pretrained_embeds.shape[0] < num_items:
            raise ValueError(
                f"Pre-trained embeddings have {pretrained_embeds.shape[0]} items, "
                f"but the dataset has {num_items} items."
            )

        # Pad: index 0 = mean embedding (padding), indices 1..num_items = real items
        pad_embed = pretrained_embeds[:num_items].mean(dim=0, keepdim=True)
        full_embeds = torch.cat([pad_embed, pretrained_embeds[:num_items]], dim=0)

        self.input_embeds = nn.Embedding.from_pretrained(full_embeds, freeze=True)
        llm_hidden_size = self.llm_model.config.hidden_size
        # Place non-LLM parameters on the same device as the backbone so that
        # forward() works without explicit device transfers.
        llm_device = next(self.llm_model.parameters()).device
        self.input_embeds = self.input_embeds.to(llm_device)
        self.input_proj = nn.Linear(
            self.config.item_embed_dim, llm_hidden_size, device=llm_device
        )
        # num_items + 1 to account for the padding position at index 0
        self.score = nn.Linear(
            llm_hidden_size, num_items + 1, bias=False, device=llm_device
        )

        # -- Gradient checkpointing ----------------------------------------
        if self.config.use_gradient_checkpointing:
            self.llm_model.gradient_checkpointing_enable()
            # PEFT requires enable_input_require_grads() after enabling
            # gradient checkpointing; without it the frozen backbone drops
            # input gradients and LoRA adapters receive no signal.
            self.llm_model.enable_input_require_grads()

    @staticmethod
    def _resolve_device_map(device_map: str) -> str | dict[str, int]:
        """Resolve ``device_map`` for DDP or single-GPU training.

        When launched via ``accelerate launch`` (DDP), each process should
        pin the model to its own GPU.  ``device_map="auto"`` would shard the
        model across all visible GPUs, which breaks DDP.
        """
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            rank = int(local_rank)
            torch.cuda.set_device(rank)
            return {"" : rank}
        # Single-GPU: if CUDA is available, pin to device 0 to avoid
        # auto-sharding overhead; otherwise "auto" for CPU-only.
        if torch.cuda.is_available():
            return {"" : 0}
        return device_map

    def build_train_collator(self, prepared_data) -> BaseCollator:
        return E4SRecCollator(self.config, prepared_data, include_labels=True)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        return E4SRecCollator(self.config, prepared_data, include_labels=False)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return only trainable params — excludes the frozen LLM backbone.

        HF Trainer calls ``state_dict()`` on every checkpoint save, so this
        override ensures per-epoch checkpoints are small (LoRA + non-LLM)
        instead of containing the full frozen backbone.
        """
        full = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in full.items()
            if not k.startswith("llm_model.") or 'lora' in k
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the LLM backbone on a prompted item sequence and score all items."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Embed item history via frozen CF embeddings + projection
        item_embeds = self.input_embeds(input_ids)       # (B, S, embed_dim)
        item_embeds = self.input_proj(item_embeds)       # (B, S, llm_hidden)

        # 2. Expand fixed prompt embeddings to batch size
        embed = self.llm_model.get_input_embeddings()
        instruct_embeds = embed(self._instruct_ids.to(device)).expand(batch_size, -1, -1)
        response_embeds = embed(self._response_ids.to(device)).expand(batch_size, -1, -1)
        instruct_mask = self._instruct_mask.to(device).expand(batch_size, -1)
        response_mask = self._response_mask.to(device).expand(batch_size, -1)

        # 3. Concatenate: [instruct | items | response]
        inputs_embeds = torch.cat([instruct_embeds, item_embeds, response_embeds], dim=1)
        full_attention_mask = torch.cat(
            [instruct_mask, attention_mask, response_mask], dim=1
        )

        # 4. Run LLM backbone
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            return_dict=True,
        )

        # 5. Score all items from the final (response) hidden state
        pooled_output = outputs.last_hidden_state[:, -1, :]  # (B, llm_hidden)
        logits = self.score(pooled_output)                   # (B, num_items + 1)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, batch, outputs):
        return outputs.loss
    
    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-*k* item IDs (0-indexed framework convention)."""
        outputs = self.forward(**model_inputs)
        logits = outputs["logits"]  # (B, num_items + 1)

        # Mask the padding position (internal index 0)
        logits[:, 0] = float("-inf")

        # -- Sampled evaluation: score only the provided candidates ---------
        if candidate_item_ids is not None:
            candidate_item_ids = candidate_item_ids.to(
                device=logits.device, dtype=torch.long
            )
            # candidate_item_ids are 0-indexed (framework convention);
            # add ITEM_ID_OFFSET to index into the 1-indexed logits.
            candidate_internal = candidate_item_ids + ITEM_ID_OFFSET
            candidate_logits = logits.gather(1, candidate_internal)
            # Mask candidate padding (0-indexed 0 maps to internal 1, but
            # padding candidates may be flagged via 0 in candidate_item_ids).
            # Guard: mask internal index 1 when the framework item ID is 0
            # but is actually padding (no valid candidate).
            # We treat framework item 0 + mask → internal 1; if it's a real
            # candidate it stays, otherwise it's already harmless.
            topk_indices_in_candidates = torch.topk(
                candidate_logits, k=min(k, candidate_logits.shape[1]), dim=1
            ).indices
            return torch.gather(candidate_item_ids, 1, topk_indices_in_candidates)

        # -- Full evaluation: exclude seen items ---------------------------
        if (
            exclude_item_ids is not None
            and exclude_mask is not None
            and exclude_item_ids.numel() > 0
        ):
            exclude_item_ids = exclude_item_ids.to(
                device=logits.device, dtype=torch.long
            )
            exclude_mask_bool = exclude_mask.to(
                device=logits.device, dtype=torch.bool
            )
            # Convert 0-indexed framework IDs to 1-indexed internal indices
            exclude_internal = exclude_item_ids + ITEM_ID_OFFSET
            B = logits.shape[0]
            for b in range(B):
                row_exclude = exclude_internal[b][exclude_mask_bool[b]]
                if row_exclude.numel() > 0:
                    logits[b, row_exclude] = float("-inf")

        # -- Top-k over all items ------------------------------------------
        topk_internal = torch.topk(logits, k=k, dim=1).indices  # (B, k), 1-indexed
        return topk_internal - ITEM_ID_OFFSET  # back to 0-indexed framework convention

__all__ = ["E4SRecModel"]
