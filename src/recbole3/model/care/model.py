from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers import Qwen2ForCausalLM, BeamSearchScorer  # noqa: F401
from transformers.models.qwen2.modeling_qwen2 import *  # noqa: F401,F403
from transformers.models.qwen2.modeling_qwen2 import _CONFIG_FOR_DOC  # noqa: F401
from transformers.generation.utils import *  # noqa: F401,F403
from transformers.generation.utils import _split_model_inputs

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.care.config import CAREConfig
from recbole3.model.care.data import CAREEvalCollator, CAREModelDataset, CARETokenCodec, CARETrainCollator


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cache_position: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None


class Qwen2Model_AdaptiveAttn_Custom(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.test = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_dict: dict = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, attention_mask_dict, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        attention_mask_dict: dict,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            attention_mask_dict,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    def update_attention_mask_stragety(self, mode, test=False):
        if mode == "hard":
            self._build_stage_attention_mask_across_items_fast = self._build_stage_attention_mask_across_items_fast_V2
        else:
            raise NotImplementedError
        self.test = test

    def _visible_prompt_columns(
        self,
        *,
        attention_mask: torch.Tensor | None,
        row: int,
        input_len: int,
        len_identifier: int,
        stage_idx: int,
        progressive: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            if not progressive:
                return torch.arange(input_len, device=device, dtype=torch.long)
            num_items = input_len // len_identifier
            code_pos = torch.arange(len_identifier, device=device).repeat(num_items)
            visible = (code_pos < stage_idx + 1).nonzero(as_tuple=True)[0]
            return torch.cat([visible, torch.tensor([input_len - 1], device=device, dtype=torch.long)])

        valid_cols = attention_mask[row, :input_len].to(dtype=torch.bool).nonzero(as_tuple=True)[0].to(device=device)
        if valid_cols.numel() == 0:
            return valid_cols
        if not progressive:
            return valid_cols

        special_col = valid_cols[-1:]
        history_cols = valid_cols[:-1]
        usable = int(history_cols.numel()) // int(len_identifier) * int(len_identifier)
        if usable <= 0:
            return special_col
        history_cols = history_cols[-usable:]
        level_ids = torch.arange(usable, device=device) % int(len_identifier)
        visible_history = history_cols[level_ids < int(stage_idx) + 1]
        return torch.cat([visible_history, special_col])

    def _build_stage_attention_mask_across_items_fast_V2(
        self,
        batch_size: int,
        input_len: int,
        len_identifier: int,
        generation_code_idx_start_from_1: int,
        query_list: list,
        progressive_list: list,
        dtype: torch.dtype,
        device: torch.device,
        attention_mask: torch.Tensor | None = None,
    ):
        if not self.test:
            stage_lens = [q + 1 for q in query_list]
            total_reason_tokens = sum(stage_lens)
            total_len = input_len + total_reason_tokens
            min_val = torch.finfo(dtype).min
            mask = torch.full((batch_size, 1, total_len, total_len), fill_value=min_val, dtype=dtype, device=device)

            base_causal = torch.tril(torch.ones(input_len, input_len, device=device, dtype=torch.bool))
            mask[:, :, :input_len, :input_len] = torch.where(
                base_causal,
                torch.zeros_like(mask[:, :, :input_len, :input_len]),
                mask[:, :, :input_len, :input_len],
            )
            reasoning_start = input_len
            cur_ptr = input_len
            for k in range(generation_code_idx_start_from_1):
                n_query = query_list[k]
                start = cur_ptr
                end = start + n_query + 1
                q_start = start
                q_end = start + n_query
                for row in range(batch_size):
                    visible_cols = self._visible_prompt_columns(
                        attention_mask=attention_mask,
                        row=row,
                        input_len=input_len,
                        len_identifier=len_identifier,
                        stage_idx=k,
                        progressive=bool(progressive_list[k]),
                        device=device,
                    )
                    mask[row:row + 1, :, q_start:q_end, visible_cols] = 0
                if q_start > reasoning_start:
                    mask[:, :, q_start:q_end, reasoning_start:q_start] = 0
                local_q_causal = torch.tril(torch.ones(n_query, n_query, device=device, dtype=torch.bool))
                mask[:, :, q_start:q_end, q_start:q_end] = torch.where(
                    local_q_causal,
                    torch.zeros_like(mask[:, :, q_start:q_end, q_start:q_end]),
                    mask[:, :, q_start:q_end, q_start:q_end],
                )
                gold_pos = end - 1
                mask[:, :, gold_pos, :gold_pos] = 0
                cur_ptr = end
        else:
            stage_idx = generation_code_idx_start_from_1 - 1
            total_len = input_len + sum(q + 1 for q in query_list[:stage_idx]) + query_list[stage_idx]
            min_val = torch.finfo(dtype).min
            mask = torch.full((batch_size, 1, total_len, total_len), fill_value=min_val, dtype=dtype, device=device)

            base_causal = torch.tril(torch.ones(input_len, input_len, device=device, dtype=torch.bool))
            mask[:, :, :input_len, :input_len] = torch.where(
                base_causal,
                torch.zeros_like(mask[:, :, :input_len, :input_len]),
                mask[:, :, :input_len, :input_len],
            )
            reasoning_start = input_len
            for k in range(generation_code_idx_start_from_1):
                start = input_len + sum(q + 1 for q in query_list[:k])
                n_query = query_list[k]
                end = start + (n_query + 1)
                q_start = start
                q_end = start + n_query
                for row in range(batch_size):
                    visible_cols = self._visible_prompt_columns(
                        attention_mask=attention_mask,
                        row=row,
                        input_len=input_len,
                        len_identifier=len_identifier,
                        stage_idx=k,
                        progressive=bool(progressive_list[k]),
                        device=device,
                    )
                    mask[row:row + 1, :, q_start:q_end, visible_cols] = 0
                if q_start > reasoning_start:
                    mask[:, :, q_start:q_end, reasoning_start:q_start] = 0
                local_q_causal = torch.tril(torch.ones(n_query, n_query, device=device, dtype=torch.bool))
                mask[:, :, q_start:q_end, q_start:q_end] = torch.where(
                    local_q_causal,
                    torch.zeros_like(mask[:, :, q_start:q_end, q_start:q_end]),
                    mask[:, :, q_start:q_end, q_start:q_end],
                )
                if k < (generation_code_idx_start_from_1 - 1):
                    gold_pos = end - 1
                    mask[:, :, gold_pos, :gold_pos] = 0
        return mask

    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        attention_mask_dict: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        spec = None
        if isinstance(attention_mask_dict, dict) and "input_len" in attention_mask_dict:
            spec = attention_mask_dict
            causal_mask = self._build_stage_attention_mask_across_items_fast(
                batch_size=batch_size,
                input_len=spec["input_len"],
                len_identifier=spec["len_identifier"],
                generation_code_idx_start_from_1=spec["generation_code_idx_start_from_1"],
                query_list=spec["query_list"],
                progressive_list=spec["progressive_list"],
                dtype=dtype,
                device=device,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            )

        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            valid = attention_mask.to(dtype=torch.bool)
            causal_mask = causal_mask.masked_fill(~valid[:, None, None, :], torch.finfo(causal_mask.dtype).min)

        if self.test and spec is not None and spec["generation_code_idx_start_from_1"] > 1:
            causal_mask = causal_mask[
                :, :, -(spec["query_list"][spec["generation_code_idx_start_from_1"] - 1] + 1):, :
            ]
        return causal_mask


class CARE(Qwen2ForCausalLM):
    """CARE: multiple query vectors for parallel reasoning per identifier code."""

    def __init__(
        self,
        config,
        query_list=[1, 1, 1, 1],
        identifier_len=4,
        query_div_scale=0,
        progressive_attn=None,
        progressive_list=[True, True, True, True],
        attention_strategy=None,
        use_cache=True,
    ):
        super().__init__(config)

        if not hasattr(config, "progressive_attn"):
            if progressive_attn:
                self.model = Qwen2Model_AdaptiveAttn_Custom(config)
                self.model.update_attention_mask_stragety(attention_strategy)
        else:
            if config.progressive_attn:
                self.model = Qwen2Model_AdaptiveAttn_Custom(config)
                self.model.update_attention_mask_stragety(config.attention_strategy, getattr(config, "test", False))

        self.identifier_len = identifier_len
        self.train_use_cache = use_cache

        if hasattr(config, "query_list"):
            self.n_query = sum(config.query_list)
            self.query_list = list(config.query_list)
            self.progressive_list = list(config.progressive_list)
        else:
            self.n_query = sum(query_list)
            self.query_list = list(query_list)
            self.progressive_list = list(progressive_list)

        if self.n_query:
            self.query_vector = nn.Embedding(self.n_query, self.model.config.hidden_size)
        if self.n_query == 0:
            zeros = torch.zeros(1, self.model.config.hidden_size)
            self.query_vector = nn.Embedding.from_pretrained(zeros, freeze=True)

        self.query_div_scale = query_div_scale

    def fixed_cross_entropy(self, source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        source = source.float()
        target = target.to(source.device)
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

    def update_config(self, query_list, progressive_list):
        self.n_query = sum(query_list)
        self.query_list = list(query_list)
        self.progressive_list = list(progressive_list)
        self.config.progressive_list = list(progressive_list)
        self.config.query_list = list(query_list)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is not None:
            return self.forward_training(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        return self.forward_inference(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def forward_training(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.model.test = False

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        for code_idx in range(self.identifier_len):
            reasoning_step = self.query_list[code_idx]
            lookup_idx = torch.arange(sum(self.query_list[:code_idx]), sum(self.query_list[:code_idx]) + reasoning_step)
            lookup_idx = lookup_idx.to(inputs_embeds.device).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
            query_vector = self.query_vector(lookup_idx)
            inputs_embeds = torch.cat([inputs_embeds, query_vector], dim=1)

            code_idx_tensor = torch.LongTensor([code_idx]).to(labels.device)
            gold_code_idx = labels[:, code_idx_tensor]
            gold_code_emb = self.model.get_input_embeddings()(gold_code_idx)
            inputs_embeds = torch.cat([inputs_embeds, gold_code_emb], dim=1)
            added_mask = torch.ones(
                (attention_mask.shape[0], reasoning_step + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, added_mask], dim=1)

        attention_mask_dict = {
            "input_len": input_ids.size(1),
            "len_identifier": self.identifier_len,
            "generation_code_idx_start_from_1": self.identifier_len,
            "query_list": self.query_list,
            "progressive_list": self.progressive_list,
        }

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            attention_mask_dict=attention_mask_dict,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        input_len = input_ids.size(1)
        pred_positions = []
        for code_idx in range(self.identifier_len):
            pred_pos = input_len + sum(self.query_list[:code_idx]) + code_idx + self.query_list[code_idx] - 1
            pred_positions.append(pred_pos)
        pred_positions = torch.tensor(pred_positions, device=logits.device)
        selected_logits = logits[:, pred_positions, :]
        targets = labels[:, :-1]

        logits_flat = selected_logits.view(-1, self.config.vocab_size)
        targets_flat = targets.reshape(-1)
        loss = self.fixed_cross_entropy(source=logits_flat, target=targets_flat, num_items_in_batch=len(targets_flat))

        qv = F.normalize(self.query_vector.weight, dim=1)
        sim_matrix = torch.matmul(qv, qv.T)
        eye = torch.eye(sim_matrix.size(0), device=qv.device).bool()
        sim_matrix = sim_matrix.masked_fill(eye, 0.0)
        loss = loss + self.query_div_scale * sim_matrix.mean()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=selected_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_inference(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        code_idx: Optional[int] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.model.test = True

        code_idx = min(code_idx, self.identifier_len - 1)
        if input_ids is not None:
            assert code_idx == 0
        n_query = self.query_list[code_idx]

        if code_idx == 0:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                else:
                    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)
            if cache_position is None:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            cache_position = torch.cat(
                [cache_position, torch.arange(cache_position.shape[0], cache_position.shape[0] + n_query).to(cache_position.device)],
                dim=0,
            )
            add_tensor = torch.arange(1, n_query + 1, device=position_ids.device)
            position_ids = torch.cat([position_ids, add_tensor + position_ids[:, -1:]], dim=1)
        else:
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids[:, -1:]
                elif cache_position is not None:
                    position_ids = cache_position[-1:].view(1, 1).repeat(inputs_embeds.shape[0], 1)
                else:
                    position_ids = torch.zeros((inputs_embeds.shape[0], 1), dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids[:, -1:].repeat(1, n_query + 1)
            for idx in range(n_query + 1):
                position_ids[:, idx] = position_ids[:, idx] + (idx + 1)
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past = past_key_values.get_seq_length()
            elif cache_position is not None:
                past = int(cache_position[-1].item()) + 1
            else:
                past = inputs_embeds.shape[1]
            cache_position = torch.arange(past, past + 1 + n_query, device=inputs_embeds.device)

        lookup_idx = torch.arange(sum(self.query_list[:code_idx]), sum(self.query_list[:code_idx]) + self.query_list[code_idx])
        lookup_idx = lookup_idx.to(inputs_embeds.device).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
        query_vector = self.query_vector(lookup_idx)
        inputs_embeds = torch.cat([inputs_embeds, query_vector], dim=1)

        input_len = attention_mask.size(1) - (sum(self.query_list[:code_idx]) + code_idx)
        added_mask = torch.ones(
            (attention_mask.shape[0], self.query_list[code_idx]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, added_mask], dim=1)

        attention_mask_dict = {
            "input_len": input_len,
            "len_identifier": self.identifier_len,
            "generation_code_idx_start_from_1": code_idx + 1,
            "query_list": self.query_list,
            "progressive_list": self.progressive_list,
        }

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            attention_mask_dict=attention_mask_dict,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        reason_states = hidden_states[:, -1:, :]
        logits = self.lm_head(reason_states)
        logits = torch.mean(logits, dim=1, keepdim=True)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cache_position=cache_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        generation_config: GenerationConfig | None = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        generation_config = generation_config if generation_config is not None else self.generation_config
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if beam_scorer is None:
            num_beams = int(generation_config.num_beams)
            if input_ids.shape[0] % num_beams != 0:
                raise ValueError(
                    "Expanded input batch size must be divisible by generation_config.num_beams, "
                    f"got batch={input_ids.shape[0]}, num_beams={num_beams}."
                )
            beam_scorer = BeamSearchScorer(
                batch_size=input_ids.shape[0] // num_beams,
                num_beams=num_beams,
                device=input_ids.device,
                length_penalty=float(generation_config.length_penalty),
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=int(generation_config.num_return_sequences),
                max_length=getattr(generation_config, "max_length", None),
            )

        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False
        decoder_prompt_len = input_ids.shape[-1]
        code_idx = 0
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if sequential:
                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs,
                    split_size=batch_size,
                    full_batch_size=batch_beam_size,
                    config=self.config.get_text_config(),
                )
                outputs_per_sub_batch = [
                    self.forward_inference(**inputs_per_sub_batch, return_dict=True, code_idx=code_idx)
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]
                outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())
            else:
                outputs = self.forward_inference(**model_inputs, return_dict=True, code_idx=code_idx)

            code_idx += 1
            model_kwargs["cache_position"] = outputs.cache_position
            model_kwargs["position_ids"] = outputs.position_ids
            model_kwargs["attention_mask"] = outputs.attention_mask
            model_kwargs["inputs_embeds"] = outputs.inputs_embeds

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=self.query_list[min(code_idx, self.identifier_len - 1)] + 1,
            )
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            beam_next_tokens_embeds = self.model.get_input_embeddings()(beam_next_tokens.unsqueeze(-1).to(input_ids.device))
            model_kwargs["inputs_embeds"] = torch.cat(
                [model_kwargs["inputs_embeds"][beam_idx, ...], beam_next_tokens_embeds], dim=1
            )
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(model_kwargs["past_key_values"], beam_idx)
            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            cur_len = cur_len + 1
            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return sequence_outputs["sequences"]

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        elif cache_position is None:
            raise NotImplementedError

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None and cache_position is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        input_ids_key = "input_ids"
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["inputs_ids"] = input_ids
        else:
            assert input_ids is not None, "input_ids must be provided"
            model_inputs["input_ids"] = input_ids

        attention_mask_key = "attention_mask"
        position_ids_key = "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            if attention_mask.shape[1] == input_ids.shape[1]:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                kwargs[position_ids_key] = position_ids

        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        model_inputs.pop("labels", None)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
        self.append_trie = None
        self.bos_token_id = None

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence, trie_dict, append_trie=None, bos_token_id=None):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(prefix_sequence[1:], trie_dict[prefix_sequence[0]], append_trie, bos_token_id)
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            return []


def prefix_allowed_tokens_fn(candidate_trie, tokenizer, special_token_for_answer="|start_of_answer|"):
    sep = tokenizer(special_token_for_answer)["input_ids"]
    bos = [1]

    def prefix_allowed_tokens(batch_id, sentence):
        sentence_ = bos
        for i in range(len(sentence), -1, -1):
            if sentence[i - len(sep):i].tolist() == sep:
                sentence_ = bos if i == len(sentence) else [1] + sentence[i:].tolist()
                break
        return candidate_trie.get(sentence_)

    return prefix_allowed_tokens


def _normalize_generated_identifier(text: Any) -> str:
    return str(text).strip().replace(" ", "").replace("\n", "").replace("\t", "")


def get_topk_results(predictions, scores, k, all_items=None, special_token_for_answer="|start_of_answer|"):
    B = len(predictions) // k
    predictions = [_normalize_generated_identifier(_.split(special_token_for_answer)[-1]) for _ in predictions]
    if all_items is not None:
        valid_items = {_normalize_generated_identifier(item) for item in all_items}
        for i, seq in enumerate(predictions):
            if seq not in valid_items:
                scores[i] = -1000
    batch_pred = []
    for b in range(B):
        batch_seqs = predictions[b * k:(b + 1) * k]
        batch_scores = scores[b * k:(b + 1) * k]
        pairs = [(a, s) for a, s in zip(batch_seqs, batch_scores)]
        results = sorted(pairs, key=lambda x: x[1], reverse=True)
        batch_pred.append([r[0] for r in results])
    return batch_pred


class CAREModel(BaseRetrievalModel):
    """RecBole3.0 adapter wrapping the faithful CARE implementation."""

    config: CAREConfig

    def __init__(self, config: CAREConfig):
        super().__init__(config)
        self._tokenizer: Any | None = None
        self._codec: CARETokenCodec | None = None
        self._care_lm: CARE | None = None
        self._prefix_fn: Callable | None = None

    def ensure_initialized(self, prepared_data) -> None:
        self._ensure_initialized(prepared_data)

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        self._care_lm.model.test = False
        return CARETrainCollator(self.config, prepared_data, tokenizer=self._tokenizer, codec=self._codec)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return CAREEvalCollator(self.config, prepared_data, tokenizer=self._tokenizer, codec=self._codec)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        self._care_lm.model.test = False
        outputs = self._care_lm.forward_training(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, Any]) -> torch.Tensor:
        del batch
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("CARE forward outputs did not include a loss.")
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
        device = model_inputs["input_ids"].device
        batch_size = int(model_inputs["input_ids"].shape[0])
        if k <= 0:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        if candidate_item_ids is not None:
            raise NotImplementedError("CARE supports full evaluation only.")

        codec = self._codec
        tokenizer = self._tokenizer
        care_lm = self._care_lm
        previous_test = bool(getattr(care_lm.model, "test", False))
        previous_use_cache = bool(getattr(care_lm.config, "use_cache", True))
        care_lm.model.test = True
        care_lm.config.use_cache = True
        beam_width = max(int(k), int(self.config.num_beams))

        try:
            output = care_lm.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=int(self.config.max_new_token),
                prefix_allowed_tokens_fn=self._require_prefix_fn(),
                num_beams=beam_width,
                num_return_sequences=beam_width,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                do_sample=False,
            )
        finally:
            care_lm.model.test = previous_test
            care_lm.config.use_cache = previous_use_cache
        decoded = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
        scores = output["sequences_scores"].detach().cpu().tolist()
        topk_text = get_topk_results(
            decoded,
            scores,
            beam_width,
            all_items=codec.all_items if bool(self.config.filter_items) else None,
            special_token_for_answer=self.config.special_token_for_answer,
        )
        excluded = self._excluded_item_sets(exclude_item_ids, exclude_mask, batch_size=batch_size)
        predictions: list[list[int]] = []
        for row in range(batch_size):
            selected: list[int] = []
            selected_set: set[int] = set()
            for code_text in topk_text[row]:
                item_id = codec.code_text_to_id(code_text)
                if item_id is None or item_id in selected_set or item_id in excluded[row]:
                    continue
                selected.append(int(item_id))
                selected_set.add(int(item_id))
                if len(selected) == int(k):
                    break
            for item_id in codec.all_item_ids:
                if len(selected) == int(k):
                    break
                if item_id not in selected_set and item_id not in excluded[row]:
                    selected.append(int(item_id))
                    selected_set.add(int(item_id))
            predictions.append(selected[: int(k)])
        return torch.tensor(predictions, dtype=torch.long, device=device)

    def _ensure_initialized(self, prepared_data: CAREModelDataset) -> None:
        if not hasattr(prepared_data, "care_codec"):
            raise RuntimeError("CAREModel requires CAREModelDataset prepared data.")
        codec = prepared_data.care_codec
        if self._codec is not None:
            return

        from transformers import AutoTokenizer, Qwen2Config

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=int(self.config.model_max_length),
            padding_side="left",
            trust_remote_code=bool(self.config.trust_remote_code),
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        new_tokens = list(codec.all_new_tokens)
        if self.config.special_token_for_answer:
            new_tokens.append(self.config.special_token_for_answer)
        tokenizer.add_tokens(sorted(set(new_tokens)))

        config = Qwen2Config.from_pretrained(self.config.base_model, trust_remote_code=bool(self.config.trust_remote_code))
        config.query_list = list(self.config.query_list)
        config.progressive_list = list(self.config.progressive_list)

        kwargs: dict[str, Any] = {
            "config": config,
            "query_list": list(self.config.query_list),
            "progressive_list": list(self.config.progressive_list),
            "progressive_attn": bool(self.config.progressive_attn),
            "attention_strategy": self.config.attention_strategy,
            "query_div_scale": float(self.config.query_div_scale),
            "identifier_len": codec.identifier_len,
            "low_cpu_mem_usage": bool(self.config.low_cpu_mem_usage),
            "trust_remote_code": bool(self.config.trust_remote_code),
        }
        dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        care_lm = CARE.from_pretrained(self.config.base_model, **kwargs)
        care_lm.identifier_len = codec.identifier_len
        care_lm.update_config(list(self.config.query_list), list(self.config.progressive_list))
        care_lm.resize_token_embeddings(len(tokenizer))
        care_lm.config.pad_token_id = tokenizer.pad_token_id
        care_lm.config.eos_token_id = tokenizer.eos_token_id

        candidate_trie = Trie(
            [[1] + tokenizer.encode(candidate) + [tokenizer.eos_token_id] for candidate in codec.all_items]
        )
        self._prefix_fn = prefix_allowed_tokens_fn(candidate_trie, tokenizer, self.config.special_token_for_answer)

        self._codec = codec
        self._tokenizer = tokenizer
        self._care_lm = care_lm

    def _require_prefix_fn(self) -> Callable:
        if self._prefix_fn is None:
            raise RuntimeError("CARE prefix function is not initialized.")
        return self._prefix_fn

    @staticmethod
    def _excluded_item_sets(exclude_item_ids, exclude_mask, *, batch_size: int) -> list[set[int]]:
        excluded = [set() for _ in range(batch_size)]
        if exclude_item_ids is None or exclude_mask is None or exclude_item_ids.numel() == 0:
            return excluded
        ids = exclude_item_ids.detach().cpu()
        mask = exclude_mask.detach().cpu().to(dtype=torch.bool)
        for row in range(min(batch_size, int(ids.shape[0]))):
            excluded[row] = {
                int(item_id)
                for item_id, keep in zip(ids[row].tolist(), mask[row].tolist())
                if keep
            }
        return excluded


def _resolve_torch_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized == "auto":
        return "auto"
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported CARE torch_dtype: {value!r}")


__all__ = ["CAREModel", "CARE", "Qwen2Model_AdaptiveAttn_Custom", "Trie", "prefix_allowed_tokens_fn"]