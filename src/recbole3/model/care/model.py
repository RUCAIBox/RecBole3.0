from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.care.config import CAREConfig
from recbole3.model.care.data import CAREEvalCollator, CAREModelDataset, CARETokenCodec, CARETrainCollator


class Trie:
    def __init__(self, sequences: Sequence[Sequence[int]] = ()) -> None:
        self.trie_dict: dict[int, dict] = {}
        self.len = 0
        for sequence in sequences:
            self._add_to_trie([int(token) for token in sequence], self.trie_dict)
            self.len += 1
        self.append_trie = None
        self.bos_token_id = None

    def get(self, prefix_sequence: Sequence[int]) -> list[int]:
        return self._get_from_trie([int(token) for token in prefix_sequence], self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def _add_to_trie(sequence: list[int], trie_dict: dict[int, dict]) -> None:
        if sequence:
            trie_dict.setdefault(sequence[0], {})
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence: list[int], trie_dict: dict[int, dict], append_trie=None, bos_token_id=None) -> list[int]:
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        if prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(prefix_sequence[1:], trie_dict[prefix_sequence[0]], append_trie, bos_token_id)
        if append_trie:
            return append_trie.get(prefix_sequence)
        return []


def build_prefix_allowed_tokens_fn(
    *,
    tokenizer: Any,
    item_code_texts: Sequence[str],
    special_token_for_answer: str,
) -> Callable[[int, torch.Tensor], list[int]]:
    candidate_trie = Trie(
        [[1] + tokenizer.encode(candidate, add_special_tokens=False) + [int(tokenizer.eos_token_id)] for candidate in item_code_texts]
    )
    sep = tokenizer(special_token_for_answer)["input_ids"]
    bos = [1]

    def prefix_allowed_tokens(batch_id: int, sentence: torch.Tensor) -> list[int]:
        del batch_id
        sentence_list = sentence.tolist()
        sentence_prefix = bos
        for i in range(len(sentence_list), -1, -1):
            if sentence_list[i - len(sep) : i] == sep:
                sentence_prefix = bos if i == len(sentence_list) else [1] + sentence_list[i:]
                break
        return candidate_trie.get(sentence_prefix)

    return prefix_allowed_tokens


@dataclass
class CARECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    cache_position: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None


class Qwen2ModelAdaptiveAttnCustom(Qwen2Model):
    """Qwen2Model with CARE progressive attention mask injected internally."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self._care_attention_spec: dict[str, Any] | None = None
        self._care_test = False

    def set_care_attention_spec(self, spec: dict[str, Any] | None, *, test: bool) -> None:
        self._care_attention_spec = spec
        self._care_test = bool(test)

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        spec = self._care_attention_spec
        if not isinstance(spec, dict) or "input_len" not in spec:
            return super()._update_causal_mask(attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        batch_size = int(input_tensor.shape[0])
        query_length = int(input_tensor.shape[1])
        key_length = int(attention_mask.shape[-1]) if attention_mask is not None else int(cache_position[-1].item()) + 1
        dtype = input_tensor.dtype
        device = input_tensor.device
        min_val = torch.finfo(dtype).min
        input_len = int(spec["input_len"])
        identifier_len = int(spec["len_identifier"])
        stage_count = int(spec["generation_code_idx_start_from_1"])
        query_list = [int(value) for value in spec["query_list"]]
        progressive_list = [bool(value) for value in spec["progressive_list"]]

        full_len = max(key_length, input_len + sum(q + 1 for q in query_list[:stage_count]))
        if self._care_test:
            full_len = max(key_length, input_len + sum(q + 1 for q in query_list[: stage_count - 1]) + query_list[stage_count - 1])
        mask = torch.full((batch_size, 1, full_len, full_len), min_val, dtype=dtype, device=device)
        base_causal = torch.tril(torch.ones(input_len, input_len, dtype=torch.bool, device=device))
        mask[:, :, :input_len, :input_len] = torch.where(base_causal, torch.zeros((), dtype=dtype, device=device), mask[:, :, :input_len, :input_len])

        special_token_idx = input_len - 1
        history_tokens = max(0, input_len - 1)
        code_pos = torch.arange(identifier_len, device=device).repeat((history_tokens + identifier_len - 1) // identifier_len)[:history_tokens]
        reasoning_start = input_len
        for stage_idx in range(stage_count):
            q_start = input_len + sum(query_list[i] + 1 for i in range(stage_idx))
            n_query = query_list[stage_idx]
            q_end = q_start + n_query
            if progressive_list[stage_idx] and code_pos.numel() > 0:
                visible_cols = (code_pos < stage_idx + 1).nonzero(as_tuple=True)[0]
                visible_cols = torch.cat([visible_cols, torch.tensor([special_token_idx], device=device)])
            else:
                visible_cols = torch.arange(input_len, device=device)
            if n_query > 0:
                mask[:, :, q_start:q_end, visible_cols] = 0
                if q_start > reasoning_start:
                    mask[:, :, q_start:q_end, reasoning_start:q_start] = 0
                local = torch.tril(torch.ones(n_query, n_query, dtype=torch.bool, device=device))
                mask[:, :, q_start:q_end, q_start:q_end] = torch.where(local, torch.zeros((), dtype=dtype, device=device), mask[:, :, q_start:q_end, q_start:q_end])
            gold_pos = q_end
            if not self._care_test or stage_idx < stage_count - 1:
                mask[:, :, gold_pos, :gold_pos] = 0

        if attention_mask is not None and attention_mask.dim() == 2:
            mask = mask[:, :, : attention_mask.shape[1], : attention_mask.shape[1]]
            mask = mask.masked_fill(~attention_mask.to(torch.bool)[:, None, None, :], min_val)
        if self._care_test:
            mask = mask[:, :, -query_length:, :]
        else:
            mask = mask[:, :, :query_length, :]
        return mask


class CARECausalLM(Qwen2ForCausalLM):
    """Original CARE-style Qwen2ForCausalLM adapted in-place for RecBole3.0."""

    def __init__(
        self,
        config: Qwen2Config,
        *,
        query_list: Sequence[int] = (1, 1, 1, 1),
        progressive_list: Sequence[bool] = (True, True, True, True),
        progressive_attn: bool = True,
        query_div_scale: float = 0.0,
        identifier_len: int | None = None,
    ) -> None:
        super().__init__(config)
        self.query_list = tuple(int(value) for value in query_list)
        self.progressive_list = tuple(bool(value) for value in progressive_list)
        self.progressive_attn = bool(progressive_attn)
        self.query_div_scale = float(query_div_scale)
        self.identifier_len = int(identifier_len or len(self.query_list))
        self.config.query_list = list(self.query_list)
        self.config.progressive_list = list(self.progressive_list)
        self.config.progressive_attn = bool(self.progressive_attn)
        n_query = max(1, sum(self.query_list))
        self.query_vector = nn.Embedding(n_query, int(self.config.hidden_size))
        if sum(self.query_list) == 0:
            with torch.no_grad():
                self.query_vector.weight.zero_()
            self.query_vector.weight.requires_grad_(False)
        self._install_care_qwen_model()

    def _install_care_qwen_model(self) -> None:
        if isinstance(self.model, Qwen2ModelAdaptiveAttnCustom):
            return
        old_model = self.model
        care_model = Qwen2ModelAdaptiveAttnCustom(self.config)
        care_model.load_state_dict(old_model.state_dict(), strict=True)
        self.model = care_model

    @classmethod
    def from_care_pretrained(
        cls,
        base_model: str,
        *,
        query_list: Sequence[int],
        progressive_list: Sequence[bool],
        progressive_attn: bool,
        attention_strategy: str,
        query_div_scale: float,
        identifier_len: int,
        torch_dtype: torch.dtype | str | None,
        attn_implementation: str | None,
        trust_remote_code: bool,
        low_cpu_mem_usage: bool,
    ) -> "CARECausalLM":
        config = Qwen2Config.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        config.query_list = list(query_list)
        config.progressive_list = list(progressive_list)
        config.progressive_attn = bool(progressive_attn)
        config.attention_strategy = attention_strategy
        kwargs: dict[str, Any] = {
            "config": config,
            "query_list": query_list,
            "progressive_list": progressive_list,
            "progressive_attn": progressive_attn,
            "query_div_scale": query_div_scale,
            "identifier_len": identifier_len,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = cls.from_pretrained(base_model, **kwargs)
        model._install_care_qwen_model()
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any):
        if labels is None:
            return self.forward_inference(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return self.forward_training(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def forward_training(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        selected_logits, targets = self.identifier_logits(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = F.cross_entropy(selected_logits.reshape(-1, selected_logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
        loss = loss + self.query_div_scale * self._query_diversity_loss()
        return {"loss": loss, "logits": selected_logits}

    def forward_inference(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        code_idx: int = 0,
        prompt_length: int | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> CARECausalLMOutputWithPast:
        if attention_mask is None:
            raise ValueError("CARE forward_inference requires attention_mask.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("CARE forward_inference requires input_ids or inputs_embeds.")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if prompt_length is None:
            prompt_length = int(attention_mask.shape[1])

        stage_idx = min(int(code_idx), self.identifier_len - 1)
        n_query = self.query_list[stage_idx]
        has_cache = bool(use_cache and past_key_values is not None)
        if has_cache:
            inputs_embeds = inputs_embeds[:, -1:, :]
            if position_ids is None:
                last_pos = attention_mask.long().sum(dim=-1, keepdim=True) - 1
            else:
                last_pos = position_ids[:, -1:]
        else:
            if position_ids is None or int(position_ids.shape[1]) != int(inputs_embeds.shape[1]):
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            last_pos = position_ids[:, -1:]

        if n_query > 0:
            query_start = sum(self.query_list[:stage_idx])
            lookup = torch.arange(query_start, query_start + n_query, device=inputs_embeds.device)
            lookup = lookup.unsqueeze(0).expand(inputs_embeds.size(0), -1)
            query_embeds = self.query_vector(lookup)
            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask[:, -1:].expand(-1, n_query)], dim=1)
            query_pos = last_pos + torch.arange(1, n_query + 1, device=inputs_embeds.device).unsqueeze(0)
            position_ids = torch.cat([last_pos, query_pos], dim=1) if has_cache else torch.cat([position_ids, query_pos], dim=1)
        elif has_cache:
            position_ids = last_pos

        if cache_position is None or (not has_cache and int(cache_position.numel()) != int(inputs_embeds.shape[1])):
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        elif has_cache:
            cache_start = int(cache_position[-1].item()) if cache_position.numel() else int(last_pos.max().item())
            cache_position = torch.arange(cache_start, cache_start + inputs_embeds.shape[1], device=inputs_embeds.device)

        input_len = int(attention_mask.shape[1]) - (sum(self.query_list[:stage_idx]) + stage_idx + n_query)
        if isinstance(self.model, Qwen2ModelAdaptiveAttnCustom):
            if self.progressive_attn:
                self.model.set_care_attention_spec(
                    {
                        "input_len": input_len,
                        "len_identifier": self.identifier_len,
                        "generation_code_idx_start_from_1": stage_idx + 1,
                        "query_list": self.query_list,
                        "progressive_list": self.progressive_list,
                    },
                    test=True,
                )
            else:
                self.model.set_care_attention_spec(None, test=True)
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=bool(use_cache),
            return_dict=True,
        )
        logits = self.lm_head(outputs.logits[:, -1:, :]) if outputs.logits.shape[-1] != self.config.vocab_size else outputs.logits[:, -1:, :]
        return CARECausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cache_position=cache_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]],
        num_beams: int,
        num_return_sequences: int,
        output_scores: bool = True,
        return_dict_in_generate: bool = True,
        early_stopping: bool = True,
        do_sample: bool = False,
        use_cache: bool = True,
        **_: Any,
    ) -> Any:
        del max_new_tokens, output_scores, return_dict_in_generate, early_stopping, do_sample
        return self._beam_search(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=int(num_beams),
            num_return_sequences=int(num_return_sequences),
            use_cache=bool(use_cache),
        )

    def _beam_search(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]],
        num_beams: int,
        num_return_sequences: int,
        use_cache: bool,
    ) -> Any:
        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        beam_sequences = input_ids.repeat_interleave(num_beams, dim=0)
        beam_attention = attention_mask.repeat_interleave(num_beams, dim=0)
        beam_embeds = self.get_input_embeddings()(beam_sequences)
        beam_position_ids = beam_attention.long().cumsum(dim=-1) - 1
        beam_position_ids = beam_position_ids.masked_fill(beam_attention == 0, 0)
        beam_cache_position = torch.arange(beam_embeds.shape[1], device=device)
        prompt_length = int(input_ids.shape[1])
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.reshape(-1)
        past_key_values = None
        for code_idx in range(self.identifier_len + 1):
            outputs = self.forward_inference(
                attention_mask=beam_attention,
                inputs_embeds=beam_embeds,
                position_ids=beam_position_ids,
                cache_position=beam_cache_position,
                past_key_values=past_key_values,
                code_idx=code_idx,
                prompt_length=prompt_length,
                use_cache=bool(use_cache),
            )
            next_scores = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
            for beam_row in range(beam_sequences.shape[0]):
                allowed = prefix_allowed_tokens_fn(beam_row, beam_sequences[beam_row])
                if allowed:
                    mask = torch.full_like(next_scores[beam_row], -torch.inf)
                    mask[torch.tensor(allowed, dtype=torch.long, device=device)] = 0
                    next_scores[beam_row] = next_scores[beam_row] + mask
            vocab_size = int(next_scores.shape[-1])
            next_scores = next_scores + beam_scores[:, None]
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            top_scores, top_tokens = torch.topk(next_scores, k=num_beams, dim=1)
            next_beam_indices = torch.div(top_tokens, vocab_size, rounding_mode="floor")
            next_token_ids = top_tokens % vocab_size
            flat_offsets = (torch.arange(batch_size, device=device) * num_beams).unsqueeze(1)
            gather_indices = (next_beam_indices + flat_offsets).reshape(-1)
            flat_next_tokens = next_token_ids.reshape(-1, 1)
            beam_sequences = torch.cat([beam_sequences[gather_indices], flat_next_tokens], dim=1)
            beam_attention = torch.cat(
                [outputs.attention_mask[gather_indices], torch.ones((batch_size * num_beams, 1), dtype=beam_attention.dtype, device=device)],
                dim=1,
            )
            next_token_embeds = self.get_input_embeddings()(flat_next_tokens)
            next_position = outputs.position_ids[gather_indices, -1:] + 1
            if use_cache:
                beam_embeds = next_token_embeds
                beam_position_ids = next_position
                beam_cache_position = outputs.cache_position[-1:] + 1
                past_key_values = self._reorder_care_cache(outputs.past_key_values, gather_indices)
            else:
                beam_embeds = torch.cat([outputs.inputs_embeds[gather_indices], next_token_embeds], dim=1)
                beam_position_ids = torch.cat([outputs.position_ids[gather_indices], next_position], dim=1)
                beam_cache_position = torch.arange(beam_embeds.shape[1], device=device)
                past_key_values = None
            beam_scores = top_scores.reshape(-1)
        width = min(int(num_return_sequences), int(num_beams))
        sequences = beam_sequences.view(batch_size, num_beams, -1)[:, :width, :].reshape(batch_size * width, -1)
        sequences_scores = beam_scores.view(batch_size, num_beams)[:, :width].reshape(-1)
        return SimpleNamespace(sequences=sequences, sequences_scores=sequences_scores)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        code_idx: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if position_ids is None or int(position_ids.shape[1]) != int(inputs_embeds.shape[1]):
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        if cache_position is None or int(cache_position.numel()) != int(inputs_embeds.shape[1]):
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        return {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "code_idx": code_idx,
            **kwargs,
        }

    def _reorder_care_cache(self, past_key_values: Any | None, beam_idx: torch.Tensor) -> Any | None:
        if past_key_values is None:
            return None
        if hasattr(self, "_temporary_reorder_cache"):
            return self._temporary_reorder_cache(past_key_values, beam_idx)
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        if isinstance(past_key_values, tuple):
            return tuple(
                tuple(state.index_select(0, beam_idx.to(state.device)) for state in layer)
                for layer in past_key_values
            )
        return past_key_values

    def identifier_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels.ndim != 2:
            raise ValueError(f"CARE labels must be rank-2 [batch, identifier_len + 1], got {tuple(labels.shape)}.")
        if int(labels.shape[1]) < self.identifier_len:
            raise ValueError(
                f"CARE labels are shorter than identifier_len={self.identifier_len}: got shape {tuple(labels.shape)}."
            )

        device = input_ids.device
        input_embeds = self.get_input_embeddings()(input_ids)
        cur_embeds = input_embeds
        cur_attention = attention_mask.to(device=device)
        targets = labels[:, : self.identifier_len].to(device=device)

        for code_idx in range(self.identifier_len):
            n_query = self.query_list[code_idx]
            if n_query > 0:
                query_start = sum(self.query_list[:code_idx])
                lookup = torch.arange(query_start, query_start + n_query, device=device)
                lookup = lookup.unsqueeze(0).expand(cur_embeds.size(0), -1)
                query_embeds = self.query_vector(lookup)
                cur_embeds = torch.cat([cur_embeds, query_embeds], dim=1)
                cur_attention = torch.cat([cur_attention, cur_attention[:, -1:].expand(-1, n_query)], dim=1)

            gold_token = labels[:, code_idx : code_idx + 1].to(device=device)
            gold_embed = self.get_input_embeddings()(gold_token)
            cur_embeds = torch.cat([cur_embeds, gold_embed], dim=1)
            cur_attention = torch.cat([cur_attention, cur_attention[:, -1:]], dim=1)

        if isinstance(self.model, Qwen2ModelAdaptiveAttnCustom):
            if self.progressive_attn:
                self.model.set_care_attention_spec(
                    {
                        "input_len": int(input_ids.shape[1]),
                        "len_identifier": self.identifier_len,
                        "generation_code_idx_start_from_1": self.identifier_len,
                        "query_list": self.query_list,
                        "progressive_list": self.progressive_list,
                    },
                    test=False,
                )
            else:
                self.model.set_care_attention_spec(None, test=False)

        outputs = super().forward(
            input_ids=None,
            inputs_embeds=cur_embeds,
            attention_mask=cur_attention,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

        pred_positions = []
        original_input_len = int(input_ids.shape[1])
        for code_idx in range(self.identifier_len):
            pred_pos = original_input_len + sum(self.query_list[:code_idx]) + code_idx + self.query_list[code_idx] - 1
            pred_positions.append(pred_pos)
        pred_positions_tensor = torch.tensor(pred_positions, dtype=torch.long, device=logits.device)
        selected_logits = logits.index_select(dim=1, index=pred_positions_tensor)
        return selected_logits, targets

    def next_code_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
        code_idx: int,
    ) -> torch.Tensor:
        """Return logits for the next CARE identifier token during constrained beam search."""
        if code_idx < 0 or code_idx >= self.identifier_len:
            raise ValueError(f"code_idx must be in [0, {self.identifier_len}), got {code_idx}.")
        device = input_ids.device
        n_query = self.query_list[code_idx]
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if n_query > 0:
            query_start = sum(self.query_list[:code_idx])
            lookup = torch.arange(query_start, query_start + n_query, device=device)
            lookup = lookup.unsqueeze(0).expand(input_ids.size(0), -1)
            query_embeds = self.query_vector(lookup)
            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask[:, -1:].expand(-1, n_query)], dim=1)

        model_attention = self._build_inference_attention_mask(
            base_attention=attention_mask,
            prompt_length=int(prompt_length),
            code_idx=int(code_idx),
            n_query=int(n_query),
            dtype=inputs_embeds.dtype,
            device=device,
        )
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=model_attention,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[:, -1, :]

    def _build_attention_mask(
        self,
        *,
        base_attention: torch.Tensor,
        original_input_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.progressive_attn:
            return base_attention

        if str(getattr(self.config, "_attn_implementation", "")).lower() in {"flash_attention_2", "flash_attention"}:
            raise RuntimeError(
                "CARE progressive_attn builds a 4D attention mask, which is not compatible with flash attention in many "
                "transformers/Qwen2 versions. Set model.attn_implementation=null/eager or model.progressive_attn=false."
            )

        batch_size = int(base_attention.shape[0])
        total_len = int(base_attention.shape[1])
        min_value = torch.finfo(dtype).min
        mask = torch.full((batch_size, 1, total_len, total_len), min_value, dtype=dtype, device=device)

        base_causal = torch.tril(torch.ones(original_input_len, original_input_len, dtype=torch.bool, device=device))
        mask[:, :, :original_input_len, :original_input_len] = torch.where(
            base_causal,
            torch.zeros((), dtype=dtype, device=device),
            mask[:, :, :original_input_len, :original_input_len],
        )

        reasoning_start = original_input_len
        cur_ptr = original_input_len
        for stage_idx, n_query in enumerate(self.query_list):
            q_start = cur_ptr
            q_end = q_start + n_query
            gold_pos = q_end

            if n_query > 0:
                if self.progressive_list[stage_idx]:
                    for row in range(batch_size):
                        valid_prompt = base_attention[row, :original_input_len].to(dtype=torch.bool).nonzero(as_tuple=True)[0]
                        if valid_prompt.numel() == 0:
                            continue
                        special_idx = valid_prompt[-1:]
                        history_cols = valid_prompt[:-1]
                        if history_cols.numel() > 0:
                            offsets = torch.arange(history_cols.numel(), device=device) % self.identifier_len
                            history_cols = history_cols[offsets < (stage_idx + 1)]
                        visible_cols = torch.cat([history_cols, special_idx])
                        mask[row, :, q_start:q_end, visible_cols] = 0
                else:
                    mask[:, :, q_start:q_end, :original_input_len] = 0
                if q_start > reasoning_start:
                    mask[:, :, q_start:q_end, reasoning_start:q_start] = 0
                q_causal = torch.tril(torch.ones(n_query, n_query, dtype=torch.bool, device=device))
                mask[:, :, q_start:q_end, q_start:q_end] = torch.where(
                    q_causal,
                    torch.zeros((), dtype=dtype, device=device),
                    mask[:, :, q_start:q_end, q_start:q_end],
                )

            mask[:, :, gold_pos, :gold_pos] = 0
            cur_ptr = gold_pos + 1

        valid = base_attention.to(dtype=torch.bool)
        mask = mask.masked_fill(~valid[:, None, None, :], min_value)
        return mask

    def _build_inference_attention_mask(
        self,
        *,
        base_attention: torch.Tensor,
        prompt_length: int,
        code_idx: int,
        n_query: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.progressive_attn:
            return base_attention
        batch_size, total_len = int(base_attention.shape[0]), int(base_attention.shape[1])
        min_value = torch.finfo(dtype).min
        mask = torch.full((batch_size, 1, total_len, total_len), min_value, dtype=dtype, device=device)
        causal = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
        mask[:, :, :, :] = torch.where(causal, torch.zeros((), dtype=dtype, device=device), mask)

        q_start = total_len - n_query
        q_end = total_len
        if n_query > 0:
            mask[:, :, q_start:q_end, :] = min_value
            for row in range(batch_size):
                valid_prompt = base_attention[row, :prompt_length].to(dtype=torch.bool).nonzero(as_tuple=True)[0]
                if valid_prompt.numel() == 0:
                    continue
                special_idx = valid_prompt[-1:]
                history_cols = valid_prompt[:-1]
                if self.progressive_list[code_idx] and history_cols.numel() > 0:
                    offsets = torch.arange(history_cols.numel(), device=device) % self.identifier_len
                    history_cols = history_cols[offsets < (code_idx + 1)]
                visible_cols = torch.cat([history_cols, special_idx, torch.arange(prompt_length, q_start, device=device)])
                mask[row, :, q_start:q_end, visible_cols] = 0
                local = torch.tril(torch.ones(n_query, n_query, dtype=torch.bool, device=device))
                mask[row, :, q_start:q_end, q_start:q_end] = torch.where(
                    local,
                    torch.zeros((), dtype=dtype, device=device),
                    mask[row, :, q_start:q_end, q_start:q_end],
                )
        valid = base_attention.to(dtype=torch.bool)
        return mask.masked_fill(~valid[:, None, None, :], min_value)

    def _query_diversity_loss(self) -> torch.Tensor:
        if self.query_vector.weight.shape[0] <= 1:
            return self.query_vector.weight.new_zeros(())
        query = F.normalize(self.query_vector.weight, dim=1)
        sim = torch.matmul(query, query.T)
        eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        return sim.masked_fill(eye, 0.0).mean()


class CAREModel(BaseRetrievalModel):
    """RecBole3.0 adapter for CARE."""

    config: CAREConfig

    def __init__(self, config: CAREConfig):
        super().__init__(config)
        self._tokenizer: Any | None = None
        self._codec: CARETokenCodec | None = None
        self._care_lm: CARECausalLM | None = None
        self._sid_token_cache: tuple[dict[tuple[int, ...], list[int]], dict[tuple[int, ...], int]] | None = None

    def ensure_initialized(self, prepared_data) -> None:
        self._ensure_initialized(prepared_data)

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return CARETrainCollator(
            self.config,
            prepared_data,
            tokenizer=self._require_tokenizer(),
            codec=self._require_codec(),
        )

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return CAREEvalCollator(
            self.config,
            prepared_data,
            tokenizer=self._require_tokenizer(),
            codec=self._require_codec(),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        care_lm = self._require_care_lm()
        return care_lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

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
        if k <= 0:
            batch_size = int(model_inputs["input_ids"].shape[0])
            return torch.empty((batch_size, 0), dtype=torch.long, device=model_inputs["input_ids"].device)
        if candidate_item_ids is not None:
            raise NotImplementedError("CARE currently supports full evaluation only, not sampled candidate evaluation.")

        tokenizer = self._require_tokenizer()
        codec = self._require_codec()
        care_lm = self._require_care_lm()
        _, token_tuple_to_item = self._sid_token_maps()
        device = model_inputs["input_ids"].device
        batch_size = int(model_inputs["input_ids"].shape[0])
        beam_width = max(int(k), int(self.config.num_beams))
        prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(
            tokenizer=tokenizer,
            item_code_texts=[codec.item_code_text(item_id) for item_id in codec.all_item_ids],
            special_token_for_answer=self.config.special_token_for_answer,
        )
        generated = care_lm.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=codec.identifier_len + 1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            do_sample=False,
            use_cache=True,
        )
        sequences = generated.sequences.reshape(batch_size, beam_width, -1)
        prompt_len = int(model_inputs["input_ids"].shape[1])
        token_tuples = sequences[:, :, prompt_len : prompt_len + codec.identifier_len].detach().cpu()
        excluded = self._excluded_item_sets(exclude_item_ids, exclude_mask, batch_size=batch_size)
        predictions: list[list[int]] = []
        for row_index in range(batch_size):
            selected: list[int] = []
            selected_set: set[int] = set()
            for beam_index in range(beam_width):
                item_id = token_tuple_to_item.get(tuple(int(v) for v in token_tuples[row_index, beam_index].tolist()))
                if item_id is None or item_id in selected_set or item_id in excluded[row_index]:
                    continue
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) == int(k):
                    break
            for item_id in codec.all_item_ids:
                if len(selected) == int(k):
                    break
                if item_id not in selected_set and item_id not in excluded[row_index]:
                    selected.append(int(item_id))
                    selected_set.add(int(item_id))
            predictions.append(selected[: int(k)])
        return torch.tensor(predictions, dtype=torch.long, device=device)

    def _ensure_initialized(self, prepared_data: CAREModelDataset) -> None:
        if not hasattr(prepared_data, "care_codec"):
            raise RuntimeError("CAREModel requires CAREModelDataset prepared data.")
        codec = prepared_data.care_codec
        if self._codec is not None:
            if self._codec.identifier_len != codec.identifier_len:
                raise ValueError("CAREModel was already initialized with an incompatible identifier length.")
            return

        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("CARE requires `transformers`. Install a compatible transformers version.") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=int(self.config.model_max_length),
            padding_side="left",
            trust_remote_code=bool(self.config.trust_remote_code),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        new_tokens = list(codec.all_new_tokens)
        if self.config.special_token_for_answer:
            new_tokens.append(self.config.special_token_for_answer)
        tokenizer.add_tokens(sorted(set(new_tokens)))

        care_lm = CARECausalLM.from_care_pretrained(
            self.config.base_model,
            query_list=self.config.query_list,
            progressive_list=self.config.progressive_list,
            progressive_attn=bool(self.config.progressive_attn),
            attention_strategy=self.config.attention_strategy,
            query_div_scale=float(self.config.query_div_scale),
            identifier_len=codec.identifier_len,
            torch_dtype=_resolve_torch_dtype(self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
            trust_remote_code=bool(self.config.trust_remote_code),
            low_cpu_mem_usage=bool(self.config.low_cpu_mem_usage),
        )
        care_lm.resize_token_embeddings(len(tokenizer))
        care_lm.config.pad_token_id = tokenizer.pad_token_id
        care_lm.config.eos_token_id = tokenizer.eos_token_id

        self._codec = codec
        self._tokenizer = tokenizer
        self._care_lm = care_lm

    def _sid_token_maps(self) -> tuple[dict[tuple[int, ...], list[int]], dict[tuple[int, ...], int]]:
        if self._sid_token_cache is not None:
            return self._sid_token_cache
        tokenizer = self._require_tokenizer()
        codec = self._require_codec()
        prefix_allowed: dict[tuple[int, ...], set[int]] = {}
        token_tuple_to_item: dict[tuple[int, ...], int] = {}
        for item_id in codec.all_item_ids:
            token_ids: list[int] = []
            for code in codec.item_codes(item_id):
                encoded = tokenizer.encode(code, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError(f"CARE SID code token {code!r} must tokenize to exactly one token, got {encoded}.")
                token_ids.append(int(encoded[0]))
            token_tuple = tuple(token_ids)
            if token_tuple in token_tuple_to_item:
                raise ValueError(
                    "CARE sid_file contains tokenization-colliding identifiers: "
                    f"items {token_tuple_to_item[token_tuple]} and {item_id} both map to {token_tuple}."
                )
            token_tuple_to_item[token_tuple] = int(item_id)
            for idx, token_id in enumerate(token_tuple):
                prefix_allowed.setdefault(token_tuple[:idx], set()).add(token_id)
        frozen_allowed = {prefix: sorted(tokens) for prefix, tokens in prefix_allowed.items()}
        self._sid_token_cache = (frozen_allowed, token_tuple_to_item)
        return self._sid_token_cache

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
        for row in range(min(batch_size, int(ids.shape[0]))):
            excluded[row] = {
                int(item_id)
                for item_id, keep in zip(ids[row].tolist(), mask[row].tolist(), strict=False)
                if keep
            }
        return excluded

    @staticmethod
    def _mask_excluded_scores(
        *,
        scores: torch.Tensor,
        all_item_ids: torch.Tensor,
        exclude_item_ids: torch.Tensor | None,
        exclude_mask: torch.Tensor | None,
    ) -> None:
        if exclude_item_ids is None or exclude_mask is None or exclude_item_ids.numel() == 0:
            return
        id_to_col = {int(item_id): col for col, item_id in enumerate(all_item_ids.detach().cpu().tolist())}
        excluded_ids = exclude_item_ids.detach().cpu()
        excluded_mask = exclude_mask.detach().cpu().to(dtype=torch.bool)

        for row in range(min(scores.shape[0], excluded_ids.shape[0])):
            for item_id, keep in zip(excluded_ids[row].tolist(), excluded_mask[row].tolist(), strict=False):
                if keep:
                    col = id_to_col.get(int(item_id))
                    if col is not None:
                        scores[row, col] = -torch.inf

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("CARE tokenizer is not initialized. Call build_train_collator/build_eval_collator first.")
        return self._tokenizer

    def _require_codec(self) -> CARETokenCodec:
        if self._codec is None:
            raise RuntimeError("CARE codec is not initialized. Call build_train_collator/build_eval_collator first.")
        return self._codec

    def _require_care_lm(self) -> CARECausalLM:
        if self._care_lm is None:
            raise RuntimeError("CARE model is not initialized. Call build_train_collator/build_eval_collator first.")
        return self._care_lm


def _resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | str | None:
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


__all__ = ["CAREModel"]