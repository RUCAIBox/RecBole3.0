from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.rpg.config import RPGConfig
from recbole3.model.rpg.data import (
    RPG_ATTENTION_MASK,
    RPG_INPUT_IDS,
    RPG_LABELS,
    RPG_SEQ_LENS,
    RPGEvalCollator,
    RPGTrainCollator,
)
from recbole3.model.rpg.tokenizer import (
    RPG_IGNORED_LABEL,
    RPG_ITEM_ID_OFFSET,
    RPGSemanticTokenizer,
)


class ResBlock(nn.Module):
    """Residual prediction head used by the original RPG implementation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class RPGModel(BaseRetrievalModel):
    """RPG model adapted to RecBole3's retrieval interface."""

    def __init__(self, config: RPGConfig):
        super().__init__(config)
        self._num_items: int | None = None
        self._n_pred_head: int | None = None
        self._codebook_size: int | None = None
        self._eos_token: int | None = None
        self._vocab_size: int | None = None
        self.gpt2: Any | None = None
        self.pred_heads: nn.Sequential | None = None
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=RPG_IGNORED_LABEL)
        self.temperature = float(config.temperature)
        self.generate_w_decoding_graph = bool(config.use_decoding_graph)
        self.init_flag = False
        self.adjacency: torch.Tensor | None = None

    @property
    def n_parameters(self) -> str:
        total_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        emb_params = sum(
            parameter.numel()
            for parameter in self._gpt2_module().get_input_embeddings().parameters()
            if parameter.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return RPGTrainCollator(self.config, prepared_data=prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return RPGEvalCollator(self.config, prepared_data=prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor], return_loss: bool = True) -> dict[str, torch.Tensor]:
        gpt2 = self._gpt2_module()
        item_id2tokens = self._item_id2tokens()
        input_ids = batch[RPG_INPUT_IDS].to(device=item_id2tokens.device, dtype=torch.long)
        attention_mask = batch[RPG_ATTENTION_MASK].to(device=item_id2tokens.device, dtype=torch.long)

        input_tokens = item_id2tokens[input_ids]
        input_embs = gpt2.wte(input_tokens).mean(dim=-2)
        outputs = gpt2(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
        )
        final_states = torch.cat(
            [
                self._pred_heads_module()[head_index](outputs.last_hidden_state).unsqueeze(-2)
                for head_index in range(self._require_n_pred_head())
            ],
            dim=-2,
        )
        result = {
            "last_hidden_state": outputs.last_hidden_state,
            "final_states": final_states,
        }
        if return_loss and RPG_LABELS in batch:
            result["loss"] = self._compute_rpg_loss(batch, final_states)
        return result

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return outputs["loss"]

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
            batch_size = int(model_inputs[RPG_INPUT_IDS].shape[0])
            return torch.empty((batch_size, 0), dtype=torch.long, device=self._device())

        if candidate_item_ids is not None:
            return self._predict_from_candidates(model_inputs, candidate_item_ids, k=k)

        graph_width = min(max(1, int(self.config.num_beams)), self._require_num_items())
        if self.generate_w_decoding_graph and k <= graph_width and exclude_item_ids is None and exclude_mask is None:
            generated = self.generate(model_inputs, n_return_sequences=k)
            return generated.squeeze(-1).to(dtype=torch.long) - RPG_ITEM_ID_OFFSET

        scores = self._score_all_items(model_inputs)
        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
            history_mask = torch.zeros_like(scores, dtype=torch.bool)
            history_mask.scatter_(
                1,
                exclude_item_ids.to(device=scores.device, dtype=torch.long),
                exclude_mask.to(device=scores.device, dtype=torch.bool),
            )
            scores = scores.masked_fill(history_mask, float("-inf"))
        return torch.topk(scores, k=k, dim=1).indices.to(dtype=torch.long)

    def generate(self, batch: Mapping[str, torch.Tensor], n_return_sequences: int = 1) -> torch.Tensor:
        token_logits = self._token_logits(batch)
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            return self.graph_propagation(token_logits=token_logits, n_return_sequences=n_return_sequences)

        scores = self._score_token_logits_for_model_item_ids(
            token_logits,
            torch.arange(1, self._require_num_items() + 1, device=token_logits.device, dtype=torch.long),
        )
        preds = scores.topk(n_return_sequences, dim=-1).indices + RPG_ITEM_ID_OFFSET
        return preds.unsqueeze(-1)

    def build_ii_sim_mat(self) -> torch.Tensor:
        n_items_with_padding = self._require_num_items() + RPG_ITEM_ID_OFFSET
        n_digit = self._require_n_pred_head()
        codebook_size = self._require_codebook_size()
        token_embs = self._gpt2_module().wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)

        item_id2tokens = self._item_id2tokens().to(device=token_embs.device)
        chunk_size = max(1, int(self.config.chunk_size))
        item_item_sim = torch.zeros(
            (n_items_with_padding, n_items_with_padding),
            device=item_id2tokens.device,
            dtype=torch.float32,
        )
        for i_start in range(1, n_items_with_padding, chunk_size):
            i_end = min(i_start + chunk_size, n_items_with_padding)
            tokens_i = item_id2tokens[i_start:i_end]
            for j_start in range(1, n_items_with_padding, chunk_size):
                j_end = min(j_start + chunk_size, n_items_with_padding)
                tokens_j = item_id2tokens[j_start:j_end]
                sum_block = torch.zeros(
                    (i_end - i_start, j_end - j_start),
                    device=item_id2tokens.device,
                    dtype=torch.float32,
                )
                for digit in range(n_digit):
                    row_inds = tokens_i[:, digit] - digit * codebook_size - 1
                    col_inds = tokens_j[:, digit] - digit * codebook_size - 1
                    temp = token_sims_01[digit].index_select(0, row_inds)
                    sum_block += temp.index_select(1, col_inds)
                item_item_sim[i_start:i_end, j_start:j_end] = sum_block / n_digit
        item_item_sim[:, 0] = float("-inf")
        return item_item_sim

    def build_adjacency_list(self, item_item_sim: torch.Tensor) -> torch.Tensor:
        k = min(max(1, int(self.config.n_edges)), max(1, int(item_item_sim.shape[-1]) - 1))
        return torch.topk(item_item_sim, k=k, dim=-1).indices

    def init_graph(self) -> None:
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)

    def graph_propagation(self, token_logits: torch.Tensor, n_return_sequences: int) -> torch.Tensor:
        num_items = self._require_num_items()
        batch_size = int(token_logits.shape[0])
        beam_width = min(max(1, int(self.config.num_beams)), num_items)
        return_width = min(int(n_return_sequences), beam_width)

        topk_nodes_sorted = torch.randint(
            1,
            num_items + 1,
            (batch_size, beam_width),
            dtype=torch.long,
            device=token_logits.device,
        )

        adjacency = self._adjacency_tensor().to(device=token_logits.device)
        for _ in range(int(self.config.propagation_steps)):
            all_neighbors = adjacency[topk_nodes_sorted].view(batch_size, -1)
            next_nodes = []
            for batch_id in range(batch_size):
                neighbors = torch.unique(all_neighbors[batch_id])
                neighbors = neighbors[(neighbors > 0) & (neighbors <= num_items)]
                if neighbors.numel() == 0:
                    neighbors = topk_nodes_sorted[batch_id]

                scores = self._score_token_logits_for_model_item_ids(
                    token_logits[batch_id : batch_id + 1],
                    neighbors.view(1, -1),
                ).squeeze(0)
                current_beam_width = min(beam_width, int(scores.shape[0]))
                selected = neighbors[torch.topk(scores, current_beam_width).indices]
                if current_beam_width < beam_width:
                    selected = F.pad(selected, (0, beam_width - current_beam_width), value=int(selected[-1].item()))
                next_nodes.append(selected)
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        return topk_nodes_sorted[:, :return_width].unsqueeze(-1)

    def _compute_rpg_loss(self, batch: Mapping[str, torch.Tensor], final_states: torch.Tensor) -> torch.Tensor:
        labels = batch[RPG_LABELS].to(device=final_states.device, dtype=torch.long)
        label_mask = labels.view(-1) != RPG_IGNORED_LABEL
        if not torch.any(label_mask):
            return final_states.sum() * 0.0

        n_pred_head = self._require_n_pred_head()
        selected_states = final_states.view(-1, n_pred_head, int(self.config.n_embd))[label_mask]
        selected_states = F.normalize(selected_states, dim=-1)
        selected_states_by_head = torch.chunk(selected_states, n_pred_head, dim=1)

        token_emb = self._gpt2_module().wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, n_pred_head, dim=0)

        token_logits = [
            torch.matmul(selected_states_by_head[head_index].squeeze(dim=1), token_embs[head_index].T) / self.temperature
            for head_index in range(n_pred_head)
        ]
        token_labels = self._item_id2tokens()[labels.view(-1)[label_mask]]
        losses = [
            self.loss_fct(
                token_logits[head_index],
                token_labels[:, head_index] - head_index * self._require_codebook_size() - 1,
            )
            for head_index in range(n_pred_head)
        ]
        return torch.mean(torch.stack(losses))

    def _predict_from_candidates(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        candidate_item_ids: torch.Tensor,
        *,
        k: int,
    ) -> torch.Tensor:
        candidate_item_ids = candidate_item_ids.to(device=self._device(), dtype=torch.long)
        self._validate_rec_item_ids(candidate_item_ids)
        token_logits = self._token_logits(model_inputs)
        candidate_model_item_ids = candidate_item_ids + RPG_ITEM_ID_OFFSET
        scores = self._score_token_logits_for_model_item_ids(token_logits, candidate_model_item_ids)
        topk_indices = torch.topk(scores, k=k, dim=1).indices
        return torch.gather(candidate_item_ids, 1, topk_indices)

    def _score_all_items(self, model_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        token_logits = self._token_logits(model_inputs)
        model_item_ids = torch.arange(
            1,
            self._require_num_items() + 1,
            device=token_logits.device,
            dtype=torch.long,
        )
        return self._score_token_logits_for_model_item_ids(token_logits, model_item_ids)

    def _score_token_logits_for_model_item_ids(
        self,
        token_logits: torch.Tensor,
        model_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        item_id2tokens = self._item_id2tokens().to(device=token_logits.device)
        item_tokens = item_id2tokens[model_item_ids.to(device=token_logits.device, dtype=torch.long)]
        token_indices = item_tokens - 1
        if model_item_ids.ndim == 1:
            expanded_logits = token_logits.unsqueeze(1).expand(-1, int(model_item_ids.shape[0]), -1)
            gather_indices = token_indices.unsqueeze(0).expand(int(token_logits.shape[0]), -1, -1)
        elif model_item_ids.ndim == 2:
            expanded_logits = token_logits.unsqueeze(1).expand(-1, int(model_item_ids.shape[1]), -1)
            gather_indices = token_indices
        else:
            raise ValueError(f"RPG model_item_ids must be 1D or 2D, got shape {tuple(model_item_ids.shape)}.")
        return torch.gather(expanded_logits, dim=-1, index=gather_indices).mean(dim=-1)

    def _token_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.forward(batch, return_loss=False)
        final_states = outputs["final_states"]
        n_pred_head = self._require_n_pred_head()
        seq_lens = batch[RPG_SEQ_LENS].to(device=final_states.device, dtype=torch.long)
        gather_positions = torch.clamp(seq_lens - 1, min=0, max=max(0, int(final_states.shape[1]) - 1))
        states = final_states.gather(
            dim=1,
            index=gather_positions.view(-1, 1, 1, 1).expand(-1, 1, n_pred_head, int(self.config.n_embd)),
        )
        states = F.normalize(states, dim=-1)

        token_emb = self._gpt2_module().wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, n_pred_head, dim=0)
        logits = [
            torch.matmul(states[:, 0, head_index, :], token_embs[head_index].T) / self.temperature
            for head_index in range(n_pred_head)
        ]
        return torch.cat([F.log_softmax(logit, dim=-1) for logit in logits], dim=-1)

    def _ensure_initialized(self, prepared_data) -> None:
        num_items = int(prepared_data.get_num_items())
        if self._num_items is not None:
            return

        tokenizer = RPGSemanticTokenizer(self.config, prepared_data)
        from transformers import GPT2Config, GPT2Model

        self._num_items = num_items
        self._n_pred_head = tokenizer.n_digit
        self._codebook_size = tokenizer.codebook_size
        self._eos_token = tokenizer.eos_token
        self._vocab_size = tokenizer.vocab_size
        if "item_id2tokens" in self._buffers:
            self._buffers["item_id2tokens"] = tokenizer.item_id2tokens
        else:
            self.register_buffer("item_id2tokens", tokenizer.item_id2tokens, persistent=False)

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=int(self.config.n_embd),
            n_layer=int(self.config.n_layer),
            n_head=int(self.config.n_head),
            n_inner=int(self.config.n_inner),
            activation_function=self.config.activation_function,
            resid_pdrop=float(self.config.resid_pdrop),
            embd_pdrop=float(self.config.embd_pdrop),
            attn_pdrop=float(self.config.attn_pdrop),
            layer_norm_epsilon=float(self.config.layer_norm_epsilon),
            initializer_range=float(self.config.initializer_range),
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)
        self.pred_heads = nn.Sequential(*[ResBlock(int(self.config.n_embd)) for _ in range(tokenizer.n_digit)])

    def _validate_rec_item_ids(self, item_ids: torch.Tensor) -> None:
        num_items = self._require_num_items()
        if item_ids.numel() == 0:
            return
        invalid = (item_ids < 0) | (item_ids >= num_items)

    def _require_num_items(self) -> int:
        return self._num_items

    def _require_n_pred_head(self) -> int:
        return self._n_pred_head

    def _require_codebook_size(self) -> int:
        return self._codebook_size

    def _gpt2_module(self) -> Any:
        return self.gpt2

    def _pred_heads_module(self) -> nn.Sequential:
        return self.pred_heads

    def _item_id2tokens(self) -> torch.Tensor:
        item_id2tokens = self._buffers.get("item_id2tokens")
        return item_id2tokens

    def _adjacency_tensor(self) -> torch.Tensor:
        return self.adjacency

    def _device(self) -> torch.device:
        return self._item_id2tokens().device


__all__ = ["RPGModel", "ResBlock"]
