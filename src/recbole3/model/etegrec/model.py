from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.etegrec.config import ETEGRecConfig
from recbole3.model.etegrec.data import ETEGRecEvalCollator, ETEGRecModelDataset, ETEGRecTrainCollator
from recbole3.model.etegrec.layers import MLPLayers
from recbole3.model.etegrec.vq import RQVAE


@dataclass
class ETEGRecForwardOutput:
    logits: torch.FloatTensor | None = None
    seq_latents: torch.FloatTensor | None = None
    seq_project_latents: torch.FloatTensor | None = None
    dec_latents: torch.FloatTensor | None = None


class ETEGRecModel(BaseRetrievalModel):
    """ETEGRec generative retrieval model with an internal RQVAE tokenizer."""

    config: ETEGRecConfig

    def __init__(self, config: ETEGRecConfig):
        super().__init__(config)
        self.semantic_embedding: nn.Embedding | None = None
        self.token_embeddings: nn.ModuleList | None = None
        self.enc_adapter: MLPLayers | None = None
        self.dec_adapter: MLPLayers | None = None
        self._t5: nn.Module | None = None
        self._rqvae: RQVAE | None = None
        self._num_items: int | None = None
        self._item_codes: torch.Tensor | None = None
        self._code_to_item_ids: dict[tuple[int, ...], list[int]] = {}

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self.ensure_initialized(prepared_data)
        return ETEGRecTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self.ensure_initialized(prepared_data)
        return ETEGRecEvalCollator(self.config, prepared_data)

    def ensure_initialized(self, prepared_data) -> None:
        if not isinstance(prepared_data, ETEGRecModelDataset):
            raise TypeError("ETEGRecModel requires ETEGRecModelDataset prepared data.")
        if not hasattr(prepared_data, "semantic_embeddings"):
            raise RuntimeError("ETEGRec prepared data is missing semantic_embeddings.")
        semantic_embeddings = prepared_data.semantic_embeddings
        num_items = int(prepared_data.get_num_items())
        if int(semantic_embeddings.shape[0]) != num_items:
            raise ValueError(
                "ETEGRec semantic embedding row count changed after model-data construction. "
                f"Expected {num_items}, got {int(semantic_embeddings.shape[0])}."
            )
        if int(semantic_embeddings.shape[1]) != int(self.config.semantic_hidden_size):
            raise ValueError(
                "ETEGRec semantic embedding dimension does not match model.semantic_hidden_size. "
                f"Expected {self.config.semantic_hidden_size}, got {int(semantic_embeddings.shape[1])}."
            )
        if self.semantic_embedding is not None:
            if self._num_items != num_items:
                raise ValueError("ETEGRecModel was already initialized with a different item count.")
            return

        embedding = nn.Embedding(num_items + 1, int(self.config.semantic_hidden_size), padding_idx=0)
        with torch.no_grad():
            embedding.weight.zero_()
            embedding.weight[1:].copy_(semantic_embeddings.to(dtype=embedding.weight.dtype))
        embedding.requires_grad_(False)
        self.semantic_embedding = embedding
        self._num_items = num_items
        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(int(self.config.code_num), int(self.config.d_model)) for _ in range(int(self.config.code_length))]
        )
        self.enc_adapter = MLPLayers([int(self.config.d_model), int(self.config.e_dim)])
        self.dec_adapter = MLPLayers([int(self.config.d_model), int(self.config.semantic_hidden_size)])
        self._t5 = self._build_t5()
        self._rqvae = self._build_rqvae()
        self.apply(self._init_weights)
        self._load_rqvae_if_configured()
        # Keep externally supplied semantic vectors intact after module-wide initialization.
        with torch.no_grad():
            self.semantic_embedding.weight.zero_()
            self.semantic_embedding.weight[1:].copy_(semantic_embeddings.to(dtype=self.semantic_embedding.weight.dtype))
        self.semantic_embedding.requires_grad_(False)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        mode: str = "rec",
        use_alignment: bool = False,
        rec_code_loss: float = 1.0,
        rec_kl_loss: float = 0.0,
        rec_dec_cl_loss: float = 0.0,
        id_vq_loss: float = 1.0,
        id_code_loss: float = 0.0,
        id_kl_loss: float = 0.0,
        id_dec_cl_loss: float = 0.0,
    ) -> dict[str, Any]:
        if mode == "rqvae":
            parts = self.compute_rqvae_loss(batch["targets"])
            return {key: value if key == "loss" else value.detach() for key, value in parts.items()}
        if mode == "tokenizer_loss":
            rqvae_parts = self.compute_rqvae_loss(batch["targets"])
            vq_loss = rqvae_parts["loss"]
            zero = vq_loss * 0
            code_loss = zero
            kl_loss = zero
            dec_cl_loss = zero
            if use_alignment:
                alignment_outputs = self.forward(batch)
                alignment_parts = self.compute_alignment_loss_parts(batch, alignment_outputs)
                code_loss = alignment_parts["code_loss"]
                kl_loss = alignment_parts["kl_loss"]
                dec_cl_loss = alignment_parts["dec_cl_loss"]
            total_loss = (
                float(id_vq_loss) * vq_loss
                + float(id_code_loss) * code_loss
                + float(id_kl_loss) * kl_loss
                + float(id_dec_cl_loss) * dec_cl_loss
            )
            return {
                "loss": total_loss,
                "vq_loss": vq_loss.detach(),
                "code_loss": code_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "dec_cl_loss": dec_cl_loss.detach(),
                "recon_loss": rqvae_parts["recon_loss"].detach(),
                "rq_loss": rqvae_parts["rq_loss"].detach(),
            }
        if mode not in {"rec", "rec_loss", "finetune"}:
            raise ValueError(f"Unknown ETEGRec forward mode: {mode!r}.")

        input_item_tokens = batch["input_ids"]
        target_item_tokens = batch.get("targets")
        if target_item_tokens is None:
            raise ValueError("ETEGRec forward requires train batches with `targets`.")

        input_codes = self._item_tokens_to_code_tokens(input_item_tokens)
        labels = self._item_tokens_to_code_tokens(target_item_tokens).reshape(target_item_tokens.shape[0], -1)
        input_codes = input_codes.reshape(input_codes.shape[0], -1)
        attention_mask = input_codes.ne(-1)
        outputs = self._forward_code_tokens(input_codes=input_codes, attention_mask=attention_mask, labels=labels)
        loss = F.cross_entropy(outputs.logits.reshape(-1, int(self.config.code_num)), labels.detach().reshape(-1))
        if mode == "finetune":
            return {"loss": loss}
        if mode == "rec_loss":
            forward_outputs = {"loss": loss, "outputs": outputs}
            zero = loss * 0
            kl_loss = zero
            dec_cl_loss = zero
            if use_alignment:
                alignment_parts = self.compute_alignment_loss_parts(batch, forward_outputs)
                kl_loss = alignment_parts["kl_loss"]
                dec_cl_loss = alignment_parts["dec_cl_loss"]
            total_loss = (
                float(rec_code_loss) * loss
                + float(rec_kl_loss) * kl_loss
                + float(rec_dec_cl_loss) * dec_cl_loss
            )
            return {
                "loss": total_loss,
                "code_loss": loss.detach(),
                "kl_loss": kl_loss.detach(),
                "dec_cl_loss": dec_cl_loss.detach(),
            }
        return {"loss": loss, "logits": outputs.logits, "outputs": outputs, "labels": labels}

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> torch.Tensor:
        del batch
        loss = outputs["loss"]
        if loss is None:
            raise ValueError("ETEGRec forward outputs did not include a loss.")
        return loss

    @staticmethod
    def compute_discrete_contrastive_loss_kl(x_logits: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
        code_num = x_logits.size(-1)
        x_log_probs = F.log_softmax(x_logits.reshape(-1, code_num), dim=-1)
        y_log_probs = F.log_softmax(y_logits.reshape(-1, code_num), dim=-1)
        return F.kl_div(x_log_probs, y_log_probs, reduction="batchmean", log_target=True)

    @staticmethod
    def compute_contrastive_loss(
        query_embeds: torch.Tensor,
        semantic_embeds: torch.Tensor,
        *,
        temperature: float = 0.07,
        sim: str = "cos",
    ) -> torch.Tensor:
        if query_embeds.ndim != 2 or semantic_embeds.ndim != 2:
            raise ValueError("ETEGRec contrastive loss expects 2D query and semantic embeddings.")
        if query_embeds.shape != semantic_embeds.shape:
            raise ValueError(
                "ETEGRec contrastive loss expects matching query and semantic shapes, "
                f"got {tuple(query_embeds.shape)} and {tuple(semantic_embeds.shape)}."
            )
        if str(sim).lower() == "cos":
            query_embeds = F.normalize(query_embeds, dim=-1)
            semantic_embeds = F.normalize(semantic_embeds, dim=-1)
        labels = torch.arange(query_embeds.size(0), dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(query_embeds, semantic_embeds.transpose(0, 1)) / float(temperature)
        return F.cross_entropy(similarities, labels)

    @staticmethod
    def _first_occurrence_indices(item_tokens: torch.Tensor) -> torch.Tensor:
        flat_tokens = item_tokens.reshape(-1)
        seen: set[int] = set()
        indices: list[int] = []
        for index, token in enumerate(flat_tokens.detach().cpu().tolist()):
            token_id = int(token)
            if token_id in seen:
                continue
            seen.add(token_id)
            indices.append(index)
        return torch.tensor(indices, dtype=torch.long, device=item_tokens.device)

    def compute_alignment_loss_parts(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        target_item_tokens = batch.get("targets")
        if target_item_tokens is None:
            raise ValueError("ETEGRec alignment loss requires train batches with `targets`.")
        target_flatten = target_item_tokens.reshape(-1).long()
        target_flatten = target_flatten[target_flatten > 0]
        if target_flatten.numel() == 0:
            zero = self._semantic_embedding().weight.sum() * 0
            return {"code_loss": self.compute_loss(batch, outputs), "kl_loss": zero, "dec_cl_loss": zero}

        forward_outputs = outputs.get("outputs")
        if not isinstance(forward_outputs, ETEGRecForwardOutput):
            raise ValueError("ETEGRec alignment loss requires `outputs` from ETEGRecModel.forward(...).")
        if forward_outputs.seq_project_latents is None or forward_outputs.dec_latents is None:
            raise ValueError("ETEGRec forward outputs are missing alignment latents.")
        if int(target_flatten.numel()) != int(forward_outputs.seq_project_latents.shape[0]):
            raise ValueError(
                "ETEGRec alignment expects one target item per sequence, "
                f"got {int(target_flatten.numel())} targets and {int(forward_outputs.seq_project_latents.shape[0])} sequences."
            )

        target_semantic_embs = self._semantic_embedding()(target_flatten)
        target_recon_embs, _, _, _, target_code_logits = self._rqvae_module()(target_semantic_embs)
        _, _, _, _, seq_code_logits = self._rqvae_module().rq(forward_outputs.seq_project_latents)
        first_indices = self._first_occurrence_indices(target_flatten)

        seq_unique_logits = seq_code_logits[first_indices]
        target_unique_logits = target_code_logits[first_indices]
        kl_loss = self.compute_discrete_contrastive_loss_kl(seq_unique_logits, target_unique_logits)
        kl_loss = kl_loss + self.compute_discrete_contrastive_loss_kl(target_unique_logits, seq_unique_logits)

        target_unique_recon = target_recon_embs[first_indices]
        dec_unique_latents = forward_outputs.dec_latents[first_indices]
        dec_cl_loss = self.compute_contrastive_loss(
            target_unique_recon,
            dec_unique_latents,
            temperature=float(self.config.tau),
        )
        dec_cl_loss = dec_cl_loss + self.compute_contrastive_loss(
            dec_unique_latents,
            target_unique_recon,
            temperature=float(self.config.tau),
        )
        return {
            "code_loss": self.compute_loss(batch, outputs),
            "kl_loss": kl_loss,
            "dec_cl_loss": dec_cl_loss,
        }

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
            raise NotImplementedError("ETEGRec stage 5 supports only full evaluation, not sampled evaluation.")
        input_ids = model_inputs["input_ids"]
        if k <= 0:
            return torch.empty((int(input_ids.shape[0]), 0), dtype=torch.long, device=input_ids.device)
        beam_width = max(int(self.config.num_beams), int(k))
        generated_codes = self._generate_code_tokens(
            input_ids=input_ids,
            num_beams=beam_width,
            num_return_sequences=beam_width,
        )
        excluded = self._excluded_item_sets(exclude_item_ids, exclude_mask, batch_size=int(input_ids.shape[0]))
        fallback_item_ids = self._fallback_item_ids()

        predictions: list[list[int]] = []
        generated_codes_cpu = generated_codes.detach().cpu()
        for row_index in range(int(input_ids.shape[0])):
            selected: list[int] = []
            selected_set: set[int] = set()
            for beam_index in range(beam_width):
                code_tuple = tuple(int(token) for token in generated_codes_cpu[row_index, beam_index].tolist())
                for item_id in self._code_to_item_ids.get(code_tuple, ()):
                    zero_based_item_id = int(item_id) - 1
                    if zero_based_item_id in selected_set or zero_based_item_id in excluded[row_index]:
                        continue
                    selected.append(zero_based_item_id)
                    selected_set.add(zero_based_item_id)
                    break
                if len(selected) == k:
                    break
            if len(selected) < k:
                for item_id in fallback_item_ids:
                    if item_id in selected_set or item_id in excluded[row_index]:
                        continue
                    selected.append(item_id)
                    selected_set.add(item_id)
                    if len(selected) == k:
                        break
            if len(selected) < k:
                raise ValueError(f"ETEGRec could not produce {k} predictions from {len(fallback_item_ids)} items.")
            predictions.append(selected)
        return torch.tensor(predictions, dtype=torch.long, device=input_ids.device)

    def recommender_parameters(self):
        rqvae_ids = {id(param) for param in self._rqvae_module().parameters()}
        semantic_ids = {id(param) for param in self._semantic_embedding().parameters()}
        for parameter in self.parameters():
            if id(parameter) not in rqvae_ids and id(parameter) not in semantic_ids:
                yield parameter

    def rqvae_parameters(self):
        yield from self._rqvae_module().parameters()

    def refresh_item_codes(self) -> torch.Tensor:
        codes = self._build_item_codes(device=self._semantic_embedding().weight.device)
        self._item_codes = codes
        self._code_to_item_ids = {}
        for item_token, code in enumerate(codes.detach().cpu().tolist()):
            if item_token == 0:
                continue
            self._code_to_item_ids.setdefault(tuple(int(token) for token in code), []).append(item_token)
        return codes

    def compute_rqvae_loss(self, item_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        item_tokens = item_tokens.reshape(-1).long()
        item_tokens = item_tokens[item_tokens > 0]
        if item_tokens.numel() == 0:
            zero = self._semantic_embedding().weight.sum() * 0
            return {"loss": zero, "recon_loss": zero, "rq_loss": zero}
        unique_item_tokens = torch.unique(item_tokens)
        semantic_embs = self._semantic_embedding()(unique_item_tokens)
        recon_embs, rq_loss, _, _, _ = self._rqvae_module()(semantic_embs)
        if self.config.loss_type == "mse":
            recon_loss = F.mse_loss(recon_embs, semantic_embs.detach(), reduction="mean")
        elif self.config.loss_type == "l1":
            recon_loss = F.l1_loss(recon_embs, semantic_embs.detach(), reduction="mean")
        else:
            raise ValueError(f"ETEGRec stage 5 does not support loss_type={self.config.loss_type!r}.")
        loss = recon_loss + float(getattr(self.config, "alpha", 1.0)) * rq_loss
        return {"loss": loss, "recon_loss": recon_loss, "rq_loss": rq_loss}

    def _build_t5(self) -> nn.Module:
        try:
            from transformers import T5Config, T5ForConditionalGeneration
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("ETEGRec requires `transformers`. Install it before using model.name=etegrec.") from exc

        t5_config = T5Config(
            num_layers=int(self.config.num_layers),
            num_decoder_layers=int(self.config.num_decoder_layers),
            d_model=int(self.config.d_model),
            d_ff=int(self.config.d_ff),
            num_heads=int(self.config.num_heads),
            d_kv=int(self.config.d_kv),
            dropout_rate=float(self.config.dropout_rate),
            activation_function=str(self.config.activation_function),
            vocab_size=int(self.config.code_num),
            pad_token_id=0,
            eos_token_id=0,
            decoder_start_token_id=0,
            feed_forward_proj=str(self.config.feed_forward_proj),
            n_positions=int(self.config.max_positions),
        )
        return T5ForConditionalGeneration(config=t5_config)

    def _build_rqvae(self) -> RQVAE:
        if len(tuple(self.config.num_emb_list)) != int(self.config.code_length) - 1:
            raise ValueError(
                "ETEGRec expects code_length to equal len(num_emb_list) + 1 because the final token is "
                "the collision counter."
            )
        for codebook_size in self.config.num_emb_list:
            if int(codebook_size) > int(self.config.code_num):
                raise ValueError("ETEGRec num_emb_list entries must be <= model.code_num.")
        return RQVAE(self.config, in_dim=int(self.config.semantic_hidden_size))

    def _load_rqvae_if_configured(self) -> None:
        path = str(self.config.rqvae_path or "").strip()
        if not path:
            return
        rqvae = self._rqvae_module()
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ETEGRec rqvae_path does not exist: {checkpoint_path}")
        state = _load_rqvae_checkpoint(checkpoint_path)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        rqvae.load_state_dict(state)
        self._mark_rqvae_quantizers_initialized(rqvae)

    @staticmethod
    def _mark_rqvae_quantizers_initialized(rqvae: RQVAE) -> None:
        rq = getattr(rqvae, "rq", None)
        vq_layers = getattr(rq, "vq_layers", ())
        for quantizer in vq_layers:
            if hasattr(quantizer, "initted"):
                quantizer.initted = True

    def _forward_code_tokens(
        self,
        *,
        input_codes: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        encoder_outputs: Any = None,
    ) -> ETEGRecForwardOutput:
        t5 = self._t5_module()
        inputs_embeds = self._code_input_embeddings(input_codes, attention_mask)

        if decoder_input_ids is None and labels is None:
            decoder_input_ids = torch.zeros((input_codes.shape[0], int(self.config.code_length)), dtype=torch.long, device=input_codes.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_inputs_embeds = self._decoder_input_embeddings(decoder_input_ids)
        model_outputs = t5(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs,
        )

        decoder_outputs = model_outputs.decoder_hidden_states[-1]
        token_embeddings = self._token_embeddings()
        code_logits = []
        for index in range(min(decoder_inputs_embeds.shape[1], int(self.config.code_length))):
            code_logits.append(torch.matmul(decoder_outputs[:, index], token_embeddings[index].weight.t()))
        logits = torch.stack(code_logits, dim=1)

        seq_latents = model_outputs.encoder_last_hidden_state.clone()
        seq_latents = seq_latents.masked_fill(~attention_mask.unsqueeze(-1), 0)
        denom = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        seq_last_latents = torch.sum(seq_latents, dim=1) / denom
        seq_project_latents = self._enc_adapter()(seq_last_latents)
        dec_latents = self._dec_adapter()(decoder_outputs[:, 0])
        return ETEGRecForwardOutput(
            logits=logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents,
        )

    def _item_tokens_to_code_tokens(self, item_tokens: torch.Tensor) -> torch.Tensor:
        item_codes = self._current_item_codes(device=item_tokens.device)
        return item_codes[item_tokens.long()]

    def _current_item_codes(self, *, device: torch.device) -> torch.Tensor:
        if self._item_codes is not None:
            return self._item_codes.to(device=device)
        return self._build_item_codes(device=device)

    def _build_item_codes(self, *, device: torch.device) -> torch.Tensor:
        rqvae = self._rqvae_module()
        semantic_embedding = self._semantic_embedding()
        with torch.no_grad():
            prefixes = rqvae.get_indices(semantic_embedding.weight.data[1:].to(device)).detach().cpu().tolist()
        tokens_to_count: dict[tuple[int, ...], int] = {}
        all_item_codes: list[list[int]] = [[-1 for _ in range(int(self.config.code_length))]]
        max_conflict = 0
        for prefix in prefixes:
            prefix_tuple = tuple(int(token) for token in prefix)
            collision_index = tokens_to_count.get(prefix_tuple, 0)
            tokens_to_count[prefix_tuple] = collision_index + 1
            max_conflict = max(max_conflict, collision_index + 1)
            all_item_codes.append([*prefix_tuple, collision_index])
        if max_conflict > int(self.config.code_num):
            raise ValueError(
                "ETEGRec RQVAE semantic ID collisions exceed code_num. "
                f"Got maximum conflict {max_conflict} > {self.config.code_num}."
            )
        return torch.tensor(all_item_codes, dtype=torch.long, device=device)

    def _generate_code_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        num_beams: int,
        num_return_sequences: int,
    ) -> torch.Tensor:
        input_codes = self._item_tokens_to_code_tokens(input_ids).reshape(input_ids.shape[0], -1)
        attention_mask = input_codes.ne(-1)
        batch_size = int(input_ids.shape[0])
        input_codes = input_codes.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        decoder_input_ids = torch.zeros((batch_size * num_beams, 1), dtype=torch.long, device=input_ids.device)
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.reshape(-1)
        beam_offset = torch.arange(batch_size, device=input_ids.device).repeat_interleave(num_beams) * num_beams

        encoder_outputs = self._t5_module().get_encoder()(
            inputs_embeds=self._code_input_embeddings(input_codes, attention_mask),
            attention_mask=attention_mask,
            return_dict=True,
        )
        while int(decoder_input_ids.shape[1]) < int(self.config.code_length) + 1:
            outputs = self._forward_code_tokens(
                input_codes=input_codes,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_scores = torch.log_softmax(next_token_logits, dim=-1) + beam_scores[:, None]
            vocab_size = int(next_scores.shape[-1])
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            beam_scores = next_scores[:, :num_beams].reshape(-1)
            beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
            beam_idx = next_indices[:, :num_beams].reshape(-1)
            decoder_input_ids = torch.cat(
                [decoder_input_ids[beam_idx + beam_offset, :], beam_next_tokens.unsqueeze(-1)],
                dim=-1,
            )

        selection_mask = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=input_ids.device)
        selection_mask[:, :num_return_sequences] = True
        return decoder_input_ids[selection_mask.view(-1), 1:].reshape(batch_size, num_return_sequences, int(self.config.code_length))

    def _code_input_embeddings(self, input_codes: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self._token_embeddings()
        safe_codes = input_codes.masked_fill(input_codes.eq(-1), 0)
        inputs_embeds = torch.zeros(
            (*safe_codes.shape, int(self.config.d_model)),
            dtype=token_embeddings[0].weight.dtype,
            device=safe_codes.device,
        )
        for index in range(int(self.config.code_length)):
            inputs_embeds[:, index:: int(self.config.code_length)] = token_embeddings[index](
                safe_codes[:, index:: int(self.config.code_length)]
            )
        pad_embedding = self._t5_module().shared.weight[0].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds = torch.where(attention_mask.unsqueeze(-1), inputs_embeds, pad_embedding.view(1, 1, -1))
        return inputs_embeds

    def _decoder_input_embeddings(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self._token_embeddings()
        embeddings = []
        for index in range(min(decoder_input_ids.shape[1], int(self.config.code_length))):
            embedding = self._t5_module().shared if index == 0 else token_embeddings[index - 1]
            embeddings.append(embedding(decoder_input_ids[:, index]))
        return torch.stack(embeddings, dim=1)

    @staticmethod
    def _shift_right(input_ids: torch.Tensor) -> torch.Tensor:
        start = torch.zeros(input_ids.shape[:-1] + (1,), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([start, input_ids], dim=-1)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _semantic_embedding(self) -> nn.Embedding:
        if self.semantic_embedding is None:
            raise RuntimeError("ETEGRecModel must be initialized with prepared data before forward.")
        return self.semantic_embedding

    def _token_embeddings(self) -> nn.ModuleList:
        if self.token_embeddings is None:
            raise RuntimeError("ETEGRec token embeddings are not initialized.")
        return self.token_embeddings

    def _t5_module(self) -> nn.Module:
        if self._t5 is None:
            raise RuntimeError("ETEGRec T5 module is not initialized.")
        return self._t5

    def _rqvae_module(self) -> RQVAE:
        if self._rqvae is None:
            raise RuntimeError("ETEGRec RQVAE module is not initialized.")
        return self._rqvae

    def _enc_adapter(self) -> MLPLayers:
        if self.enc_adapter is None:
            raise RuntimeError("ETEGRec encoder adapter is not initialized.")
        return self.enc_adapter

    def _dec_adapter(self) -> MLPLayers:
        if self.dec_adapter is None:
            raise RuntimeError("ETEGRec decoder adapter is not initialized.")
        return self.dec_adapter

    def _fallback_item_ids(self) -> list[int]:
        if self._num_items is None:
            raise RuntimeError("ETEGRec item count is not initialized.")
        return list(range(int(self._num_items)))

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


def _load_rqvae_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError:
        # Original ETEGRec RQVAE checkpoints store trainer metadata with pickle
        # protocol 4. PyTorch 2.6+ defaults to weights_only=True, which can reject
        # those trusted, user-provided checkpoint files.
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


__all__ = ["ETEGRecModel"]
