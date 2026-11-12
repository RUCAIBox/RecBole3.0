"""RecBole model adapter for the original ManCAR training and fixed-step inference."""

# Upstream: https://github.com/FuCongResearchSquad/ManCAR/tree/0a259aa1988892b3edd938437671e6b90190a423
# Copyright (c) 2026 ManCAR Authors. See the MIT notice in this package's __init__.py.

import torch
from torch import nn
from torch.nn import functional as F

from recbole3.dataset import ITEM_ID
from recbole3.model.base import BaseRetrievalModel

from . import layers
from .data import ManCARCollator, build_swing
from .layers import MAX_ITEM_SEQ_LEN

ITEM_SEQ_ID = "history_id"
ITEM_SEQ_LEN = "item_seq_len"


class ManCARModel(BaseRetrievalModel):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(nn, "RMSNorm"):
            raise ImportError("ManCAR requires PyTorch 2.4 or later.")
        if config.embedding_dim != 256:
            raise ValueError("ManCAR's thought adapter requires embedding_dim=256.")
        if not 1 <= config.history_max_length <= MAX_ITEM_SEQ_LEN:
            raise ValueError("ManCAR requires history_max_length between 1 and 50.")
        if not 1 <= config.trigger_num <= MAX_ITEM_SEQ_LEN or config.items_per_trigger < 1:
            raise ValueError("Invalid ManCAR prompt size.")
        if config.reason_step < 0 or config.temperature <= 0:
            raise ValueError("Invalid ManCAR reasoning configuration.")
        self.core = None
        self.neighbors = {}
        self._num_items = 0
        self.register_buffer("_task_to_model", None, persistent=False)
        self.register_buffer("_model_to_task", None, persistent=False)

    def ensure_initialized(self, prepared_data):
        if self.core is None:
            self._num_items = prepared_data.get_num_items()
            item_ids = prepared_data.mancar_item_ids
            self._task_to_model = torch.tensor(item_ids, dtype=torch.long)
            self._model_to_task = torch.cat((
                torch.tensor([-1]), self._task_to_model.argsort()
            ))
            self.neighbors = build_swing(
                prepared_data.get_train_dataset().frame, self.config, item_ids=item_ids
            )
            self.core = _ManCARCore(self.config, self._num_items + 1)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.core is not None:
            self.core.all_item_embs = None
        return result

    def train(self, mode=True):
        super().train(mode)
        if not mode and self.core is not None:
            self.core.encode_all_items()
        return self

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        if self.core is not None:
            self.core.all_item_embs = None
        return result

    def build_train_collator(self, prepared_data):
        self.ensure_initialized(prepared_data)
        return ManCARCollator(
            self.config, prepared_data, neighbors=self.neighbors, training=True
        )

    def build_eval_collator(self, prepared_data):
        self.ensure_initialized(prepared_data)
        return ManCARCollator(
            self.config, prepared_data, neighbors=self.neighbors, training=False
        )

    def _feed(self, batch):
        feed = dict(batch)
        if ITEM_ID not in feed:
            feed[ITEM_ID] = torch.zeros(
                len(feed["history_id"]),
                dtype=torch.long,
                device=feed["history_id"].device,
            )
        if self.core.all_item_embs is None:
            self.core.encode_all_items()
        return feed

    def forward(self, batch):
        return self.core(self._feed(batch), stage="train")

    def compute_loss(self, batch, outputs):
        return self.core.loss(outputs)

    @torch.no_grad()
    def predict(
        self,
        model_inputs,
        *,
        k,
        candidate_item_ids=None,
        exclude_item_ids=None,
        exclude_mask=None,
    ):
        scores = self.core.inference(self._feed(model_inputs))["prediction"].clone()
        scores[:, 0] = -torch.inf
        if exclude_item_ids is not None:
            ids = exclude_item_ids.to(scores.device)
            mask = (
                torch.ones_like(ids, dtype=torch.bool)
                if exclude_mask is None
                else exclude_mask.to(scores.device).bool()
            )
            valid = mask & (ids >= 0) & (ids < self._num_items)
            rows = torch.arange(len(scores), device=scores.device)[:, None].expand_as(
                ids
            )
            scores[rows[valid], self._task_to_model[ids[valid]]] = -torch.inf
        if candidate_item_ids is not None:
            raise ValueError(
                "ManCAR supports full-catalog evaluation only."
            )
        count = min(max(int(k), 0), self._num_items)
        if count == 0:
            return torch.empty((len(scores), 0), device=scores.device, dtype=torch.long)
        values, ids = scores.topk(count, dim=1)
        ids = self._model_to_task[ids].masked_fill(~torch.isfinite(values), -1)
        if count < k:
            ids = torch.cat(
                [
                    ids,
                    torch.full(
                        (len(scores), k - count),
                        -1,
                        dtype=torch.long,
                        device=scores.device,
                    ),
                ],
                dim=1,
            )
        return ids


class _ManCARCore(nn.Module):
    def __init__(self, config, item_num):
        super().__init__()
        self.item_num = item_num
        self.emb_size = self.hidden_size = config.embedding_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.inner_size = config.inner_size
        self.dropout = config.dropout
        self.hidden_act = config.hidden_act
        self.rms_norm_eps = config.rms_norm_eps
        self.temperature = config.temperature
        self.pl_weight = config.pl_weight
        self.noise = config.noise_factor
        self.reason_step = config.reason_step
        self._define_params()
        self.apply(self.init_weights)
        self.all_item_embs = None

    @torch.no_grad()
    def encode_all_items(self):
        # ID-only item encoding in the source runner caches a detached snapshot.
        self.all_item_embs = self.item_id_emb.weight.detach().clone()

    @staticmethod
    def init_weights(m):
        if "Linear" in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif "Embedding" in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _define_params(self):
        self.item_id_emb = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_ITEM_SEQ_LEN + 2, self.emb_size, padding_idx=0)
        self.trm_encoder = layers.TransformerBlockv2(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps,
        )
        self.model = layers.PromptReaRecBertWrapper(
            self.trm_encoder, self.hidden_size, self.reason_step, self.noise
        )


        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        item_seq_ids, item_seq_len = feed_dict[ITEM_SEQ_ID], feed_dict[ITEM_SEQ_LEN]
        B = item_seq_ids.size(0)

        # Left Padding for item_seq_ids
        hist_padding_mask = item_seq_ids != 0
        valid_pos_ids = torch.cumsum(hist_padding_mask.long(), dim=1)
        pos_ids = torch.where(hist_padding_mask, valid_pos_ids, 0)
        pos_embs = self.pos_emb(pos_ids)

        # prompt_id and mask
        prompt_ids = feed_dict["prompt_ids"]

        item_embs = self.item_id_emb(item_seq_ids) + pos_embs
        prompt_embs = self.item_id_emb(prompt_ids)
        input_embs = torch.cat([prompt_embs, item_embs], dim=1)

        # Calculate sequence embeddings
        output, prompt_latent = self.model(
            input_embs,
            item_seq_len,
            prompt_ids,
            self.all_item_embs,
            epoch,
            feed_dict['neighbor_item_i2i'],
            feed_dict[ITEM_ID],
            stage,
            noise_factor=(
                self.noise
            ),
        )
        seq_embs = output[:B, -1, :]
        if stage == "all_steps":
            seq_embs = output

        feed_dict["seq_embs"] = seq_embs
        feed_dict["seq_output"] = output
        feed_dict["prompt_latent"] = prompt_latent

        feed_dict['epoch'] = epoch
        return feed_dict

    def kl_rank_softlabel_full_vocab(
        self,
        r_t: torch.Tensor,              # [B, D]
        pos_ids: torch.Tensor,          # [B, K]  (K<=10)
        target_ids: torch.Tensor,       # [B]
        temperature: float,
        gamma: float = 1.0,             # rank weight temperature
        use_cosine: bool = False,
    ):

        E = self.all_item_embs  # [N, D]
        B = r_t.size(0)
        device = r_t.device
        K = pos_ids.size(1)

        if use_cosine:
            r = F.normalize(r_t, dim=-1)
            En = F.normalize(E, dim=-1)
            scores = (r @ En.T) / temperature          # [B, N]
        else:
            scores = (r_t @ E.T) / temperature         # [B, N]

        # logZ: [B]
        logZ = torch.logsumexp(scores, dim=-1)         # [B]

        sup_ids = torch.cat([target_ids.unsqueeze(1), pos_ids], dim=1)  # [B, 1+K]

        sup_scores = scores.gather(1, sup_ids)

        logp_sup = sup_scores - logZ.unsqueeze(1)      # [B, 1+K]
        rank = torch.arange(K+1, device=device).float()          # 0..K-1
        q_sup = F.softmax(-rank / gamma, dim=0)                    # [K]
        q_sup = q_sup.unsqueeze(0).expand(B, -1)                       # [B, K]


        logq_sup = torch.log(q_sup.clamp_min(1e-12))
        kl = (q_sup * (logq_sup - logp_sup)).sum(dim=1).mean()
        return kl

    def loss(self, out_dict: dict) -> torch.Tensor:
        # self.encode_all_items()
        """
        Compute the loss for the model output.
        :param out_dict: Dictionary containing model outputs
        :return: Loss value
        """
        seq_embs = out_dict["seq_embs"]
        target_item_ids = out_dict[ITEM_ID]
        batch_size = len(seq_embs)

        logits = torch.matmul(seq_embs, self.all_item_embs.T)

        thinking_embs = out_dict["seq_output"]  # (B, T, D)
        T = thinking_embs.shape[1]


        thinking_sup_loss = 0.0
        if out_dict['epoch'] >= 0:
            for i in range(T):
                thinking_item = out_dict['neighbor_item_i2i'][:,:10]
                thinking_emb = thinking_embs[:batch_size, i, :]
                thinking_sup_loss += self.kl_rank_softlabel_full_vocab(
                                            r_t=thinking_emb,
                                            pos_ids=thinking_item,
                                            target_ids=target_item_ids,
                                            temperature=1.0,
                                            gamma=1.0 * (T - i),
                                        )


        all_logits = torch.einsum(
            "btd,nd->btn", thinking_embs[:batch_size], self.all_item_embs
        )
        temp_scales = self.temperature * (torch.arange(1, T+1).to(logits.device) ** 2)
        scaled_logits = all_logits / temp_scales.view(1, T, 1)
        pl_loss = self.loss_fct(
            scaled_logits.view(-1, self.all_item_embs.size(0)),
            target_item_ids.repeat_interleave(T)
        )

        return  pl_loss * self.pl_weight  + thinking_sup_loss

    @torch.no_grad()
    def inference(self, feed_dict: dict) -> dict:
        """
        Inference method for the model.
        :param feed_dict: Dictionary containing input data
        :return: Dictionary with inference results
        """
        if self.all_item_embs is None:
            self.encode_all_items()
        seq_embs = self.forward(feed_dict, stage="infer")["seq_embs"]

        logits = torch.matmul(seq_embs, self.all_item_embs.transpose(0, 1))
        return {"prediction": logits}
