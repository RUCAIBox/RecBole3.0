"""ManCAR attention and latent reasoning layers, adapted from the upstream MIT code."""

# Upstream: https://github.com/FuCongResearchSquad/ManCAR/tree/0a259aa1988892b3edd938437671e6b90190a423
# Copyright (c) 2026 ManCAR Authors. See the MIT notice in this package's __init__.py.

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

MAX_ITEM_SEQ_LEN = 50


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        rms_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, kv_cache=None):
        query = self.query_layer(input_tensor)
        key = self.key_layer(input_tensor)
        value = self.value_layer(input_tensor)

        if kv_cache is not None:
            key = torch.cat([kv_cache["key"], key], dim=1)
            value = torch.cat([kv_cache["value"], value], dim=1)
        else:
            kv_cache = {}

        kv_cache["key"] = key
        kv_cache["value"] = value

        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.


        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states, kv_cache, attention_scores


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, rms_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": F.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        attention_output, new_kv_cache, attention_scores = self.multi_head_attention(
            hidden_states, attention_mask, kv_cache
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, new_kv_cache, attention_scores, attention_mask


class TransformerBlockv2(nn.Module):
    r"""One TransformerBlock consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        rms_norm_eps=1e-12,
    ):
        super(TransformerBlockv2, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attn_mask, *,
            kv_caches=None, output_all_encoded_layers=False, fp16_kv=False):

        use_half = torch.float16 if fp16_kv else hidden_states.dtype
        out_list = [] if output_all_encoded_layers else None
        new_kv   = []
        x = hidden_states                    # will hold “current sequence” slice

        attention_scores_list = []
        attention_mask_list = []

        for i, layer in enumerate(self.layers):
            kvc = None if kv_caches is None else kv_caches[i]

            # ---- 1. choose the slice we send to this layer ----
            if kvc is None:                  # full pass (step‑0)
                inp = x                      # [B, S, D]
            else:                            # incremental step
                inp = x[:, -1:, :]           # [B, 1, D]

            # ---- 2. run the layer (no checkpoint inside incremental) ----
            h, kv, attention_scores, attention_mask = layer(inp, attn_mask, kvc)

            # ---- 3. cast KV once, store back as dict ----
            if kv is not None:
                kv = {"key": kv["key"].to(use_half), "value": kv["value"].to(use_half)}
            new_kv.append(kv)

            # ---- 4. update running sequence only AFTER last layer ----
            if i == len(self.layers) - 1:
                x = torch.cat([x, h], dim=1) if kvc is not None else h
            else:
                x = x if kvc is not None else h   # keep same length inside block

            if output_all_encoded_layers:
                out_list.append(x)
            attention_mask_list.append(attention_mask)
            attention_scores_list.append(attention_scores)
        if output_all_encoded_layers:
            return out_list, new_kv
        return x, new_kv, attention_scores_list, attention_mask_list


class ThoughtAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    @torch.no_grad()
    def target_norm(self, emb_weight):
        return emb_weight.norm(dim=-1).mean()

    def forward(self, h, emb_weight):

        tn = self.target_norm(emb_weight).detach()
        hn = h.norm(dim=-1, keepdim=True)
        h = h * (tn / hn) * self.scale
        return h


class PromptReaRecBertWrapper(nn.Module):
    """
    Wrapper that adds (optional) autoregressive “reasoning” steps on top of a base Transformer.

    • Inputs: prompt+history embeddings (already concatenated), their lengths, and prompt IDs.
    • If `reason_step == 0`: run the whole sequence once (no AR).
      Return the last token embedding and the prompt segment embedding.
    • If `reason_step > 0`: iteratively append `reason_step` new tokens.
        - At each step: build masks, reuse KV cache, take the newest token, (optionally) add noise,
          add a positional embedding, and feed it back.
    • Padding mask = right‑padded prompt  + left‑padded history.
      Attention mask = causal for “thinking” tokens, full for prefix, then padded with -1e10.
    • Noise path duplicates the batch (clean + noisy) to compute CL later.

    """

    def __init__(self, transformer, hidden_size, reason_step, noise_factor):
        super().__init__()
        self.transformer = transformer
        self.n_layers = len(transformer.layers)
        self.reason_step = reason_step
        self.hidden_size = hidden_size
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.5)
        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)  # pos emb for each thinking step
        self.reason_step = reason_step

        self.adpter_norm = ThoughtAdapter(256)


    def _prepare_padding_mask(self, input_lens, prompt_ids, step, device):
        """
        Build a padding mask (True = pad) for prompt (right-pad) + history (left-pad) at current step.

        input_lens: [B]  real history lengths (without left-pad)
        prompt_ids : [B, P]
        step       : int, number of generated thinking tokens so far
        return     : [B,1,1,P+L] bool mask (broadcastable to attention scores)
        """
        B = len(input_lens)
        L = MAX_ITEM_SEQ_LEN + step

        prompt_pad = prompt_ids.eq(0)  # [B,P]
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        hist_start = L - input_lens.unsqueeze(1)
        hist_pad = idx < hist_start     # [B,L]  left side is pad

        combined_pad = torch.cat([prompt_pad, hist_pad], dim=1)  # [B,P+L]
        return combined_pad.unsqueeze(1).unsqueeze(2)

    def _prepare_attention_mask(self, batch_size, seq_len, think_len, device, padding_mask):
        """
        Build the full attention mask (float) with causal part for thinking tokens.

        seq_len   : prompt+history length
        think_len : current number of generated tokens
        padding_mask: [B,1,1,P+L]
        return    : [B,1,total,total] float mask with -1e10 on disallowed/pad, 0 on allowed
        """
        total_len = seq_len + think_len
        mask = torch.zeros((total_len, total_len), device=device)
        mask[:seq_len, :seq_len] = 1  # full visibility for prefix
        if think_len > 0:
            mask[-think_len:, -think_len:] = torch.tril(torch.ones((think_len, think_len), device=device))
            mask[seq_len:, :seq_len] = 1  # thinking tokens can see prefix
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def forward(self, input_embs, input_lens, prompt_ids, item_embs, epoch, i2i_label=None, target_id=None, stage='infer', noise_factor=0.01, reason_step=None):
        """
        input_embs : [B, P+L, D]  prompt + history embeddings
        input_lens : [B]         history lengths
        prompt_ids : [B, P]
        noise_factor: float      >0 duplicates batch with gaussian noise
        reason_step : override default steps if not None

        returns:
            outputs      : [B*(1 or 2), steps+1, D]  (or [B,1,D] if reason_step==0)
            prompt_latent: [B, P, D]
        """
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device
        _, prompt_len = prompt_ids.size()

        # ----- fast path: no reasoning -----
        if self.reason_step == 0:
            repeat_batch = (noise_factor > 0.0) + 1
            input_embs = input_embs.repeat(repeat_batch, 1, 1)
            prompt_ids = prompt_ids.repeat(repeat_batch, 1)
            input_lens = input_lens.repeat(repeat_batch)
            pad_mask = self._prepare_padding_mask(input_lens, prompt_ids, 0, device)
            attn_mask = self._prepare_attention_mask(batch_size * repeat_batch, seq_len, 0, device, pad_mask).bool()

            inp = self.dropout(self.LayerNorm(input_embs))
            hs, _, attention_scores, attention_mask = self.transformer(inp, attn_mask, kv_caches=None)

            # Prompt Latents used for swing loss, final for classification
            prompt_latent = hs[:batch_size, :prompt_len, :]
            final_token   = hs[:, -1:, :]
            return final_token, prompt_latent

        # ----- AR path -----
        repeat_batch = (noise_factor > 0.0) + 1
        input_embs = input_embs.repeat(repeat_batch, 1, 1)
        prompt_ids = prompt_ids.repeat(repeat_batch, 1)
        input_lens = input_lens.repeat(repeat_batch)

        past_kv = [None] * self.n_layers
        all_outputs, all_noise_outputs = [], []

        reason_step = self.reason_step if reason_step is None else reason_step
        for step in range(reason_step + 1):
            pad_mask = self._prepare_padding_mask(input_lens, prompt_ids, step, device)
            attn_mask = self._prepare_attention_mask(batch_size * repeat_batch,
                                                     seq_len, step, device, pad_mask).bool()

            input_embs = self.dropout(self.LayerNorm(input_embs))
            hs, past_kv, attention_scores, attention_mask = self.transformer(input_embs, attn_mask, kv_caches=past_kv)

            last_h = hs[:batch_size, -1:, :]

            all_outputs.append(last_h)

            last_h = self.adpter_norm(last_h, item_embs)


            if noise_factor > 0:
                all_noise_outputs.append(hs[batch_size:, -1:, :])

            if step == reason_step:
                break

            new_pos = self.reason_pos_emb(torch.tensor([step], device=device)).expand(batch_size, 1, -1)
            input_embs = last_h + new_pos
            if noise_factor > 0.0:
                noise = torch.randn_like(input_embs) * noise_factor
                input_embs = torch.cat([input_embs, input_embs + noise], dim=0)

        prompt_latent = hs[:batch_size, :prompt_len, :]
        outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            outputs = torch.cat([outputs, torch.cat(all_noise_outputs, dim=1)], dim=0)

        return outputs, prompt_latent
