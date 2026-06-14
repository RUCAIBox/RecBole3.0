"""ReaRec neural network layers.

Provides:
  - SASRec-style Transformer encoder with optional KV cache.
  - ``SequenceBackbone``: abstract base class defining the two-phase interface
    (initial_encode + step_encode) used by the reasoning loop.
  - ``SASRecBackbone``: KV-cache-accelerated SASRec Transformer backbone.
  - ``HSTUBackbone``: HSTU-based backbone wrapping the framework's ``HSTUModel``;
    performs a full HSTU re-encode at each reasoning step (no KV cache).
  - ``ReaRecAutoRegressiveWrapper``: drives the K-step reasoning loop over any
    ``SequenceBackbone`` and manages the noise-augmented batch for PRL.

KV-cache design (SASRec)
------------------------
KV tensors are stored in **head-split** form ``[B, H, L, head_dim]``.
At reasoning step k (k ≥ 1), only the single new reasoning token is projected
(Q/K/V of shape ``[B, H, 1, head_dim]``); the cached K/V from all prior steps are
prepended before computing scaled dot-product attention.  The attention mask passed
at step k is the *last row* of the full causal mask,
``full_mask[:, :, -1:, :]`` → ``[B, 1, 1, L+k]``, which broadcasts correctly with
the query shape ``[B, H, 1, L+k]`` and avoids redundant computation.

HSTU backbone design
---------------------
HSTU uses SiLU-based attention with relative time + position bias and requires
FBGEMM jagged tensor operations.  Each reasoning step concatenates the new
reasoning token *compactly* after the last real item using ``scatter_`` (no gaps),
then re-runs the full HSTU preprocessor + encoder on the growing sequence.
Timestamps for reasoning positions are set to the last real item's timestamp
so the time delta remains 0 relative to the most recent interaction.
The inner ``HSTUModel`` instance is held as a registered sub-module so all
parameters participate in gradient updates.
"""
from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Type alias for a per-layer KV cache list
KVCacheList = list[dict[str, torch.Tensor] | None]


# ---------------------------------------------------------------------------
# Transformer building blocks (KV-cache-aware)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional KV cache and pre-LN residual."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.sqrt_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, H, L, head_dim]."""
        B, L, _ = x.shape
        return x.view(B, L, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,                    # [B, Lq, D]
        attention_mask: torch.Tensor,                   # [B, 1, Lq, Lkv] additive
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Attend and return (output [B, Lq, D], new_kv_cache).

        When ``kv_cache`` is provided (reasoning steps 1..K):
          - ``hidden_states`` is the new single token ``[B, 1, D]``.
          - Cached K/V (from all previous positions) are prepended.
          - ``attention_mask`` should be ``[B, 1, 1, L+k]`` (last-row slice).
        """
        q = self._split_heads(self.query(hidden_states))   # [B, H, Lq, hd]
        k = self._split_heads(self.key(hidden_states))     # [B, H, Lq, hd]
        v = self._split_heads(self.value(hidden_states))   # [B, H, Lq, hd]

        if kv_cache is not None:
            # Prepend cached keys/values: [B, H, L_prev, hd] + [B, H, Lq, hd]
            k = torch.cat([kv_cache["k"], k], dim=2)      # [B, H, Lkv, hd]
            v = torch.cat([kv_cache["v"], v], dim=2)      # [B, H, Lkv, hd]

        new_kv_cache: dict[str, torch.Tensor] = {"k": k, "v": v}

        # Scaled dot-product: [B, H, Lq, hd] × [B, H, hd, Lkv] = [B, H, Lq, Lkv]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_head_size
        scores = scores + attention_mask                   # additive mask broadcast
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)              # [B, H, Lq, hd]
        context = context.permute(0, 2, 1, 3).contiguous().view(
            hidden_states.shape[0], -1, self.num_attention_heads * self.attention_head_size
        )  # [B, Lq, D]

        out = self.out_dropout(self.dense(context))
        return self.layer_norm(out + hidden_states), new_kv_cache


class FeedForward(nn.Module):
    """Point-wise feed-forward with post-LN residual."""

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.act_fn = self._get_act_fn(hidden_act)

    @staticmethod
    def _get_act_fn(name: str):  # type: ignore[return]
        fns = {
            "gelu": lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
            "relu": F.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if name not in fns:
            raise ValueError(f"Unknown activation '{name}'. Choose from: {list(fns)}.")
        return fns[name]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # [B, L, D] -> [B, L, D]
        h = self.dropout(self.dense_2(self.act_fn(self.dense_1(hidden_states))))
        return self.layer_norm(h + hidden_states)


class TransformerLayer(nn.Module):
    """Single Transformer block: multi-head attention + feed-forward."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.ffn = FeedForward(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,                    # [B, Lq, D]
        attention_mask: torch.Tensor,                   # [B, 1, Lq, Lkv]
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # ([B, Lq, D], new_kv_cache)
        attn_out, new_kv = self.attention(hidden_states, attention_mask, kv_cache)
        return self.ffn(attn_out), new_kv


class TransformerEncoder(nn.Module):
    """Stack of Transformer layers with optional per-layer KV caches."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        single = TransformerLayer(
            n_heads, hidden_size, inner_size,
            hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
        )
        self.layers = nn.ModuleList([copy.deepcopy(single) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,        # [B, Lq, D]
        attention_mask: torch.Tensor,       # [B, 1, Lq, Lkv]
        kv_caches: KVCacheList | None = None,
    ) -> tuple[torch.Tensor, KVCacheList]:  # ([B, Lq, D], new_kv_caches)
        """Forward pass with optional KV cache.

        Args:
            kv_caches: One ``dict | None`` per layer from the previous step.
                       ``None`` entries mean no cache (initial pass or no-cache mode).

        Returns:
            Tuple of (output hidden states, updated per-layer KV cache list).
        """
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        new_caches: KVCacheList = []
        for layer, kv in zip(self.layers, kv_caches):
            hidden_states, new_kv = layer(hidden_states, attention_mask, kv)
            new_caches.append(new_kv)
        return hidden_states, new_caches


# ---------------------------------------------------------------------------
# Attention mask builder
# ---------------------------------------------------------------------------

def build_causal_attention_mask(
    seq_len: int,
    history_lengths: torch.Tensor,   # [B] actual (non-padded) lengths
    original_seq_len: int,            # length of the initial left-padded sequence L
) -> torch.Tensor:
    """Build additive causal attention mask ``[B, 1, seq_len, seq_len]``.

    Positions in the original sequence that are left-padding
    (column index < original_seq_len - history_length[b]) are set to -1e10 so
    no token can attend to them.  Appended reasoning token positions
    (column index ≥ original_seq_len) are never treated as padding.

    Returns 0.0 for attended positions, -1e10 for masked positions.
    """
    device = history_lengths.device
    B = int(history_lengths.shape[0])

    # Lower-triangular causal mask, additive form
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
    additive = causal.masked_fill(causal == 0, -1e10).masked_fill(causal == 1, 0.0)
    mask = additive.unsqueeze(0).unsqueeze(0).expand(B, 1, seq_len, seq_len).clone()

    # Left-padding: block columns in the original portion that precede actual content
    orig_positions = torch.arange(original_seq_len, device=device)               # [L]
    left_pad_count = (original_seq_len - history_lengths).clamp(min=0)           # [B]
    padding_cols = orig_positions.unsqueeze(0) < left_pad_count.unsqueeze(1)     # [B, L]

    if seq_len > original_seq_len:
        # Reasoning tokens appended after L are never padding
        no_pad = torch.zeros(B, seq_len - original_seq_len, dtype=torch.bool, device=device)
        padding_cols = torch.cat([padding_cols, no_pad], dim=1)                  # [B, seq_len]

    mask = mask.masked_fill(padding_cols.unsqueeze(1).unsqueeze(2), -1e10)
    return mask  # [B, 1, seq_len, seq_len]


# ---------------------------------------------------------------------------
# Abstract backbone interface
# ---------------------------------------------------------------------------

class SequenceBackbone(ABC, nn.Module):
    """Abstract backbone for the ReaRec autoregressive reasoning loop.

    Implementations must provide two methods:

    ``initial_encode``
        Processes the *full* left-padded item sequence (step 0).
        Returns the last-position hidden state and an opaque ``state`` that
        will be forwarded to each subsequent ``step_encode`` call.

    ``step_encode``
        Processes *one new reasoning token* (steps 1..K).
        Receives the state from the preceding step and returns the updated
        last-position hidden state together with the updated state.

    The ``ReaRecAutoRegressiveWrapper`` owns LayerNorm + Dropout; both inputs
    arrive *pre-normalised*.

    For backbones that own their own embedding table (e.g. ``HSTUBackbone``),
    ``initial_encode`` receives a ``raw_context`` dict containing raw item IDs
    and timestamps instead of pre-computed embeddings.  SASRec-style backbones
    ignore ``raw_context`` and use the pre-normalised ``seq_embs`` as usual.
    """

    def get_item_embs(self) -> torch.Tensor | None:
        """Return scoring embeddings ``[num_items, D]`` if owned by this backbone.

        Returns ``None`` for backbones (e.g. SASRec) whose item embeddings are
        managed by the outer ``ReaRecModel``.
        """
        return None

    @abstractmethod
    def initial_encode(
        self,
        seq_embs: torch.Tensor,                          # [B, L, D]  pre-normalised
        history_lengths: torch.Tensor,                   # [B]
        raw_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, Any]:                       # (last_hidden [B, 1, D], state)
        """Encode the full sequence. Return last hidden state and backbone state."""

    @abstractmethod
    def step_encode(
        self,
        new_token: torch.Tensor,         # [B, 1, D]  pre-normalised
        history_lengths: torch.Tensor,   # [B]
        step: int,                       # 1-indexed reasoning step
        original_seq_len: int,           # L (length of original left-padded sequence)
        state: Any,
    ) -> tuple[torch.Tensor, Any]:       # (last_hidden [B, 1, D], new_state)
        """Encode one new reasoning token. Return updated last hidden and state."""


# ---------------------------------------------------------------------------
# SASRec backbone (KV-cache-accelerated)
# ---------------------------------------------------------------------------

class SASRecBackbone(SequenceBackbone):
    """SASRec Transformer backbone with KV-cache-accelerated reasoning steps.

    At reasoning step 0 the full sequence is processed (O(L²) attention).
    For each subsequent step only the single new reasoning token is projected;
    the KV pairs from all prior positions are retrieved from the cache,
    reducing per-step cost to O(L+k) instead of O((L+k)²).
    """

    def __init__(self, encoder: TransformerEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def initial_encode(
        self,
        seq_embs: torch.Tensor,                          # [B, L, D]
        history_lengths: torch.Tensor,                   # [B]
        raw_context: dict[str, torch.Tensor] | None = None,  # unused by SASRec
    ) -> tuple[torch.Tensor, Any]:
        L = seq_embs.shape[1]
        attn_mask = build_causal_attention_mask(L, history_lengths, original_seq_len=L)
        # [B, 1, L, L]

        output, kv_caches = self.encoder(seq_embs, attn_mask)
        # output: [B, L, D],  kv_caches: list of {k: [B,H,L,hd], v: [B,H,L,hd]}

        last_hidden = output[:, -1:, :]  # [B, 1, D]
        state = {"kv_caches": kv_caches, "original_seq_len": L}
        return last_hidden, state

    def step_encode(
        self,
        new_token: torch.Tensor,         # [B, 1, D]
        history_lengths: torch.Tensor,   # [B]
        step: int,
        original_seq_len: int,
        state: Any,
    ) -> tuple[torch.Tensor, Any]:
        kv_caches: KVCacheList = state["kv_caches"]
        orig_L: int = state["original_seq_len"]
        curr_len = orig_L + step  # total sequence length after appending this token

        # Build the [B, 1, 1, curr_len] mask directly instead of constructing the full
        # [B, 1, curr_len, curr_len] causal mask and slicing the last row. The last row
        # of the causal mask is all-attended (it's the bottom row of a lower-triangular),
        # so the only thing to mask is left-padding columns in the original portion.
        device = history_lengths.device
        B = int(history_lengths.shape[0])
        orig_positions = torch.arange(orig_L, device=device)                  # [L]
        left_pad_count = (orig_L - history_lengths).clamp(min=0)              # [B]
        padding_cols = orig_positions.unsqueeze(0) < left_pad_count.unsqueeze(1)  # [B, L]
        if curr_len > orig_L:
            # Reasoning tokens appended after L are never padding
            no_pad = torch.zeros(B, curr_len - orig_L, dtype=torch.bool, device=device)
            padding_cols = torch.cat([padding_cols, no_pad], dim=1)           # [B, curr_len]
        step_mask = torch.zeros(B, 1, 1, curr_len, device=device, dtype=new_token.dtype)
        step_mask = step_mask.masked_fill(padding_cols.unsqueeze(1).unsqueeze(2), -1e10)
        # [B, 1, 1, curr_len]

        output, new_kv_caches = self.encoder(new_token, step_mask, kv_caches=kv_caches)
        # output: [B, 1, D]

        last_hidden = output[:, -1:, :]  # [B, 1, D]
        new_state = {"kv_caches": new_kv_caches, "original_seq_len": orig_L}
        return last_hidden, new_state


# ---------------------------------------------------------------------------
# HSTU backbone (wraps the framework's HSTUModel)
# ---------------------------------------------------------------------------

class HSTUBackbone(SequenceBackbone):
    """HSTU-based backbone for ReaRec's reasoning loop.

    Wraps an already-initialised ``HSTUModel`` instance and reuses its item
    embedding table, positional preprocessor, and HSTU encoder stack.

    Encoding strategy
    -----------------
    * ``initial_encode``: delegates to ``HSTUModel._encode_sequence_embeddings``
      (full HSTU forward on the item history).
    * ``step_encode``: builds a **compact** dense tensor that places real items at
      positions 0..L_b-1 and the accumulated reasoning tokens immediately after at
      positions L_b..L_b+k-1 using ``scatter_``.  The HSTU preprocessor and
      encoder from the inner model are then called directly on this compact tensor.
      No KV cache is used; each step is a fresh full encode of the growing sequence.

    Reasoning-token timestamps
    --------------------------
    HSTU's relative time-bucketed bias requires timestamps for every sequence
    position.  Reasoning tokens are assigned the timestamp of the last real item
    (time delta = 0 → bucket 0), which is a safe neutral value.

    Requirements
    ------------
    Requires ``fbgemm_gpu``; this is inherited from the inner ``HSTUModel``.
    """

    def __init__(self, hstu_model: Any) -> None:  # HSTUModel, typed as Any to avoid circular import
        super().__init__()
        self._hstu = hstu_model  # registered as nn.Module child

    # ------------------------------------------------------------------
    # Backbone API
    # ------------------------------------------------------------------

    def get_item_embs(self) -> torch.Tensor:
        """Return scoring embeddings ``[num_items, D]`` (skip HSTU padding slot 0)."""
        from recbole3.model.hstu.config import ITEM_ID_OFFSET
        return self._hstu._item_embedding_module().weight[ITEM_ID_OFFSET:]  # [num_items, D]

    def initial_encode(
        self,
        seq_embs: torch.Tensor,                          # [B, L, D] — ignored; HSTU owns embedding
        history_lengths: torch.Tensor,                   # [B]
        raw_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Full HSTU encode of the item history.

        Returns the hidden state at the last valid position and a lightweight
        state dict carrying ``item_ids`` and ``timestamps`` for use in
        ``step_encode``.
        """
        assert raw_context is not None, (
            "HSTUBackbone.initial_encode requires raw_context={'item_ids': ..., 'timestamps': ...}"
        )
        from recbole3.model.hstu.data import HISTORY_TIMESTAMPS
        from recbole3.model.sequential import HISTORY_ITEM_IDS

        item_ids: torch.Tensor = raw_context["item_ids"]      # [B, L]
        timestamps: torch.Tensor = raw_context["timestamps"]  # [B, L]

        # _encode_sequence_embeddings expects a batch dict; omit ITEM_ID so the
        # target is NOT appended to the history sequence (HSTU's AR training mode).
        batch = {
            HISTORY_ITEM_IDS: item_ids,
            HISTORY_TIMESTAMPS: timestamps,
            "history_lengths": history_lengths,
        }
        encoded = self._hstu._encode_sequence_embeddings(batch)  # [B, L, D]

        B, _, D = encoded.shape
        # Gather the last valid position for each batch item
        user_pos = (history_lengths - 1).clamp(min=0)           # [B]
        last_hidden = encoded.gather(
            1, user_pos.view(B, 1, 1).expand(B, 1, D)
        )  # [B, 1, D]

        state: dict[str, Any] = {
            "item_ids": item_ids,      # kept lightweight; re-embedded in step_encode
            "timestamps": timestamps,
        }
        return last_hidden, state

    def step_encode(
        self,
        new_token: torch.Tensor,         # [B, 1, D]  wrapper-normalised reasoning token
        history_lengths: torch.Tensor,   # [B]
        step: int,                       # 1-indexed reasoning step
        original_seq_len: int,           # L = history_max_length
        state: Any,
    ) -> tuple[torch.Tensor, Any]:
        """HSTU re-encode with reasoning tokens compactly appended after real items.

        Compact layout per batch item b::

            [item_0, ..., item_{L_b-1}, r_1, ..., r_k, <zero-pad>]
             ← history_lengths[b] →   ← step →

        Items are re-embedded from ``state["item_ids"]``; padding slots remain 0.
        Reasoning tokens are scattered to their correct positions via
        ``scatter_`` so there is no gap between real items and reasoning tokens.
        """
        from recbole3.model.hstu.config import HSTU_PADDING_ITEM_ID, ITEM_ID_OFFSET

        item_ids: torch.Tensor = state["item_ids"]            # [B, L]
        timestamps: torch.Tensor = state["timestamps"]        # [B, L]
        prev_reasoning: torch.Tensor | None = state.get("reasoning_embs")  # [B, step-1, D] | None

        device = history_lengths.device
        B = int(history_lengths.shape[0])
        D = int(new_token.shape[-1])
        L = int(item_ids.shape[1])  # history_max_length

        # ── 1. Accumulate reasoning tokens ──────────────────────────────
        all_reasoning = (
            torch.cat([prev_reasoning, new_token], dim=1)
            if prev_reasoning is not None
            else new_token
        )  # [B, step, D]
        k = step

        # ── 2. Re-embed item history (apply ITEM_ID_OFFSET to valid positions) ──
        positions = torch.arange(L, device=device)
        valid_mask = positions.unsqueeze(0) < history_lengths.unsqueeze(1)  # [B, L]
        model_item_ids = torch.full_like(item_ids, HSTU_PADDING_ITEM_ID)
        model_item_ids[valid_mask] = item_ids[valid_mask] + ITEM_ID_OFFSET
        raw_item_embs = self._hstu._item_embedding_module()(model_item_ids)  # [B, L, D]

        # ── 3. Build compact dense tensor [B, max_L+k, D] ───────────────
        # Items occupy 0..max_L-1 (zeros at positions ≥ history_lengths[b]);
        # reasoning tokens are scattered to positions history_lengths[b]+j.
        max_L = int(history_lengths.max().item())
        max_compact = max_L + k
        compact = raw_item_embs.new_zeros(B, max_compact, D)
        compact[:, :max_L, :] = raw_item_embs[:, :max_L, :]  # [B, max_L, D]

        reason_target = (
            history_lengths.unsqueeze(1)                              # [B, 1]
            + torch.arange(k, device=device).unsqueeze(0)            # [1, k]
        )  # [B, k] — position in compact for each reasoning token
        compact.scatter_(
            1,
            reason_target.unsqueeze(-1).expand(B, k, D),
            all_reasoning,
        )

        # ── 4. Build extended timestamps [B, max_compact] ───────────────
        # Reasoning positions get the last real item's timestamp (delta = 0).
        last_ts_idx = (history_lengths - 1).clamp(min=0)              # [B]
        last_ts = timestamps[torch.arange(B, device=device), last_ts_idx]  # [B]
        synthetic_ts = last_ts.unsqueeze(1).expand(B, k)              # [B, k]
        extended_ts = timestamps.new_zeros(B, max_compact)
        extended_ts[:, :max_L] = timestamps[:, :max_L]
        extended_ts.scatter_(1, reason_target, synthetic_ts)

        # ── 5. Preprocessor: add position embeddings, dropout, zero padding ──
        sequence_lengths = history_lengths + k                         # [B]
        dummy_ids = torch.zeros(B, max_compact, dtype=torch.long, device=device)
        _, processed, _ = self._hstu._input_preprocessor_module()(
            past_lengths=sequence_lengths,
            past_ids=dummy_ids,
            past_embeddings=compact,
            past_payloads={"timestamps": extended_ts},
        )  # processed: [B, max_compact, D]

        # ── 6. HSTU encode ───────────────────────────────────────────────
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            sequence_lengths.to(torch.int32)
        )
        valid_pos = (
            torch.arange(max_compact, device=device).view(1, max_compact)
            < sequence_lengths.view(-1, 1)
        )  # [B, max_compact]
        causal = torch.tril(
            torch.ones(max_compact, max_compact, dtype=torch.bool, device=device)
        )
        attn_mask = (
            valid_pos.unsqueeze(1) & valid_pos.unsqueeze(2) & causal.unsqueeze(0)
        ).to(processed.dtype)  # [B, max_compact, max_compact]

        encoded = self._hstu._encoder_module()(
            x=processed,
            x_offsets=x_offsets,
            all_timestamps=extended_ts,
            invalid_attn_mask=attn_mask,
        )  # [B, max_compact, D]

        # ── 7. Extract last valid position ──────────────────────────────
        user_pos = history_lengths + k - 1                             # [B]
        last_hidden = encoded.gather(
            1, user_pos.view(B, 1, 1).expand(B, 1, D)
        )  # [B, 1, D]

        new_state: dict[str, Any] = {
            "item_ids": item_ids,
            "timestamps": timestamps,
            "reasoning_embs": all_reasoning,
        }
        return last_hidden, new_state


# ---------------------------------------------------------------------------
# ReaRec autoregressive reasoning wrapper
# ---------------------------------------------------------------------------

class ReaRecAutoRegressiveWrapper(nn.Module):
    """Drive the K-step ReaRec reasoning loop over any ``SequenceBackbone``.

    Algorithm (per-forward call)
    ----------------------------
    1. Apply LayerNorm + Dropout to the full item-sequence embeddings.
    2. Call ``backbone.initial_encode`` → last hidden state h_0 + state_0.
    3. For step k = 1 .. K:
       a. Compute reasoning token: ``r_k = h_{k-1} + RPE[k-1]``.
       b. If PRL noise is active, create a noisy copy: ``r̃_k = r_k + ε``.
       c. Apply LayerNorm + Dropout to the new token(s).
       d. Call ``backbone.step_encode`` with the KV-cached state → h_k, state_k.
    4. Stack all h_0..h_K along dim-1 → ``[B, K+1, D]``.

    PRL noise doubles the effective batch: first B rows = clean, last B rows = noisy.
    The wrapper owns LayerNorm and Dropout; the backbone receives pre-normalised input.

    Args:
        backbone: Any ``SequenceBackbone`` implementation.
        hidden_size: Hidden dimension D.
        reason_step: Number of reasoning steps K (0 = standard SASRec forward).
    """

    def __init__(
        self,
        backbone: SequenceBackbone,
        hidden_size: int,
        reason_step: int,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.reason_step = reason_step
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Default matches official ReaRec, which hard-codes wrapper input dropout to 0.2.
        # This is intentionally distinct from (and lighter than) the transformer's internal
        # dropout (cfg.dropout=0.5): in PRL, the contrastive loss compares clean vs noisy
        # reasoning trajectories whose only intended difference is the cfg.noise_factor
        # perturbation. If the wrapper dropout is set too high, independent dropout masks
        # on the two halves dominate the actual noise signal and ruin the CL objective.
        self.dropout = nn.Dropout(p=float(dropout_p))
        if reason_step > 0:
            # RPE: one embedding per reasoning step (step index 0..K-1)
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)

    def forward(
        self,
        input_embs: torch.Tensor,                            # [B, L, D]  raw item+position embeddings
        history_lengths: torch.Tensor,                       # [B]
        noise_factor: float = 0.0,
        raw_context: dict[str, torch.Tensor] | None = None, # passed through to backbone.initial_encode
    ) -> torch.Tensor:
        """Run the reasoning loop.

        Args:
            input_embs:      Pre-computed item+position embeddings (SASRec) or a
                             dummy zero tensor (HSTU, which owns its own embedding).
            history_lengths: Actual sequence lengths per batch item.
            noise_factor:    PRL Gaussian noise scale; 0 disables noise.
            raw_context:     For HSTU backbone only — dict with ``'item_ids'`` and
                             ``'timestamps'`` tensors that the backbone uses instead
                             of ``input_embs``.

        Returns:
            ``[B, K+1, D]`` when noise is off, or ``[2B, K+1, D]`` when noise is
            active (first B = clean trajectory, last B = noise-augmented trajectory).
        """
        B, L, _ = input_embs.shape
        device = input_embs.device

        use_noise = (noise_factor > 0.0) and self.training and (self.reason_step >= 1)
        repeat = 2 if use_noise else 1

        # Double the batch for clean + noisy runs; both halves start identically
        seq_embs = input_embs.repeat(repeat, 1, 1)                # [B*repeat, L, D]
        eff_lengths = history_lengths.repeat(repeat)               # [B*repeat]

        # Also double raw_context tensors for the noise-augmented copy
        eff_raw_context: dict[str, torch.Tensor] | None = None
        if raw_context is not None:
            eff_raw_context = {
                k: v.repeat(repeat, *([1] * (v.dim() - 1)))
                for k, v in raw_context.items()
            }

        # ── Step 0: encode the full item sequence ───────────────────────────
        normed_seq = self.dropout(self.layer_norm(seq_embs))       # [B*repeat, L, D]
        last_hidden, state = self.backbone.initial_encode(
            normed_seq, eff_lengths, raw_context=eff_raw_context
        )
        # last_hidden: [B*repeat, 1, D]

        all_clean: list[torch.Tensor] = [last_hidden[:B]]          # [B, 1, D]
        all_noisy: list[torch.Tensor] = ([last_hidden[B:]] if use_noise else [])

        # ── Steps 1..K: autoregressive reasoning ────────────────────────────
        for step in range(1, self.reason_step + 1):
            rpe = self.reason_pos_emb(
                torch.tensor([step - 1], device=device, dtype=torch.long)
            )  # [1, D]
            clean_next = last_hidden[:B] + rpe.unsqueeze(0)        # [B, 1, D]

            if use_noise:
                noise = torch.randn_like(clean_next) * noise_factor
                noisy_next = clean_next + noise                     # [B, 1, D]
                next_token = torch.cat([clean_next, noisy_next], dim=0)  # [2B, 1, D]
            else:
                next_token = clean_next                             # [B, 1, D]

            normed_token = self.dropout(self.layer_norm(next_token))
            last_hidden, state = self.backbone.step_encode(
                normed_token, eff_lengths,
                step=step, original_seq_len=L, state=state,
            )
            # last_hidden: [B*repeat, 1, D]

            all_clean.append(last_hidden[:B])
            if use_noise:
                all_noisy.append(last_hidden[B:])

        clean_out = torch.cat(all_clean, dim=1)                    # [B, K+1, D]
        if use_noise:
            noisy_out = torch.cat(all_noisy, dim=1)                # [B, K+1, D]
            return torch.cat([clean_out, noisy_out], dim=0)        # [2B, K+1, D]
        return clean_out                                            # [B, K+1, D]


__all__ = [
    "HSTUBackbone",
    "KVCacheList",
    "ReaRecAutoRegressiveWrapper",
    "SASRecBackbone",
    "SequenceBackbone",
    "TransformerEncoder",
    "build_causal_attention_mask",
]

