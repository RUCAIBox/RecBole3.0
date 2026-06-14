# ReaRec in RecBole3.0

## Overview

This document describes the `rearec` model implemented in this repository, including:

- the high-level architecture
- the autoregressive reasoning mechanism
- the two learning strategies (ERL and PRL)
- the two supported backbone encoders (SASRec and HSTU)
- how to launch common experiments
- the meaning of frequently used configuration options

ReaRec (Reasoning-enhanced Recommendation) is a sequential recommendation model that introduces K additional autoregressive "thinking" steps before scoring. Instead of scoring directly from the sequence encoder output, the model chains K reasoning steps that iteratively refine the user representation before computing retrieval scores. This approach is inspired by chain-of-thought reasoning in language models and applied to the recommendation setting.

The model supports two learning strategies:

- `erl` — Ensemble Reasoning Learning: ensemble all K+1 step representations for the final score
- `prl` — Progressive Reasoning Learning: train with progressive difficulty and contrastive noise augmentation

The model supports two backbone sequence encoders:

- `sasrec` — Transformer with multi-head self-attention, supporting KV caching for efficient autoregressive reasoning
- `hstu` — Hierarchical Sequential Transduction Unit, requiring fbgemm-gpu and timestamp-enriched data

## File Structure

The main ReaRec implementation lives in these files:

- `src/recbole3/model/rearec/config.py`
  - ReaRecConfig dataclass, all model hyperparameters
- `src/recbole3/model/rearec/data.py`
  - model-side dataset and collators for both SASRec and HSTU variants
- `src/recbole3/model/rearec/layers.py`
  - Transformer building blocks, SequenceBackbone ABC, SASRecBackbone, HSTUBackbone, ReaRecAutoRegressiveWrapper
- `src/recbole3/model/rearec/model.py`
  - ReaRecModel: forward pass, loss computation, and predict
- `configs/model/rearec.yaml`
  - default ReaRec model and trainer configuration

## High-Level Architecture

The forward pass has three conceptual stages.

### Stage 0: Initial Encode

The item history `[item_0, ..., item_{L-1}]` is encoded by the backbone. The hidden state at the last valid position, `h_0`, serves as the first "thought" — the user representation before any reasoning.

### Stage k = 1 .. K: Reasoning Steps

Each reasoning step k performs:

1. compute reasoning token: `r_k = h_{k-1} + RPE[k-1]`
   - RPE is a Reasoning Position Embedding, one learnable vector per step, of shape `[K, D]`
2. append `r_k` to the input sequence and encode through the backbone
3. extract the hidden state at the new last position as `h_k`

For SASRec, KV caching means each step only projects one new token against the cached K, V tensors from the previous full encode. This reduces per-step cost from O((L+k)^2) to O(L+k).

For HSTU, there is no KV cache. The full growing sequence is re-encoded from scratch at each step. Reasoning tokens are appended compactly after the real items using scatter_, and reasoning-token timestamps are set to the last real item's timestamp so that the time-delta relative time bias resolves to bucket 0.

### Output

The K+1 hidden states `[h_0, h_1, ..., h_K]` are stacked into a tensor of shape `[B, K+1, D]`. How this tensor is consumed depends on the learning strategy.

## Learning Strategy: ERL

ERL (Ensemble Reasoning Learning) is described in Section 3.2 of the ReaRec paper.

The final user representation is the mean over all K+1 step embeddings:

```
u = mean([h_0, h_1, ..., h_K])   # shape [B, D]
loss_ce = CrossEntropy(u · item_embeddings / temperature, target)
```

A KL diversity term is subtracted from the loss to encourage the K+1 step representations to produce diverse score distributions, preventing degenerate collapse to identical reasoning outputs:

```
loss = loss_ce - kl_weight * KL_diversity
```

When `kl_weight=0`, the KL term is disabled and ERL reduces to a simple ensemble cross-entropy loss.

Note that ERL loss can be negative during training. This is expected behaviour when the KL term is large relative to the CE term.

### ERL parameters

- `model.learning_strategy: erl`
- `model.reason_step`
  - number of additional reasoning steps K
- `model.temperature`
  - softmax temperature for retrieval scores
- `model.kl_weight`
  - weight lambda for the KL diversity term; 0 disables KL

## Learning Strategy: PRL

PRL (Progressive Reasoning Learning) is described in Section 3.3 of the ReaRec paper.

The loss has three components:

```
loss = CE(h_K, target)
     + pl_weight * progressive_CE
     + cl_weight * contrastive_CE
```

### Progressive CE

Earlier reasoning steps receive easier targets (higher temperature) and later steps receive harder targets (lower temperature). The temperature for step k is:

```
tau_k = temperature * temp_scale ^ (K - k)
```

Step K uses `temperature` directly. Step 0 uses `temperature * temp_scale^K`. This progressive schedule trains the model so that each successive step makes a more refined prediction.

### Contrastive CE

A noise-augmented copy of the reasoning trajectory is constructed by injecting Gaussian noise (scale `noise_factor`) into each reasoning token. The noisy copy doubles the effective batch to 2B during forward. The contrastive loss encourages the clean trajectory to score higher than the noisy one.

Noise injection is controlled by `warmup_epochs`. When the current training epoch is less than or equal to `warmup_epochs`, no noise is injected and the contrastive term is zero for that epoch. When `warmup_epochs=0`, noise is always active from epoch 1 onward.

When `reason_step=0`, there are no reasoning tokens to augment and the contrastive term is skipped.

### PRL parameters

- `model.learning_strategy: prl`
- `model.reason_step`
  - number of additional reasoning steps K
- `model.temperature`
  - base softmax temperature
- `model.pl_weight`
  - weight for the progressive learning CE term
- `model.temp_scale`
  - progressive temperature decay base alpha
- `model.noise_factor`
  - standard deviation of Gaussian noise for contrastive augmentation
- `model.cl_weight`
  - weight for the contrastive loss term
- `model.warmup_epochs`
  - number of initial epochs during which noise injection is suppressed; 0 means always on

## Backbone: SASRec

SASRec uses a standard left-padded Transformer architecture.

- item sequences are left-padded: `[PAD, ..., PAD, item_{t-N}, ..., item_{t-1}]`
- `PADDING_ID = num_items` (one beyond the valid item ID range 0..num_items-1)
- position IDs are computed via cumsum of the valid-item mask, 0-indexed from the first real item; padding tokens receive sentinel position L which maps to a zero position embedding vector
- after the initial encode, K and V tensors of shape `[B, H, L, head_dim]` are cached per Transformer layer; each reasoning step only projects one new token and extends the cached K, V, reducing per-step cost significantly

SASRec-specific architecture parameters are only active when `backbone=sasrec`:

- `model.embedding_dim`
  - item embedding and hidden dimension; default 256
- `model.num_layers`
  - number of Transformer encoder layers; default 2
- `model.num_heads`
  - number of self-attention heads; default 2
- `model.inner_size`
  - feed-forward inner (intermediate) dimension; default 300
- `model.hidden_act`
  - activation function in the FFN; options: `gelu`, `relu`, `swish`, `tanh`, `sigmoid`; default `gelu`
- `model.layer_norm_eps`
  - epsilon for LayerNorm; default 1e-12
- `model.dropout`
  - dropout probability applied to embeddings and attention outputs; default 0.5
- `model.initializer_range`
  - standard deviation for truncated-normal weight initialisation; default 0.02
- `model.history_max_length`
  - maximum sequence length L used for padding and position embeddings; default 50

## Backbone: HSTU

HSTU (Hierarchical Sequential Transduction Unit) is a more expressive backbone that operates on timestamp-enriched item sequences.

- item sequences are right-padded: `[item_0, ..., item_{L_b-1}, PAD, ..., PAD]`
- `HSTU_PADDING_ITEM_ID = 0`; valid item IDs are shifted by `ITEM_ID_OFFSET=1` internally so that the padding sentinel is unambiguous
- no KV cache; each reasoning step re-encodes the full growing sequence
- reasoning tokens are appended compactly after the last real item using scatter_ (no gaps between real items and reasoning tokens)
- reasoning-token timestamps are set equal to the last real item's timestamp, so the time-delta resolves to 0 and maps to bucket 0 in the relative time bias
- `max_encoder_length` is expanded at init to `history_max_length + max(reason_step, 1)` to accommodate all K reasoning tokens
- requires `fbgemm_gpu`; a `RuntimeError` is raised at startup if the package is not found

HSTU-specific architecture parameters are only active when `backbone=hstu`:

- `model.attention_dim`
  - per-head attention projection dimension; default 32
- `model.linear_hidden_dim`
  - per-head value and output projection dimension; default 32
- `model.input_dropout_rate`
  - dropout on input embeddings after position encoding; default 0.0
- `model.attn_dropout_rate`
  - dropout on SiLU attention weights; default 0.0
- `model.linear_dropout_rate`
  - dropout on HSTU linear outputs; default 0.0
- `model.num_time_buckets`
  - number of time-delta buckets for the relative time bias; default 128

## Dataset and Collators

### SASRec path

`ReaRecModelDataset` detects `backbone='sasrec'` and dispatches to `BaseSequentialModelDataset._build_model_datasets`. This builds `history_item_ids` only; timestamps are not required.

- `ReaRecTrainCollator`
  - left-pads sequences to `history_max_length`
  - outputs: `history_item_ids [B, L]`, `history_lengths [B]`, `item_id [B]`
- `ReaRecEvalCollator`
  - same as the train collator but omits `item_id`

### HSTU path

`ReaRecModelDataset` detects `backbone='hstu'` and dispatches to `HSTUModelDataset._build_model_datasets`. This builds both `history_item_ids` and `history_timestamps`. Datasets without timestamps will raise a `ValueError` at this stage.

- `ReaRecHSTUTrainCollator`
  - right-pads sequences to width `history_max_length + 1` (one extra column reserved for the virtual query-timestamp slot)
  - outputs: `history_item_ids [B, L+1]`, `history_timestamps [B, L+1]` (with the query timestamp written into the virtual slot at position `history_lengths[b]`), `history_lengths [B]`, `item_id [B]`
  - the target item is NOT appended to the sequence (unlike the base `HSTUTrainCollator`)
- `ReaRecHSTUEvalCollator`
  - same as the HSTU train collator but omits `item_id`

## Configuration Reference

All parameters below are fields of `ReaRecConfig` and can be overridden from the command line or the YAML file.

### Backbone selection

- `model.backbone`
  - which sequence encoder to use; `sasrec` (default) or `hstu`

### Reasoning

- `model.learning_strategy`
  - learning strategy; `erl` or `prl`; default `prl`
- `model.reason_step`
  - number of autoregressive reasoning steps K; 0 disables reasoning and reduces to a plain SASRec baseline; default 2
- `model.temperature`
  - softmax temperature used in retrieval score computation; default 0.07

### ERL-specific

- `model.kl_weight`
  - weight lambda for the KL diversity loss term; 0 disables KL; default 0.05

### PRL-specific

- `model.pl_weight`
  - weight for the progressive learning CE term; default 1.0
- `model.temp_scale`
  - progressive temperature decay base alpha; tau for step k is `temperature * temp_scale^(K-k)`; default 5.0
- `model.noise_factor`
  - Gaussian noise standard deviation for contrastive augmentation; default 0.01
- `model.cl_weight`
  - weight for the contrastive loss term; default 1.0
- `model.warmup_epochs`
  - epochs for which noise injection is suppressed; 0 means noise is always active; default 0

### Trainer

- `trainer.batch_size`
  - training and evaluation batch size; default 512
- `trainer.shuffle`
  - whether to shuffle the training data each epoch; default true
- `trainer.monitor`
  - metric to monitor for early stopping; default `ndcg@10`
- `trainer.max_epochs`
  - maximum number of training epochs; default 10
- `trainer.early_stopping.enabled`
  - whether to use early stopping; default true
- `trainer.early_stopping.patience`
  - number of non-improving epochs before stopping; default 2
- `trainer.optimizer.name`
  - optimizer class name; default `Adam`
- `trainer.optimizer.kwargs.lr`
  - learning rate; default 0.001

## How to Run ReaRec

Run from the repository root. All examples assume the `recbole3` conda environment is active.

### Basic SASRec + ERL

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=rearec \
  dataset.category=Video_Games \
  model.backbone=sasrec \
  model.learning_strategy=erl \
  model.reason_step=2 \
  runtime.output_dir=outputs/rearec_sasrec_erl
```

### SASRec + PRL with warmup

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=rearec \
  dataset.category=Video_Games \
  model.backbone=sasrec \
  model.learning_strategy=prl \
  model.reason_step=2 \
  model.warmup_epochs=3 \
  model.noise_factor=0.01 \
  runtime.output_dir=outputs/rearec_sasrec_prl
```

### Disable reasoning (SASRec baseline)

Setting `reason_step=0` skips all reasoning steps. The output is `[B, 1, D]` and the model behaves as a plain SASRec encoder.

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=rearec \
  dataset.category=Video_Games \
  model.backbone=sasrec \
  model.reason_step=0 \
  runtime.output_dir=outputs/rearec_sasrec_baseline
```

### HSTU backbone (requires fbgemm-gpu and timestamp data)

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=rearec \
  dataset.category=Video_Games \
  model.backbone=hstu \
  model.learning_strategy=prl \
  model.reason_step=2 \
  model.attention_dim=32 \
  model.linear_hidden_dim=32 \
  runtime.output_dir=outputs/rearec_hstu_prl
```

### Override trainer settings from the command line

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=rearec \
  dataset.category=Video_Games \
  model.backbone=sasrec \
  model.learning_strategy=prl \
  trainer.max_epochs=20 \
  trainer.early_stopping.patience=5 \
  trainer.optimizer.kwargs.lr=0.0005 \
  runtime.output_dir=outputs/rearec_custom_trainer
```

## Suggested Debugging Workflow

When validating a new setup, proceed in this order:

1. run `backbone=sasrec` with `reason_step=0` and verify baseline SASRec metrics are reasonable
2. set `reason_step=2` and `learning_strategy=erl` with `kl_weight=0` to confirm reasoning steps compile and produce improving metrics without the KL term
3. enable `kl_weight` and confirm ERL loss goes negative as expected when diversity is enforced
4. switch to `learning_strategy=prl` with `warmup_epochs=0` and verify progressive and contrastive terms are non-zero
5. then introduce `warmup_epochs` and `noise_factor` tuning
6. only after SASRec is stable, attempt `backbone=hstu` with a timestamp-enriched dataset

This isolates whether problems come from:

- the backbone encoder itself
- the KV cache logic in the reasoning loop
- the loss function formulation
- data collation (padding direction, timestamp availability)
- noise augmentation timing

## Common Pitfalls

- using `backbone=hstu` without `fbgemm_gpu` installed
  - raises `RuntimeError` on model initialization; install fbgemm-gpu before proceeding

- using `backbone=hstu` on a dataset without timestamps
  - `HSTUModelDataset` raises `ValueError` at dataset build time; switch to a dataset with timestamp fields or use `backbone=sasrec`

- setting `reason_step=0` with `learning_strategy=prl`
  - no reasoning tokens exist to augment; the contrastive CE term is skipped automatically and the output is `[B, 1, D]` using only `h_0`

- setting `warmup_epochs > 0` without calling `build_train_collator` before the first forward call
  - `steps_per_epoch` is `None` and the epoch counter cannot be computed; noise injection stays off conservatively for the entire run until the collator is properly built

- ERL loss going negative during training
  - this is expected behaviour; the KL diversity term is subtracted, so large `kl_weight` values or well-separated reasoning step distributions can push the total ERL loss below zero

- HSTU backbone requires datasets with timestamps
  - pure interaction-ID datasets without a timestamp field will fail at the collator stage even if the backbone initialises successfully

- KV cache shape mismatch after changing `history_max_length` mid-run
  - cached K, V tensors are allocated based on the sequence length at encode time; changing `history_max_length` between the initial encode and a reasoning step will cause a shape error; always reload a fresh model if you change this parameter

## Summary

ReaRec in RecBole3.0 extends the standard sequential recommendation setup with an autoregressive reasoning loop:

- a backbone encoder (SASRec or HSTU) produces an initial user representation
- K additional reasoning steps iteratively refine that representation using learnable position embeddings
- ERL ensembles all K+1 representations and adds a KL diversity regulariser
- PRL trains with progressively harder targets per step and optionally adds a contrastive noise-augmented trajectory loss
- the SASRec backbone uses KV caching to keep the per-step cost linear in the sequence length
- the HSTU backbone provides a more expressive encoder for timestamp-enriched datasets at the cost of no caching and an external fbgemm-gpu dependency
