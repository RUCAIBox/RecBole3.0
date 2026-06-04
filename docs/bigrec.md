# BIGRec in RecBole3.0

## Overview

This document describes the `bigrec` model implemented in this repository, including:

- the high-level two-step grounding paradigm
- the runtime flow for training and evaluation
- how prompt construction, beam-search generation, and embedding grounding work
- the optional grounding weight injection (Eq. 3 from the paper)
- how to launch common experiments
- the meaning of frequently used configuration options

BIGRec (Bi-Step Grounding Paradigm for Large Language Models in Recommendation,
arXiv:2308.08434) uses a two-step approach:

1. **Step 1 — Language space → Recommendation space**: Fine-tune a causal LLM
   (LLaMA + LoRA) to generate the title of the next item given a user's
   interaction history in an Alpaca-style instruction-following format.
2. **Step 2 — Recommendation space → Actual item space**: Embed the generated
   title with the LLM, compute L2 distances to pre-computed item embeddings,
   and return the closest items as recommendations.

Current implementation notes:

- BIGRec uses a custom `BIGRecPipeline` and `BIGRecTrainer` instead of the
  standard RecBole3.0 `Pipeline` / `Trainer`, mirroring the LCRec integration
  pattern.
- Training delegates the optimisation loop to HuggingFace `Trainer`.
- LoRA adapters are supported via the `peft` library.
- Evaluation is split into three phases (generation → embedding → ranking) so
  that each heavy model can be freed from VRAM before the next is loaded.
- An optional vLLM backend accelerates generation by ~10-30×.
- An optional grounding weight injection step (Eq. 3) reweights L2 distances
  by item popularity or pre-computed CF scores to improve ranking.

## File Structure

The main BIGRec implementation lives in these files:

- `src/recbole3/model/bigrec/config.py`
  - `BIGRecConfig` dataclass — all hyperparameters for training and evaluation
- `src/recbole3/model/bigrec/data.py`
  - `BIGRecModelDataset` — injects `history_item_ids` into all dataset splits
  - `BIGRecSFTDataset` — tokenises (history, target) pairs for LoRA fine-tuning
  - Prompt utilities: `build_instruction`, `build_input_block`, `build_prompt`,
    `build_eval_prompts`, `build_item_text_lookup`, `batchify`
- `src/recbole3/model/bigrec/trainer.py`
  - `BIGRecTrainer` — LoRA SFT fine-tuning (`fit`) and embedding grounding
    evaluation (`evaluate`), including Eq. 3 weight injection
- `src/recbole3/model/bigrec/pipeline.py`
  - `BIGRecPipeline` — orchestrates the full training and/or evaluation run
- `src/recbole3/model/bigrec/__init__.py`
  - Public exports for the sub-package
- `configs/model/bigrec.yaml`
  - Default BIGRec model config

Related framework files:

- `src/recbole3/model/__init__.py`
  - MODEL_TABLE entry for `bigrec`
- `src/recbole3/model/sequential.py`
  - `BaseSequentialModelDataset`, `HISTORY_ITEM_IDS` (used by `BIGRecModelDataset`)
- `src/recbole3/evaluation/metric.py`
  - `RecallMetric`, `NDCGMetric`, `RetrievalEvalData` (used by `BIGRecTrainer`)

## High-Level Architecture

```
BIGRecPipeline.run()
│
├── BIGRecModelDataset.from_task_dataset()    # inject history_item_ids
│
├── [pipeline_stage = 'training']
│   ├── BIGRecTrainer.fit()                   # Step 1: LoRA SFT fine-tuning
│   │   ├── _load_tokenizer(padding_side='right')
│   │   ├── build_item_text_lookup()
│   │   ├── BIGRecSFTDataset(train_frame, tokenizer, ...)
│   │   ├── _load_model(inference_mode=False)
│   │   └── HuggingFace Trainer.train()
│   │
│   └── BIGRecTrainer.evaluate()             # Step 2: embedding grounding eval
│
└── [pipeline_stage = 'evaluation']          # load checkpoint → evaluate only
    └── BIGRecTrainer.evaluate()
```

The evaluation path runs in three sequential phases so that each model is freed
from VRAM before the next is loaded:

```
BIGRecTrainer.evaluate()
│
├── Phase 1: Generation
│   ├── [use_vllm=False]  _load_trained_model()        # base + LoRA adapter
│   │                     _generate_all_titles()        # beam-search loop
│   │                     → free gen_model from VRAM
│   │
│   └── [use_vllm=True]   _generate_all_titles_vllm()  # continuous batching
│                         → vLLM engine torn down after generation
│
├── Phase 2: Embedding
│   ├── _load_base_model_for_embedding()    # base LLM (no LoRA)
│   │   [or _load_trained_model() when embedding_use_base_model=False]
│   ├── _precompute_item_embeddings()       # [num_items, H], cached to disk
│   └── _build_grounding_weights()         # [num_items] or None
│
└── Phase 3: Ranking
    ├── [grounding_gamma_search=True]
    │   ├── _extract_embeddings(valid_texts)    # oracle embs for val split
    │   ├── torch.cdist()                       # [N_valid, num_items]
    │   ├── _run_gamma_search()                 # best γ per metric@K on val
    │   ├── _extract_embeddings(eval_texts)     # oracle embs for test split
    │   ├── torch.cdist()                       # [N_eval, num_items]
    │   └── _evaluate_from_dist_per_k_gammas()  # apply per-K best γ
    │
    └── [standard path]
        └── _rank_from_texts()                  # embed + L2 rank + metrics
            ├── _extract_embeddings()           # oracle embs [B, H]
            ├── torch.cdist()                   # L2 distances [B, num_items]
            ├── _apply_grounding_weights()  [optional]
            ├── argsort(effective_dist)         # top-K item ids
            └── _compute_metrics()             # Recall@K, NDCG@K
```

## Runtime Flow in Detail

### 1. Dataset Preparation

`BIGRecPipeline` calls the framework's standard dataset pipeline, then wraps
the resulting task dataset in `BIGRecModelDataset.from_task_dataset()`.

`BIGRecModelDataset` adds a `history_item_ids` column to every split (train,
valid, test) by accumulating each user's interaction sequence up to — but not
including — the current interaction.  Cross-split accumulation ensures that the
valid and test histories include all interactions from earlier splits.

Truncation to `history_max_length` most-recent items is applied at prompt
construction time (not during dataset preparation), so the stored history is
always the full prefix sequence.

### 2. LoRA Fine-tuning (Step 1)

`BIGRecTrainer.fit()` fine-tunes the CausalLM backbone with LoRA:

1. Load tokenizer (right-padded for SFT).
2. Build item text lookup — a list indexed by framework `item_id`.
3. Optionally subsample the training frame (controlled by `sample_num`).
4. Resolve `effective_max_steps`: when `num_train_epochs` finishes within
   `max_steps`, the epoch limit is used naturally; otherwise `max_steps` caps.
5. Tokenize all training rows with `BIGRecSFTDataset`:
   - Each sample is an Alpaca-format prompt ending with the target item title
     as the response.
   - When `train_on_inputs=False`, prompt tokens are masked to `-100` so only
     the response tokens contribute to cross-entropy loss.
   - When `train_on_inputs=True` (official BIGRec default), loss is computed
     over the full sequence.
6. Load model from `llm_path` and wrap with a `LoraConfig` (if `use_lora=True`).
7. Delegate the training loop to HuggingFace `Trainer` with `TrainingArguments`
   mirroring the `BIGRecConfig` fields.  Early stopping monitors validation LM
   loss via `EarlyStoppingCallback`.
8. Save the checkpoint to `output_dir`.
9. Explicitly free the model from VRAM (`del model; torch.cuda.empty_cache()`)
   so the subsequent `evaluate()` call can load without OOM.

### 3. Embedding Grounding (Step 2)

`BIGRecTrainer.evaluate()` runs in three sequential phases.

#### Phase 1 — Generation

For each evaluation row, an Alpaca-format prompt is built from the user's
interaction history and fed to the model for beam-search generation.  The
decoded output is stripped of surrounding double-quotes to produce the oracle
title text.

Two backends are supported:

- **HuggingFace (default, `use_vllm=False`)**: The fine-tuned model is loaded
  once and used for a batched beam-search loop.  The model is freed from VRAM
  after generation completes.
- **vLLM (`use_vllm=True`)**: All prompts are passed to vLLM's continuous
  batching engine in a single call, achieving ~10-30× higher throughput.
  The vLLM engine is torn down and GPU memory freed after generation.

When `grounding_gamma_search=True`, Phase 1 is repeated on the validation
split so that the best γ can be found before evaluating the test split.

#### Phase 2 — Item embedding pre-computation

For each item title in the item table, the last-token hidden state of the last
transformer layer is extracted as the item embedding.  The LLM uses left-padding
so that position `-1` always contains the last real token.

When `embedding_use_base_model=True` (official BIGRec default), a fresh base
LLM without the LoRA adapter is loaded.  This ensures item embeddings and oracle
embeddings are computed in the same vector space as the pre-trained model.

Embeddings are cached as a `.pt` file under `embedding_cache_dir` and reloaded
on subsequent runs (unless `refresh_embedding_cache=True`).

#### Phase 3 — Ranking

Oracle embeddings are extracted from the Phase 1 generated titles, then L2
distances to all pre-computed item embeddings are computed via `torch.cdist`.
Items are ranked by ascending effective distance and the top-K ids are returned.

When `grounding_gamma_search=True`:

1. The same process is run on the validation split.
2. `_run_gamma_search` evaluates every γ in the candidate grid and records the
   best γ independently per metric×K combination.
3. The best-found γ values are applied on the test split via
   `_evaluate_from_dist_per_k_gammas`.

### 4. Grounding Weight Injection (Eq. 3)

When `grounding_mode` is not `'none'`, raw L2 distances are reweighted before
ranking:

```
D̂ᵢ = (Dᵢ − min_j Dⱼ) / (max_j Dⱼ − min_j Dⱼ)    [per-row min-max normalisation]
D̃ᵢ = D̂ᵢ × (1 + Wᵢ)^(−γ)                           [popularity / CF adjustment]
```

A higher Wᵢ lowers D̃ᵢ and promotes the item in the ranking.  The exponent γ
controls the strength of the weight signal.

Weight sources:

| `grounding_mode` | Weight Wᵢ source |
|---|---|
| `none` | No reweighting — pure L2 ranking |
| `popularity` | Cᵢ = Nᵢ / Σ Nⱼ (training interaction counts), min-max normalised to [0, 1] |
| `cf` | Pre-computed CF model scores from `cf_score_path` (.pt file, shape [num_items]), min-max normalised |
| `popularity+cf` | Sum of both signals, then re-normalised to [0, 1] |

**Key property of Eq. 3**: the item with the absolute minimum L2 distance always
has D̂ = 0 and therefore D̃ = 0, placing it at rank 1 regardless of weights.
Weight injection is most effective at reordering items that are NOT the uniquely
closest to the oracle embedding.

The paper searches γ over [0, 100] on the validation split per top-K cutoff.
Setting `grounding_gamma_search=True` automates this search using the official
199-value grid (0.00–0.99 in steps of 0.01, then 1–99 in steps of 1), or a
custom grid via `grounding_gamma_search_values`.

## Prompt Format

BIGRec uses an Alpaca-style three-section prompt.  The exact wording is
controlled by `domain` (`'product'`, `'movie'`, `'item'`).

For `domain='product'`:

```
Below is an instruction that describes a task, paired with an input that
provides further context. Write a response that appropriately completes the
request.

### Instruction:
Given a list of products the user has purchased before, please recommend
a new product that the user likes to the user.

### Input:
The user has purchased the following products before:"<title_1>", "<title_2>", …, "<title_N>"


### Response:
"<target_title>"     ← present during SFT; absent during beam-search eval
```

For `domain='movie'`, the wording changes to "movies … watched"; for
`domain='item'`, it uses "items … interacted with".

## File Structure of the Embedding Cache

Item embeddings are stored as:

```
<embedding_cache_dir>/<dataset_name>_<split>_item_embs.pt
```

The cache is a float32 CPU tensor of shape `[num_items, hidden_size]`.  Index
`i` corresponds to framework `item_id = i`; index 0 is the placeholder item
reserved by the framework.

The cache path encodes the dataset name and split but **not** the model name.
When switching to a different checkpoint or base model, either change
`embedding_cache_dir` or set `refresh_embedding_cache=true`.

## Common Configuration Parameters

Below are the most important parameters from `configs/model/bigrec.yaml`.

### LLM Backbone

- `model.llm_path`
  - **Required.** Local directory or HuggingFace Hub identifier of the
    pretrained LLaMA model.

- `model.torch_dtype`
  - Weight dtype: `'float16'` or `'bfloat16'`.

- `model.load_in_8bit`
  - Load model in INT8 quantisation via `bitsandbytes` (reduces VRAM by ~50%).
  - Requires `bitsandbytes` to be installed separately.

- `model.attn_implementation`
  - `'eager'` (default) or `'flash_attention_2'` (requires the `flash-attn` package).

- `model.pipeline_parallel` / `model.pipeline_parallel_gpus`
  - Enable single-process pipeline parallelism to shard the model across
    `pipeline_parallel_gpus` consecutive GPUs (starting from `device_id`).
  - Does not require `torchrun`; uses HuggingFace `device_map='auto'`.
  - Default: `false` / `2`.

### LoRA Fine-tuning

- `model.use_lora`
  - Whether to apply LoRA adapters (strongly recommended; keeps base model frozen).

- `model.lora_r`
  - LoRA rank.  Paper default: 8.

- `model.lora_alpha`
  - LoRA scaling factor.  Paper default: 16.

- `model.lora_target_modules`
  - List of linear sub-module names to replace with LoRA adapters.
  - Default: `[q_proj, v_proj]`.

### Training

- `model.train_batch_size` / `model.gradient_accumulation_steps`
  - Effective batch size = `train_batch_size × gradient_accumulation_steps × num_gpus`.

- `model.num_train_epochs`

- `model.learning_rate`
  - Official BIGRec default: `3e-4`.

- `model.lr_scheduler_type`
  - `'cosine'` (default), `'linear'`, `'constant'`, etc.

- `model.gradient_checkpointing`
  - Recomputes activations to reduce peak VRAM at extra compute cost.

- `model.train_on_inputs`
  - `true` (default): full-sequence loss — matches official BIGRec training.
  - `false`: response-only supervision — prompt tokens masked to `-100`.

- `model.sample_num`
  - Randomly subsample this many training rows (integer count) before training begins.
  - Mirrors the official BIGRec `--sample` flag.  `-1` (default) uses the
    full training set.  Unlike `max_steps`, the dataset is shrunk and then
    trained to completion for `num_train_epochs` epochs.

- `model.max_steps`
  - Hard cap on the total number of optimiser steps.  Useful for quick
    sanity-checks on large datasets.  `-1` disables the cap and lets
    `num_train_epochs` control training.  Default: `500`.

- `model.deepspeed`
  - Path to a DeepSpeed JSON config for multi-GPU / ZeRO optimisation, or `null`.

### Tokenization and Generation

- `model.max_input_length`
  - Maximum token length for the prompt (instruction + input).

- `model.max_new_tokens`
  - Maximum tokens generated per beam-search call.

- `model.num_beams`
  - Beam-search width.  Paper default: 4.

### Embedding Grounding

- `model.history_max_length`
  - Number of most-recent history items included in the prompt.  Paper uses 10.

- `model.item_text_field`
  - Column in `item_table` used as the natural-language item name.
  - Default: `title`.

- `model.fallback_item_text_field`
  - Fallback column when `item_text_field` is absent or empty.
  - Default: `metadata_text`.

- `model.domain`
  - Prompt wording domain: `'movie'`, `'product'` (default), or `'item'`.

- `model.embedding_batch_size`
  - Batch size for encoding item titles into embeddings.

- `model.embedding_cache_dir`
  - Directory for pre-computed item embedding `.pt` cache files.

- `model.refresh_embedding_cache`
  - Re-compute item embeddings even when a cached file exists.

- `model.embedding_use_base_model`
  - `true` (default): use the base LLM (no LoRA) for both item and oracle
    embeddings, so both live in the same vector space.
  - `false`: use the fine-tuned LoRA model for embedding extraction.

### Grounding Weight Injection (Eq. 3)

- `model.grounding_mode`
  - `'none'` (default): pure L2 ranking.
  - `'popularity'`: inject training-interaction-count weights.
  - `'cf'`: inject pre-computed CF model scores (requires `cf_score_path`).
  - `'popularity+cf'`: sum both signals and re-normalise.

- `model.grounding_gamma`
  - γ exponent.  γ = 0 disables reweighting even when `grounding_mode` is set.
  - Default: `1.0`.  Paper searches [0, 100] per top-K on the validation split.

- `model.grounding_gamma_search`
  - `true`: auto grid-search the best γ independently per metric×K on the
    validation split (official BIGRec procedure), then apply the best-found
    values on the test split.
  - Default: `false`.

- `model.grounding_gamma_search_values`
  - Custom γ grid for the search.  Empty list (default) uses the official
    199-value grid: `[0.00, 0.01, …, 0.99, 1, 2, …, 99]`.

- `model.cf_score_path`
  - Path to a `.pt` file containing a 1-D float tensor of shape `[num_items]`
    with pre-computed CF scores (e.g., from BPR or SASRec).
  - Required when `grounding_mode` contains `'cf'`.

### Evaluation

- `model.eval_batch_size`
  - Per-device batch size for beam-search generation at eval time.

- `model.eval_metrics`
  - Metrics to compute: `recall`, `ndcg`.

- `model.eval_topk`
  - Top-K cutoffs, e.g. `[1, 5, 10, 20]`.

- `model.eval_protocol`
  - `'sampled'`: rank pre-defined `candidate_item_ids` sets.
  - `'full'`: rank all items (expensive; matches the paper's all-rank setting).

- `model.eval_user_num`
  - Number of users to evaluate per split.  Default: `5000`, mirroring the
    official BIGRec test set (random subsample with `random_state=42`).
  - `-1` evaluates all users in the split.

### vLLM Acceleration

- `model.use_vllm`
  - Use vLLM for generation during evaluation.  Achieves ~10-30× faster
    throughput than HuggingFace `generate()`.
  - Requires `pip install 'vllm>=0.4.0'`.  Default: `false`.

- `model.vllm_gpu_memory_utilization`
  - Fraction of GPU memory reserved by the vLLM KV-cache.  Default: `0.85`.

### Pipeline Control

- `model.pipeline_stage`
  - `'training'` (default): LoRA fine-tuning followed by test evaluation.
  - `'evaluation'`: load a saved checkpoint and run evaluation only.

- `model.checkpoint_path`
  - Path to a saved LoRA adapter directory.
  - Required when `pipeline_stage='evaluation'`.

## How to Run BIGRec

Run from the repository root inside the `recbole3` conda environment.

### 1. Full training + evaluation (default)

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  runtime.output_dir=outputs/bigrec_vg
```

### 2. Training with LoRA, then evaluation

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.lora_r=8 \
  model.lora_alpha=16 \
  model.num_train_epochs=3 \
  model.train_batch_size=4 \
  model.gradient_accumulation_steps=8 \
  model.num_beams=4 \
  runtime.output_dir=outputs/bigrec_vg_lora
```

### 3. Evaluation only from a saved checkpoint

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.pipeline_stage=evaluation \
  model.checkpoint_path=outputs/bigrec_vg/checkpoint \
  runtime.output_dir=outputs/bigrec_vg_eval
```

### 4. Training with popularity-based grounding weight injection

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.grounding_mode=popularity \
  model.grounding_gamma=10.0 \
  runtime.output_dir=outputs/bigrec_vg_pop
```

### 5. Grounding with automatic γ search on the validation split

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.grounding_mode=popularity \
  model.grounding_gamma_search=true \
  runtime.output_dir=outputs/bigrec_vg_pop_search
```

### 6. CF-weighted grounding (provide pre-computed CF scores)

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.grounding_mode=cf \
  model.grounding_gamma=5.0 \
  model.cf_score_path=outputs/bpr_scores.pt \
  runtime.output_dir=outputs/bigrec_vg_cf
```

### 7. Flash Attention 2 + bfloat16

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.attn_implementation=flash_attention_2 \
  model.torch_dtype=bfloat16 \
  model.bf16=true \
  runtime.output_dir=outputs/bigrec_vg_fa2
```

### 8. Fast evaluation with vLLM

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.pipeline_stage=evaluation \
  model.checkpoint_path=outputs/bigrec_vg/checkpoint \
  model.use_vllm=true \
  runtime.output_dir=outputs/bigrec_vg_vllm
```

### 9. Small-scale ablation with subsampled training data

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=bigrec \
  dataset.category=Video_Games \
  model.llm_path=/path/to/llama-2-7b-hf \
  model.sample_num=1024 \
  model.max_steps=-1 \
  runtime.output_dir=outputs/bigrec_vg_ablation
```

## Installing Dependencies

BIGRec requires the `bigrec` optional dependency group:

```bash
pip install "recbole3[bigrec]"
```

Or, to install manually:

```bash
pip install "transformers>=4.40,<5.0" "peft>=0.9.0,<0.19.0" "tqdm>=4.60.0"
```

For INT8 quantisation (`load_in_8bit=True`):

```bash
pip install bitsandbytes
```

For Flash Attention 2 (`attn_implementation=flash_attention_2`):

```bash
pip install flash-attn --no-build-isolation
```

For vLLM acceleration (`use_vllm=True`):

```bash
pip install "vllm>=0.4.0"
```

## Suggested Debugging Workflow

When setting up a new experiment, proceed in this order:

1. Verify `llm_path` is correct by checking the tokenizer loads cleanly.
2. Run a smoke test with `max_steps=5`, `train_batch_size=1`,
   `gradient_accumulation_steps=1`, small `max_input_length=128`.
3. Check that item embeddings are pre-computed and saved to `embedding_cache_dir`.
4. Inspect sample generated titles to verify beam-search decodes sensible text.
5. Confirm base L2 evaluation runs end-to-end before enabling grounding weight
   injection.
6. When using `grounding_mode != 'none'`, run `grounding_gamma_search=true` on
   the validation split to find the best γ before committing to a test evaluation.

## Common Pitfalls

- `llm_path` is empty or points to a missing directory
  - `AutoModelForCausalLM.from_pretrained` will raise `OSError` / `ValueError`.
  - Set `model.llm_path` to a local directory or a valid HuggingFace Hub ID.

- OOM during training
  - Reduce `train_batch_size` and increase `gradient_accumulation_steps`
    proportionally.
  - Enable `gradient_checkpointing=true` (already on by default).
  - Use `load_in_8bit=true` or `torch_dtype=bfloat16` to reduce VRAM usage.
  - The training model is explicitly freed from VRAM before `evaluate()` runs,
    but this requires `del` + `torch.cuda.empty_cache()` to take effect.

- OOM during evaluation on large datasets
  - Use `use_vllm=true` (more memory-efficient than HF batched generation).
  - Reduce `eval_user_num` to limit the number of users evaluated.
  - Reduce `eval_batch_size` or `embedding_batch_size`.

- Generated titles are empty or contain only quotes
  - Empty titles are replaced by `[empty_i]` placeholders to avoid zero-vector
    embeddings.  This indicates the LoRA model is not converging.
  - Check `learning_rate`, `lora_r`, and `num_train_epochs`.

- Embedding cache is stale after changing `llm_path` or LoRA checkpoint
  - Set `refresh_embedding_cache=true` to force re-computation.
  - The cache path encodes the dataset name and split (e.g.,
    `Video_Games_test_item_embs.pt`), not the model name; rename the cache
    directory or change `embedding_cache_dir` when switching models.

- `cf_score_path` shape mismatch
  - The `.pt` file must contain a 1-D float tensor of length exactly `num_items`
    (as reported by `data.get_num_items()`).
  - Shape is validated at load time; a `ValueError` is raised on mismatch.

- Training runs but evaluation metrics are all zero
  - Verify `eval_protocol` matches the dataset preparation.  Use `'full'` if
    the dataset does not provide `candidate_item_ids`.

- `max_steps` stops training too early on a small dataset
  - When `num_train_epochs` finishes within `max_steps`, the trainer
    automatically disables the `max_steps` cap so epoch count controls training.
  - For full training runs, set `max_steps=-1` explicitly.

## Summary

BIGRec in RecBole3.0 follows the original bi-step grounding paradigm:

1. LoRA fine-tuning teaches the LLM to generate item titles from interaction history.
2. L2 distance between the oracle embedding (generated title) and pre-computed item
   embeddings ranks all candidate items.
3. An optional grounding weight injection step (Eq. 3) reweights distances by
   popularity or CF signals to further improve ranking.

Evaluation runs in three phases (generation → embedding → ranking) so each
large model can be fully freed from VRAM before the next is loaded.  An
optional vLLM backend accelerates generation by ~10-30×.

The implementation is self-contained within `src/recbole3/model/bigrec/` and
integrates with the RecBole3.0 evaluation protocol through `RetrievalEvalData`,
`RecallMetric`, and `NDCGMetric`.
