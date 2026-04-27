# LLMRank in RecBole3.0

## Overview

This document describes the `llmrank` model implemented in this repository, including:

- the high-level architecture
- the runtime flow
- how candidate generation, reranking, and evaluation work
- how to launch common experiments
- the meaning of frequently used configuration options

This implementation is built on top of the RecBole3.0 framework, but follows the core idea of the original LLMRank project:

1. use one backbone retriever to produce one candidate list
2. feed the candidate list plus user history into one LLM
3. let the LLM rerank that candidate list
4. evaluate the final ranked list with retrieval metrics such as `ndcg@10` and `recall@10`

## File Structure

The main LLMRank implementation lives in these files:

- `src/recbole3/model/llmrank/config.py`
  - LLMRank model config dataclass
- `src/recbole3/model/llmrank/pipeline.py`
  - end-to-end pipeline entry for LLMRank
- `src/recbole3/model/llmrank/candidates.py`
  - candidate generation and backbone wrapping
- `src/recbole3/model/llmrank/model.py`
  - prompt construction, backend calls, response parsing, and reranking
- `src/recbole3/model/llmrank/trainer.py`
  - final reranking evaluation logic
- `configs/model/llmrank.yaml`
  - default LLMRank model and trainer config

Related framework files:

- `src/recbole3/trainer.py`
  - generic model training and evaluation
- `src/recbole3/evaluation/methods/base.py`
  - shared evaluation helpers
- `src/recbole3/evaluation/methods/full.py`
  - full-sort retrieval evaluation method

## High-Level Architecture

The runtime flow is:

1. prepare the task dataset
2. generate backbone candidates for `valid` and `test`
3. inject those candidates into the eval frames
4. build model-side prepared data
5. run LLMRank evaluation on the injected candidate sets

In code, the main path is:

- `LLMRankPipeline.run()`
- `build_candidate_frames(...)`
- `LLMRankTrainer.run(...)`
- `LLMRankModel.predict(...)`

## Runtime Flow in Detail

### 1. Dataset Preparation

`LLMRankPipeline` first prepares the base task dataset using the framework's normal dataset pipeline.

At this stage, the repository's common dataset logic has already built:

- train split
- valid split
- test split
- user history fields such as `SEEN_ITEM_IDS`

### 2. Candidate Generation

Before reranking, LLMRank constructs one candidate list for each evaluation row.

This is handled in `src/recbole3/model/llmrank/candidates.py`.

The implementation supports three kinds of candidate sources:

- `random`
  - random sampling from all items except seen history
- `bm25`
  - lexical retrieval using item text and user history text
- any registered retrieval backbone model
  - for example `hstu`
  - the generic wrapper can also support future retrieval backbones, as long as they are registered in the model table and implement retrieval `predict(...)`

### 3. Selected Users Subset

Like the original LLMRank workflow, this implementation can evaluate only a subset of users.

- `model.selected_user_count=200`
  - sample 200 users and only evaluate them
- `model.selected_user_count=-1`
  - use all evaluable users

This user filtering is applied during candidate-frame construction.

### 4. Candidate Injection

After candidates are built, `LLMRankPipeline._inject_candidates(...)` replaces the valid and test eval datasets with new `FrameDataset`s that contain `candidate_item_ids`.

Those candidate lists then become the only items that LLMRank reranks.

### 5. Final Reranking

`LLMRankModel` is an inference-only retrieval model.

It does not train with a loss. Instead, during evaluation it:

1. receives one user history plus one candidate list
2. constructs one prompt
3. sends the prompt to one backend
4. parses the response back into reordered item ids
5. returns the top-k ranked item ids

### 6. Final Evaluation

The final LLMRank evaluation is handled by `src/recbole3/model/llmrank/trainer.py`.

Important points:

- evaluation protocol is `full`
- but LLMRank only reranks the provided candidate subset
- the final metrics are computed on the reranked candidate outputs
- the trainer supports the official-style controls:
  - `has_gt`
  - `fix_pos`
  - `shuffle`
  - `recall_budget`

## Candidate Generation Design

### Candidate Source = `random`

`random` is useful for probing zero-shot ranking ability.

Behavior:

- sample candidates from all items
- exclude the user's seen history via `SEEN_ITEM_IDS`
- output `backbone_topk` candidates per row

### Candidate Source = `bm25`

`bm25` uses item text and user interaction history text.

Behavior:

- tokenize item text
- build one BM25 index over the full item corpus
- use the user's history items as one query
- exclude seen items via `SEEN_ITEM_IDS`
- return the top `backbone_topk` candidates

### Candidate Source = one retrieval backbone model

Example:

- `model.candidate_source=hstu`

Behavior:

- load the registered model spec
- read `configs/model/<backbone>.yaml`
- merge `model.backbone_model` overrides into the backbone model config
- merge `model.backbone_trainer` overrides into the backbone trainer config
- load one checkpoint if available, otherwise auto-train the backbone
- use the backbone's retrieval `predict(...)` to generate top-k candidates

This generic wrapper is the reason future retrieval models can be plugged in more easily.

## Backbone Training

When `candidate_source` is one retrieval backbone model and `model.backbone_checkpoint_path` is not provided, LLMRank will auto-train the backbone.

### Which config controls backbone training?

Backbone training is controlled by:

- `model.backbone_model`
- `model.backbone_trainer`

These are merged into the default backbone config file:

- `configs/model/<backbone>.yaml`

For `hstu`, that means:

- `configs/model/hstu.yaml`

### Early Stopping

Backbone training now supports early stopping again.

You can control it in two ways:

- legacy style used in this repository:
  - `stopping_step`
- fully expanded trainer style:
  - `early_stopping.enabled`
  - `early_stopping.patience`
  - `early_stopping.min_delta`

If `stopping_step` is provided, it is mapped into the framework's early stopping config.

### Best vs Last Checkpoint

Backbone auto-training now:

- saves both `best_model.pt` and `last_model.pt`
- loads `best_model.pt` by default
- falls back to `last_model.pt` only if `best_model.pt` is unavailable

This behavior is important because LLMRank candidate quality depends directly on backbone retrieval quality.

## Prompt Construction

Prompt construction is implemented in `src/recbole3/model/llmrank/model.py`.

Supported prompt strategies:

- `sequential`
  - use ordered interaction history directly
- `recency_focused`
  - ordered history plus extra emphasis on the most recent interaction
- `in_context_learning`
  - include one demonstration-style example in the prompt

Supported parsing strategies:

- `title`
  - LLM outputs item titles
- `index`
  - LLM outputs candidate order indices

## Reranking Backends

Supported backends:

- `identity`
  - do not rerank; directly use the candidate list order
- `heuristic_overlap`
  - simple non-LLM heuristic reranker
- `openai`
  - call one OpenAI-compatible API endpoint, including local vLLM service
- `local_hf`
  - load and run one local Hugging Face model directly

### `identity`

This is useful for measuring backbone quality without LLM reranking.

### `openai`

This backend is appropriate when you:

- call the real OpenAI API, or
- run one local vLLM server that exposes an OpenAI-compatible endpoint

### `local_hf`

This backend runs the local model directly in-process.

It is simple, but can be much slower than serving the model via vLLM.

## Common Configuration Parameters

Below are the most important parameters from `configs/model/llmrank.yaml`.

### Candidate Construction

- `model.candidate_source`
  - candidate generator source
  - examples: `random`, `bm25`, `hstu`

- `model.backbone_topk`
  - number of raw candidates produced by the backbone

- `model.recall_budget`
  - number of candidates finally passed into LLMRank evaluation

- `model.selected_user_count`
  - number of users used for evaluation
  - `-1` means all users

- `model.has_gt`
  - whether to ensure the ground-truth item is inserted into the final candidate list

- `model.fix_pos`
  - where to insert the ground-truth item if `has_gt=true`
  - `-1` means append before optional shuffle

- `model.shuffle`
  - whether to shuffle the final candidate list order before reranking
  - this is especially important when `has_gt=false` and when using `identity`

### Candidate Cache and Files

- `model.candidate_cache_dir`
  - internal cache directory for generated candidates and auto-trained backbone outputs

- `model.candidate_file_dir`
  - external candidate-file directory

- `model.use_candidate_file`
  - whether to read external candidate files first

- `model.refresh_candidate_cache`
  - whether to ignore existing candidate caches and rebuild them

### Backbone Training

- `model.backbone_checkpoint_path`
  - explicit path to one trained backbone checkpoint

- `model.backbone_model`
  - overrides for the backbone model config

- `model.backbone_trainer`
  - overrides for the backbone trainer config

Useful subfields inside `model.backbone_trainer`:

- `batch_size`
- `shuffle`
- `max_epochs`
- `eval_steps`
- `monitor`
- `stopping_step`
- `optimizer.name`
- `optimizer.kwargs.lr`
- `eval.protocol`
- `eval.exclude_history`
- `eval.metrics`

### Prompt and Parsing

- `model.history_max_length`
  - max history length used in prompts

- `model.prompt_strategy`
  - `sequential`, `recency_focused`, `in_context_learning`

- `model.parsing_strategy`
  - `title` or `index`

- `model.domain`
  - controls wording in prompts
  - examples: `product`, `movie`, `item`

- `model.boots`
  - number of bootstrapping rounds for repeated prompting with shuffled candidate order

### LLM Backend

- `model.backend`
  - `identity`, `heuristic_overlap`, `openai`, `local_hf`

- `model.api_base_url`
  - OpenAI-compatible endpoint URL

- `model.api_model_name`
  - remote model name or served model identifier

- `model.api_batch`
  - concurrent request batch size for API dispatch

- `model.async_dispatch`
  - whether to dispatch API requests concurrently

- `model.temperature`
  - generation temperature

- `model.max_output_tokens`
  - output token limit for API backend

- `model.api_response_cache_path`
  - disk cache for API responses

- `model.refresh_api_response_cache`
  - whether to ignore cached API responses

### Local Hugging Face Backend

- `model.local_model_path`
- `model.local_tokenizer_path`
- `model.local_device`
- `model.local_device_map`
- `model.local_dtype`
- `model.local_batch_size`
- `model.local_max_output_tokens`
- `model.local_max_input_tokens`
- `model.local_trust_remote_code`
- `model.local_attn_implementation`
- `model.local_use_chat_template`

## How to Run LLMRank

Run from the repository root.

### 1. Backbone only: HSTU + identity

This measures the backbone candidate order directly.

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=llmrank \
  dataset.category=Video_Games \
  model.candidate_source=hstu \
  model.backend=identity \
  runtime.output_dir=outputs/llmrank_hstu_identity
```

### 2. Zero-shot style ranking: random candidates + LLM

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=llmrank \
  dataset.category=Video_Games \
  model.candidate_source=random \
  model.backbone_topk=20 \
  model.recall_budget=20 \
  model.has_gt=true \
  model.shuffle=true \
  model.backend=openai \
  model.api_base_url=http://127.0.0.1:8000/v1/chat/completions \
  model.api_model_name=/path/to/served/model \
  runtime.output_dir=outputs/llmrank_random_vllm
```

### 3. HSTU backbone + local vLLM reranking

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=llmrank \
  dataset.category=Video_Games \
  model.candidate_source=hstu \
  model.backend=openai \
  model.selected_user_count=1000 \
  model.api_base_url=http://127.0.0.1:8000/v1/chat/completions \
  model.api_model_name=/path/to/served/model \
  runtime.output_dir=outputs/llmrank_hstu_vllm
```

### 4. Reuse one trained backbone checkpoint

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=llmrank \
  dataset.category=Video_Games \
  model.candidate_source=hstu \
  model.backbone_checkpoint_path=outputs/my_hstu/checkpoints/best_model.pt \
  model.backend=identity \
  runtime.output_dir=outputs/llmrank_hstu_ckpt_identity
```

## Suggested Debugging Workflow

When debugging one new setup, it is usually best to proceed in this order:

1. run `candidate_source=hstu` with `backend=identity`
2. verify backbone metrics are reasonable
3. verify candidate cache is being written correctly
4. then switch `backend=openai` or `backend=local_hf`
5. only then compare prompt strategies and parsing strategies

This isolates whether problems come from:

- backbone candidate quality
- candidate alignment
- prompt construction
- response parsing
- API/backend behavior

## Common Pitfalls

- backbone metrics are strong, but final `identity` metrics are poor
  - candidate rows and eval rows may be misaligned

- `local_hf` is too slow
  - prefer serving the model through vLLM and using `backend=openai`

- the final candidate list looks too easy or too hard
  - check `has_gt`, `fix_pos`, and `shuffle`

- a backbone trains but LLMRank cannot use it
  - ensure the model is registered as a retrieval model and supports `predict(..., k=..., candidate_item_ids=None, exclude_item_ids=..., exclude_mask=...)`

## Summary

This LLMRank reproduction in RecBole3.0 keeps the project structure aligned with the framework while preserving the original rerank-then-evaluate idea:

- one backbone builds candidates
- one LLM reranks those candidates
- one retrieval-style evaluation reports the final result

The implementation is now also structured so that future retrieval backbones can be plugged in with much less model-specific code.
