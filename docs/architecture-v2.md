# RecBole3.0 v2 Architecture

## Goal

RecBole3.0 is a recommendation research toolkit built on explicit component registration and Hydra-driven configuration. It prioritizes:

- developer-friendly extension points
- a small number of stable concepts
- readable intermediate data (pandas DataFrames)
- low maintenance cost through explicit (non-magic) wiring

## High-Level Flow

```
hydra compose → Pipeline.run()
  ├── parse config (RuntimeConfig, dataset cfg, model cfg, trainer cfg)
  ├── resolve specs via DATASET_TABLE / MODEL_TABLE
  ├── instantiate: Dataset → Model → Trainer
  ├── dataset.prepare(eval_config)
  ├── optional: ModelDataset.from_task_dataset()  # model-side transforms
  └── trainer.run(model, prepared_data)
       ├── fit() → epochs → validation → early stopping / checkpoint
       └── evaluate() on test split
```

## Core Modules

### `config.py` — Top-Level App Configuration

`AppConfig` is the root Hydra-structured config composed from `configs/config.yaml`. It holds three opaque sub-configs (`dataset`, `model`, `trainer`) plus one structured block:

- `RuntimeConfig`: `seed`, `device` (accelerate override), `output_dir`

`instantiate_dataclass()` recursively coerces OmegaConf dicts into typed dataclass instances, validating that no unexpected keys are present.

### `run.py` — Entry Point

`main()` calls `compose_config()` (Hydra initialize + compose with CLI overrides), then delegates to `run_experiment()`, which:

1. Resolves `model.name` → `ModelSpec` via `get_model_spec()`
2. Instantiates `model_spec.pipeline_cls` (default: `Pipeline`)
3. Calls `pipeline.run()`

### `pipeline.py` — Experiment Orchestrator

`Pipeline` owns the end-to-end flow:

1. Parse config into `(RuntimeConfig, dataset_cfg, model_cfg, trainer_cfg)`
2. Resolve `dataset.name` → `DatasetSpec`, instantiate `Dataset(config)`
3. Instantiate `Model(config)` and `Trainer(trainer_config)` from `ModelSpec`
4. `dataset.prepare(eval_config=trainer_config.eval)` → prepared task data
5. Optional: `ModelDataset.from_task_dataset()` for model-side feature engineering
6. `trainer.run(model, prepared_data, output_dir=)` → fit + test
7. Print serialized summary

Model-specific pipelines (e.g., `LLMRankPipeline`, `LCRecPipeline`, `RankMixerPipeline`) can override the flow for non-standard workflows like LLM fine-tuning or two-stage training.

### `utils.py` — Framework Utilities

- `LazyImport(module, attr)`: deferred import for optional dependencies (e.g., `transformers`, `fbgemm-gpu`). Resolves on first attribute access or call.
- `require_component_cfg(cfg, component)`: validates and returns a sub-config node from the composed Hydra config.
- `require_component_name(component_cfg, component)`: extracts the `name` field used for table lookup.

---

## `dataset/` — Data Layer

The dataset layer is split into **parser** (source-specific raw data) and **task dataset** (framework-level preparation).

### Stable Concepts

| Concept | Role |
|---|---|
| `DatasetConfig` | Base config with `name` and `SplitConfig` |
| `SplitConfig` | `strategy`, `order`, `per_user`, ratios, holdout counts, `seed` |
| `BaseDatasetParser` | Abstract parser: `parse() → ParsedData` |
| `ParsedData` | `interactions` (DataFrame), optional `user_table`, `item_table` |
| `BaseTaskDataset` | Task-aware preparation, split logic, eval frame construction |
| `FrameDataset` | `torch.utils.data.Dataset` wrapping a DataFrame; `__getitems__` returns DataFrame slices directly |
| `DatasetCache` | JSONL-backed cache helper for parsers |

### Parser Layer (`parser.py`)

`BaseDatasetParser` is the abstract interface:

- Owns raw-data download (HuggingFace / ModelScope), cache management, normalization
- `parse()` returns `ParsedData`
- `data_dir` property for processed-file location

`ParsedData` contains:
- `interactions: pd.DataFrame` — must have `user_id`, `item_id`; optional `timestamp`, `label`
- `user_table: pd.DataFrame | None` — raw user metadata
- `item_table: pd.DataFrame | None` — raw item metadata

Extra columns beyond the recognized set are preserved through the pipeline.

### Task Dataset Layer (`base.py`)

`BaseTaskDataset` owns everything that happens after parsing:

- **ID remapping**: raw user/item ids → framework ids (both 0-based). `num_items` counts only real items; padding item ids are model-owned.
- **Entity table synthesis**: missing tables are derived from interaction ids; missing rows in provided tables are auto-appended.
- **Interaction ordering**: `chronological` (timestamp-based or original order) or `random` (seeded)
- **Splitting**: `ratio` or `leave_one_out`, optionally per-user
- **Eval frame construction**: builds `seen_item_ids` histories and optional `candidate_item_ids`
- **Accessor methods**: `get_train_dataset()`, `get_eval_dataset(split)`, `get_interactions()`, `get_user_table()`, `get_item_table()`, `get_num_users()`, `get_num_items()`

The `task` property infers `"ranking"` or `"retrieval"` from `EvalConfig.protocol`.

#### Split Strategies

- `ratio`: fractional split via `train_ratio` / `valid_ratio` / `test_ratio` with remainder distributed by fractional part
- `leave_one_out`: hold out last N interactions per group via `valid_holdout_num` / `test_holdout_num`

When `per_user=True`, ordering and splitting happen independently inside each user group.

#### Retrieval Eval Frames

Retrieval eval DataFrames are request-level, built from positive interactions only (`label` is positive when missing or > 0):

- `seen_item_ids` (tuple): first-seen unique item histories. Validation starts from positive train interactions; test starts from positive train + positive validation.
- `candidate_item_ids` (tuple, sampled only): deterministic seeded negative samples. Each row has the same tuple length; target item is first.

Protocol semantics:
- `full`: `candidate_item_ids` absent. Model scores all items.
- `sampled`: `candidate_item_ids` present. Model scores only candidates.
- `labeled`: per-row labels with scores (ranking).

### Schema Contracts (`utils.py`)

`FrameSchema(required, optional)` validates DataFrame columns. Column name constants:

- `USER_ID`, `ITEM_ID`, `TIMESTAMP`, `LABEL`
- `SEEN_ITEM_IDS`, `CANDIDATE_ITEM_IDS`

### Cache Helper (`cache.py`)

`DatasetCache(root)` provides JSONL read/write for parser caches:
- `read_frame()` / `write_frame()` for individual files
- `read_parsed()` / `write_parsed()` for the standard three-file layout (`interactions.jsonl`, `users.jsonl`, `items.jsonl`)
- `get_or_create_frame()` for lazy cache population with a builder callback

### Built-in Datasets

| Dataset | Task | Source |
|---|---|---|
| `amazon2023_retrieval` | retrieval | Amazon Reviews 2023 (HuggingFace / ModelScope) |
| `amazon2014_retrieval` | retrieval | Amazon Reviews 2014 (HuggingFace / ModelScope) |
| `avazu_ctr` | ranking | Avazu CTR prediction |

Each dataset provides: a config dataclass, a parser subclass, and a task dataset class (e.g., `Amazon2023RetrievalDataset`).

### Dataset Registration

```python
@dataclass(frozen=True)
class DatasetSpec:
    dataset_cls: type[BaseTaskDataset]
    config_cls: type[DatasetConfig]

DATASET_TABLE: dict[str, DatasetSpec] = { ... }
```

---

## `model/` — Model Layer

### Core Abstractions (`base.py`)

**`ModelConfig`** — base config dataclass with `name`.

**`BaseCollator`** — abstract collator called as `DataLoader.collate_fn`. Receives feature records (DataFrame or list), returns a model-ready batch dict. Initialized with `(config, prepared_data)`.

**`BaseModel(nn.Module)`** — the central model interface:

| Method | Purpose |
|---|---|
| `ensure_initialized(prepared_data)` | Lazy parameter init (e.g., embedding cardinality from `num_items`) |
| `build_train_collator(prepared_data) → BaseCollator` | Training batch collation |
| `build_eval_collator(prepared_data) → BaseCollator` | Evaluation batch collation |
| `forward(batch) → dict` | Forward pass on a collated batch |
| `compute_loss(batch, outputs) → Tensor` | Training loss |

Two task-specific subclasses:

- **`BaseRankingModel`**: `predict(model_inputs) → Tensor` returns one score per labeled eval row (shape `[B]`).
- **`BaseRetrievalModel`**: `predict(model_inputs, *, k, candidate_item_ids, exclude_item_ids, exclude_mask) → Tensor` returns top-k item ids (shape `[B, k]`).

**`ModelDatasets[TModelTrain, TModelEval]`** — container for optional model-side split replacements returned by `BaseModelDataset._build_model_datasets()`.

**`BaseModelDataset`** — model-side data transform that extends `BaseTaskDataset`:

- `from_task_dataset(dataset, model_config)` clones prepared state, calls `_build_model_datasets()`, applies replacements
- Subclasses override `_build_model_datasets()` to add model-specific columns (e.g., `history_item_ids`, `history_timestamps`)

### Sequential Model Support (`sequential.py`)

`SequentialModelConfig` adds `history_max_length: int | None`.

`build_history_item_ids(records, *, initial_histories, include_target_item, history_max_length)` builds per-row prefix histories and returns the final user-history state. Histories are iteratively constructed — each row sees the items from all prior rows of the same user.

`BaseSequentialModelDataset` uses `build_history_item_ids()` to add `history_item_ids` to train/valid/test frames, carrying history state across splits. This is the standard model-data class for sequence models.

### Model Registration

```python
@dataclass(frozen=True)
class ModelSpec:
    model_cls: type[BaseModel] | LazyImport
    config_cls: type[ModelConfig]
    model_data_cls: type[BaseModelDataset] | None = None
    trainer_cls: type[Trainer] = Trainer
    trainer_config_cls: type[TrainerConfig] = TrainerConfig
    pipeline_cls: type[Pipeline] | LazyImport = Pipeline

MODEL_TABLE: dict[str, ModelSpec] = { ... }
```

`LazyImport` allows deferring heavy dependencies until a model is actually used.

### Built-in Models

| Model | Type | Key Characteristics |
|---|---|---|
| `hstu` | Retrieval | HSTU jagged-attention sequential model; requires `fbgemm-gpu` |
| `lares` | Retrieval | SASRec-style transformer with recurrent state and GRPO RL stage |
| `rqvae` | Retrieval | RQ-VAE for semantic item IDs |
| `letter` | Retrieval | LETTER with custom trainer |
| `tiger` | Retrieval | TIGER generative retrieval |
| `rankmixer` | Ranking | Token-mixing MLP for CTR; custom pipeline |
| `rpg` | Ranking | RPG with custom trainer config |
| `lcrec` | Retrieval | LLM-based via `transformers.PreTrainedModel`; custom pipeline |
| `llmrank` | Ranking | LLM-based ranking; custom pipeline and trainer |

---

## `trainer.py` — Training Loop

`Trainer` aligns prepared data with the training loop, running through PyTorch DataLoaders + `accelerate`.

### Ownership

- Builds train/eval DataLoaders via `build_dataloader(dataset, collate_fn, shuffle)`
- Creates `accelerate.Accelerator` with mixed precision and gradient accumulation
- Constructs torch `Optimizer` and optional `LRScheduler` / `ReduceLROnPlateau` from config
- Runs `fit()`: epoch loop with loss tracking, end-of-epoch validation, monitor-driven early stopping, best/last checkpoint saves
- Runs `evaluate(model, prepared_data, split)`: standalone evaluation
- `run()`: complete `fit + test` entrypoint with logging and best-checkpoint reload

### Config (`trainer_config.py`)

`TrainerConfig` is a comprehensive dataclass:

| Field | Purpose |
|---|---|
| `batch_size`, `shuffle`, `dataloader_num_workers`, `pin_memory` | DataLoader settings |
| `mixed_precision`, `gradient_accumulation_steps` | Accelerate settings |
| `max_epochs`, `eval_steps` | Training schedule |
| `optimizer: OptimizerConfig` | `name` (torch.optim class), `kwargs` |
| `scheduler: SchedulerConfig \| None` | `name`, `interval` (step/epoch), `kwargs` |
| `monitor: str \| None` | Validation metric for early stopping and checkpointing |
| `early_stopping: EarlyStoppingConfig` | `enabled`, `patience`, `min_delta` |
| `checkpoint: CheckpointConfig` | `save_best`, `save_last` |
| `eval: EvalConfig` | Evaluation protocol and metrics |
| `save_inference_results`, `inference_topk` | Raw output capture |

### Custom Trainers

Models can ship custom trainers for specialized training logic:

- `LARESTrainer`: two-stage SL→RL with GRPO policy gradient and recurrence-scaling evaluation
- `LETTERTrainer`: custom loss and training flow
- `LLMRankTrainer`: LLM-specific training with `LLMRankTrainerConfig`
- `RQVAETrainer`: VAE-specific training
- `RPGTrainer`: RPG-specific with `RPGTrainerConfig`

Custom trainers override `run()` or `fit()` while reusing `build_dataloader`, `build_optimizer`, `build_scheduler`, and evaluation infrastructure from the base `Trainer`.

---

## `evaluation/` — Evaluation Layer

### Protocol Dispatch

Three evaluation methods, created by `create_evaluation_method(config)`:

| Protocol | Method Class | Model Requirement | Data Shape |
|---|---|---|---|
| `labeled` | `LabeledEvaluationMethod` | `BaseRankingModel` | Per-row score & label |
| `sampled` | `SampledEvaluationMethod` | `BaseRetrievalModel` | Top-k from candidate set |
| `full` | `FullEvaluationMethod` | `BaseRetrievalModel` | Top-k from all items (with optional history exclusion) |

### Evaluation Flow

1. `build_eval_collate_fn(model, prepared_data)` wraps the model's eval collator so each batch yields `(model_inputs, records)`
2. Per batch: `collect_batch(model, model_inputs, records)` calls `model.predict()` and packs results into `RankingEvalData` or `RetrievalEvalData`
3. `compute_metrics(batch_eval_data)` concatenates batch data and runs all configured metrics

### Config (`EvalConfig`)

```python
@dataclass
class EvalConfig:
    protocol: EvalProtocol              # "labeled" | "sampled" | "full"
    metrics: tuple[MetricSpec, ...]     # metric specs with optional ks
    neg_sampling_num: int = 100         # sampled negatives count
    candidate_seed: int = 42            # seed for sampled candidates
    exclude_history: bool = False       # filter seen items in full eval
```

### Metrics (`metric.py`)

`MetricSpec(name, ks)` configures a metric. `create_builtin_metrics()` instantiates:

| Metric | Protocol | Type | Top-k |
|---|---|---|---|
| `auc` | labeled | `BaseRankingMetric` | — |
| `gauc` | labeled | `BaseRankingMetric` | — |
| `logloss` | labeled | `BaseRankingMetric` | — |
| `recall` | sampled, full | `BaseRetrievalMetric` | `ks` |
| `ndcg` | sampled, full | `BaseRetrievalMetric` | `ks` |

Each metric has a `name`, `higher_is_better`, and `result_directions()` returning `{name: bool}` (with `@k` suffixes for retrieval metrics).

### Data Contracts

```python
@dataclass
class RankingEvalData:
    scores: np.ndarray      # [N] flat scores
    labels: np.ndarray      # [N] flat labels
    group_ids: np.ndarray   # [N] user/group indices for GAUC

@dataclass
class RetrievalEvalData:
    pred_item_ids: np.ndarray    # [B, K] top-k predictions
    target_item_ids: np.ndarray  # [B, T] target items (padded)
    target_mask: np.ndarray      # [B, T] boolean validity mask
```

### Full Evaluation History Exclusion

When `exclude_history=True`, the `FullEvaluationMethod` passes `seen_item_ids` as `exclude_item_ids` to `model.predict()`. The model is responsible for masking those items out of the score distribution.

---

## `logger.py` — Training Logger

`TrainingLogger` writes structured log files to `{output_dir}/logs/{model}_{dataset}_{category}_{timestamp}.log`:

- Config sections (Trainer, Model, Dataset) rendered as indented trees
- Model info: name, class, total/trainable parameter counts
- Dataset info: name, task, user/item counts, split record counts
- Per-epoch table: epoch, loss, time, batches, LR
- Validation metrics per evaluated epoch
- Early stopping and best-epoch markers
- Final test results with protocol and all metrics
- Closing summary: total epochs, best epoch, stopped-early flag, wall time

Only writes on rank 0 in distributed settings.

---

## Component Tables

The framework uses two explicit tables:

- `DATASET_TABLE` (`dataset/__init__.py`) — maps dataset name → `DatasetSpec(dataset_cls, config_cls)`
- `MODEL_TABLE` (`model/__init__.py`) — maps model name → `ModelSpec(model_cls, config_cls, model_data_cls, trainer_cls, trainer_config_cls, pipeline_cls)`

`get_dataset_spec(name)` / `get_model_spec(name)` look up entries and produce clear error messages listing available options.

---

## Config Structure

Hydra composes configs from `configs/`:
- `configs/config.yaml` — root with `runtime`, `dataset`, `model`, `trainer` defaults
- `configs/dataset/*.yaml` — selected via `dataset=...`
- `configs/model/*.yaml` — selected via `model=...`

Override syntax:
```bash
python -m recbole3.run dataset=amazon2023_retrieval model=hstu trainer.max_epochs=10
```

---

## Adding New Components

### Adding a Dataset

1. Subclass `DatasetConfig` with source-specific fields
2. Subclass `BaseDatasetParser` implementing `parse() → ParsedData`
3. Subclass `BaseTaskDataset` (or `RetrievalDataset` / `RankingDataset` pattern) binding `config_cls` and `parser_cls`
4. Add entry to `DATASET_TABLE`
5. Create YAML config under `configs/dataset/`

### Adding a Model

1. Subclass `ModelConfig` with model hyperparameters
2. Implement model class inheriting from `BaseRetrievalModel` or `BaseRankingModel`
3. Implement train and eval `BaseCollator` subclasses
4. Optionally create a `BaseModelDataset` subclass for custom data transforms
5. Optionally create a `Trainer` subclass for non-standard training
6. Add entry to `MODEL_TABLE`
7. Create YAML config under `configs/model/`

If the model depends on optional packages, use `LazyImport` in `ModelSpec` to avoid import errors until the model is actually selected.

---

## Key Patterns

- **DataFrames all the way down**: from parser output through prepared splits to collator input. `FrameDataset.__getitems__` returns DataFrame slices directly, so collators receive DataFrames — not lists of dicts.
- **Collator as boundary**: model-specific tensorization lives in the collator, not the dataset. The prepared dataset produces logical records; the collator produces model-ready tensors.
- **Lazy parameter init**: models that need `num_items` for embedding construction implement `ensure_initialized()`. The trainer calls it before logging.
- **History carry-forward**: `BaseSequentialModelDataset` and HSTU's `HSTUModelDataset` propagate user history state from train → valid → test, ensuring eval splits see only past interactions.
- **Padding item ids are model-owned**: framework ids are 0-based with no reserved slots. Models that need padding (HSTU) or offset (LaRes) manage this internally via config constants like `ITEM_ID_OFFSET`.
- **No auto-discovery**: every component must be explicitly registered in a table. `get_dataset_spec()` / `get_model_spec()` will fail with a helpful message listing available entries.
