# RecBole3.0 v1 Architecture

## Goal

RecBole3.0 v1 is a recommendation research toolkit skeleton. It deliberately prioritizes:

- developer-friendly extension points
- a small number of stable concepts
- readable intermediate data
- low maintenance cost

The repository implements the framework boundary, not concrete recommendation algorithms or datasets.

## Core Modules

### `dataset/`

`dataset` is split into one dataset-specific parser layer and one task-aware prepared-data layer.

The stable concepts are:

- `BaseDatasetParser`
- `BaseTaskDataset`
- `RankingDataset`
- `RetrievalDataset`
- `Interaction`
- `RetrievalEvalRequest`
- `RecordsDataset`

`BaseDatasetParser` owns:

- raw-data download when needed
- dataset-specific cache management
- raw-field cleanup and normalization
- producing one `ParsedData` payload

`ParsedData` contains:

- `interactions: list[Interaction]`
- `user_table`
- `item_table`

`BaseTaskDataset` owns:

- parser instantiation
- interaction ordering
- ratio and leave-one-out split helpers
- exposing prepared train and eval datasets through method accessors
- exposing prepared metadata such as `num_users` and `num_items`

`RankingDataset` owns:

- validating `labeled` evaluation protocol usage
- keeping row-based `Interaction` records for all three splits

`RetrievalDataset` owns:

- validating `full` and `sampled` evaluation protocol usage
- keeping row-based `Interaction` records for `train`
- building request-level `RetrievalEvalRequest` records for `valid` and `test`
- deterministic sampled candidate generation for `sampled`

Dataset code does not own:

- model-specific negative sampling for training
- padding or masking
- architecture-specific feature packing
- runtime metric computation

### `trainer/`

`trainer` aligns prepared datasets with the training loop and runs them through PyTorch DataLoaders plus `accelerate`.

The framework uses one trainer skeleton plus three protocol-specific evaluation methods:

- `Trainer`
- `LabeledEvaluationMethod`
- `SampledEvaluationMethod`
- `FullEvaluationMethod`

`Trainer` owns:

- building training and evaluation DataLoaders through one shared `build_dataloader(...)`
- consuming `prepared_data.get_train_dataset()` and `prepared_data.get_eval_dataset(...)`
- creating the accelerator plus config-driven torch optimizers and optional schedulers
- running epoch-based `fit()` with end-of-epoch validation
- applying optional monitor-driven early stopping and best/last checkpoint saves during `fit()`
- running standalone `evaluate()` on `valid` or `test`
- providing `run()` as the complete `fit + final test` entrypoint, including best-checkpoint reload when available
- selecting the evaluation method from `TrainerConfig.eval.protocol`

Evaluation method classes own:

- wrapping the model eval collator so one batch yields `(model_inputs, records)`
- instantiating protocol-compatible metrics from `TrainerConfig.eval.metrics`
- determining any protocol-specific top-k requirements from configured metrics
- collecting batch-level `RankingEvalData` or `RetrievalEvalData`
- aggregating batch outputs and computing protocol-compatible metrics
- protocol-specific runtime behavior such as full-history filtering

### `model/`

`model` owns the recommendation architecture and the final conversion from logical sample records into model-ready batches.

`BaseModel` is a `torch.nn.Module` and owns:

- `forward`
- `compute_loss`
- `build_train_collator`
- `build_eval_collator`

The framework exposes two task-specific model bases:

- `BaseRankingModel.predict(model_inputs)` returns one score per labeled eval row
- `BaseRetrievalModel.predict(model_inputs, k=..., candidate_item_ids=..., exclude_item_ids=..., exclude_mask=...)` returns ordered top-k item ids

`BaseCollator` is used directly as `DataLoader.collate_fn`, so batch tensorization stays close to the model.

When one model needs extra task-level data processing, bind the model to one optional model-data class through `MODEL_TABLE`. That class is implementation detail, not a user-facing config entry.

The framework also provides task-matched model-side dataset extension points:

- `BaseRankingModelDataset`
- `BaseRetrievalModelDataset`

For sequence models, the framework provides built-in model-side logical records plus one shared history builder:

- `SequentialInteraction`
- `SequentialRetrievalEvalRequest`
- `build_history_item_ids(...)`
- `BaseSequentialRankingModelDataset`
- `BaseSequentialRetrievalModelDataset`

Those sequential helpers are model-side only. They do not change dataset parser responsibilities or retrieval metric contracts.

## Component Tables

The framework uses two explicit tables rather than dynamic registration:

- `DATASET_TABLE`
- `MODEL_TABLE`

`MODEL_TABLE` binds each model to its model class, model config class, optional model-data class, trainer class, and trainer config class.

User config still supplies strings such as `dataset.name` and `model.name`. `run_experiment(...)` resolves those names through the tables, instantiates the matching implementations, and reads trainer defaults from the selected model config.

## Prepared Dataset Contract

`BaseTaskDataset.prepare(eval_config=...)` returns the dataset instance itself after hydrating all runtime state.

A prepared dataset contains:

- dataset config
- ordered interactions
- optional user and item tables
- train dataset
- valid dataset
- test dataset
- `num_users`
- `num_items`

The public prepared-data access pattern is method-based:

- `get_train_dataset()`
- `get_eval_dataset(split)`
- `get_interactions()`
- `get_user_table()`
- `get_item_table()`
- `get_num_users()`
- `get_num_items()`

Prepared datasets expose split datasets through `get_train_dataset()` and `get_eval_dataset(...)`.

## Data Contracts

### Interaction

`Interaction` is the shared row contract used by parsers and ranking datasets.

It contains:

- `user_id`
- `item_id`
- optional `timestamp`
- optional `label`

For v1, normalized `user_id` and `item_id` are expected to be contiguous integer ids starting at `0`.

### Retrieval

`RetrievalEvalRequest` contains:

- inherited `Interaction` fields such as `user_id`, `item_id`, and optional `timestamp`
- `seen_item_ids`
- optional `candidate_item_ids`

Protocol semantics are:

- `full`: `candidate_item_ids is None`
- `sampled`: `candidate_item_ids` is non-null and contains the request candidate set

### Model-side Sequential Records

`SequentialInteraction` extends `Interaction` with:

- `history_item_ids`

`SequentialRetrievalEvalRequest` extends `RetrievalEvalRequest` with:

- `history_item_ids`

`build_history_item_ids(...)` constructs one prefix history per record from an ordered `Sequence[Interaction]`, optionally seeded by one user-history mapping from earlier splits.

Because `RetrievalEvalRequest` extends `Interaction`, ranking and retrieval sequence models use the same history construction rule.

## Cache Ownership

The framework no longer manages one common dataset cache layer.

Parser implementations own all dataset-specific cache paths and policies. That keeps cache layout close to the source-specific logic that produced it and avoids framework-level path coupling.

## Split Strategy

The dataset layer owns reusable split generation.

Supported v1 strategies:

- `ratio`
- `leave_one_out`

Supported split controls live under `DatasetConfig.split`:

- `strategy`
- `order`
- `per_user`
- `train_ratio`, `valid_ratio`, `test_ratio`
- `valid_holdout_num`, `test_holdout_num`
- `seed`
