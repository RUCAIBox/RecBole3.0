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
- `ParsedData`
- `FrameDataset`

`BaseDatasetParser` owns:

- raw-data download when needed
- dataset-specific cache management
- raw-field cleanup and normalization
- producing one `ParsedData` payload

`ParsedData` contains:

- `interactions: pandas.DataFrame`
- optional `user_table`
- optional `item_table`

If `user_table` or `item_table` is omitted, `BaseTaskDataset` derives the missing table from the raw interaction ids. If a table is provided, its id column must be non-null and unique; missing interaction keys are appended before framework id remapping.

`BaseTaskDataset` owns:

- parser instantiation
- validating parser output DataFrame schemas
- deriving missing entity tables
- remapping raw interaction and entity-table ids into framework ids
- reserving `item_id=0` as the padding item id
- interaction ordering
- ratio and leave-one-out split helpers
- exposing prepared train and eval datasets through method accessors
- exposing prepared metadata such as `num_users` and `num_items`

`RankingDataset` owns:

- validating `labeled` evaluation protocol usage
- keeping interaction DataFrames for all three splits

`RetrievalDataset` owns:

- validating `full` and `sampled` evaluation protocol usage
- keeping an interaction DataFrame for `train`
- building request-level eval DataFrames for `valid` and `test`
- filtering retrieval eval rows to positive interactions
- carrying tuple-valued `seen_item_ids` histories that grow within each eval split
- deterministic sampled candidate generation for `sampled`

Dataset code does not own:

- model-specific negative sampling for training
- tensor padding or masking
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

For sequence models, the framework provides built-in model-side DataFrame transforms plus one shared history builder:

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

Prepared datasets expose split datasets through `get_train_dataset()` and `get_eval_dataset(...)`. `get_interactions()`, entity tables, and split frames use framework ids, not raw source ids. `get_item_table()` includes the padding row at `item_id=0`; all real interaction items are in `[1, get_num_items() - 1]`. Accessors return copies for DataFrame metadata views.

## Data Contracts

### Parser Output DataFrame

`ParsedData.interactions` is the parser-facing interaction table.

It must contain:

- `user_id`
- `item_id`

Optional framework-recognized columns are:

- `timestamp`
- `label`

Extra columns are allowed and are preserved through the prepared interaction frames. At parser output time, `user_id` and `item_id` are raw dataset ids and may be strings or other hashable keys, but they must be non-null.

Optional `user_table` and `item_table` use the same raw id columns. Provided entity tables must have unique non-null ids. Missing tables are synthesized from interaction ids; missing rows in provided tables are appended.

After `BaseTaskDataset.prepare(...)`, `user_id` and `item_id` are framework ids. `user_id` starts at `0`. `item_id=0` is reserved for padding, real items start at `1`, and `get_num_items()` includes the padding row.

### Retrieval

Retrieval train DataFrames preserve all interaction rows from the train split. Retrieval eval DataFrames are request-level rows built from positive interactions only; `label` is treated as positive when it is missing or greater than `0`.

Retrieval eval DataFrames contain:

- interaction fields such as `user_id`, `item_id`, and optional `timestamp`
- tuple-valued `seen_item_ids`
- optional tuple-valued `candidate_item_ids`

`seen_item_ids` uses first-seen unique item histories. Validation starts from positive train interactions and grows across validation rows for each user. Test starts from positive train plus positive validation interactions and grows across test rows.

Protocol semantics are:

- `full`: `candidate_item_ids` is absent or null
- `sampled`: `candidate_item_ids` is non-null, starts with the target item, has the same tuple length in every row, contains the request candidate set, and never contains `item_id=0`

### Model-side Sequential Records

Sequential model-side DataFrames add:

- `history_item_ids`

HSTU model-side DataFrames also add:

- `history_timestamps`

`build_history_item_ids(...)` constructs one prefix history per row from an ordered interaction DataFrame, optionally seeded by one user-history mapping from earlier splits.

## Cache Ownership

The framework does not manage one global dataset cache layout.

Parser implementations own all dataset-specific cache paths and policies. That keeps cache layout close to the source-specific logic that produced it and avoids framework-level path coupling.

For reusable local file behavior, the framework provides `recbole3.dataset.cache.DatasetCache`. It handles JSONL DataFrame read/write, force-or-missing frame creation, and standard `ParsedData` cache files. It does not download sources or choose source-specific paths.

The built-in Amazon 2023 parser is the reference example:

- raw reviews are cached under `download_dir/amazon2023/<download_source>/<category>/<kcore>/reviews.jsonl`
- raw metadata is cached beside reviews as `meta.jsonl` when `metadata_mode=sentence`
- parsed DataFrames are cached under `processed_dir/<dataset.name>/schema_v2/<download_source>/<category>/<kcore>/<metadata_mode>/`
- `refresh_cache=true` rebuilds both raw and parsed parser-managed cache files

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

`order=chronological` sorts by `timestamp` only when the whole group has timestamps; otherwise it preserves parser order. `order=random` uses `seed`. With `per_user=true`, ordering and splitting happen independently inside each user group.
