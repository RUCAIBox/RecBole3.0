# Component Authoring Guide

## Goal

This guide answers one practical question:

How do you add one new dataset, model, or trainer with the current v1 API?

The framework does not auto-discover components. Every component is added explicitly through one table entry.

## Dataset in 4 Steps

For most new datasets, you only need 4 things:

1. one `DatasetConfig` subclass
2. one `BaseDatasetParser` subclass
3. one one-line task dataset class
4. one `DATASET_TABLE` entry

### Step 1: Write the Config

Keep task-level split options in `DatasetConfig.split`.
Put source-specific options on your own config class.

```python
from dataclasses import dataclass, field

from recbole3.dataset import DatasetConfig


@dataclass(slots=True)
class MyDatasetConfig(DatasetConfig):
    name: str = field(default="my_retrieval_dataset", metadata={"help": "Dataset name."})
    source_path: str = field(default="data/raw/my_dataset.csv", metadata={"help": "Raw source path."})
    processed_dir: str = field(default="data/processed/my_retrieval_dataset", metadata={"help": "Parsed cache path."})
    refresh_cache: bool = field(default=False, metadata={"help": "Whether to rebuild parser cache."})
```

### Step 2: Write the Parser

`BaseDatasetParser.parse()` is the only required parser entrypoint.
It must return one `ParsedData` object.

Parser responsibilities:

- download raw data if needed
- manage source-specific cache files
- normalize raw fields
- return `interactions`, `user_table`, and `item_table`

For JSONL-backed parser caches, use `from recbole3.dataset.cache import DatasetCache`. Keep source-specific path layout in your parser, and use the cache helper for DataFrame and `ParsedData` read/write.

```python
import pandas as pd

from recbole3.dataset.cache import DatasetCache
from recbole3.dataset import BaseDatasetParser, ITEM_ID, LABEL, TIMESTAMP, USER_ID, ParsedData


class MyDatasetParser(BaseDatasetParser):
    config: MyDatasetConfig

    def parse(self) -> ParsedData:
        cache = DatasetCache(self.config.processed_dir)
        if not self.config.refresh_cache and cache.parsed_exists():
            return cache.read_parsed()

        frame = pd.read_csv(self.config.source_path)

        interactions = pd.DataFrame(
            {
                USER_ID: frame["reviewer_id"],
                ITEM_ID: frame["asin"],
                TIMESTAMP: frame["timestamp"],
                LABEL: frame["label"],
            }
        )
        user_table = frame.loc[:, ["reviewer_id"]].drop_duplicates("reviewer_id")
        user_table = user_table.rename(columns={"reviewer_id": USER_ID})
        item_table = frame.loc[:, ["asin", "title", "category"]].drop_duplicates("asin")
        item_table = item_table.rename(columns={"asin": ITEM_ID})
        parsed = ParsedData(interactions=interactions, user_table=user_table, item_table=item_table)
        cache.write_parsed(parsed)
        return parsed
```

`ParsedData.interactions` must be a pandas DataFrame with `user_id` and `item_id`. At parser output time those two columns are raw source ids and may be strings or other hashable keys, but they must be non-null.

`user_table` and `item_table` are optional. If you provide them, they must contain unique non-null raw `user_id` / `item_id` values. Missing entity rows are appended from interactions before framework id remapping, so sparse metadata tables are allowed.

`BaseTaskDataset.prepare(...)` remaps raw ids into framework ids. Both `user_id` and `item_id` start at `0`, and dataset `item_id` values always refer to real items. `timestamp` and `label` are optional; extra columns are allowed and preserved in prepared frames.

Retrieval eval frames use tuple-valued `seen_item_ids`; sampled retrieval also uses tuple-valued `candidate_item_ids` with the target item first and the same tuple length in every row.

### Step 3: Bind It to One Task Dataset

Use `BaseTaskDataset` for concrete datasets. The evaluation protocol selects the task contract:

- `labeled` prepares ranking splits
- `full` and `sampled` prepare retrieval splits

In most cases the concrete dataset class only binds `config_cls` and `parser_cls`.

```python
from recbole3.dataset import BaseTaskDataset


class MyDataset(BaseTaskDataset):
    config_cls = MyDatasetConfig
    parser_cls = MyDatasetParser
```

### Step 4: Add One Table Entry

```python
from recbole3.dataset import DATASET_TABLE, DatasetSpec


DATASET_TABLE["my_retrieval_dataset"] = DatasetSpec(
    dataset_cls=MyDataset,
    config_cls=MyDatasetConfig,
)
```

## Dataset Rules

Your parser should do:

- source download
- source cache
- raw cleanup
- DataFrame construction for interactions and optional entity metadata
- user and item table construction when source metadata is available

Your parser should not do:

- framework id remapping
- train negative sampling
- padding, masking, truncation, sequence packing
- model-specific feature engineering
- metric computation

Your task dataset should do:

- ordering interactions
- splitting interactions
- building eval records required by the task

Your task dataset should not do:

- source download
- source cache layout decisions
- model-specific tensor preparation

## Prepared Dataset API

After `prepare(eval_config=...)`, the public API is:

- `get_train_dataset()`
- `get_eval_dataset("valid")`
- `get_eval_dataset("test")`
- `get_interactions()`
- `get_user_table()`
- `get_item_table()`
- `get_num_users()`
- `get_num_items()`

All three splits are `FrameDataset` instances backed by pandas DataFrames. A single integer index returns one row as a dictionary, while DataLoader batch fetching returns one DataFrame to the collator.

Prepared frames and entity tables use framework ids, not raw source ids. `get_item_table()` contains only real item rows, and item ids start at `0`. Models that need a padding id reserve and map it internally. DataFrame metadata accessors return copies.

## Split Config

Common split controls live under `dataset.split`:

```yaml
dataset:
  name: my_retrieval_dataset
  split:
    strategy: leave_one_out
    order: chronological
    per_user: true
    valid_holdout_num: 1
    test_holdout_num: 1
```

Ratio split uses all three ratios:

```yaml
dataset:
  name: my_ranking_dataset
  split:
    strategy: ratio
    order: chronological
    per_user: false
    train_ratio: 8
    valid_ratio: 1
    test_ratio: 1
```

Ratio values are normalized as weights and rounded with a deterministic largest-remainder rule. Leave-one-out holdout counts are clamped to the available group size. `order=chronological` uses `timestamp` only when every row in the group has one; otherwise parser order is preserved. `order=random` is deterministic for the same `seed`.

## Model in 3 Steps

You need:

1. one `ModelConfig` subclass
2. one model implementation
3. one `MODEL_TABLE` entry

Required model methods:

- `build_train_collator(prepared_data)`
- `build_eval_collator(prepared_data)`
- `forward(batch)`
- `compute_loss(batch, outputs)`

Task-specific prediction methods:

- `BaseRankingModel.predict(model_inputs)`
- `BaseRetrievalModel.predict(model_inputs, k=..., candidate_item_ids=..., exclude_item_ids=..., exclude_mask=...)`

If one model needs extra dataset processing, implement one model-data class and bind it through `MODEL_TABLE.model_data_cls`. Keep that binding outside the model class.

### Model-side Dataset Extension

Use the shared model-data base class:

- `BaseModelDataset` for model-specific split replacement
- `BaseSequentialModelDataset` when the model needs `history_item_ids`

You do not need to override `from_task_dataset(...)`. The base class clones one prepared task dataset for you, then calls `_build_model_datasets(...)`.

```python
from recbole3.model import BaseModelDataset, MODEL_TABLE, ModelDatasets, ModelSpec


class MyModelDataset(BaseModelDataset[Any, Any]):
    def _build_model_datasets(self, *, model_config: MyModelConfig) -> ModelDatasets[Any, Any]:
        train_dataset = self.get_train_dataset()
        return ModelDatasets(train_dataset=train_dataset)
```

Rules for model-side datasets:

- keep the same task as the source task dataset
- return `ModelDatasets(...)` from `_build_model_datasets(...)`
- you may replace `train_dataset`
- you may replace eval datasets only if they still satisfy the task evaluation contract
- any split left as `None` keeps the cloned prepared split unchanged
- do not move source download or source cache logic into the model-data class

### Sequential Model-side Dataset

If your model needs `history_item_ids`, prefer the built-in sequential bases instead of rebuilding sequence logic in the collator.

- `BaseSequentialModelDataset` adds `history_item_ids` to train, valid, and test split DataFrames
- `build_history_item_ids(...)` is the shared helper that constructs one prefix history per row

Minimal example:

```python
from recbole3.model import BaseSequentialModelDataset


class MySequentialModelDataset(BaseSequentialModelDataset):
    pass
```

If you need custom sequence update rules, override `_include_target_item_in_history(...)` in your model-data class.

Bind it in `MODEL_TABLE`:

```python
MODEL_TABLE["my_model"] = ModelSpec(
    model_cls=MyModel,
    config_cls=MyModelConfig,
    model_data_cls=MyModelDataset,
    trainer_cls=Trainer,
    trainer_config_cls=TrainerConfig,
)
```

If one model does not need extra data processing, keep `model_data_cls=None`.

## Trainer in 3 Steps

You need:

1. one `TrainerConfig` subclass in the model's `config.py` if the common config is not enough
2. one `Trainer` subclass in the model directory if direct `Trainer` reuse is not enough
3. one `ModelSpec(...)` binding that points at that trainer class and trainer config class

Trainer code should consume prepared datasets through:

- `get_train_dataset()`
- `get_eval_dataset(...)`

```python
from recbole3.model import MODEL_TABLE, ModelSpec
from recbole3.trainer import Trainer
from my_model.config import MyModelConfig, MyTrainerConfig
from my_model.trainer import MyTrainer

MODEL_TABLE["my_model"] = ModelSpec(
    model_cls=MyModel,
    config_cls=MyModelConfig,
    model_data_cls=MyModelDataset,
    trainer_cls=MyTrainer,
    trainer_config_cls=MyTrainerConfig,
)
```

## YAML Layout

Each dataset should have one YAML file under:

- `configs/dataset/`

Each model should have one YAML file under:

- `configs/model/`

Dataset file example:

```yaml
name: my_retrieval_dataset
source_path: data/raw/my_dataset.csv
processed_dir: data/processed/my_retrieval_dataset
split:
  strategy: leave_one_out
  order: chronological
  per_user: true
  valid_holdout_num: 1
  test_holdout_num: 1
```

Model file example:

```yaml
# @package _global_

model:
  name: my_model
  history_max_length: 50

trainer:
  optimizer:
    name: Adam
    kwargs:
      lr: 0.001
  eval:
    protocol: sampled
    neg_sampling_num: 100
    metrics:
      - name: recall
        ks: [10]
```

Model-data parameters belong in the `model` block because they are model-specific, even when they are consumed by `BaseModelDataset`.

## Checklist

Before adding one new dataset, check these points:

- parser output `interactions` is a DataFrame with non-null raw `user_id` and `item_id`
- optional `user_table` contains unique non-null raw `user_id`
- optional `item_table` contains unique non-null raw `item_id`
- retrieval datasets produce only positive eval requests and valid `seen_item_ids` histories
- sampled retrieval candidates are equal-width tuples and target-first
- dataset class inherits `BaseTaskDataset`
- dataset name is added to `DATASET_TABLE`
- YAML `dataset.name` matches the table key exactly
