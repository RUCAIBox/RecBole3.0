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
    refresh_cache: bool = field(default=False, metadata={"help": "Whether to rebuild parser cache."})
```

### Step 2: Write the Parser

`BaseDatasetParser.parse()` is the only required parser entrypoint.
It must return one `ParsedData` object.

Parser responsibilities:

- download raw data if needed
- manage source-specific cache files
- normalize raw fields
- remap raw ids to contiguous `user_id` and `item_id`
- return `interactions`, `user_table`, and `item_table`

```python
from dataclasses import dataclass

import pandas as pd

from recbole3.dataset import BaseDatasetParser, Interaction, ParsedData


class MyDatasetParser(BaseDatasetParser):
    config: MyDatasetConfig

    def parse(self) -> ParsedData:
        frame = pd.read_csv(self.config.source_path)

        user_index = pd.Index(pd.unique(frame["raw_user_id"]), name="raw_user_id")
        item_index = pd.Index(pd.unique(frame["raw_item_id"]), name="raw_item_id")
        user_id_map = pd.Series(range(len(user_index)), index=user_index)
        item_id_map = pd.Series(range(len(item_index)), index=item_index)

        interactions = [
            Interaction(
                user_id=int(user_id_map[row.raw_user_id]),
                item_id=int(item_id_map[row.raw_item_id]),
                timestamp=int(row.timestamp),
                label=float(row.label),
            )
            for row in frame.itertuples(index=False)
        ]
        user_table = pd.DataFrame({"user_id": range(len(user_index)), "raw_user_id": user_index})
        item_table = pd.DataFrame({"item_id": range(len(item_index)), "raw_item_id": item_index})
        return ParsedData(interactions=interactions, user_table=user_table, item_table=item_table)
```

### Step 3: Bind It to One Task Dataset

Choose the task base that matches the evaluation contract:

- `RankingDataset` for labeled ranking tasks
- `RetrievalDataset` for retrieval tasks

In most cases the concrete dataset class only binds `config_cls` and `parser_cls`.

```python
from recbole3.dataset import RetrievalDataset


class MyRetrievalDataset(RetrievalDataset):
    config_cls = MyDatasetConfig
    parser_cls = MyDatasetParser
```

### Step 4: Add One Table Entry

```python
from recbole3.dataset import DATASET_TABLE, DatasetSpec


DATASET_TABLE["my_retrieval_dataset"] = DatasetSpec(
    dataset_cls=MyRetrievalDataset,
    config_cls=MyDatasetConfig,
    task="retrieval",
)
```

## Dataset Rules

Your parser should do:

- source download
- source cache
- raw cleanup
- id remapping
- user and item table construction

Your parser should not do:

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

All three splits are `torch.utils.data.Dataset` instances.

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

Use the task-matched base class:

- `BaseRetrievalModelDataset` for retrieval models
- `BaseRankingModelDataset` for ranking models
- `BaseSequentialRetrievalModelDataset` for retrieval sequence models
- `BaseSequentialRankingModelDataset` for ranking sequence models

You do not need to override `from_task_dataset(...)`. The base class clones one prepared task dataset for you, then calls `_build_model_datasets(...)`.

```python
from recbole3.model import BaseRetrievalModelDataset, MODEL_TABLE, ModelDatasets, ModelSpec


class MyModelDataset(BaseRetrievalModelDataset[Any, Any]):
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

- `BaseSequentialRankingModelDataset` converts all three splits into `SequentialInteraction`
- `BaseSequentialRetrievalModelDataset` converts `train` into `SequentialInteraction` and `valid`/`test` into `SequentialRetrievalEvalRequest`
- `build_history_item_ids(...)` is the shared helper that constructs one prefix history per record

Minimal retrieval example:

```python
from recbole3.model import BaseSequentialRetrievalModelDataset


class MySequentialModelDataset(BaseSequentialRetrievalModelDataset):
    pass
```

Minimal ranking example:

```python
from recbole3.model import BaseSequentialRankingModelDataset


class MySequentialRankingModelDataset(BaseSequentialRankingModelDataset):
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

Example:

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

- parser output uses contiguous `user_id` and `item_id`
- `user_table` contains `user_id`
- `item_table` contains `item_id`
- dataset class inherits the correct task base
- dataset name is added to `DATASET_TABLE`
- YAML `dataset.name` matches the table key exactly

