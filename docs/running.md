# Running Experiments

This project uses Hydra for config composition, but the CLI is intentionally thin:

- `python -m recbole3.run ...` forwards every command-line argument directly into `hydra.compose(overrides=[...])`
- you use Hydra override syntax such as `dataset=...` and `trainer.max_epochs=...`
- this entrypoint does not expose Hydra's full CLI helpers such as `--cfg`, `--multirun`, or automatic working-directory switching

## 1. Install Dependencies

From the repository root:

```bash
python -m pip install -e '.[dev]'
```

The default built-in example in this repository uses:

- `dataset=amazon2023_retrieval`
- `model=hstu`

That combination also needs optional extras:

```bash
python -m pip install -e '.[dev,modelscope,hstu]'
```

If you switch the Amazon dataset source to Hugging Face, install `huggingface` instead of `modelscope`:

```bash
python -m pip install -e '.[dev,huggingface,hstu]'
```

## 2. Understand Config Groups

Hydra group selection comes from filenames under `configs/`:

- `configs/dataset/amazon2023_retrieval.yaml` -> `dataset=amazon2023_retrieval`
- `configs/model/hstu.yaml` -> `model=hstu`

The root config keeps dataset and model groups optional, so a normal run should choose both explicitly.

Each model config file provides:

- the top-level `model` block
- the top-level `trainer` defaults bound to that model

## 3. Minimal Run Command

Run from the repository root:

```bash
python -m recbole3.run dataset=amazon2023_retrieval model=hstu
```

This composes `configs/config.yaml` together with the selected dataset and model files. The selected model file injects the matching `trainer` defaults.

## 4. Override Individual Fields

Hydra lets you keep the same config groups and override only the fields you care about.

Example:

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=hstu \
  dataset.category=Books \
  dataset.kcore=full \
  trainer.max_epochs=10 \
  trainer.batch_size=128 \
  trainer.eval.protocol=sampled \
  trainer.eval.neg_sampling_num=100 \
  runtime.device=cuda:0 \
  runtime.output_dir=outputs/books_sampled
```

Common patterns:

- select a config group: `model=hstu`
- override a nested field: `trainer.optimizer.kwargs.lr=1e-4`
- change a runtime setting: `runtime.device=cpu`

Each override must be passed as one shell argument. If your shell would interpret special characters, quote that single override token.

## 5. Effective Output Path

For this entrypoint, treat `runtime.output_dir` as the effective run output path.

Example:

```bash
python -m recbole3.run \
  dataset=amazon2023_retrieval \
  model=hstu \
  runtime.output_dir=outputs/debug_run
```

If checkpoint saving is enabled, checkpoints are written under:

```text
<runtime.output_dir>/checkpoints/
```

## 6. Common Issues

`Missing dataset/model configuration`

- You did not select one of the required Hydra config groups.
- Start from `dataset=... model=...`.

`Could not override 'trainer'`

- Trainer is no longer a standalone Hydra config group.
- Select the model with `model=...`, then override trainer fields such as `trainer.max_epochs=...`.

`model.name=hstu` fails with missing `fbgemm-gpu`

- Install the optional HSTU dependency: `python -m pip install -e '.[hstu]'`

`download_source='modelscope'` or `download_source='huggingface'` fails

- Install the matching dataset extra: `.[modelscope]` or `.[huggingface]`

## 7. Python API

If you want the same config composition inside Python:

```python
from recbole3.run import compose_config, run_experiment

cfg = compose_config(
    overrides=[
        "dataset=amazon2023_retrieval",
        "model=hstu",
        "trainer.max_epochs=2",
    ]
)
result = run_experiment(cfg)
```
