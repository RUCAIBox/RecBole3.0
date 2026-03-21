# RecBole3.0

A simplified and efficient generative recommendation library focusing on core components: Configuration, Data Management, Modeling, Training, and Evaluation.

## Architecture

The library is organized into five main modules:

1.  **Config**: Centralized configuration management via `Config` class
2.  **Data**: Dataset loading, parsing, and dataloader construction
3.  **Models**: Model architectures, layers, and loss functions
4.  **Trainer**: Training loop, optimization, and evaluation
5.  **Evaluator**: Recommendation metrics (Recall, NDCG, Precision, etc.)

### Module Details

#### 1. Config (`config/`)
Manages all experiment parameters. Configurations can be loaded from:
- YAML files (via `config_file_list`)
- Command-line arguments (format: `--key=value`)
- Internal defaults (`recbole/props/`)

Config precedence: defaults < YAML files < CLI args < config_dict

#### 2. Data (`data/`)
Simplifies the data pipeline:
*   **SRDataset** (`dataset.py`): Main dataset class with properties `n_users`, `n_items`, `n_interactions`, `avg_item_seq_len`
*   **Split Strategies**: `leave_one_out`, `time_split`
*   **DataLoader**: Standard PyTorch DataLoader with custom `Collator`
*   **Parsers** (`parser/`): Convert raw data to unified format
    - `BaseParser`: Abstract base class
    - `Amazon2023Parser`: Amazon 2023 dataset parser

#### 3. Models (`models/`)
Contains model definitions and building blocks.
*   **Base** (`base.py`): Abstract base class with `forward`, `calculate_loss`, `full_sort_predict`
*   **Layers** (`layer.py`): `LearnablePositionalEmbeddingInputFeaturesPreprocessor`, `RelativeBucketedTimeAndPositionBasedBias`
*   **Loss** (`loss.py`): CrossEntropy, BPR loss functions
*   **Utils** (`utils.py`): Embedding utilities (`l2_normalize`, `truncated_normal`)
*   **Sequential Models** (`sequential/`):
    - `HSTU` (`hstu.py`): Hierarchical Sequential Transfer Unit with jagged tensor attention

#### 4. Trainer (`trainer/`)
*   **Trainer** (`trainer.py`): Training and evaluation logic
    - AdamW optimizer with cosine learning rate scheduler
    - Gradient clipping, checkpoint saving
    - Early stopping based on validation metrics
    - Multi-GPU support via `accelerate` (DDP)

#### 5. Evaluator (`evaluator/`)
*   **Metrics**: Recall, NDCG, Precision, HitRate, MRR, MAP, F1
*   Configurable `metrics` list and `topk` values

## Installation

```bash
pip install -e .
```

Required dependencies:
- torch
- accelerate
- transformers
- datasets
- fbgemm_gpu (for HSTU jagged tensor operations)

## Quick Start

### Basic Usage

```bash
# Run with default settings
python main.py

# Specify model and dataset
python main.py --model=HSTU --dataset=Amazon2023 --category=Video_Games

# Custom hyperparameters
python main.py --model=HSTU --dataset=Amazon2023 --category=Musical_Instruments \
    --lr=0.001 --epochs=10 --train_batch_size=256
```

### Programmatic Usage

```python
from recbole.pipeline import Pipeline

pipeline = Pipeline(
    model_name="HSTU",
    dataset_name="Amazon2023",
    config_dict={
        "category": "Video_Games",
        "lr": 0.001,
        "epochs": 10,
    }
)
pipeline.run()
```

### Configuration

Config files are located in `recbole/props/`:
- `default.yaml`: Default hyperparameters
- `model/HSTU.yaml`: HSTU-specific settings
- `dataset/Amazon2023.yaml`: Dataset configuration

Override via CLI: `--key=value` (e.g., `--lr=0.001`, `--d_model=128`)

## Folder Structure

```text
.
├── README.md
├── CLAUDE.md
├── main.py                       # Entry point
├── recbole/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py             # Configuration class
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # SRDataset class
│   │   ├── collator.py           # Batch collation
│   │   ├── utils.py              # Data utilities
│   │   └── parser/
│   │       ├── base.py           # Base parser
│   │       └── amazon2023.py     # Amazon 2023 parser
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseModel abstract class
│   │   ├── layer.py              # Reusable layers
│   │   ├── loss.py               # Loss functions
│   │   ├── utils.py              # Model utilities
│   │   └── sequential/
│   │       ├── __init__.py
│   │       └── hstu.py           # HSTU model
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── trainer.py            # Trainer class
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── evaluator.py          # Evaluation metrics
│   │   └── metrics.py            # Metric implementations
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py           # End-to-end pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── utils.py              # Utility functions
│   │   └── logger.py             # Logging setup
│   └── props/
│       ├── default.yaml          # Default config
│       ├── model/
│       │   └── HSTU.yaml         # HSTU config
│       └── dataset/
│           └── Amazon2023.yaml   # Dataset config
└── test.ipynb
```

## Data Format

Processed datasets use JSON format in the cache directory:
- `all_item_seq.json`: User sequences with `user_id`, `item_seq`, `timestamp_seq`
- `id_mapping.json`: User/item to ID mappings (`user2id`, `item2id`)
- `metadata.<category>.json`: Optional item metadata

## License

MIT License