from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.evaluation import EvalConfig
from recbole3.model.etegrec.config import ETEGRecConfig
from recbole3.model.etegrec.data import _resolve_semantic_embedding_path
from recbole3.model.etegrec.pretrain_rqvae import RQVAE, build_pretrain_config
from recbole3.model.etegrec.pretrain_trainer import ETEGRecRQVAEPretrainTrainer, ETEGRecRQVAEPretrainTrainerConfig
from recbole3.run import compose_config
from recbole3.utils import require_component_cfg, require_component_name


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain ETEGRec's standalone RQVAE tokenizer on item embeddings.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Checkpoint output directory. Defaults to runtime.output_dir.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example: dataset=amazon2023_retrieval model.semantic_emb_file=etegrec_hstu_emb_256.npy",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_embeddings(path: Path, *, expected_dim: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"ETEGRec RQVAE pretraining semantic embedding file does not exist: {path}")
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"ETEGRec RQVAE pretraining embeddings must be 2D, got shape {embeddings.shape}.")
    if expected_dim is not None and int(embeddings.shape[1]) != int(expected_dim):
        raise ValueError(
            "ETEGRec RQVAE pretraining embedding dimension does not match model.semantic_hidden_size. "
            f"Expected {expected_dim}, got {int(embeddings.shape[1])}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("ETEGRec RQVAE pretraining embeddings contain NaN or infinite values.")
    return embeddings.astype(np.float32, copy=False)


def _build_dataloader(embeddings: np.ndarray, config: ETEGRecRQVAEPretrainTrainerConfig) -> DataLoader[torch.Tensor]:
    tensor = torch.as_tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(tensor)

    def collate(batch):
        return torch.stack([row[0] for row in batch], dim=0)

    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = compose_config(overrides=["model=etegrec", *args.overrides])
    dataset_cfg = require_component_cfg(cfg, "dataset")
    model_cfg = instantiate_dataclass(ETEGRecConfig, require_component_cfg(cfg, "model"))
    runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))

    dataset_name = require_component_name(dataset_cfg, "dataset")
    dataset_spec = get_dataset_spec(dataset_name)
    dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
    task_data = dataset.prepare(eval_config=EvalConfig(protocol="full"))
    data_dir = Path(dataset._parser.data_dir)

    embedding_path = _resolve_semantic_embedding_path(model_cfg.semantic_emb_file, data_dir=data_dir)
    embeddings = _load_embeddings(embedding_path, expected_dim=int(model_cfg.semantic_hidden_size))
    if int(embeddings.shape[0]) != int(task_data.get_num_items()):
        raise ValueError(
            "ETEGRec RQVAE pretraining embeddings must contain one row per remapped item id. "
            f"Expected {task_data.get_num_items()} rows, got {int(embeddings.shape[0])}."
        )

    trainer_config = ETEGRecRQVAEPretrainTrainerConfig(
        lr=float(getattr(cfg.trainer, "rqvae_pretrain_lr", 1e-3)),
        epochs=int(getattr(cfg.trainer, "rqvae_pretrain_epochs", getattr(cfg.trainer, "max_epochs", 10000))),
        batch_size=int(getattr(cfg.trainer, "rqvae_pretrain_batch_size", getattr(cfg.trainer, "batch_size", 1024))),
        num_workers=int(getattr(cfg.trainer, "dataloader_num_workers", 2)),
        eval_step=int(getattr(cfg.trainer, "rqvae_pretrain_eval_step", getattr(cfg.trainer, "eval_steps", 50))),
        learner=str(getattr(cfg.trainer, "rqvae_pretrain_learner", getattr(cfg.trainer.optimizer, "name", "AdamW"))),
        weight_decay=float(getattr(cfg.trainer, "rqvae_pretrain_weight_decay", getattr(cfg.trainer.optimizer, "kwargs", {}).get("weight_decay", 1e-4))),
        lr_scheduler_type=str(getattr(cfg.trainer, "rqvae_pretrain_scheduler", "linear")),
        warmup_epochs=int(getattr(cfg.trainer, "rqvae_pretrain_warmup_epochs", 50)),
        save_limit=int(getattr(cfg.trainer, "rqvae_pretrain_save_limit", 3)),
        gradient_clip_norm=float(getattr(cfg.trainer, "gradient_clip_norm", 1.0)),
        device=str(runtime_cfg.device if runtime_cfg.device != "auto" else "auto"),
    )
    pretrain_config = build_pretrain_config(
        model_cfg,
        quant_loss_weight=float(getattr(model_cfg, "quant_loss_weight", getattr(cfg.trainer, "alpha", 1.0))),
    )
    dataloader = _build_dataloader(embeddings, trainer_config)
    model = RQVAE(pretrain_config, in_dim=int(embeddings.shape[1]))
    output_dir = Path(args.output_dir or runtime_cfg.output_dir)
    trainer = ETEGRecRQVAEPretrainTrainer(trainer_config, model, output_dir=output_dir)
    result = trainer.fit(dataloader)

    print(f"ETEGRec RQVAE pretraining embedding path: {embedding_path}")
    print(f"ETEGRec RQVAE pretraining output_dir: {output_dir}")
    print(f"Best loss checkpoint: {result['best_loss_path']}")
    print(f"Best collision checkpoint: {result['best_collision_path']}")
    print(f"Best collision rate: {result['best_collision_rate']}")


if __name__ == "__main__":
    main()
