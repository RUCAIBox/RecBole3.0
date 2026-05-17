from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, get_dataset_spec
from recbole3.model import get_model_spec
from recbole3.model.etegrec.config import ETEGRecConfig
from recbole3.model.hstu.config import ITEM_ID_OFFSET
from recbole3.run import compose_config
from recbole3.utils import require_component_cfg, require_component_name


DEFAULT_EMBEDDING_DIM = ETEGRecConfig().semantic_hidden_size


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HSTU once and export item collaborative embeddings for ETEGRec.",
    )
    parser.add_argument(
        "--semantic-emb-file",
        default="etegrec_hstu_emb_256.npy",
        help=(
            "Output .npy file. Bare filenames are resolved under dataset.data_dir; "
            "paths with directories and absolute paths are used as provided."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="HSTU item embedding dimension to export. Defaults to ETEGRec semantic_hidden_size.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example: dataset=amazon2023_retrieval dataset.category=Scientific",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_prepared_data(
    model_spec: Any,
    dataset_cfg: Any,
    model_cfg: Any,
    trainer_cfg: Any,
) -> tuple[Any, BaseTaskDataset, Any, Any]:
    dataset_name = require_component_name(dataset_cfg, "dataset")
    dataset_spec = get_dataset_spec(dataset_name)

    dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
    model = model_spec.model_cls(instantiate_dataclass(model_spec.config_cls, model_cfg))
    trainer = model_spec.trainer_cls(instantiate_dataclass(model_spec.trainer_config_cls, trainer_cfg))

    task_data = dataset.prepare(eval_config=trainer.config.eval)
    if model_spec.model_data_cls is None:
        prepared_data = task_data
    else:
        prepared_data = model_spec.model_data_cls.from_task_dataset(task_data, model_config=model_cfg)
    return dataset, prepared_data, model, trainer


def _resolve_output_path(semantic_emb_file: str, data_dir: Path) -> Path:
    output_path = Path(semantic_emb_file)
    if output_path.is_absolute() or output_path.parent != Path("."):
        return output_path
    return data_dir / output_path


def _load_best_checkpoint_if_available(model: Any, fit_result: Any) -> None:
    if not isinstance(fit_result, dict):
        print("[HSTU export] No fit result dict found; exporting current model after fit().")
        model.eval()
        return

    checkpoint_paths = fit_result.get("checkpoint_paths")
    if not isinstance(checkpoint_paths, dict):
        print("[HSTU export] No checkpoint paths found; exporting current model after fit().")
        model.eval()
        return

    best_checkpoint = checkpoint_paths.get("best")
    if not best_checkpoint:
        print("[HSTU export] No best checkpoint found; exporting current model after fit().")
        model.eval()
        return

    best_checkpoint_path = Path(best_checkpoint)
    if not best_checkpoint_path.exists():
        print(f"[HSTU export] Best checkpoint not found at {best_checkpoint_path}; exporting current model after fit().")
        model.eval()
        return

    print(f"[HSTU export] Loading best checkpoint before exporting embeddings: {best_checkpoint_path}")
    try:
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            nested_state = checkpoint.get(key)
            if isinstance(nested_state, dict):
                state_dict = nested_state
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"HSTU best checkpoint must contain a state dict, got {type(state_dict).__name__}.")
    model.load_state_dict(state_dict)
    model.eval()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = compose_config(overrides=["model=hstu", f"model.embedding_dim={args.embedding_dim}", *args.overrides])
    model_spec = get_model_spec("hstu")
    runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
    dataset_cfg = require_component_cfg(cfg, "dataset")
    model_cfg = require_component_cfg(cfg, "model")
    trainer_cfg = require_component_cfg(cfg, "trainer")

    dataset, prepared_data, model, trainer = _build_prepared_data(
        model_spec,
        dataset_cfg,
        model_cfg,
        trainer_cfg,
    )

    pipeline = model_spec.pipeline_cls(cfg=cfg, model_spec=model_spec)
    with pipeline._accelerate_runtime_device(runtime_cfg.device):
        fit_result = trainer.fit(model, prepared_data, output_dir=runtime_cfg.output_dir)

    from accelerate import PartialState

    distributed_state = PartialState()
    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        _load_best_checkpoint_if_available(model, fit_result)
        item_embeddings = model._item_embedding_module().weight[ITEM_ID_OFFSET:].detach().cpu().numpy()
        item_embeddings = item_embeddings.astype(np.float32, copy=False)

        output_path = _resolve_output_path(args.semantic_emb_file, Path(dataset._parser.data_dir))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, item_embeddings)
        print(f"Saved ETEGRec HSTU item embeddings to: {output_path}")
        print(f"Embedding shape: {item_embeddings.shape}")
    distributed_state.wait_for_everyone()


if __name__ == "__main__":
    main()
