from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import BaseTaskDataset, get_dataset_spec
from recbole3.model import get_model_spec
from recbole3.model.hstu.config import ITEM_ID_OFFSET
from recbole3.run import compose_config
from recbole3.utils import require_component_cfg, require_component_name


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HSTU once and export item collaborative embeddings for LETTER.",
    )
    parser.add_argument(
        "--cf-emb-file",
        default="cf_embeddings.pt",
        help="Output collaborative embedding file path. Relative paths are resolved under dataset.data_dir.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example: dataset=amazon2023_retrieval dataset.category=Books",
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


def _resolve_output_path(cf_emb_file: str, data_dir: Path) -> Path:
    output_path = Path(cf_emb_file)
    if not output_path.is_absolute():
        output_path = data_dir / output_path
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = compose_config(overrides=["model=hstu", *args.overrides])
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
        trainer.fit(model, prepared_data, output_dir=runtime_cfg.output_dir)

    item_embeddings = model._item_embedding_module().weight[ITEM_ID_OFFSET:].detach().cpu()

    output_path = _resolve_output_path(args.cf_emb_file, Path(dataset._parser.data_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(item_embeddings, output_path)
    print(f"Saved HSTU collaborative embeddings to: {output_path}")


if __name__ == "__main__":
    main()
