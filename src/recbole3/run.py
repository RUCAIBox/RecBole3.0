from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from omegaconf import DictConfig, OmegaConf

from recbole3.config import configs_dir
from recbole3.model import get_model_spec
from recbole3.utils import require_component_name, require_component_cfg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    model_cfg = require_component_cfg(cfg, "model")
    model_name = require_component_name(model_cfg, "model")
    model_spec = get_model_spec(model_name)

    # Create and run pipeline
    pipeline = model_spec.pipeline_cls(
        cfg=cfg,
        model_spec=model_spec,
    )
    return pipeline.run()


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Main entry point for running experiments."""

    cfg = compose_config(overrides=list(argv if argv is not None else sys.argv[1:]))
    result = run_experiment(cfg)
    return result


def compose_config(overrides: Sequence[str] | None = None, config_dir: str | Path | None = None) -> DictConfig:
    """Compose the root Hydra config from a config directory and override list."""

    import hydra

    config_root = Path(config_dir).resolve() if config_dir is not None else configs_dir().resolve()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        return hydra.compose(config_name="config", overrides=list(overrides or []))


if __name__ == "__main__":
    main()