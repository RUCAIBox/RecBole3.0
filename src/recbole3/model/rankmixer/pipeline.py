from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf
import pandas as pd

from recbole3.config import RuntimeConfig, instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.dataset import ITEM_ID, LABEL, TIMESTAMP, USER_ID
from recbole3.dataset.avazu.ranking import AvazuCTRConfig, AvazuCTRParser
from recbole3.pipeline import Pipeline
from recbole3.model.rankmixer.config import RankMixerConfig
from recbole3.model.rankmixer.data import RankMixerPreparedData, resolve_rankmixer_feature_columns
from recbole3.utils import require_component_cfg, require_component_name


class RankMixerPipeline(Pipeline):
    """RankMixer pipeline that prepares point-wise CTR data without TaskDataset.prepare()."""

    def _parse_config(
        self,
        cfg: DictConfig,
    ) -> tuple[RuntimeConfig, DictConfig, DictConfig, DictConfig]:
        runtime_cfg = instantiate_dataclass(RuntimeConfig, cfg.get("runtime"))
        dataset_cfg = require_component_cfg(cfg, "dataset")
        model_cfg = require_component_cfg(cfg, "model")
        trainer_cfg = require_component_cfg(cfg, "trainer")
        return runtime_cfg, dataset_cfg, model_cfg, trainer_cfg

    def run(self) -> dict[str, Any]:
        runtime_cfg, dataset_cfg, model_cfg, trainer_cfg = self._parse_config(self.cfg)

        dataset_name = require_component_name(dataset_cfg, "dataset")
        dataset_spec = get_dataset_spec(dataset_name)

        dataset_config = instantiate_dataclass(dataset_spec.config_cls, dataset_cfg)
        rankmixer_config = instantiate_dataclass(RankMixerConfig, model_cfg)
        parser = self._build_rankmixer_parser(dataset_config)
        prepared_data = RankMixerPreparedData(
            config=dataset_config,
            train_frame=self._select_rankmixer_columns(
                parser._load_split_frame(dataset_config.train_path, split_name="train"),
                model_config=rankmixer_config,
            ),
            valid_frame=self._select_rankmixer_columns(
                parser._load_split_frame(dataset_config.valid_path, split_name="valid"),
                model_config=rankmixer_config,
            ),
            test_frame=self._select_rankmixer_columns(
                parser._load_split_frame(dataset_config.test_path, split_name="test"),
                model_config=rankmixer_config,
            ),
        )

        model = self.model_spec.model_cls(rankmixer_config)
        trainer = self.model_spec.trainer_cls(
            instantiate_dataclass(self.model_spec.trainer_config_cls, trainer_cfg)
        )

        os.makedirs(runtime_cfg.output_dir, exist_ok=True)
        with self._accelerate_runtime_device(runtime_cfg.device):
            run_result = trainer.run(model, prepared_data, output_dir=runtime_cfg.output_dir)

        printable = {
            "prepared_data": self.serialize_prepared_data(prepared_data),
            "fit": run_result["fit"],
            "test": run_result["test"],
        }
        print(OmegaConf.to_yaml(OmegaConf.create(printable), resolve=True))

        return {
            "prepared_data": prepared_data,
            **run_result,
        }

    @staticmethod
    def _build_rankmixer_parser(dataset_config: Any) -> AvazuCTRParser:
        if not isinstance(dataset_config, AvazuCTRConfig):
            raise TypeError(
                "RankMixerPipeline currently supports AvazuCTRConfig only. "
                f"Got {type(dataset_config).__name__}."
            )
        return AvazuCTRParser(dataset_config)

    @staticmethod
    def _select_rankmixer_columns(frame: pd.DataFrame, *, model_config: RankMixerConfig) -> pd.DataFrame:
        feature_columns = list(resolve_rankmixer_feature_columns(model_config))
        required_columns = [str(model_config.label_column), *feature_columns]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                "RankMixerPipeline requires labeled point-wise feature frames. "
                f"Missing columns: {missing}."
            )
        optional_configured_columns = [
            model_config.user_id_column,
            model_config.item_id_column,
            model_config.timestamp_column,
        ]
        optional_columns = [str(column) for column in optional_configured_columns if column and str(column) in frame.columns]
        selected_columns = [*optional_columns, *required_columns]
        renamed = frame.loc[:, selected_columns].reset_index(drop=True).copy()

        rename_map: dict[str, str] = {}
        if model_config.user_id_column and str(model_config.user_id_column) in renamed.columns:
            rename_map[str(model_config.user_id_column)] = USER_ID
        if model_config.item_id_column and str(model_config.item_id_column) in renamed.columns:
            rename_map[str(model_config.item_id_column)] = ITEM_ID
        if model_config.timestamp_column and str(model_config.timestamp_column) in renamed.columns:
            rename_map[str(model_config.timestamp_column)] = TIMESTAMP
        rename_map[str(model_config.label_column)] = LABEL
        renamed = renamed.rename(columns=rename_map)
        return renamed


__all__ = [
    "RankMixerPipeline",
]
