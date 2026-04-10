from __future__ import annotations

import importlib

import pytest
from hydra.errors import ConfigCompositionException

from recbole3.dataset import get_dataset_spec
from recbole3.run import compose_config


def test_get_dataset_spec_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset_spec("does_not_exist")


def test_core_modules_import_without_cycles() -> None:
    assert importlib.import_module("recbole3.model") is not None
    assert importlib.import_module("recbole3.trainer") is not None
    assert importlib.import_module("recbole3.trainer_config") is not None
    assert importlib.import_module("recbole3.run") is not None


def test_compose_config_rejects_independent_trainer_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="Could not override 'trainer'"):
        compose_config(overrides=["dataset=amazon2023_retrieval", "model=hstu", "trainer=retrieval"])
