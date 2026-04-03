from __future__ import annotations

from tests.test_helpers import ensure_stub_tables
from recbole3.dataset import get_dataset_spec
from recbole3.model import get_model_spec
from recbole3.trainer import get_trainer_spec


def test_component_tables_expose_stub_components() -> None:
    ensure_stub_tables()

    assert get_dataset_spec("stub_dataset").config_cls.__name__ == "StubDatasetConfig"
    assert get_model_spec("stub_model").config_cls.__name__ == "StubModelConfig"
    assert get_trainer_spec("stub_trainer").config_cls.__name__ == "StubTrainerConfig"
