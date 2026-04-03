from __future__ import annotations

import pytest

from recbole3.dataset import get_dataset_spec


def test_get_dataset_spec_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset_spec("does_not_exist")
