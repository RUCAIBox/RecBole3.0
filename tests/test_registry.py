from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests.test_helpers import ensure_stub_tables
from recbole3.dataset import get_dataset_spec
from recbole3.model import get_model_spec


def test_component_tables_expose_stub_components() -> None:
    ensure_stub_tables()

    assert get_dataset_spec("stub_dataset").config_cls.__name__ == "StubDatasetConfig"
    model_spec = get_model_spec("stub_model")
    assert model_spec.config_cls.__name__ == "StubModelConfig"
    assert model_spec.trainer_config_cls.__name__ == "StubTrainerConfig"


def test_model_registry_import_does_not_require_transformers() -> None:
    code = textwrap.dedent(
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "transformers" or name.startswith("transformers."):
                raise AssertionError(f"unexpected transformers import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        from recbole3.model import get_model_spec

        assert get_model_spec("hstu").config_cls.__name__ == "HSTUConfig"
        assert get_model_spec("rqvae").config_cls.__name__ == "RQVAEConfig"
        assert get_model_spec("lcrec").config_cls.__name__ == "LCRecConfig"
        assert get_model_spec("minionerec").config_cls.__name__ == "MiniOneRecConfig"
        assert "transformers" not in sys.modules
        """
    )
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
