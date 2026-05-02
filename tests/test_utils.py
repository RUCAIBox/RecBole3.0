from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_lazy_import_resolves_only_when_used() -> None:
    code = textwrap.dedent(
        """
        import sys

        sys.modules.pop("calendar", None)

        from recbole3.utils import LazyImport

        lazy_calendar = LazyImport("calendar", "Calendar")

        assert "calendar" not in sys.modules
        assert repr(lazy_calendar) == "<lazy import calendar:Calendar>"
        assert lazy_calendar.resolve().__name__ == "Calendar"
        assert "calendar" in sys.modules
        assert lazy_calendar().firstweekday == 0
        """
    )
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
