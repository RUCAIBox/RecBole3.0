from __future__ import annotations

from typing import Any

from recbole3.model.lcrec.config import LCRecConfig

__all__ = [
    "LCRecConfig",
    "LCRecItemTokenizer",
    "LCRecTrainer",
    "get_lcrec_sft_datasets",
]


def __getattr__(name: str) -> Any:
    if name == "LCRecTrainer":
        from recbole3.model.lcrec.trainer import LCRecTrainer

        globals()[name] = LCRecTrainer
        return LCRecTrainer

    if name in {"LCRecItemTokenizer", "get_lcrec_sft_datasets"}:
        from recbole3.model.lcrec.data import LCRecItemTokenizer, get_lcrec_sft_datasets

        values = {
            "LCRecItemTokenizer": LCRecItemTokenizer,
            "get_lcrec_sft_datasets": get_lcrec_sft_datasets,
        }
        globals().update(values)
        return values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
