from __future__ import annotations

from recbole3.model.sequential import BaseSequentialModelDataset


class LLM4RSModelDataset(BaseSequentialModelDataset):
    """Adds chronological prompt histories to candidate evaluation records."""

    pass


__all__ = [
    "LLM4RSModelDataset",
]
