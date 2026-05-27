from __future__ import annotations

from recbole3.model.sequential import BaseSequentialModelDataset


class STARecModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds sequential histories for STARec memory."""

    pass


__all__ = [
    "STARecModelDataset",
]
