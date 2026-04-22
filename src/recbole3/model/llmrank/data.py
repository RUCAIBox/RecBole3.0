from __future__ import annotations

from recbole3.model.sequential import BaseSequentialRetrievalModelDataset


class LLMRankModelDataset(BaseSequentialRetrievalModelDataset):
    """Model-side retrieval dataset that only adds sequential histories for prompting."""

    pass


__all__ = [
    "LLMRankModelDataset",
]
