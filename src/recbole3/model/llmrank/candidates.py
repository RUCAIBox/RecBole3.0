"""Backward-compatible imports for the shared candidate-generation module."""

from recbole3.dataset.candidates import (
    BaseCandidateGenerator,
    BM25CandidateGenerator,
    BM25Model,
    CandidateGenerationConfig,
    HSTUCandidateGenerator,
    ModelBackboneCandidateGenerator,
    RandomCandidateGenerator,
    build_candidate_frames,
)


__all__ = [
    "BaseCandidateGenerator",
    "BM25CandidateGenerator",
    "BM25Model",
    "CandidateGenerationConfig",
    "HSTUCandidateGenerator",
    "ModelBackboneCandidateGenerator",
    "RandomCandidateGenerator",
    "build_candidate_frames",
]
