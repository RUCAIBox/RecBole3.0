from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.evaluation.metric import EvalProtocol, MetricSpec


@dataclass(slots=True)
class EvalConfig:
    """Evaluation configuration shared by trainer evaluation flows."""

    protocol: EvalProtocol = field(metadata={"help": "Evaluation protocol."})
    metrics: tuple[MetricSpec, ...] = field(
        default_factory=tuple,
        metadata={"help": "Built-in metrics instantiated for validation and test."},
    )
    neg_sampling_num: int = field(
        default=100,
        metadata={"help": "Number of sampled negatives per group for sampled evaluation."},
    )
    candidate_seed: int = field(
        default=42,
        metadata={"help": "Seed used when generating sampled evaluation candidates."},
    )
    exclude_history: bool = field(
        default=True,
        metadata={"help": "Whether full retrieval evaluation filters items present in seen_item_ids."},
    )
