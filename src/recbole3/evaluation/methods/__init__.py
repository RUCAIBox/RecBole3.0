from __future__ import annotations

from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.methods.base import BaseEvaluationMethod
from recbole3.evaluation.methods.full import FullEvaluationMethod
from recbole3.evaluation.methods.labeled import LabeledEvaluationMethod
from recbole3.evaluation.methods.sampled import SampledEvaluationMethod


def create_evaluation_method(config: EvalConfig) -> BaseEvaluationMethod:
    protocol = config.protocol
    if protocol == "labeled":
        return LabeledEvaluationMethod(metric_specs=config.metrics)
    if protocol == "sampled":
        return SampledEvaluationMethod(metric_specs=config.metrics)
    if protocol == "full":
        return FullEvaluationMethod(
            metric_specs=config.metrics,
            exclude_history=bool(config.exclude_history),
        )
    raise ValueError(f"Unsupported evaluation protocol '{protocol}'.")


__all__ = [
    "BaseEvaluationMethod",
    "FullEvaluationMethod",
    "LabeledEvaluationMethod",
    "SampledEvaluationMethod",
    "create_evaluation_method",
]

