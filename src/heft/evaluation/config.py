"""Configuration for dataset-backed tracking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from heft.tracking import AnnotationQueryPolicy


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Frame coverage, metric coordinate system, and aggregation policy."""

    query_policy: AnnotationQueryPolicy = "first_visible"
    thresholds: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    reference_size: tuple[int, int] = (256, 256)
    aggregation: Literal["macro", "micro"] = "macro"
    # Include each trajectory's query frame when enabled.
    include_first_frame: bool = False
    require_full_video: bool = True

    def __post_init__(self) -> None:
        thresholds = tuple(float(value) for value in self.thresholds)
        if not thresholds or any(value <= 0 for value in thresholds):
            raise ValueError("thresholds must contain positive values")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("thresholds must be unique")
        if min(self.reference_size) <= 1:
            raise ValueError("reference_size dimensions must be greater than one")
        if self.aggregation not in ("macro", "micro"):
            raise ValueError("aggregation must be 'macro' or 'micro'")
        object.__setattr__(self, "thresholds", thresholds)
