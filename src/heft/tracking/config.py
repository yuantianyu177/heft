"""Configuration and value objects for feature tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
from torch import Tensor

from heft.attn_hook import FeatureKind

from .features import FeatureVideo

_TRACKABLE_FEATURES = frozenset(
    (FeatureKind.QUERY, FeatureKind.KEY, FeatureKind.VALUE, FeatureKind.HIDDEN_STATES)
)


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """One captured layer, optionally restricted to one attention head."""

    layer: int
    head: int | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.head is not None and self.head < 0:
            raise ValueError("head must be non-negative")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Algorithm and memory settings shared by tracking tasks."""

    query_feature: FeatureKind = FeatureKind.QUERY
    target_feature: FeatureKind = FeatureKind.KEY
    update_feature: FeatureKind | None = None
    upsample_features: bool = True
    argmax_radius: float = 35.0
    search_radius: float = 100.0
    visibility_threshold: float = 16.0
    feature_ema_alpha: float = 0.05
    feature_update_sampling_radius: int = 1
    frequency_range: tuple[float, float] = (0.0, 1.0)
    point_batch_size: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_feature", FeatureKind(self.query_feature))
        object.__setattr__(self, "target_feature", FeatureKind(self.target_feature))
        if self.update_feature is not None:
            object.__setattr__(self, "update_feature", FeatureKind(self.update_feature))
        selected = (self.query_feature, self.target_feature, self.update_feature)
        if any(
            kind not in _TRACKABLE_FEATURES for kind in selected if kind is not None
        ):
            raise ValueError(
                "tracking requires query, key, value, or hidden-state features"
            )
        if self.argmax_radius < 0 or self.search_radius < 0:
            raise ValueError("tracking radii must be non-negative")
        if self.visibility_threshold < 0:
            raise ValueError("visibility_threshold must be non-negative")
        if not 0.0 <= self.feature_ema_alpha <= 1.0:
            raise ValueError("feature_ema_alpha must be between zero and one")
        if self.feature_update_sampling_radius < 0:
            raise ValueError("feature_update_sampling_radius must be non-negative")
        if self.point_batch_size <= 0:
            raise ValueError("point_batch_size must be positive")
        lower, upper = self.frequency_range
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("frequency_range must satisfy 0 <= lower < upper <= 1")
        if FeatureKind.VALUE in selected and (lower, upper) != (0.0, 1.0):
            raise ValueError(
                "value tracking requires the full channel range (0.0, 1.0)"
            )


@dataclass(frozen=True, slots=True, init=False)
class TrackingTask:
    """One independently schedulable video tracking request."""

    features: FeatureVideo | Path
    query_points: Tensor
    selection: FeatureSelection
    video_id: str | None
    time_advance_frames: float

    def __init__(
        self,
        *,
        features: FeatureVideo | str | os.PathLike[str],
        query_points: Tensor,
        selection: FeatureSelection,
        video_id: str | None = None,
        time_advance_frames: float = 0.0,
    ) -> None:
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError("query_points must have shape [point, 3]")
        if query_points.device.type != "cpu":
            raise ValueError("query_points must be a CPU tensor for multiprocessing")
        if not isfinite(time_advance_frames):
            raise ValueError("time_advance_frames must be finite")
        normalized_points = query_points.to(dtype=torch.float32).contiguous()
        object.__setattr__(
            self,
            "features",
            features if isinstance(features, FeatureVideo) else Path(features),
        )
        object.__setattr__(self, "query_points", normalized_points)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "time_advance_frames", float(time_advance_frames))


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """CPU trajectory and visibility tensors returned by the tracker."""

    video_id: str
    frame_size: tuple[int, int]
    query_points: Tensor
    tracks: Tensor
    visibility: Tensor
    selection: FeatureSelection
    gpu_id: int | None = None
