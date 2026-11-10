"""Input-only content-time estimation with two complementary motion cues."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from torch import Tensor

from .motion import MotionTraces, estimate_trace_lag, rgb_point_traces
from .temporal import ContentLagEstimate, _partition, estimate_content_lag


@dataclass(frozen=True, slots=True)
class AdaptiveLag:
    """Frame advance and the input evidence used to choose it."""

    advance_frames: float
    source: str
    trace: ContentLagEstimate
    same_phase_motion: ContentLagEstimate | None


def _credible_trace_peak(trace: ContentLagEstimate) -> float | None:
    if trace.support < 48 or len(trace.scores) < 3:
        return None
    peak = max(
        range(len(trace.scores)),
        key=lambda i: (trace.scores[i], -abs(trace.offsets[i])),
    )
    if not 0 < peak < len(trace.scores) - 1 or trace.peak_margin is None:
        return None
    center = trace.offsets[peak]
    window = [
        value
        for delay, value in zip(trace.offsets, trace.scores, strict=True)
        if abs(delay - center) <= 2.0
    ]
    span = max(window) - min(window)
    if span <= 1.0e-6 or trace.peak_margin / span < 0.075:
        return None
    return center


def choose_advance(
    trace: ContentLagEstimate,
    motion: ContentLagEstimate | None,
    *,
    prior: float = 0.0,
) -> AdaptiveLag:
    """Prefer a distinct trace peak, then use independent same-phase motion."""
    if not isfinite(prior):
        raise ValueError("The fallback advance must be finite")
    peak = _credible_trace_peak(trace)
    if peak is not None:
        return AdaptiveLag(peak, "rgb_paths", trace, motion)
    if motion is not None and motion.reliable:
        return AdaptiveLag(motion.advance_frames, "same_phase_motion", trace, motion)
    return AdaptiveLag(prior, "prior", trace, motion)


def estimate_video_lag(
    rgb: Tensor,
    features: Tensor,
    *,
    chunks: Sequence[tuple[int, int]] | None = None,
    rgb_paths: Sequence[MotionTraces] | None = None,
    prior: float = 0.0,
    max_shift: float = 3.0,
    step: float = 0.25,
) -> AdaptiveLag:
    """Estimate lag from a video's RGB frames and dense features on the same grid of times."""
    if rgb.ndim != 4 or features.ndim != 4 or len(rgb) != len(features) or not len(rgb):
        raise ValueError("Expected aligned RGB and feature time axes")
    ranges = _partition(len(rgb), chunks)
    paths = (
        rgb_paths
        if rgb_paths is not None
        else [rgb_point_traces(rgb[a:b]) for a, b in ranges]
    )
    trace = estimate_trace_lag(
        [features[a:b] for a, b in ranges],
        paths,
        max_shift=max_shift,
        step=step,
        min_margin=0.0,
    )
    motion = None
    if _credible_trace_peak(trace) is None:
        motion = estimate_content_lag(
            rgb, features, chunks=ranges, gap=4, max_shift=max_shift, step=step
        )
    return choose_advance(trace, motion, prior=prior)
