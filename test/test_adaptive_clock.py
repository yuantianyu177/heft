"""Tests for contrast-invariant confidence and input-only clock fallbacks."""

import pytest
import torch

from heft.calibration import ContentLagEstimate
from heft.calibration.adaptive import choose_advance, estimate_video_lag


def curve(center, amplitude, *, support=256, reliable=False):
    offsets = tuple(i / 4 for i in range(-12, 13))
    values = tuple(0.8 - amplitude * (x - center) ** 2 for x in offsets)
    best = max(values)
    rival = max(
        v for t, v in zip(offsets, values, strict=True) if abs(t - center) >= 1.0
    )
    return ContentLagEstimate(
        center if reliable else 0.0,
        reliable,
        support,
        offsets,
        values,
        best,
        best - rival,
    )


def test_relative_confidence_preserves_peak_when_cosine_contrast_changes():
    flat_scale = curve(1.25, 0.0001)
    larger_scale = curve(1.25, 0.02)
    other = curve(-0.5, 0.02, reliable=True)
    assert choose_advance(flat_scale, other).advance_frames == pytest.approx(1.25)
    assert choose_advance(larger_scale, other).advance_frames == pytest.approx(1.25)


def test_limited_support_and_boundary_peaks_use_independent_motion_cue():
    other = curve(0.75, 0.02, reliable=True)
    for trace in (curve(1.25, 0.02, support=9), curve(3.0, 0.02), curve(0.0, 0.0)):
        estimate = choose_advance(trace, other, prior=1.25)
        assert estimate.source == "same_phase_motion"
        assert estimate.advance_frames == 0.75


def test_static_input_reports_fallback_and_validates_partition():
    rgb = torch.zeros((12, 3, 16, 16), dtype=torch.uint8)
    features = torch.zeros((12, 8, 4, 4))
    estimate = estimate_video_lag(rgb, features, chunks=((0, 12),), prior=1.25)
    assert estimate.source == "prior"
    assert estimate.advance_frames == 1.25
    with pytest.raises(ValueError, match="partition"):
        estimate_video_lag(rgb, features, chunks=((1, 12),))
