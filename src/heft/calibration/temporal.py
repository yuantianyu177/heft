"""Estimate a video's feature clock from its RGB motion and structure."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ContentLagEstimate:
    """A positive advance samples later features to compensate for delayed content."""

    advance_frames: float
    reliable: bool
    support: int
    offsets: tuple[float, ...]
    scores: tuple[float, ...]
    peak_score: float | None
    peak_margin: float | None


def retime_feature_volume(
    volume: Tensor, advance_frames: float, *, preserve_first: bool = True
) -> Tensor:
    """Resample within one clip and retain its original anchor by default."""
    if volume.ndim == 0 or len(volume) == 0 or not volume.is_floating_point():
        raise ValueError("Expected a nonempty floating-point time axis")
    if not math.isfinite(advance_frames):
        raise ValueError("The feature advance must be finite")
    if advance_frames == 0 or len(volume) == 1:
        return volume
    times = torch.arange(len(volume), device=volume.device, dtype=torch.float32)
    times = (times + advance_frames).clamp_(0, len(volume) - 1)
    lower = times.floor().long()
    upper = (lower + 1).clamp_max_(len(volume) - 1)
    weight = (times - lower).to(volume.dtype).reshape((-1,) + (1,) * (volume.ndim - 1))
    result = torch.lerp(volume[lower], volume[upper], weight)
    if preserve_first:
        result[0].copy_(volume[0])
    return result


def _partition(frames: int, chunks: Sequence[tuple[int, int]] | None) -> tuple[tuple[int, int], ...]:
    ranges = ((0, frames),) if chunks is None else tuple(chunks)
    previous = 0
    for start, stop in ranges:
        if start != previous or not start < stop <= frames:
            raise ValueError("Chunks must partition the video in chronological order")
        previous = stop
    if previous != frames:
        raise ValueError("Chunks must partition the video in chronological order")
    return ranges


def _motion_cues(volume: Tensor, gap: int) -> Tensor:
    # A common spatial contrast scale removes channel offsets and frame-wide gain.
    centered = volume - volume.mean(dim=(-2, -1), keepdim=True)
    contrast = centered.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    scaled = centered / contrast.clamp_min(1e-5)
    dx = F.pad((scaled[..., 1:] - scaled[..., :-1]).square().mean(1), (0, 1))
    dy = F.pad((scaled[..., 1:, :] - scaled[..., :-1, :]).square().mean(1), (0, 0, 0, 1))
    motion = (scaled[gap:] - scaled[:-gap]).square().mean(1).sqrt()
    return torch.stack((motion, dx[gap:] - dx[:-gap], dy[gap:] - dy[:-gap]), dim=1)


def _normalize_spatial(cues: Tensor) -> Tensor:
    flat = cues.flatten(-2)
    return F.normalize(flat - flat.mean(dim=-1, keepdim=True), dim=-1, eps=1e-5)


@torch.inference_mode()
def estimate_content_lag(
    rgb: Tensor,
    features: Tensor,
    *,
    chunks: Sequence[tuple[int, int]] | None = None,
    gap: int = 4,
    max_shift: float = 3.,
    step: float = .25,
    min_score: float = .12,
    min_margin: float = .01,
) -> ContentLagEstimate:
    """Compare spatial motion cues at matching timestamps, without point labels.

    RGB is [time, 3, height, width] in uint8 or float. Features must have one
    map per RGB frame in [time, channel, height, width] order. A gap of four
    compares frames from the same phase of a stride-four interleaved readout.
    """
    if (rgb.ndim != 4 or rgb.shape[1] != 3 or features.ndim != 4
        or len(rgb) != len(features) or len(rgb) == 0 or features.shape[1] == 0):
        raise ValueError("Expected RGB and features on the same nonempty frame axis")
    if min(features.shape[-2:]) < 2 or not features.is_floating_point():
        raise ValueError("Expected a two-dimensional floating-point feature grid")
    if rgb.dtype != torch.uint8 and not rgb.is_floating_point():
        raise ValueError("Expected uint8 or floating-point RGB")
    if gap < 1 or not math.isfinite(max_shift) or not math.isfinite(step) or not 0 < step <= max_shift:
        raise ValueError("Require gap >= 1 and 0 < step <= max_shift")
    if not (0 <= min_score <= 1 and 0 <= min_margin <= 2):
        raise ValueError("Invalid score or margin threshold")
    ranges = _partition(len(rgb), chunks)
    count = math.floor(max_shift / step)
    offsets = tuple(step * i for i in range(-count, count + 1))
    border = math.ceil(max(abs(shift) for shift in offsets))
    device = features.device
    weights = torch.tensor((.50, .25, .25), device=device)
    totals = torch.zeros(len(offsets), device=device)
    support = 0

    for start, stop in ranges:
        # All tested shifts use the same interior windows and omit the anchor.
        if stop - start <= gap + 2 * border + 1:
            continue
        images = rgb[start:stop].to(device=device, dtype=torch.float32)
        images = F.interpolate(images, size=features.shape[-2:], mode="area")
        rgb_cues = _motion_cues(images, gap)
        feature_cues = _motion_cues(features[start:stop].float(), gap)
        times = torch.arange(border + 1, len(rgb_cues) - border, device=device)
        moving = rgb_cues[times, 0].mean(dim=(-2, -1)) > 1e-4
        informative = moving & (feature_cues[times, 0].mean(dim=(-2, -1)) > 1e-4)
        valid_times = times[informative]
        support += len(valid_times)
        if not len(valid_times):
            continue

        feature_unit = _normalize_spatial(feature_cues[valid_times])
        for index, shift in enumerate(offsets):
            locations = valid_times.to(torch.float32) - shift
            lower = locations.floor().long()
            upper = (lower + 1).clamp_max(len(rgb_cues) - 1)
            blend = (locations - lower).reshape(-1, 1, 1, 1)
            rgb_unit = _normalize_spatial(torch.lerp(rgb_cues[lower], rgb_cues[upper], blend))
            totals[index] += ((feature_unit * rgb_unit).sum(-1) * weights).sum()

    if support == 0:
        return ContentLagEstimate(0., False, support, offsets, (), None, None)
    scores = tuple(float(x) for x in (totals / support).cpu())
    if not all(math.isfinite(score) for score in scores):
        return ContentLagEstimate(0., False, support, offsets, (), None, None)
    best = max(range(len(offsets)), key=lambda i: (scores[i], -abs(offsets[i])))
    separation = min(1., count * step)
    competitor = max(score for shift, score in zip(offsets, scores, strict=True)
                     if abs(shift - offsets[best]) >= separation - 1e-9)
    margin = scores[best] - competitor
    reliable = support >= 6 and 0 < best < len(offsets) - 1 and scores[best] >= min_score and margin >= min_margin
    return ContentLagEstimate(offsets[best] if reliable else 0., reliable, support, offsets, scores, scores[best], margin)
