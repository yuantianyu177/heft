"""Label-free content clocks from the agreement of RGB traces and descriptors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .temporal import ContentLagEstimate


@dataclass(frozen=True, slots=True)
class MotionTraces:
    """Normalized [point, time, xy] RGB tracks and their visibility."""

    points: Tensor
    visible: Tensor


def rgb_point_traces(
    rgb: Tensor, *, raster: tuple[int, int] = (256, 256), max_points: int = 96
) -> MotionTraces:
    """Trace RGB corners with adjacent-frame and anchor consistency checks."""
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not len(rgb) or min(raster) < 16 or max_points < 1:
        raise ValueError("Expected RGB [time, 3, height, width] and a nonempty raster")
    pixels = rgb.detach().cpu()
    if pixels.is_floating_point():
        pixels = (pixels * 255).round().clamp_(0, 255).to(torch.uint8)
    if pixels.dtype != torch.uint8:
        raise ValueError("RGB must be uint8 or floating point in [0, 1]")
    height, width = raster
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (width, height), interpolation=cv2.INTER_AREA)
        for frame in pixels.permute(0, 2, 3, 1).numpy()
    ]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[6:-6, 6:-6] = 255
    anchors = cv2.goodFeaturesToTrack(gray[0], max_points, .015, 5., mask=mask)
    if anchors is None:
        return MotionTraces(torch.zeros((0, len(gray), 2)), torch.zeros((0, len(gray)), dtype=torch.bool))

    points = np.zeros((len(anchors), len(gray), 2), dtype=np.float32)
    valid = np.zeros(points.shape[:-1], dtype=bool)
    points[:, 0] = anchors[:, 0]
    valid[:, 0] = True
    tracking = dict(winSize=(17, 17), maxLevel=3)
    for time in range(1, len(gray)):
        rows = np.flatnonzero(valid[:, time - 1])
        if not len(rows):
            break
        previous = np.ascontiguousarray(points[rows, time-1, None])
        current, forward_ok, _ = cv2.calcOpticalFlowPyrLK(gray[time-1], gray[time], previous, None, **tracking)
        returned, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray[time], gray[time-1], current, None, **tracking)
        direct, anchor_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray[0], gray[time], anchors[rows], current.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **tracking
        )
        good = (forward_ok & back_ok & anchor_ok).reshape(-1).astype(bool)
        good &= np.linalg.norm(returned[:, 0] - previous[:, 0], axis=1) <= .65
        good &= np.linalg.norm(direct[:, 0] - current[:, 0], axis=1) <= 1.25
        xy = current[:, 0]
        good &= ((xy >= 2) & (xy <= np.array([width-3, height-3]))).all(axis=1)
        points[rows[good], time] = xy[good]
        valid[rows[good], time] = True
    points /= np.array([width - 1, height - 1], dtype=np.float32)
    return MotionTraces(torch.from_numpy(points), torch.from_numpy(valid))


def _trace_scores(
    feature: Tensor, traces: MotionTraces, offsets: Tensor, border: int, min_speed: float
) -> tuple[Tensor, int]:
    time_count = len(feature)
    coordinates = traces.points.to(device=feature.device, dtype=torch.float32).transpose(0, 1)
    visible = traces.visible.to(feature.device).T
    if not coordinates.shape[1] or time_count < 2 * border + 4:
        return offsets.new_zeros(len(offsets)), 0

    times = torch.arange(border + 1, time_count-border-1, device=feature.device)
    neighbors = times[:, None] + torch.arange(-border, border+1, device=feature.device)
    support = visible[neighbors].all(dim=1) & visible[0]
    speed = ((coordinates[times+1] - coordinates[times-1]) * 128.).norm(dim=-1)
    traveled = ((coordinates[times] - coordinates[0]) * 256.).norm(dim=-1)
    support &= (speed >= min_speed) & (traveled >= 2.)
    count = int(support.sum())
    if not count:
        return offsets.new_zeros(len(offsets)), 0

    anchor_grid = (coordinates[0] * 2 - 1)[None, :, None, :]
    anchors = F.grid_sample(feature[:1].float(), anchor_grid, align_corners=False, padding_mode="border")
    anchors = F.normalize(anchors, dim=1)
    # A delayed descriptor at nominal time t best describes the RGB point at
    # t - lag. The full search keeps the same track/time pairs for every lag.
    shifted = times[:, None] - offsets
    lo = shifted.floor().long()
    hi = (lo + 1).clamp_max(time_count-1)
    grid = torch.lerp(coordinates[lo], coordinates[hi], (shifted-lo)[..., None, None])
    grid = grid.transpose(1, 2) * 2 - 1
    samples = F.grid_sample(feature[times].float(), grid, align_corners=False, padding_mode="border")
    similarity = (F.normalize(samples, dim=1) * anchors).sum(dim=1)
    weights = torch.where(support, speed.clamp_max(3.), 0.)
    return (similarity * weights[..., None]).sum(dim=(0, 1)) / weights.sum() * count, count


@torch.inference_mode()
def estimate_trace_lag(
    feature_chunks: Sequence[Tensor],
    traces: Sequence[MotionTraces],
    *,
    max_shift: float = 3.,
    step: float = .25,
    min_speed: float = .6,
    min_support: int = 48,
    min_margin: float = .002,
) -> ContentLagEstimate:
    """Estimate a single signed offset from RGB traces anchored at clip starts."""
    if not feature_chunks or len(feature_chunks) != len(traces):
        raise ValueError("Provide matching feature and trace chunks")
    if not math.isfinite(step) or not math.isfinite(max_shift) or not 0 < step <= max_shift:
        raise ValueError("Require 0 < step <= max_shift")
    if min_support < 1 or min_speed < 0 or min_margin < 0:
        raise ValueError("Invalid motion confidence thresholds")
    bins = math.floor(max_shift / step)
    offsets = tuple(step * i for i in range(-bins, bins + 1))
    device = feature_chunks[0].device
    shift_tensor = torch.as_tensor(offsets, device=device, dtype=torch.float32)
    border = math.ceil(max(abs(x) for x in offsets))
    totals = torch.zeros_like(shift_tensor)
    total_pairs = 0
    for feature, tracks in zip(feature_chunks, traces, strict=True):
        if (feature.ndim != 4 or not len(feature) or feature.device != device
            or tracks.points.ndim != 3 or tracks.points.shape[1:] != (len(feature), 2)
            or tracks.visible.shape != tracks.points.shape[:-1]):
            raise ValueError("Invalid aligned feature and trace shapes")
        scores, count = _trace_scores(feature, tracks, shift_tensor, border, min_speed)
        totals += scores
        total_pairs += count

    if not total_pairs:
        return ContentLagEstimate(0., False, total_pairs, offsets, (), None, None)
    scores = tuple(float(x) for x in (totals / total_pairs).cpu())
    if not all(math.isfinite(score) for score in scores):
        return ContentLagEstimate(0., False, total_pairs, offsets, (), None, None)
    best = max(range(len(scores)), key=lambda i: (scores[i], -abs(offsets[i])))
    separation = min(1., bins * step)
    runner = max(score for offset, score in zip(offsets, scores, strict=True)
                 if abs(offset - offsets[best]) >= separation - 1e-9)
    margin = scores[best] - runner
    reliable = 0 < best < len(scores) - 1 and total_pairs >= min_support and margin >= min_margin
    return ContentLagEstimate(offsets[best] if reliable else 0., reliable, total_pairs, offsets,
                              scores, scores[best], margin)
