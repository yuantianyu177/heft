"""Vectorized TAP-Vid-style point tracking metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .result import MetricValues


@dataclass(frozen=True, slots=True)
class MetricCounts:
    occlusion_correct: int
    positions: int
    visible: int
    within_correct: tuple[int, ...]
    true_positives: tuple[int, ...]
    jaccard_denominators: tuple[int, ...]

    def __add__(self, other: MetricCounts) -> MetricCounts:
        return MetricCounts(
            occlusion_correct=self.occlusion_correct + other.occlusion_correct,
            positions=self.positions + other.positions,
            visible=self.visible + other.visible,
            within_correct=tuple(
                left + right
                for left, right in zip(self.within_correct, other.within_correct)
            ),
            true_positives=tuple(
                left + right
                for left, right in zip(self.true_positives, other.true_positives)
            ),
            jaccard_denominators=tuple(
                left + right
                for left, right in zip(
                    self.jaccard_denominators, other.jaccard_denominators
                )
            ),
        )


def compute_counts(
    predicted_tracks: Tensor,
    ground_truth_tracks: Tensor,
    predicted_visibility: Tensor,
    ground_truth_visibility: Tensor,
    thresholds: tuple[float, ...],
    *,
    evaluation_mask: Tensor | None = None,
) -> MetricCounts:
    """Compute reusable integer counts for all thresholds in one pass."""

    squared_distance = (predicted_tracks - ground_truth_tracks).square().sum(dim=-1)
    threshold_tensor = torch.tensor(thresholds, dtype=squared_distance.dtype)
    within = squared_distance.unsqueeze(0) < threshold_tensor[:, None, None].square()
    ground_truth_visibility = ground_truth_visibility.bool()
    predicted_visibility = predicted_visibility.bool()
    if evaluation_mask is None:
        evaluation_mask = torch.ones_like(ground_truth_visibility)
    visible = (ground_truth_visibility & evaluation_mask).unsqueeze(0)
    predicted_visible = (predicted_visibility & evaluation_mask).unsqueeze(0)

    within_correct = (within & visible).sum(dim=(1, 2))
    true_positives = (within & visible & predicted_visible).sum(dim=(1, 2))
    false_positives = (predicted_visible & (~visible | ~within)).sum(dim=(1, 2))
    visible_count = int(visible.sum().item())
    correct_visibility = predicted_visibility == ground_truth_visibility

    return MetricCounts(
        occlusion_correct=int((correct_visibility & evaluation_mask).sum().item()),
        positions=int(evaluation_mask.sum().item()),
        visible=visible_count,
        within_correct=tuple(int(value) for value in within_correct.tolist()),
        true_positives=tuple(int(value) for value in true_positives.tolist()),
        jaccard_denominators=tuple(
            visible_count + int(value) for value in false_positives.tolist()
        ),
    )


def values_from_counts(
    counts: MetricCounts,
    thresholds: tuple[float, ...],
) -> MetricValues:
    pts_within = {
        threshold: _divide(correct, counts.visible)
        for threshold, correct in zip(thresholds, counts.within_correct)
    }
    jaccard = {
        threshold: _divide(true_positive, denominator)
        for threshold, true_positive, denominator in zip(
            thresholds,
            counts.true_positives,
            counts.jaccard_denominators,
        )
    }
    return MetricValues(
        occlusion_accuracy=_divide(counts.occlusion_correct, counts.positions),
        pts_within=pts_within,
        jaccard=jaccard,
        average_pts_within_thresh=sum(pts_within.values()) / len(pts_within),
        average_jaccard=sum(jaccard.values()) / len(jaccard),
    )


def macro_average(
    values: tuple[MetricValues, ...],
    thresholds: tuple[float, ...],
) -> MetricValues:
    count = len(values)
    pts_within = {
        threshold: sum(item.pts_within[threshold] for item in values) / count
        for threshold in thresholds
    }
    jaccard = {
        threshold: sum(item.jaccard[threshold] for item in values) / count
        for threshold in thresholds
    }
    return MetricValues(
        occlusion_accuracy=sum(item.occlusion_accuracy for item in values) / count,
        pts_within=pts_within,
        jaccard=jaccard,
        average_pts_within_thresh=sum(pts_within.values()) / len(pts_within),
        average_jaccard=sum(jaccard.values()) / len(jaccard),
    )


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
