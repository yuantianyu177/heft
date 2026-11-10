"""Match tracking results to dataset ground truth and evaluate them."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from heft.data import AnnotatedVideo, TrackingDataset
from heft.tracking import AnnotationQueries, TrackingResult

from .config import EvaluationConfig
from .metrics import MetricCounts, compute_counts, macro_average, values_from_counts
from .result import EvaluationReport, VideoMetrics


def evaluate_dataset(
    results: Iterable[TrackingResult],
    dataset: TrackingDataset,
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Evaluate tracking results against GT loaded exclusively from ``dataset``."""

    resolved_config = EvaluationConfig() if config is None else config
    result_list = tuple(results)
    if not result_list:
        raise ValueError("results must not be empty")

    result_ids = [result.video_id for result in result_list]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("tracking result video IDs must be unique")

    requested = set(result_ids)
    samples = {
        sample.video_id: sample
        for index in range(len(dataset))
        if (sample := dataset[index]).video_id in requested
    }
    missing = requested.difference(samples)
    if missing:
        raise KeyError(f"video IDs not found in dataset: {sorted(missing)}")

    video_metrics: list[VideoMetrics] = []
    counts: list[MetricCounts] = []
    for result in result_list:
        metric_counts = _evaluate_video(
            result,
            samples[result.video_id],
            resolved_config,
        )
        counts.append(metric_counts)
        video_metrics.append(
            VideoMetrics(
                video_id=result.video_id,
                metrics=values_from_counts(
                    metric_counts,
                    resolved_config.thresholds,
                ),
            )
        )

    per_video_values = tuple(item.metrics for item in video_metrics)
    if resolved_config.aggregation == "macro":
        aggregate = macro_average(per_video_values, resolved_config.thresholds)
    else:
        total = counts[0]
        for item in counts[1:]:
            total += item
        aggregate = values_from_counts(total, resolved_config.thresholds)

    return EvaluationReport(
        aggregate=aggregate,
        videos=tuple(video_metrics),
        config=resolved_config,
    )


def _evaluate_video(
    result: TrackingResult,
    sample: AnnotatedVideo,
    config: EvaluationConfig,
) -> MetricCounts:
    annotations = sample.annotations
    num_frames = result.tracks.shape[1]
    if config.require_full_video and num_frames != sample.video.num_frames:
        raise ValueError(
            f"{sample.video_id}: full-video evaluation expects "
            f"{sample.video.num_frames} frames, got {num_frames}"
        )
    selected, query_frames = AnnotationQueries(
        annotations=annotations,
        policy=config.query_policy,
    ).indices(num_frames)
    ground_truth_tracks = annotations.tracks[selected, :num_frames]
    ground_truth_visibility = annotations.visibility[selected, :num_frames]

    if result.tracks.shape != ground_truth_tracks.shape:
        raise ValueError(
            f"{sample.video_id}: predicted and GT tracks must have identical shapes"
        )
    if result.visibility.shape != ground_truth_visibility.shape:
        raise ValueError(
            f"{sample.video_id}: predicted and GT visibility must have identical shapes"
        )

    height, width = result.frame_size
    frame_scale = torch.tensor((width - 1, height - 1), dtype=torch.float32)
    expected_xy = (
        ground_truth_tracks[torch.arange(selected.shape[0]), query_frames] * frame_scale
    )
    expected_queries = torch.cat(
        (expected_xy, query_frames[:, None].float()),
        dim=1,
    )
    if result.query_points.shape != expected_queries.shape or not torch.allclose(
        result.query_points,
        expected_queries,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError(f"{sample.video_id}: query points do not match dataset GT")

    reference_height, reference_width = config.reference_size
    reference_scale = torch.tensor(
        (reference_width, reference_height),
        dtype=torch.float32,
    )
    first_scored_frames = (
        query_frames if config.include_first_frame else query_frames + 1
    )
    evaluation_mask = torch.arange(num_frames) >= first_scored_frames[:, None]
    predicted_tracks = result.tracks.float() / frame_scale * reference_scale
    ground_truth_tracks = ground_truth_tracks * reference_scale
    return compute_counts(
        predicted_tracks,
        ground_truth_tracks,
        result.visibility,
        ground_truth_visibility,
        config.thresholds,
        evaluation_mask=evaluation_mask,
    )
