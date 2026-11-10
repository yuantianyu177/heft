"""Tests for dataset-backed tracking evaluation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from heft.data import AnnotatedVideo, ArrayVideoSource, TrackAnnotations
from heft.evaluation import EvaluationConfig, evaluate_dataset
from heft.tracking import AnnotationQueries, FeatureSelection, TrackingResult


class MemoryDataset:
    def __init__(self, samples: tuple[AnnotatedVideo, ...]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AnnotatedVideo:
        return self.samples[index]


def _sample(
    video_id: str,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
) -> AnnotatedVideo:
    frames = np.zeros((tracks.shape[1], 4, 5, 3), dtype=np.uint8)
    return AnnotatedVideo(
        video_id=video_id,
        video=ArrayVideoSource(frames),
        annotations=TrackAnnotations(tracks=tracks, visibility=visibility),
    )


def _result(
    sample: AnnotatedVideo,
    *,
    visibility: torch.Tensor | None = None,
    frame_size: tuple[int, int] = (11, 21),
) -> TrackingResult:
    selected, frames = AnnotationQueries(sample.annotations).indices()
    normalized_tracks = sample.annotations.tracks[selected]
    height, width = frame_size
    scale = torch.tensor((width - 1, height - 1), dtype=torch.float32)
    tracks = normalized_tracks * scale
    query_points = torch.cat(
        (
            tracks[torch.arange(len(frames)), frames],
            frames[:, None].float(),
        ),
        dim=1,
    )
    return TrackingResult(
        video_id=sample.video_id,
        frame_size=frame_size,
        query_points=query_points,
        tracks=tracks,
        visibility=(
            sample.annotations.visibility[selected]
            if visibility is None
            else visibility
        ),
        selection=FeatureSelection(layer=0),
    )


def test_evaluation_loads_gt_from_dataset_and_selects_first_visible_queries() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor(
            [
                [[0.25, 0.50], [0.50, 0.75]],
                [[0.10, 0.20], [0.20, 0.30]],
            ]
        ),
        visibility=torch.tensor(
            [
                [True, True],
                [False, True],
            ]
        ),
    )

    report = evaluate_dataset((_result(sample),), MemoryDataset((sample,)))

    assert report.videos[0].video_id == "scene"
    assert report.aggregate.occlusion_accuracy == 1.0
    assert report.aggregate.average_pts_within_thresh == 1.0
    assert report.aggregate.average_jaccard == 1.0


def test_evaluation_scores_only_frames_after_each_query_by_default() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor(
            [
                [[0.25, 0.50], [0.50, 0.75], [0.75, 1.00]],
                [[0.10, 0.20], [0.20, 0.30], [0.30, 0.40]],
            ]
        ),
        visibility=torch.tensor([[True, True, True], [False, True, True]]),
    )
    result = _result(
        sample, visibility=torch.tensor([[True, True, True], [True, True, False]])
    )
    result.tracks[1, 2] = 0.0
    dataset = MemoryDataset((sample,))

    excluded = evaluate_dataset((result,), dataset).aggregate
    included = evaluate_dataset(
        (result,), dataset, config=EvaluationConfig(include_first_frame=True)
    ).aggregate

    assert excluded.occlusion_accuracy == pytest.approx(2 / 3)
    assert excluded.average_pts_within_thresh == pytest.approx(2 / 3)
    assert excluded.average_jaccard == pytest.approx(2 / 3)
    assert included.occlusion_accuracy == 0.8
    assert included.average_pts_within_thresh == 0.8
    assert included.average_jaccard == 0.8


def test_evaluation_uses_256_as_the_coordinate_scale() -> None:
    sample = _sample(
        "scene",
        tracks=torch.zeros((1, 2, 2)),
        visibility=torch.ones((1, 2), dtype=torch.bool),
    )
    result = _result(sample, frame_size=(257, 257))
    result.tracks[0, 1, 0] = 1.0
    metrics = evaluate_dataset((result,), MemoryDataset((sample,))).aggregate

    assert metrics.pts_within[1.0] == metrics.jaccard[1.0] == 0.0
    assert metrics.pts_within[2.0] == metrics.jaccard[2.0] == 1.0


def test_evaluation_matches_video_ids_and_supports_macro_or_micro_average() -> None:
    first = _sample(
        "first",
        tracks=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
        visibility=torch.tensor([[True, False]]),
    )
    second = _sample(
        "second",
        tracks=torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ),
        visibility=torch.ones((2, 2), dtype=torch.bool),
    )
    first_result = _result(first, visibility=torch.tensor([[True, True]]))
    second_result = _result(second)
    dataset = MemoryDataset((second, first))

    macro = evaluate_dataset((first_result, second_result), dataset)
    micro = evaluate_dataset(
        (first_result, second_result),
        dataset,
        config=EvaluationConfig(aggregation="micro"),
    )

    assert [item.video_id for item in macro.videos] == ["first", "second"]
    assert macro.aggregate.occlusion_accuracy == 0.5
    assert micro.aggregate.occlusion_accuracy == pytest.approx(2 / 3)
    assert macro.videos[0].metrics.jaccard[1.0] == 0.0


def test_evaluation_rejects_unaligned_queries_and_frame_counts() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor([[[0.25, 0.50], [0.50, 0.75]]]),
        visibility=torch.ones((1, 2), dtype=torch.bool),
    )
    result = _result(sample)
    shifted_queries = result.query_points.clone()
    shifted_queries[0, 0] += 1

    with pytest.raises(ValueError, match="query points"):
        evaluate_dataset(
            (replace(result, query_points=shifted_queries),),
            MemoryDataset((sample,)),
        )

    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_dataset(
            (replace(result, visibility=result.visibility[:, :1]),),
            MemoryDataset((sample,)),
        )


def test_evaluation_requires_full_video_unless_partial_coverage_is_enabled() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor([[[0.25, 0.50], [0.50, 0.75], [0.75, 1.00]]]),
        visibility=torch.ones((1, 3), dtype=torch.bool),
    )
    result = _result(sample)
    result = replace(
        result,
        tracks=result.tracks[:, :2],
        visibility=result.visibility[:, :2],
    )

    dataset = MemoryDataset((sample,))
    with pytest.raises(
        ValueError, match="full-video evaluation expects 3 frames, got 2"
    ):
        evaluate_dataset((result,), dataset)

    report = evaluate_dataset(
        (result,), dataset, config=EvaluationConfig(require_full_video=False)
    )

    assert report.aggregate.occlusion_accuracy == 1.0
    assert report.aggregate.average_pts_within_thresh == 1.0
