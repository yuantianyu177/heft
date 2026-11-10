"""Tests for feature-video manifests and query generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from heft.data import TrackAnnotations
from heft.tracking import (
    AnnotationQueries,
    ExplicitQueries,
    FeatureVideo,
    FrameRange,
    GridQueries,
    MaskGridQueries,
    RopePairing,
    RopeSpec,
)


def _feature_video(tmp_path: Path) -> FeatureVideo:
    return FeatureVideo(
        root=tmp_path / "video-a",
        name="video-a",
        model="fake",
        num_frames=5,
        frame_size=(6, 8),
        feature_size=(3, 4),
        chunks=(
            FrameRange(index=0, start=0, stop=3, feature_frames=2),
            FrameRange(index=1, start=3, stop=5, feature_frames=1),
        ),
        step=4,
        layers=(1, 3),
        heads=(0, 2),
        feature_kinds=("query", "key"),
        rope=RopeSpec(
            pairing=RopePairing.ADJACENT,
            temporal_pairs=2,
            height_pairs=1,
            width_pairs=1,
        ),
    )


def test_feature_video_manifest_round_trip(tmp_path: Path) -> None:
    expected = _feature_video(tmp_path)

    manifest_path = expected.write()

    assert manifest_path == expected.root / "manifest.json"
    assert FeatureVideo.open(expected.root) == expected
    assert not list(expected.root.glob("*.tmp"))


def test_feature_video_refuses_to_overwrite_manifest_by_default(tmp_path: Path) -> None:
    feature_video = _feature_video(tmp_path)
    feature_video.write()

    with pytest.raises(FileExistsError, match="manifest already exists"):
        feature_video.write()

    feature_video.write(overwrite=True)


def test_rope_frequency_selection_keeps_complete_adjacent_pairs() -> None:
    rope = RopeSpec(
        pairing=RopePairing.ADJACENT,
        temporal_pairs=2,
        height_pairs=1,
        width_pairs=1,
    )

    assert rope.channel_indices((0.0, 0.5)) == (0, 1)


def test_cosmos_rope_frequency_selection_uses_split_half_pairs() -> None:
    rope = RopeSpec(
        pairing=RopePairing.SPLIT_HALF,
        temporal_pairs=22,
        height_pairs=21,
        width_pairs=21,
    )

    selected = rope.channel_indices((0.0, 0.5))

    assert selected == tuple(
        list(range(11))
        + list(range(22, 32))
        + list(range(43, 53))
        + list(range(64, 75))
        + list(range(86, 96))
        + list(range(107, 117))
    )


def test_grid_queries_use_pixel_centres_and_requested_frame(tmp_path: Path) -> None:
    feature_video = _feature_video(tmp_path)

    queries = GridQueries(stride=4, frame=2).generate(feature_video)

    torch.testing.assert_close(
        queries,
        torch.tensor(
            [
                [2.0, 2.0, 2.0],
                [6.0, 2.0, 2.0],
            ]
        ),
    )


def test_mask_grid_queries_resize_and_filter_the_grid(tmp_path: Path) -> None:
    feature_video = _feature_video(tmp_path)
    mask = torch.tensor(
        [
            [True, False],
            [False, False],
        ]
    )

    queries = MaskGridQueries(mask=mask, stride=4, frame=1).generate(feature_video)

    torch.testing.assert_close(queries, torch.tensor([[2.0, 2.0, 1.0]]))


def test_explicit_queries_are_normalized_for_tracking(tmp_path: Path) -> None:
    feature_video = _feature_video(tmp_path)
    source = torch.tensor([[1, 2, 0], [7, 5, 4]], dtype=torch.int64)

    queries = ExplicitQueries(source).generate(feature_video)

    assert queries.dtype is torch.float32
    assert queries.device.type == "cpu"
    assert queries.is_contiguous()
    torch.testing.assert_close(queries, source.float())


def test_annotation_queries_exclude_tracks_not_visible_on_first_frame(
    tmp_path: Path,
) -> None:
    feature_video = _feature_video(tmp_path)
    tracks = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]],
            [[0.2, 0.4], [0.3, 0.5], [0.4, 0.6], [0.5, 0.7], [0.6, 0.8]],
            [[0.9, 0.9], [0.8, 0.8], [0.7, 0.7], [0.6, 0.6], [0.5, 0.5]],
        ]
    )
    visibility = torch.tensor(
        [
            [False, False, False, False, False],
            [True, True, True, True, True],
            [False, True, True, True, True],
        ]
    )
    annotations = TrackAnnotations(tracks=tracks, visibility=visibility)

    queries = AnnotationQueries(
        annotations=annotations,
        policy="first_frame_visible",
    ).generate(feature_video)

    torch.testing.assert_close(queries, torch.tensor([[1.4, 2.0, 0.0]]))


def test_annotation_queries_default_to_each_tracks_first_visible_frame(
    tmp_path: Path,
) -> None:
    feature_video = _feature_video(tmp_path)
    tracks = torch.tensor(
        [
            [[0.0, 0.0], [0.25, 0.4], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            [[0.0, 0.0], [0.0, 0.0], [0.75, 0.8], [0.8, 0.8], [0.8, 0.8]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    visibility = torch.tensor(
        [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, False, False],
        ]
    )

    queries = AnnotationQueries(
        annotations=TrackAnnotations(tracks=tracks, visibility=visibility),
    ).generate(feature_video)

    torch.testing.assert_close(
        queries,
        torch.tensor(
            [
                [1.75, 2.0, 1.0],
                [5.25, 4.0, 2.0],
            ]
        ),
    )


def test_annotation_queries_ignore_visibility_after_discarded_tail() -> None:
    annotations = TrackAnnotations(
        tracks=torch.zeros(2, 5, 2),
        visibility=torch.tensor(
            [
                [False, True, True, True, True],
                [False, False, False, False, True],
            ]
        ),
    )

    selected, frames = AnnotationQueries(
        annotations,
        policy="first_visible",
    ).indices(num_frames=4)

    assert selected.tolist() == [0]
    assert frames.tolist() == [1]
