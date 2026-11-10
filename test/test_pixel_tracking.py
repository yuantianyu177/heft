"""Regression tests for projecting descriptors before pixel interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from heft.attn_hook import FeatureKind
from heft.tracking import (
    FeatureSelection,
    FeatureTracker,
    FeatureVideo,
    FrameRange,
    RopePairing,
    RopeSpec,
    TrackingConfig,
    tracker,
)
from heft.tracking.ops import DescriptorState, TrackingOps

_DEVICES = (
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("head", [0, None])
@pytest.mark.parametrize("upsample_features", [True, False])
def test_pixel_tracker_matches_dense_reference_without_upsampling_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    head: int | None,
    upsample_features: bool,
) -> None:
    kinds = (FeatureKind("query"), FeatureKind("key"), FeatureKind("value"))
    video = FeatureVideo(
        root=tmp_path,
        name="pixel-tracking",
        model="fake",
        num_frames=5,
        frame_size=(11, 17),
        feature_size=(4, 6),
        chunks=(
            FrameRange(0, 0, 3, 2, feature_frame_stride=2),
            FrameRange(1, 3, 5, 2, feature_frame_stride=1),
        ),
        step=4,
        layers=(0,),
        heads=(0, 1),
        feature_kinds=tuple(kind.value for kind in kinds),
        rope=RopeSpec(RopePairing.ADJACENT, 1, 1, 1),
    )
    generator = torch.Generator().manual_seed(28)
    for chunk in video.chunks:
        folder = tmp_path / f"chunk_{chunk.index:03d}" / "step_004" / "layer_000"
        folder.mkdir(parents=True)
        for kind in kinds:
            save_file(
                {
                    f"head_{index:03d}": torch.randn(
                        1, chunk.feature_frames * 24, 6, generator=generator
                    )
                    for index in (0, 1)
                },
                folder / f"{kind.value}.safetensors",
            )
    config = TrackingConfig(
        query_feature=kinds[0],
        target_feature=kinds[1],
        update_feature=kinds[2],
        upsample_features=upsample_features,
        argmax_radius=1.8,
        search_radius=4.3,
        visibility_threshold=5,
        feature_ema_alpha=0.1,
        point_batch_size=2,
    )
    queries = torch.tensor([[1.25, 2.75, 0.0], [8.7, 5.3, 0.0], [16.0, 10.0, 0.0]])
    selection = FeatureSelection(layer=0, head=head)
    feature_tracker = FeatureTracker(config=config, device=device)
    with monkeypatch.context() as reference:
        reference.setattr(tracker, "TrackingOps", TrackingOps)
        expected = feature_tracker.track(video, queries, selection)

    interpolate = F.interpolate

    def guard_interpolate(tensor, *args, **kwargs):
        if kwargs.get("mode") == "bilinear" and kwargs.get("size") == video.frame_size:
            assert tensor.shape[1] == 1, (
                "pixel interpolation must operate on scalar maps"
            )
        return interpolate(tensor, *args, **kwargs)

    monkeypatch.setattr(F, "interpolate", guard_interpolate)
    actual = feature_tracker.track(video, queries, selection)
    torch.testing.assert_close(actual.tracks, expected.tracks, atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(actual.visibility, expected.visibility)


@pytest.mark.parametrize(
    ("frame_size", "grid_size"),
    [((9, 11), (4, 5)), ((1, 13), (2, 5)), ((9, 1), (3, 2)), ((4, 5), (4, 5))],
)
def test_pixel_samples_preserve_bilinear_then_fractional_sampling(
    frame_size: tuple[int, int], grid_size: tuple[int, int]
) -> None:
    config = TrackingConfig()
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(63)
    volume = torch.randn(2, 6, *grid_size, generator=generator)
    height, width = frame_size
    points = torch.tensor(
        [
            [0.0, 0.0],
            [width - 1, height - 1],
            [0.31 * width, 0.37 * height],
            [width - 0.2, height - 0.4],
            [-0.25, -0.25],
        ],
    )
    reference = TrackingOps(frame_size=frame_size, config=config, device=device)
    ops = tracker.TrackingOps(frame_size=frame_size, config=config, device=device)
    expected = reference.sample(reference.frame(volume, 1), points)
    source = ops.frame(volume, 1)
    assert source.shape == volume.shape[1:]
    torch.testing.assert_close(
        ops.sample(source, points), expected, atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("device_name", _DEVICES)
def test_pixel_search_preserves_cosine_when_corner_features_nearly_cancel(
    device_name: str,
) -> None:
    device = torch.device(device_name)
    generator = torch.Generator().manual_seed(4)
    left = torch.randn(128, generator=generator).to(device)
    residual = torch.randn(128, generator=generator).to(device)
    right = -left + 1e-7 * residual
    volume = torch.stack((left, right), dim=-1).reshape(1, 128, 1, 2)
    config = TrackingConfig(argmax_radius=0, search_radius=1.1)
    ops = tracker.TrackingOps(frame_size=(1, 5), config=config, device=device)
    state = DescriptorState.from_values((-left)[None])
    prediction = ops.predict(
        state,
        ops.frame(volume, 0),
        previous_positions=torch.tensor([[2.0, 0.0]], device=device),
        previous_visibility=torch.ones(1, dtype=torch.bool, device=device),
        apply_search_mask=True,
        softmax_before_resize=False,
    )
    torch.testing.assert_close(prediction, torch.tensor([[3.0, 0.0]], device=device))
