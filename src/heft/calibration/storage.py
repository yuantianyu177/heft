"""Estimate the content clock of features saved by the extractor."""

from __future__ import annotations

import os

import torch
from torch import Tensor

from heft.attn_hook import FeatureKind
from heft.tracking.config import FeatureSelection
from heft.tracking.features import FeatureVideo
from heft.tracking.pixel_ops import PixelTrackingOps
from heft.tracking.reader import FeatureReader

from .adaptive import AdaptiveLag, estimate_video_lag


@torch.inference_mode()
def estimate_feature_video_lag(
    features: FeatureVideo | str | os.PathLike[str],
    rgb: Tensor,
    selection: FeatureSelection,
    *,
    kind: FeatureKind = FeatureKind.HIDDEN_STATES,
    frequency_range: tuple[float, float] = (0.0, 1.0),
    prior: float = 0.0,
    device: str | torch.device = "cpu",
) -> AdaptiveLag:
    """Estimate one video offset using its saved descriptors and original RGB frames."""
    from heft.tracking.tracker import _local_window

    video = features if isinstance(features, FeatureVideo) else FeatureVideo.open(features)
    if rgb.ndim != 4 or rgb.shape[:2] != (video.num_frames, 3):
        raise ValueError("Supply exactly the captured RGB frames in [T, 3, H, W] order")
    kind = FeatureKind(kind)
    reader = FeatureReader(video, device=torch.device(device))
    dense = None
    for chunk in video.chunks:
        volume = reader.load_chunk(
            chunk, selection=selection, kinds=(kind,), frequency_range=frequency_range,
        )[kind]
        local_start, local_frames = _local_window(chunk)
        aligned = PixelTrackingOps.interpolate_time(
            volume,
            local_frames,
            feature_frame_stride=chunk.feature_frame_stride,
            first_feature_frame=chunk.first_feature_frame,
            first_rgb_frame=local_start - chunk.start,
        )
        if dense is None:
            dense = volume.new_empty(video.num_frames, *volume.shape[1:])
        dense[chunk.start:chunk.stop].copy_(aligned[chunk.start - local_start:])

    assert dense is not None
    ranges = tuple((chunk.start, chunk.stop) for chunk in video.chunks)
    return estimate_video_lag(rgb, dense, chunks=ranges, prior=prior)
