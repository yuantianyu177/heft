"""Marker prompting and mask-constrained regeneration for a video denoiser."""

from collections.abc import Callable
from math import isfinite

import cv2
import numpy as np
import torch
from torch import Tensor


def _video_shape(video: np.ndarray) -> tuple[int, int, int]:
    if (
        video.ndim != 4
        or video.shape[-1] != 3
        or video.dtype != np.uint8
        or not video.size
    ):
        raise ValueError(
            "Expected nonempty RGB video [time, height, width, 3] with uint8 pixels"
        )
    return video.shape[:3]


def balance_red(video: np.ndarray) -> np.ndarray:
    """Desaturate bright red in the input using the Point Prompting HSV ellipse."""
    _video_shape(video)
    output = video.copy()
    for i, frame in enumerate(video):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hue = (hsv[..., 0].astype(np.int16) * 2 + 180) % 360 - 180
        saturation, value = hsv[..., 1:].astype(np.float32).transpose(2, 0, 1)
        ellipse = ((saturation - 255) / 80) ** 2 + ((value - 255) / 30) ** 2
        eligible = (hue >= -30) & (hue <= 10) & (ellipse <= 1.0)
        hsv[..., 1][eligible] = np.minimum(hsv[..., 1][eligible], 80)
        output[i][eligible] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[eligible]
    return output


def marker_disk(
    centers: np.ndarray, size: tuple[int, int], radius: float
) -> np.ndarray:
    """Return a binary disk around each subpixel (x, y) center."""
    xy = np.asarray(centers, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[-1] != 2 or min(size) < 1 or radius <= 0:
        raise ValueError(
            "Expected [time, xy] centers and a positive mask size and radius"
        )
    y, x = np.ogrid[: size[0], : size[1]]
    return (
        (x - xy[:, 0, None, None]) ** 2 + (y - xy[:, 1, None, None]) ** 2
    ) <= radius**2


def trace_marker(
    video: np.ndarray,
    xy: tuple[float, float],
    *,
    radius: float = 90.0,
    max_radius: float = 150.0,
    expansion: float = 1.1,
    blob_radius: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow the saturated red marker and carry its last position through misses."""
    count, height, width = _video_shape(video)
    center = np.asarray(xy, dtype=np.float32)
    if (
        center.shape != (2,)
        or not np.all(center >= 0)
        or not np.all(center <= [width - 1, height - 1])
        or not (0 < radius <= max_radius and expansion >= 1.0 and blob_radius > 0)
    ):
        raise ValueError("Invalid query coordinates or marker search parameters")
    tracks = np.empty((count, 2), dtype=np.float32)
    visible = np.zeros(count, dtype=bool)
    tracks[0], visible[0] = center, True
    reach = radius
    for i, frame in enumerate(video[1:], start=1):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hue = (hsv[..., 0].astype(np.int16) * 2 + 180) % 360 - 180
        points = np.argwhere(
            (hue >= -10) & (hue <= 5) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 150)
        )
        points = points[:, ::-1].astype(np.float32)
        local = points[((points - center) ** 2).sum(-1) <= reach**2]
        if len(local):
            nearest = local[((local - center) ** 2).sum(-1).argmin()]
            nearby = local[((local - nearest) ** 2).sum(-1) <= blob_radius**2]
            center = nearby.mean(0)
            reach = radius
            visible[i] = True
        else:
            reach = min(max_radius, reach * expansion)
        tracks[i] = center
    return tracks, visible


def counterfactual_denoise(
    clean: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    predict: Callable[[Tensor, Tensor, bool], Tensor],
    step: Callable[[Tensor, Tensor, int], Tensor],
    add_noise: Callable[[Tensor, Tensor, Tensor], Tensor],
    *,
    guidance: float = 8.0,
    mask: Tensor | None = None,
) -> Tensor:
    """Apply edited-image guidance and preserve known latents outside the mask."""
    if (
        not len(timesteps)
        or noise.shape != clean.shape
        or not (isfinite(guidance) and guidance >= 0.0)
    ):
        raise ValueError(
            "Expected a nonempty schedule, matching noise and valid guidance"
        )
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("Inpainting masks must be boolean")
    sample = add_noise(clean, noise, timesteps[0])
    for i, time in enumerate(timesteps):
        marked = predict(sample, time, True).float()
        original = predict(sample, time, False).float()
        sample = step(sample, marked + guidance * (marked - original), i)
        if mask is not None:
            known = (
                add_noise(clean, noise, timesteps[i + 1])
                if i + 1 < len(timesteps)
                else clean
            )
            sample = torch.where(mask, sample, known)
    return sample
