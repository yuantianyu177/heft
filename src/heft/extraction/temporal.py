"""Map RGB clips to DiT inputs and interleave independently captured phases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import TemporalMode


@dataclass(frozen=True, slots=True)
class Phase:
    prefix_frames: int
    latent_times: tuple[int, ...]


def four_phases(frames: int, stride: int) -> tuple[Phase, ...]:
    """Share frame zero and take the four offset native sampling grids."""

    if stride != 4:
        raise ValueError("four_phase requires a native temporal stride of four")
    return tuple(
        Phase((-offset) % stride, (0, *range(offset or stride, frames, stride)))
        for offset in range(stride)
    )


def encode_video_latents(
    vae: Any,
    pixels: Tensor,
    *,
    mode: TemporalMode,
    native_stride: int,
    patch: int,
    phase: Phase | None = None,
) -> Tensor:
    """Encode normalized [B,C,T,H,W] pixels and complete the DiT time patches."""

    frames = pixels.shape[2]
    if mode is TemporalMode.FRAMEWISE:
        clean = torch.cat(
            [_encode(vae, frame, 1) for frame in pixels.split(1, dim=2)], dim=2
        )
    elif phase is not None:
        if phase.prefix_frames:
            prefix = pixels[:, :, :1].expand(-1, -1, phase.prefix_frames, -1, -1)
            pixels = torch.cat((prefix, pixels), dim=2)
        latent_frames = len(phase.latent_times)
        rgb_frames = 1 + (latent_frames - 1) * native_stride
        clean = _encode(vae, pixels[:, :, :rgb_frames].contiguous(), latent_frames)
    else:
        latent_frames = (frames - 1 + native_stride - 1) // native_stride + 1
        rgb_frames = 1 + (latent_frames - 1) * native_stride
        pixels = _repeat_tail(pixels, rgb_frames)
        if mode is TemporalMode.STRIDE1:
            clean = _encode(vae, pixels, rgb_frames)[:, :, :frames]
        else:
            clean = _encode(vae, pixels, latent_frames)
            if mode is TemporalMode.PRE_DIT:
                clean = _sample_time(
                    clean, range(0, rgb_frames, native_stride), range(frames), dim=2
                )

    padded_frames = (clean.shape[2] + patch - 1) // patch * patch
    return _repeat_tail(clean, padded_frames).contiguous()


def stitch_phase_features(
    features: Sequence[Tensor],
    phases: Sequence[Phase],
    *,
    frames: int,
    spatial_tokens: int,
    patch: int,
) -> Tensor:
    """Interleave phase descriptors at their RGB times, averaging the shared anchor."""

    first = features[0]
    batch, _, channels = first.shape
    dense = first.new_empty(batch, frames, spatial_tokens, channels)
    anchors = []
    for tensor, phase in zip(features, phases, strict=True):
        times = phase.latent_times
        # A partial DiT patch repeats its last latent. Its center uses the nominal
        # grid, including the padded position, just as in a single-phase readout.
        padding = (-len(times)) % patch
        padded = times + tuple(times[-1] + 4 * i for i in range(1, padding + 1))
        centers = tuple(
            sum(padded[i : i + patch]) / patch for i in range(0, len(padded), patch)
        )
        retained = tuple(time for time in times if time < frames)
        volume = tensor.reshape(batch, len(centers), spatial_tokens, channels)
        aligned = _sample_time(volume, centers, retained, dim=1)
        anchors.append(aligned[:, 0].float())
        dense[:, list(retained[1:])] = aligned[:, 1:]
    dense[:, 0] = torch.stack(anchors).mean(dim=0).to(dense.dtype)
    return dense.flatten(1, 2)


def _encode(vae: Any, pixels: Tensor, expected_frames: int) -> Tensor:
    clean = vae.encode(pixels).latent_dist.mode()
    if clean.shape[2] != expected_frames:
        raise ValueError(
            f"VAE produced {clean.shape[2]} latent frames, expected {expected_frames}"
        )
    return clean


def _repeat_tail(volume: Tensor, frames: int) -> Tensor:
    padding = frames - volume.shape[2]
    if padding == 0:
        return volume
    tail = volume[:, :, -1:].expand(-1, -1, padding, -1, -1)
    return torch.cat((volume, tail), dim=2)


def _sample_time(
    tensor: Tensor,
    source: Sequence[float],
    target: Sequence[float],
    *,
    dim: int,
) -> Tensor:
    if tuple(source) == tuple(target):
        return tensor
    grid = torch.tensor(source, dtype=torch.float32, device=tensor.device)
    query = torch.tensor(target, dtype=torch.float32, device=tensor.device)
    query = query.clamp(grid[0], grid[-1])
    right = torch.searchsorted(grid, query)
    left = (right - 1).clamp_min(0)
    distance = grid[right] - grid[left]
    weight = (query - grid[left]) / distance.clamp_min(1e-6)
    shape = [1] * tensor.ndim
    shape[dim] = len(target)
    return torch.lerp(
        tensor.index_select(dim, left),
        tensor.index_select(dim, right),
        weight.to(tensor.dtype).reshape(shape),
    )
