"""Interleaved anchored query-key tracking following DiffTrack, Appendix C."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor

ChunkReadout = Callable[[tuple[int, ...]], tuple[Tensor, Tensor]]


def interleaved_windows(
    frames: int, anchor: int, chunk_size: int = 13
) -> tuple[tuple[int, ...], ...]:
    """Assign every later frame to a uniformly spaced window with the same anchor."""
    if frames < 1 or not 0 <= anchor < frames or chunk_size < 2:
        raise ValueError("Invalid frame count, anchor, or chunk size")
    if anchor == frames - 1:
        return ()
    targets_per_chunk = chunk_size - 1
    stride = max(1, (frames - anchor) // targets_per_chunk)
    # Slide by one frame far enough to cover a remainder after the initial stride.
    last = max(anchor + stride, frames - 1 - (targets_per_chunk - 1) * stride)
    return tuple(
        (anchor, *range(first, min(frames, first + targets_per_chunk * stride), stride))
        for first in range(anchor + 1, last + 1)
    )


def bidirectional_cost(
    anchor_q: Tensor,
    anchor_k: Tensor,
    target_q: Tensor,
    target_k: Tensor,
    indices: Tensor,
    weights: Tensor,
) -> Tensor:
    """Read rows of the averaged forward and transposed reverse attention maps."""
    scale = anchor_q.shape[-1] ** -.5
    forward = ((anchor_q[indices] @ target_k.T) * scale).softmax(dim=-1)
    # The reverse normalization uses every anchor key, even though we only
    # retain the four probabilities surrounding each RGB query.
    reverse = ((target_q @ anchor_k.T) * scale).softmax(dim=-1)
    reverse = reverse.T[indices]
    return ((forward + reverse) * (.5 * weights[..., None])).sum(dim=1)


def _anchor_sampling(xy: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
    height, width = size
    x = (xy[:, 0] * width - .5).clamp(0., width - 1)
    y = (xy[:, 1] * height - .5).clamp(0., height - 1)
    left, top = x.floor().long(), y.floor().long()
    right, bottom = (left + 1).clamp_max(width - 1), (top + 1).clamp_max(height - 1)
    dx, dy = x - left, y - top
    indices = torch.stack((top * width + left, top * width + right,
                           bottom * width + left, bottom * width + right), dim=1)
    weights = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy),
                           (1 - dx) * dy, dx * dy), dim=1)
    return indices, weights


@torch.inference_mode()
def track_difftrack(
    frames: int,
    queries: Tensor,
    feature_size: tuple[int, int],
    readout: ChunkReadout,
    *,
    chunk_size: int = 13,
) -> Tensor:
    """Track normalized [x, y, t] queries using framewise-VAE DiT Q and K.

    A readout returns two [window time, spatial token, channel] tensors. The
    result is [point, video time, xy] in normalized RGB coordinates. Entries
    before a point's query time are zero.
    """
    if (queries.ndim != 2 or queries.shape[1] != 3 or frames < 1
        or min(feature_size) < 1 or chunk_size < 2):
        raise ValueError("Expected valid [point, 3] queries and a nonempty feature grid")
    if not queries.is_floating_point() or not torch.isfinite(queries).all():
        raise ValueError("Queries must contain finite floating point coordinates")
    xy, times = queries[:, :2].float(), queries[:, 2]
    if ((xy < 0) | (xy > 1)).any() or ((times < 0) | (times >= frames) | (times != times.floor())).any():
        raise ValueError("Expected normalized points and integer query times within the video")

    height, width = feature_size
    tracks = xy.new_zeros((len(xy), frames, 2))
    for anchor in map(int, times.unique(sorted=True).tolist()):
        points = torch.where(times == anchor)[0]
        starting = xy.index_select(0, points)
        tracks[points, anchor] = starting
        if anchor == frames - 1:
            continue
        indices, weights = _anchor_sampling(starting, feature_size)
        sums = xy.new_zeros((frames - anchor - 1, len(points), height * width))
        counts = xy.new_zeros((len(sums),))
        for window in interleaved_windows(frames, anchor, chunk_size):
            query, key = readout(window)
            if (query.ndim != 3 or query.shape != key.shape
                or query.shape[:2] != (len(window), height * width)):
                raise ValueError("Readout has incompatible Q/K shapes")
            query, key = query.float(), key.float()
            for i, target in enumerate(window[1:], 1):
                offset = target - anchor - 1
                sums[offset].add_(bidirectional_cost(query[0], key[0], query[i], key[i], indices, weights))
                counts[offset] += 1
        costs = sums / counts[:, None, None]
        best = costs.argmax(dim=-1)
        prediction = torch.stack(((best % width + .5) / width, (best // width + .5) / height), dim=-1)
        tracks[points, anchor + 1:] = prediction.transpose(0, 1)
    return tracks


def ddim_inverse_step(
    sample: Tensor, prediction: Tensor, alpha: float, next_alpha: float, prediction_type: str
) -> Tensor:
    """Transport a deterministic DDIM sample from alpha to a noisier alpha."""
    if not 0 < next_alpha <= alpha <= 1:
        raise ValueError("DDIM inversion requires 0 < next_alpha <= alpha <= 1")
    signal, noise = math.sqrt(alpha), math.sqrt(1 - alpha)
    if prediction_type == "epsilon":
        clean = (sample - noise * prediction) / signal
        epsilon = prediction
    elif prediction_type == "v_prediction":
        clean = signal * sample - noise * prediction
        epsilon = signal * prediction + noise * sample
    else:
        raise ValueError(f"Unsupported DDIM prediction type {prediction_type!r}")
    return math.sqrt(next_alpha) * clean + math.sqrt(1 - next_alpha) * epsilon
