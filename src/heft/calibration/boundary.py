"""Feature-time resampling at the boundary between four-phase windows."""

from collections.abc import Sequence
from math import ceil, floor

import torch
from torch import Tensor

from .temporal import _partition, retime_feature_volume


def retime_with_anchors(
    volume: Tensor,
    chunks: Sequence[tuple[int, int]],
    advance_frames: float,
) -> Tensor:
    """Interpolate to the adjacent chunk's exact anchor when delay exhausts a clip."""
    ranges = _partition(len(volume), chunks)
    if advance_frames == 0.0:
        return volume
    pieces = []
    for start, stop in ranges:
        original = volume[start:stop]
        aligned = retime_feature_volume(original, advance_frames)
        if advance_frames > 0.0 and stop < len(volume) and stop - start > 1:
            time_last = stop - 1 - advance_frames
            first = max(start + 1, floor(time_last) + 1)
            targets = torch.arange(
                first, stop, device=volume.device, dtype=torch.float32
            )
            weight = (targets - time_last) / (stop - time_last)
            shape = (-1,) + (1,) * (volume.ndim - 1)
            aligned[first - start :] = torch.lerp(
                original[-1], volume[stop], weight.to(volume.dtype).reshape(shape)
            )
        if advance_frames < 0.0 and stop - start > 1:
            time_next = 1.0 - advance_frames
            last = min(stop - start, ceil(time_next))
            targets = torch.arange(1, last, device=volume.device, dtype=torch.float32)
            shape = (-1,) + (1,) * (volume.ndim - 1)
            weight = (targets / time_next).to(volume.dtype).reshape(shape)
            aligned[1:last] = torch.lerp(original[0], original[1], weight)
        pieces.append(aligned)
    return torch.cat(pieces)
