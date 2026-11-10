"""Fused CUDA kernels for spatially bounded feature matching."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_CHANNELS = 32
_BLOCK_PIXELS = 64


@triton.jit
def _search_correlation_kernel(
    query,
    target,
    positions,
    output,
    channels,
    height: tl.constexpr,
    width: tl.constexpr,
    extent: tl.constexpr,
    side: tl.constexpr,
    pixels: tl.constexpr,
    radius_squared: tl.constexpr,
    block_channels: tl.constexpr,
    block_pixels: tl.constexpr,
):
    point = tl.program_id(0)
    offsets = tl.program_id(1) * block_pixels + tl.arange(0, block_pixels)
    offset_x = offsets % side - extent
    offset_y = offsets // side - extent
    position_x = tl.load(positions + point * 2)
    position_y = tl.load(positions + point * 2 + 1)
    pixel_x = tl.floor(position_x).to(tl.int32) + offset_x
    pixel_y = tl.floor(position_y).to(tl.int32) + offset_y
    distance_x = pixel_x - position_x
    distance_y = pixel_y - position_y
    distance_squared = distance_x * distance_x + distance_y * distance_y
    valid = (
        (offsets < pixels)
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
        & (distance_squared <= radius_squared)
    )
    accumulator = tl.zeros((block_pixels,), dtype=tl.float32)
    spatial_offset = pixel_y * width + pixel_x

    for channel_start in tl.range(0, channels, block_channels):
        channel = channel_start + tl.arange(0, block_channels)
        channel_valid = channel < channels
        query_value = tl.load(
            query + point * channels + channel,
            mask=channel_valid,
            other=0.0,
        )
        target_value = tl.load(
            target + channel[None, :] * height * width + spatial_offset[:, None],
            mask=valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        accumulator += tl.sum(target_value * query_value[None, :], axis=1)

    result = tl.where(valid, accumulator, -float("inf"))
    tl.store(output + point * pixels + offsets, result, mask=offsets < pixels)


def search_region_correlation(
    query: Tensor,
    target: Tensor,
    positions: Tensor,
    radius: float,
) -> Tensor:
    """Return cosine logits only for each point's square search neighborhood."""

    points, channels = query.shape
    height, width = target.shape[-2:]
    extent = math.ceil(radius)
    side = extent * 2 + 1
    pixels = side * side
    output = torch.empty((points, pixels), device=query.device, dtype=torch.float32)
    grid = (points, triton.cdiv(pixels, _BLOCK_PIXELS))
    _search_correlation_kernel[grid](
        query,
        target,
        positions,
        output,
        channels,
        height,
        width,
        extent,
        side,
        pixels,
        radius * radius,
        _BLOCK_CHANNELS,
        _BLOCK_PIXELS,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return output


@triton.jit
def _projected_search_kernel(
    projection,
    normalizer,
    positions,
    output,
    grid_height: tl.constexpr,
    grid_width: tl.constexpr,
    frame_height: tl.constexpr,
    frame_width: tl.constexpr,
    extent: tl.constexpr,
    side: tl.constexpr,
    radius_squared: tl.constexpr,
    block_pixels: tl.constexpr,
):
    point = tl.program_id(0)
    offsets = tl.program_id(1) * block_pixels + tl.arange(0, block_pixels)
    position_x = tl.load(positions + point * 2)
    position_y = tl.load(positions + point * 2 + 1)
    x = tl.floor(position_x).to(tl.int32) + offsets % side - extent
    y = tl.floor(position_y).to(tl.int32) + offsets // side - extent
    dx, dy = x - position_x, y - position_y
    valid = (
        (offsets < side * side)
        & (x >= 0)
        & (x < frame_width)
        & (y >= 0)
        & (y < frame_height)
        & (dx * dx + dy * dy <= radius_squared)
    )

    source_x = tl.maximum((x + 0.5) * (grid_width / frame_width) - 0.5, 0.0)
    source_y = tl.maximum((y + 0.5) * (grid_height / frame_height) - 0.5, 0.0)
    left, top = source_x.to(tl.int32), source_y.to(tl.int32)
    right = tl.minimum(left + 1, grid_width - 1)
    bottom = tl.minimum(top + 1, grid_height - 1)
    wx, wy = source_x - left, source_y - top
    base = projection + point * grid_height * grid_width
    top_left = tl.load(base + top * grid_width + left, mask=valid, other=0.0)
    top_right = tl.load(base + top * grid_width + right, mask=valid, other=0.0)
    bottom_left = tl.load(base + bottom * grid_width + left, mask=valid, other=0.0)
    bottom_right = tl.load(base + bottom * grid_width + right, mask=valid, other=0.0)
    upper = (1.0 - wx) * top_left + wx * top_right
    lower = (1.0 - wx) * bottom_left + wx * bottom_right
    numerator = (1.0 - wy) * upper + wy * lower
    denominator = tl.load(normalizer + y * frame_width + x, mask=valid, other=1.0)
    logits = tl.where(valid, tl.div_rn(numerator, denominator), -float("inf"))
    tl.store(output + point * side * side + offsets, logits, mask=offsets < side * side)


def projected_search_logits(
    projection: Tensor, normalizer: Tensor, positions: Tensor, radius: float
) -> Tensor:
    """Interpolate scalar projections only at pixels within the search regions."""

    points, grid_h, grid_w = projection.shape
    frame_h, frame_w = normalizer.shape
    extent = math.ceil(radius)
    side = extent * 2 + 1
    logits = projection.new_empty((points, side * side))
    _projected_search_kernel[(points, triton.cdiv(side * side, 128))](
        projection.contiguous(),
        normalizer.contiguous(),
        positions.contiguous(),
        logits,
        grid_h,
        grid_w,
        frame_h,
        frame_w,
        extent,
        side,
        radius * radius,
        128,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return logits
