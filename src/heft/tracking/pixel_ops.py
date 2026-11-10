"""Pixel cosine matching from low-resolution, unnormalized feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import TrackingConfig
from .kernels import projected_search_logits
from .ops import DescriptorState, TrackingOps

_CANCELLATION_ENERGY_RATIO = 1e-6
_MAX_CORRECTION_ELEMENTS = 1 << 20


def _bilinear_weights(fraction: Tensor) -> Tensor:
    x, y = fraction.unbind(dim=-1)
    return torch.stack(((1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y), dim=-1)


@dataclass(slots=True)
class _PixelGrid:
    feature_size: tuple[int, int]
    cell_indices: Tensor
    gram_weights: Tensor


@dataclass(slots=True)
class _PixelNorm:
    lengths: Tensor
    cancellation_pixels: Tensor


class PixelTrackingOps(TrackingOps):
    """Project on the feature grid and evaluate normalized logits at video pixels."""

    def __init__(
        self,
        *,
        frame_size: tuple[int, int],
        config: TrackingConfig,
        device: torch.device,
    ) -> None:
        super().__init__(frame_size=frame_size, config=config, device=device)
        self._grid: _PixelGrid | None = None
        self._normalizers: WeakKeyDictionary[
            torch.UntypedStorage, dict[int, _PixelNorm]
        ] = WeakKeyDictionary()
        self._pixel_size = torch.tensor(
            [self.frame_w, self.frame_h], dtype=torch.float32, device=device
        )
        self._corner_offsets = self._pixel_size.new_tensor(
            [[0, 0], [1, 0], [0, 1], [1, 1]]
        )

    def frame(self, volume: Tensor, index: int) -> Tensor:
        return volume[index]

    def sample(self, frame: Tensor, points: Tensor) -> Tensor:
        if not self._projects_pixels(frame):
            return super().sample(frame, points)

        # First interpolate at the four integer video pixels surrounding each
        # query. Then preserve the original fractional sampling on that raster.
        bounds = self._pixel_size - 1
        normalized = points.to(torch.float32) * (2.0 / bounds.clamp_min(1)) - 1
        positions = ((normalized + 1) * 0.5 * bounds).clamp_min_(0).minimum(bounds)
        base = positions.floor()
        pixel_corners = (base[:, None] + self._corner_offsets).minimum(bounds)
        features = self._sample_pixels(frame, pixel_corners).permute(1, 2, 0)
        weights = _bilinear_weights(positions - base)
        return (features * weights[..., None]).sum(dim=1)

    def _sample_pixels(self, frame: Tensor, pixels: Tensor) -> Tensor:
        feature_grid = (pixels + 0.5) * (2.0 / self._pixel_size) - 1
        return F.grid_sample(
            frame[None], feature_grid[None], align_corners=False, padding_mode="border"
        )[0]

    def predict(
        self,
        descriptors: DescriptorState,
        target_frame: Tensor,
        *,
        previous_positions: Tensor,
        previous_visibility: Tensor,
        apply_search_mask: bool,
        softmax_before_resize: bool,
    ) -> Tensor:
        if not self._projects_pixels(target_frame):
            return super().predict(
                descriptors,
                target_frame,
                previous_positions=previous_positions,
                previous_visibility=previous_visibility,
                apply_search_mask=apply_search_mask,
                softmax_before_resize=softmax_before_resize,
            )

        normalizer = self._pixel_normalizer(target_frame)
        output = descriptors.values.new_empty(
            (len(descriptors.values), 2), dtype=torch.float32
        )
        batch_size = self.config.point_batch_size
        for start in range(0, len(output), batch_size):
            stop = min(start + batch_size, len(output))
            query = descriptors.normalized[start:stop]
            projection = (query @ target_frame.flatten(1)).reshape(
                len(query), *target_frame.shape[-2:]
            )
            positions = previous_positions[start:stop]
            visibility = previous_visibility[start:stop]
            if (
                not apply_search_mask
                or projection.device.type != "cuda"
                or normalizer.cancellation_pixels.numel()
            ):
                logits = self._pixel_logits(projection, normalizer.lengths)
                self._repair_cancellation(
                    logits, query, target_frame, normalizer.cancellation_pixels
                )
                if apply_search_mask:
                    output[start:stop] = self._dense_expectation(
                        logits,
                        positions,
                        visibility,
                        apply_search_mask=True,
                        values_are_logits=True,
                    )
                else:
                    output[start:stop] = self._expect_global(logits.flatten(1))
                continue

            visible = visibility.nonzero(as_tuple=False).flatten()
            invisible = (~visibility).nonzero(as_tuple=False).flatten()
            batch = output[start:stop]
            if visible.numel():
                visible_positions = positions.index_select(0, visible)
                logits = projected_search_logits(
                    projection.index_select(0, visible),
                    normalizer.lengths,
                    visible_positions,
                    self.config.search_radius,
                )
                batch[visible] = self._expect_search_regions(logits, visible_positions)
            if invisible.numel():
                logits = self._pixel_logits(
                    projection.index_select(0, invisible), normalizer.lengths
                )
                batch[invisible] = self._expect_global(logits.flatten(1))
        return output

    def _projects_pixels(self, frame: Tensor) -> bool:
        return self.config.upsample_features and frame.shape[-2:] != (
            self.frame_h,
            self.frame_w,
        )

    def _pixel_logits(self, projection: Tensor, normalizer: Tensor) -> Tensor:
        # Inner products are linear, so resizing scalar projections gives the
        # numerator of the pixel-space cosine. Normalize only after resizing.
        resized = F.interpolate(
            projection[:, None],
            size=(self.frame_h, self.frame_w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return resized.div_(normalizer)

    def _repair_cancellation(
        self, logits: Tensor, query: Tensor, frame: Tensor, indices: Tensor
    ) -> None:
        if not indices.numel():
            return
        # Nearly cancelling corners can lose most of the significant digits in
        # the scalar projection. Resample just those pixels in bounded batches.
        batch_size = max(1, _MAX_CORRECTION_ELEMENTS // frame.shape[0])
        for batch in indices.split(batch_size):
            pixels = torch.stack(
                (batch.remainder(self.frame_w), batch // self.frame_w), dim=-1
            )
            features = self._sample_pixels(frame, pixels[:, None])[:, :, 0]
            logits.flatten(1).index_copy_(
                1, batch, query @ F.normalize(features, dim=0)
            )

    def _pixel_normalizer(self, frame: Tensor) -> _PixelNorm:
        feature_size = (frame.shape[-2], frame.shape[-1])
        grid = self._grid
        if grid is None or grid.feature_size != feature_size:
            grid = self._build_grid(feature_size)
            self._grid = grid
            self._normalizers.clear()

        # Backward consistency revisits read-only targets. Weak storage keys
        # retain the scalar norms while each source volume is alive.
        storage = frame.untyped_storage()
        frames = self._normalizers.setdefault(storage, {})
        offset = int(frame.storage_offset())
        normalizer = frames.get(offset)
        if normalizer is not None:
            return normalizer

        padded = F.pad(frame[None], (0, 1, 0, 1), mode="replicate")[0].permute(1, 2, 0)
        corners = torch.stack(
            (padded[:-1, :-1], padded[:-1, 1:], padded[1:, :-1], padded[1:, 1:]),
            dim=2,
        ).reshape(-1, 4, frame.shape[0])
        # Squared norms of interpolated features are w^T G w, where G contains
        # the inner products of a cell's four corners. Compute the Gram and
        # weighted sum in double precision to reduce cancellation error.
        corners = corners.to(torch.float64)
        gram = torch.bmm(corners, corners.transpose(1, 2))
        terms = gram.index_select(0, grid.cell_indices).mul_(grid.gram_weights)
        squared_norm = terms.sum(dim=(1, 2))
        corner_energy = terms.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cancellation = (corner_energy > 0) & (
            squared_norm <= _CANCELLATION_ENERGY_RATIO * corner_energy
        )
        lengths = squared_norm.clamp_min_(0).sqrt_().clamp_min_(1e-12)
        normalizer = _PixelNorm(
            lengths.to(frame.dtype).reshape(self.frame_h, self.frame_w),
            cancellation.nonzero(as_tuple=False).flatten(),
        )
        frames[offset] = normalizer
        return normalizer

    def _build_grid(self, feature_size: tuple[int, int]) -> _PixelGrid:
        grid_h, grid_w = feature_size
        y, x = torch.meshgrid(self.y_coordinates, self.x_coordinates, indexing="ij")
        source_x = ((x + 0.5) * (grid_w / self.frame_w) - 0.5).clamp_min_(0)
        source_y = ((y + 0.5) * (grid_h / self.frame_h) - 0.5).clamp_min_(0)
        left, top = source_x.floor(), source_y.floor()
        cells = (top.long() * grid_w + left.long()).flatten()
        weights = (
            _bilinear_weights(torch.stack((source_x - left, source_y - top), dim=-1))
            .reshape(-1, 4)
            .to(torch.float64)
        )
        return _PixelGrid(
            feature_size, cells, weights[:, :, None] * weights[:, None, :]
        )
