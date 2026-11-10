"""Memory-efficient tensor operations used by the tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import TrackingConfig
from .kernels import search_region_correlation

_SEARCH_TILE_SIZE = 128


@dataclass(slots=True)
class DescriptorState:
    """Raw descriptors and their incrementally maintained normalization."""

    values: Tensor
    normalized: Tensor

    @classmethod
    def from_values(cls, values: Tensor) -> DescriptorState:
        return cls(values=values, normalized=F.normalize(values, dim=1))

    def index_select(self, indices: Tensor) -> DescriptorState:
        return DescriptorState(
            values=self.values.index_select(0, indices),
            normalized=self.normalized.index_select(0, indices),
        )

    def update(self, indices: Tensor, values: Tensor, alpha: float) -> None:
        current = self.values.index_select(0, indices)
        updated = torch.lerp(current, values, alpha)
        self.values.index_copy_(0, indices, updated)
        self.normalized.index_copy_(0, indices, F.normalize(updated, dim=1))


class TrackingOps:
    """Run spatially bounded cosine matching and descriptor updates."""

    def __init__(
        self,
        *,
        frame_size: tuple[int, int],
        config: TrackingConfig,
        device: torch.device,
    ) -> None:
        self.frame_h, self.frame_w = frame_size
        self.config = config
        self.device = device
        self.x_coordinates = torch.arange(
            self.frame_w, device=device, dtype=torch.float32
        )
        self.y_coordinates = torch.arange(
            self.frame_h, device=device, dtype=torch.float32
        )
        self.argmax_offsets = self._make_circle_offsets(config.argmax_radius)
        self.offsets, self.offset_weights = self._make_update_neighborhood()

    @staticmethod
    def interpolate_time(
        volume: Tensor,
        frames: int,
        *,
        feature_frame_stride: int | None = None,
        first_feature_frame: float = 0.0,
        first_rgb_frame: int = 0,
    ) -> Tensor:
        """Interpolate only the temporal axis of a ``[T,C,H,W]`` volume."""

        if feature_frame_stride is not None:
            times = torch.arange(frames, device=volume.device, dtype=torch.float32)
            indices = (
                times + first_rgb_frame - first_feature_frame
            ) / feature_frame_stride
            indices = indices.clamp(0, volume.shape[0] - 1)
            left = indices.floor().long()
            right = indices.ceil().long()
            weight = (indices - left).to(volume.dtype).view(-1, 1, 1, 1)
            return torch.lerp(volume[left], volume[right], weight)
        if volume.shape[0] == frames:
            return volume
        interpolated = F.interpolate(
            volume.permute(1, 0, 2, 3).unsqueeze(0),
            size=(frames, volume.shape[2], volume.shape[3]),
            mode="trilinear",
            align_corners=False,
        )
        return interpolated[0].permute(1, 0, 2, 3).contiguous()

    def frame(self, volume: Tensor, index: int) -> Tensor:
        """Materialize one spatially upsampled frame, or retain the feature grid."""

        frame = volume[index]
        if not self.config.upsample_features:
            return frame
        return F.interpolate(
            frame.unsqueeze(0),
            size=(self.frame_h, self.frame_w),
            mode="bilinear",
            align_corners=False,
        )[0]

    def sample(self, frame: Tensor, points: Tensor) -> Tensor:
        """Bilinearly sample ``[C,H,W]`` using full-resolution pixel coordinates."""

        normalized = points.to(dtype=torch.float32).clone()
        if self.frame_w > 1:
            normalized[:, 0].mul_(2.0 / (self.frame_w - 1)).sub_(1.0)
        if self.frame_h > 1:
            normalized[:, 1].mul_(2.0 / (self.frame_h - 1)).sub_(1.0)
        grid = normalized.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            frame.unsqueeze(0),
            grid,
            align_corners=True,
            padding_mode="border",
        )
        return sampled[0, :, :, 0].transpose(0, 1)

    def make_descriptors(self, frame: Tensor, points: Tensor) -> DescriptorState:
        return DescriptorState.from_values(self.sample(frame, points))

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
        """Correlate descriptors and compute radius-localized soft argmax positions."""

        normalized_target = F.normalize(target_frame.flatten(1), dim=0).reshape_as(
            target_frame
        )
        output = torch.empty(
            descriptors.values.shape[0], 2, device=self.device, dtype=torch.float32
        )
        batch_size = self.config.point_batch_size
        full_resolution = target_frame.shape[-2:] == (self.frame_h, self.frame_w)

        for start in range(0, descriptors.values.shape[0], batch_size):
            stop = min(start + batch_size, descriptors.values.shape[0])
            query = descriptors.normalized[start:stop]
            positions = previous_positions[start:stop]
            visibility = previous_visibility[start:stop]

            if not full_resolution:
                output[start:stop] = self._predict_resized(
                    query,
                    normalized_target,
                    positions,
                    visibility,
                    apply_search_mask=apply_search_mask,
                    softmax_before_resize=softmax_before_resize,
                )
                continue

            if not apply_search_mask:
                output[start:stop] = self._predict_global(query, normalized_target)
                continue

            visible = visibility.nonzero(as_tuple=False).flatten()
            invisible = (~visibility).nonzero(as_tuple=False).flatten()
            batch_output = output[start:stop]
            if visible.numel() > 0:
                batch_output[visible] = self._predict_search_regions(
                    query.index_select(0, visible),
                    normalized_target,
                    positions.index_select(0, visible),
                )
            if invisible.numel() > 0:
                batch_output[invisible] = self._predict_global(
                    query.index_select(0, invisible), normalized_target
                )

        return output

    def update_descriptor(
        self,
        descriptors: DescriptorState,
        update_frame: Tensor,
        positions: Tensor,
        visibility: Tensor,
    ) -> None:
        """Apply the original visibility-gated, neighborhood-weighted EMA update."""

        indices = visibility.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return
        centers = positions.index_select(0, indices)
        samples = centers[:, None, :] + self.offsets[None, :, :]
        samples[..., 0].clamp_(0, self.frame_w - 1)
        samples[..., 1].clamp_(0, self.frame_h - 1)
        sampled = self.sample(update_frame, samples.reshape(-1, 2)).reshape(
            indices.numel(), self.offsets.shape[0], -1
        )
        weighted = (sampled * self.offset_weights[None, :, None]).sum(dim=1)
        descriptors.update(indices, weighted, self.config.feature_ema_alpha)

    def _predict_search_regions(
        self,
        query: Tensor,
        normalized_target: Tensor,
        positions: Tensor,
    ) -> Tensor:
        if query.device.type == "cuda":
            return self._predict_search_regions_cuda(
                query, normalized_target, positions
            )
        return self._predict_search_regions_tiled(query, normalized_target, positions)

    def _predict_search_regions_cuda(
        self,
        query: Tensor,
        normalized_target: Tensor,
        positions: Tensor,
    ) -> Tensor:
        logits = search_region_correlation(
            query, normalized_target, positions, self.config.search_radius
        )
        return self._expect_search_regions(logits, positions)

    def _expect_search_regions(self, logits: Tensor, positions: Tensor) -> Tensor:
        extent = math.ceil(self.config.search_radius)
        side = extent * 2 + 1
        argmax = logits.argmax(dim=1)
        base = positions.floor().long()
        centers = base + torch.stack(
            (
                argmax.remainder(side) - extent,
                torch.div(argmax, side, rounding_mode="floor") - extent,
            ),
            dim=1,
        )
        output = torch.zeros(logits.shape[0], 2, device=self.device)
        matched = logits.isfinite().any(dim=1).nonzero(as_tuple=False).flatten()
        if matched.numel() > 0:
            output[matched] = self._search_region_expectation(
                logits.index_select(0, matched),
                centers.index_select(0, matched),
                positions.index_select(0, matched),
                base.index_select(0, matched),
                extent,
                side,
            )
        return output

    def _predict_search_regions_tiled(
        self,
        query: Tensor,
        normalized_target: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Match visible points only inside tiles carrying their search regions."""

        output = torch.empty(query.shape[0], 2, device=self.device)
        tile_columns = math.ceil(self.frame_w / _SEARCH_TILE_SIZE)
        pixel_x = positions[:, 0].floor().long().clamp_(0, self.frame_w - 1)
        pixel_y = positions[:, 1].floor().long().clamp_(0, self.frame_h - 1)
        tile_ids = torch.div(
            pixel_y, _SEARCH_TILE_SIZE, rounding_mode="floor"
        ) * tile_columns + torch.div(pixel_x, _SEARCH_TILE_SIZE, rounding_mode="floor")
        halo = math.ceil(self.config.search_radius) + 1

        for tile_id in tile_ids.unique().tolist():
            rows = (tile_ids == tile_id).nonzero(as_tuple=False).flatten()
            tile_y, tile_x = divmod(tile_id, tile_columns)
            core_x = tile_x * _SEARCH_TILE_SIZE
            core_y = tile_y * _SEARCH_TILE_SIZE
            x0 = max(0, core_x - halo)
            y0 = max(0, core_y - halo)
            x1 = min(self.frame_w, core_x + _SEARCH_TILE_SIZE + halo)
            y1 = min(self.frame_h, core_y + _SEARCH_TILE_SIZE + halo)
            crop_w = x1 - x0

            crop = normalized_target[:, y0:y1, x0:x1].reshape(
                normalized_target.shape[0], -1
            )
            logits = query.index_select(0, rows) @ crop
            row_positions = positions.index_select(0, rows)
            valid = self._crop_search_mask(row_positions, x0, x1, y0, y1)
            masked_logits = logits.masked_fill(~valid.flatten(1), -torch.inf)
            has_match = valid.flatten(1).any(dim=1)
            argmax = masked_logits.argmax(dim=1)
            centers = torch.stack(
                (
                    argmax.remainder(crop_w) + x0,
                    torch.div(argmax, crop_w, rounding_mode="floor") + y0,
                ),
                dim=1,
            )
            matched = has_match.nonzero(as_tuple=False).flatten()
            tile_output = torch.zeros(rows.shape[0], 2, device=self.device)
            if matched.numel() > 0:
                tile_output[matched] = self._local_expectation(
                    logits.index_select(0, matched),
                    centers.index_select(0, matched),
                    bounds=(x0, x1, y0, y1),
                    search_positions=row_positions.index_select(0, matched),
                )
            output[rows] = tile_output

        return output

    def _search_region_expectation(
        self,
        logits: Tensor,
        centers: Tensor,
        positions: Tensor,
        base: Tensor,
        extent: int,
        side: int,
    ) -> Tensor:
        coordinates = centers[:, None, :] + self.argmax_offsets[None, :, :]
        distance = coordinates.to(torch.float32) - positions[:, None, :]
        valid = (
            (coordinates[..., 0] >= 0)
            & (coordinates[..., 0] < self.frame_w)
            & (coordinates[..., 1] >= 0)
            & (coordinates[..., 1] < self.frame_h)
            & (distance.square().sum(dim=2) <= self.config.search_radius**2)
        )
        local = coordinates - base[:, None, :] + extent
        valid &= (
            (local[..., 0] >= 0)
            & (local[..., 0] < side)
            & (local[..., 1] >= 0)
            & (local[..., 1] < side)
        )
        indices = local[..., 1].clamp_(0, side - 1) * side + local[..., 0].clamp_(
            0, side - 1
        )
        local_logits = logits.gather(1, indices)
        weights = local_logits.masked_fill(~valid, -torch.inf).softmax(dim=1)
        coordinates = coordinates.to(torch.float32)
        x = (weights * coordinates[..., 0]).sum(dim=1)
        y = (weights * coordinates[..., 1]).sum(dim=1)
        return torch.stack((x, y), dim=1)

    def _predict_global(self, query: Tensor, normalized_target: Tensor) -> Tensor:
        return self._expect_global(query @ normalized_target.flatten(1))

    def _expect_global(self, logits: Tensor) -> Tensor:
        argmax = logits.argmax(dim=1)
        centers = torch.stack(
            (
                argmax.remainder(self.frame_w),
                torch.div(argmax, self.frame_w, rounding_mode="floor"),
            ),
            dim=1,
        )
        return self._local_expectation(
            logits,
            centers,
            bounds=(0, self.frame_w, 0, self.frame_h),
        )

    def _predict_resized(
        self,
        query: Tensor,
        normalized_target: Tensor,
        positions: Tensor,
        visibility: Tensor,
        *,
        apply_search_mask: bool,
        softmax_before_resize: bool,
    ) -> Tensor:
        """Preserve the original probability/feature-grid resize ordering."""

        source_h, source_w = normalized_target.shape[-2:]
        correlation = query @ normalized_target.flatten(1)
        if softmax_before_resize:
            scores = (correlation - correlation.amax(dim=1, keepdim=True)).exp()
            values = F.interpolate(
                scores.reshape(-1, 1, source_h, source_w),
                size=(self.frame_h, self.frame_w),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            return self._dense_expectation(
                values,
                positions,
                visibility,
                apply_search_mask=apply_search_mask,
                values_are_logits=False,
            )

        logits = F.interpolate(
            correlation.reshape(-1, 1, source_h, source_w),
            size=(self.frame_h, self.frame_w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return self._dense_expectation(
            logits,
            positions,
            visibility,
            apply_search_mask=apply_search_mask,
            values_are_logits=True,
        )

    def _dense_expectation(
        self,
        values: Tensor,
        positions: Tensor,
        visibility: Tensor,
        *,
        apply_search_mask: bool,
        values_are_logits: bool,
    ) -> Tensor:
        valid = torch.ones_like(values, dtype=torch.bool)
        if apply_search_mask:
            valid = self._radius_mask(positions, self.config.search_radius)
            valid |= ~visibility[:, None, None]
        fill_value = -torch.inf if values_are_logits else 0.0
        masked = values.masked_fill(~valid, fill_value)
        has_match = valid.flatten(1).any(dim=1)
        argmax = masked.flatten(1).argmax(dim=1)
        centers = torch.stack(
            (
                argmax.remainder(self.frame_w),
                torch.div(argmax, self.frame_w, rounding_mode="floor"),
            ),
            dim=1,
        )
        output = torch.zeros(values.shape[0], 2, device=self.device)
        matched = has_match.nonzero(as_tuple=False).flatten()
        if matched.numel() > 0:
            output[matched] = self._local_expectation(
                values.flatten(1).index_select(0, matched),
                centers.index_select(0, matched),
                bounds=(0, self.frame_w, 0, self.frame_h),
                search_positions=(
                    positions.index_select(0, matched) if apply_search_mask else None
                ),
                search_visibility=(
                    visibility.index_select(0, matched) if apply_search_mask else None
                ),
                values_are_logits=values_are_logits,
            )
        return output

    def _local_expectation(
        self,
        values: Tensor,
        centers: Tensor,
        *,
        bounds: tuple[int, int, int, int],
        search_positions: Tensor | None = None,
        search_visibility: Tensor | None = None,
        values_are_logits: bool = True,
    ) -> Tensor:
        x0, x1, y0, y1 = bounds
        width = x1 - x0
        coordinates = centers[:, None, :] + self.argmax_offsets[None, :, :]
        valid = (
            (coordinates[..., 0] >= x0)
            & (coordinates[..., 0] < x1)
            & (coordinates[..., 1] >= y0)
            & (coordinates[..., 1] < y1)
        )
        if search_positions is not None:
            distance = coordinates.to(torch.float32) - search_positions[:, None, :]
            in_search = distance.square().sum(dim=2) <= self.config.search_radius**2
            if search_visibility is not None:
                in_search |= ~search_visibility[:, None]
            valid &= in_search

        local_x = (coordinates[..., 0] - x0).clamp_(0, width - 1)
        local_y = (coordinates[..., 1] - y0).clamp_(0, y1 - y0 - 1)
        indices = local_y * width + local_x
        local_values = values.gather(1, indices)
        if values_are_logits:
            weights = local_values.masked_fill(~valid, -torch.inf).softmax(dim=1)
        else:
            weights = local_values.masked_fill(~valid, 0.0)
            weights /= weights.sum(dim=1, keepdim=True).clamp_min_(
                torch.finfo(weights.dtype).tiny
            )
        coordinates = coordinates.to(torch.float32)
        x = (weights * coordinates[..., 0]).sum(dim=1)
        y = (weights * coordinates[..., 1]).sum(dim=1)
        return torch.stack((x, y), dim=1)

    def _crop_search_mask(
        self,
        positions: Tensor,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> Tensor:
        x_distance = (
            self.x_coordinates[x0:x1][None, None, :] - positions[:, 0, None, None]
        )
        y_distance = (
            self.y_coordinates[y0:y1][None, :, None] - positions[:, 1, None, None]
        )
        return x_distance.square() + y_distance.square() <= self.config.search_radius**2

    def _radius_mask(self, centers: Tensor, radius: float) -> Tensor:
        x_distance = self.x_coordinates[None, None, :] - centers[:, 0, None, None]
        y_distance = self.y_coordinates[None, :, None] - centers[:, 1, None, None]
        return x_distance.square() + y_distance.square() <= radius**2

    def _make_circle_offsets(self, radius: float) -> Tensor:
        extent = math.ceil(radius)
        axis = torch.arange(-extent, extent + 1, device=self.device)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2)
        return offsets[offsets.square().sum(dim=1) <= radius**2]

    def _make_update_neighborhood(self) -> tuple[Tensor, Tensor]:
        radius = self.config.feature_update_sampling_radius
        axis = torch.arange(
            -radius, radius + 1, device=self.device, dtype=torch.float32
        )
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2)
        if radius == 0:
            weights = torch.ones(1, device=self.device)
        else:
            weights = torch.exp(-offsets.norm(dim=1) / (radius / 3.0))
            weights.div_(weights.sum())
        return offsets, weights
