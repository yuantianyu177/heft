"""Forward/backward-consistency point tracking over extracted attention features."""

from __future__ import annotations

import os
from math import isfinite

import torch
from torch import Tensor

from heft.attn_hook import FeatureKind
from heft.calibration.temporal import retime_feature_volume

from .config import (
    FeatureSelection,
    TrackingConfig,
    TrackingResult,
)
from .features import FeatureVideo, FrameRange
from .ops import DescriptorState
from .pixel_ops import PixelTrackingOps as TrackingOps
from .query import _validate_queries
from .reader import FeatureReader


class FeatureTracker:
    """Run the original tracking algorithm on one explicitly selected device."""

    def __init__(
        self,
        *,
        config: TrackingConfig | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self.config = TrackingConfig() if config is None else config
        self.device = torch.device(device)

    @torch.inference_mode()
    def track(
        self,
        features: FeatureVideo | str | os.PathLike[str],
        query_points: Tensor,
        selection: FeatureSelection,
        *,
        video_id: str | None = None,
        time_advance_frames: float = 0.0,
    ) -> TrackingResult:
        """Track each query from its start frame through the end of the video."""

        if not isfinite(time_advance_frames):
            raise ValueError("time_advance_frames must be finite")
        video = (
            features
            if isinstance(features, FeatureVideo)
            else FeatureVideo.open(features)
        )
        points = query_points.to(device=self.device, dtype=torch.float32).contiguous()
        self._validate_inputs(video, points, selection)
        tracks = torch.zeros(
            (points.shape[0], video.num_frames, 2),
            device=self.device,
            dtype=torch.float32,
        )
        visibility = torch.zeros(
            (points.shape[0], video.num_frames),
            device=self.device,
            dtype=torch.bool,
        )
        if points.shape[0] == 0:
            return self._result(video, points, tracks, visibility, selection, video_id)

        query_frames, order = points[:, 2].long().sort(stable=True)
        point_indices = torch.arange(points.shape[0], device=self.device)
        tracks[point_indices, query_frames] = points[order, :2]
        visibility[point_indices, query_frames] = True

        frames, sizes = query_frames.unique_consecutive(return_counts=True)
        groups: list[tuple[int, slice]] = []
        group_start = 0
        for frame, size in zip(frames.tolist(), sizes.tolist(), strict=True):
            groups.append((frame, slice(group_start, group_start + size)))
            group_start += size

        reader = FeatureReader(video, device=self.device)
        ops = TrackingOps(
            frame_size=video.frame_size,
            config=self.config,
            device=self.device,
        )
        required_kinds = self._required_kinds()
        descriptors_by_frame: dict[int, DescriptorState] = {}
        first_query_frame = groups[0][0]

        for chunk, volumes in reader.prefetch_chunks(
            (chunk for chunk in video.chunks if chunk.stop > first_query_frame),
            selection=selection,
            kinds=required_kinds,
            frequency_range=self.config.frequency_range,
        ):
            local_start, local_frames = _local_window(chunk)
            volumes = {
                kind: ops.interpolate_time(
                    volume,
                    local_frames,
                    feature_frame_stride=chunk.feature_frame_stride,
                    first_feature_frame=chunk.first_feature_frame,
                    first_rgb_frame=local_start - chunk.start,
                )
                for kind, volume in volumes.items()
            }
            if time_advance_frames:
                prefix = chunk.start - local_start
                for kind, volume in volumes.items():
                    shifted = retime_feature_volume(
                        volume[prefix:], time_advance_frames
                    )
                    volumes[kind] = (
                        torch.cat((volume[:prefix], shifted)) if prefix else shifted
                    )
            for query_frame, rows in groups:
                if query_frame >= chunk.stop:
                    break
                # The first chunk for a group starts at its query. Later chunks
                # start at the existing handoff frame.
                start = max(local_start, query_frame)
                offset = start - local_start
                descriptors_by_frame[query_frame] = self._track_chunk(
                    {kind: volume[offset:] for kind, volume in volumes.items()},
                    tracks[rows, start : chunk.stop],
                    visibility[rows, start : chunk.stop],
                    descriptors_by_frame.get(query_frame),
                    ops=ops,
                    apply_search_mask=selection.head is not None,
                )

        input_order = order.argsort()
        return self._result(
            video,
            points,
            tracks[input_order],
            visibility[input_order],
            selection,
            video_id,
        )

    def _track_chunk(
        self,
        volumes: dict[FeatureKind, Tensor],
        tracks: Tensor,
        visibility: Tensor,
        descriptors: DescriptorState | None,
        *,
        ops: TrackingOps,
        apply_search_mask: bool,
    ) -> DescriptorState:
        if descriptors is None:
            query_frame = ops.frame(volumes[self.config.query_feature], 0)
            descriptors = ops.make_descriptors(query_frame, tracks[:, 0])

        update_kind = self._active_update_feature
        for frame_index in range(1, tracks.shape[1]):
            target_frame = ops.frame(volumes[self.config.target_feature], frame_index)
            predicted = ops.predict(
                descriptors,
                target_frame,
                previous_positions=tracks[:, frame_index - 1],
                previous_visibility=visibility[:, frame_index - 1],
                apply_search_mask=apply_search_mask,
                softmax_before_resize=True,
            )
            query_frame = (
                target_frame
                if self.config.query_feature is self.config.target_feature
                else ops.frame(volumes[self.config.query_feature], frame_index)
            )
            visibility[:, frame_index] = True
            current_visibility = self._visibility(
                predicted,
                frame_index=frame_index,
                query_frame=query_frame,
                volumes=volumes,
                tracks=tracks,
                visibility=visibility,
                ops=ops,
            )
            tracks[:, frame_index] = predicted
            visibility[:, frame_index] = current_visibility

            if update_kind is not None:
                if update_kind is self.config.target_feature:
                    update_frame = target_frame
                elif update_kind is self.config.query_feature:
                    update_frame = query_frame
                else:
                    update_frame = ops.frame(volumes[update_kind], frame_index)
                ops.update_descriptor(
                    descriptors, update_frame, predicted, current_visibility
                )
        return descriptors

    def _visibility(
        self,
        target_points: Tensor,
        *,
        frame_index: int,
        query_frame: Tensor,
        volumes: dict[FeatureKind, Tensor],
        tracks: Tensor,
        visibility: Tensor,
        ops: TrackingOps,
    ) -> Tensor:
        backward_descriptors = ops.make_descriptors(query_frame, target_points)
        active_indices = torch.arange(
            target_points.shape[0], device=self.device, dtype=torch.long
        )
        backward_positions = target_points.clone()
        squared_error = torch.zeros(
            target_points.shape[0], device=self.device, dtype=torch.float32
        )
        visible_count = (
            visibility[:, :frame_index].sum(dim=1, dtype=torch.float32).clamp_min_(1.0)
        )
        failure_limit = visible_count * self.config.visibility_threshold**2
        update_kind = self._active_update_feature

        for previous_index in range(frame_index - 1, -1, -1):
            if active_indices.numel() == 0:
                break
            target_frame = ops.frame(
                volumes[self.config.target_feature], previous_index
            )
            backward_positions = ops.predict(
                backward_descriptors,
                target_frame,
                previous_positions=backward_positions,
                previous_visibility=visibility[active_indices, previous_index + 1],
                apply_search_mask=True,
                softmax_before_resize=False,
            )
            valid = visibility[active_indices, previous_index]
            difference = backward_positions - tracks[active_indices, previous_index]
            squared_error.add_(difference.square().sum(dim=1) * valid)

            keep = squared_error <= failure_limit[active_indices]
            kept = keep.nonzero(as_tuple=False).flatten()
            active_indices = active_indices.index_select(0, kept)
            backward_positions = backward_positions.index_select(0, kept)
            backward_descriptors = backward_descriptors.index_select(kept)
            squared_error = squared_error.index_select(0, kept)
            valid = valid.index_select(0, kept)

            if update_kind is not None and active_indices.numel() > 0:
                update_frame = (
                    target_frame
                    if update_kind is self.config.target_feature
                    else ops.frame(volumes[update_kind], previous_index)
                )
                ops.update_descriptor(
                    backward_descriptors,
                    update_frame,
                    backward_positions,
                    valid,
                )

        result = torch.zeros(
            target_points.shape[0], device=self.device, dtype=torch.bool
        )
        result[active_indices] = True
        return result

    @property
    def _active_update_feature(self) -> FeatureKind | None:
        if self.config.feature_ema_alpha == 0.0:
            return None
        return self.config.update_feature

    def _required_kinds(self) -> tuple[FeatureKind, ...]:
        kinds = [self.config.query_feature, self.config.target_feature]
        update_kind = self._active_update_feature
        if update_kind is not None:
            kinds.append(update_kind)
        return tuple(dict.fromkeys(kinds))

    def _validate_inputs(
        self,
        video: FeatureVideo,
        points: Tensor,
        selection: FeatureSelection,
    ) -> None:
        _validate_queries(points, video)
        if selection.layer not in video.layers:
            raise ValueError(f"layer {selection.layer} was not captured")
        if selection.head is not None and (
            video.heads is not None and selection.head not in video.heads
        ):
            raise ValueError(f"head {selection.head} was not captured")

    @staticmethod
    def _result(
        video: FeatureVideo,
        points: Tensor,
        tracks: Tensor,
        visibility: Tensor,
        selection: FeatureSelection,
        video_id: str | None,
    ) -> TrackingResult:
        return TrackingResult(
            video_id=video.name if video_id is None else video_id,
            frame_size=video.frame_size,
            query_points=points.cpu(),
            tracks=tracks.cpu(),
            visibility=visibility.cpu(),
            selection=selection,
        )


def _local_window(chunk: FrameRange) -> tuple[int, int]:
    start = chunk.start if chunk.index == 0 else chunk.start - 1
    return start, chunk.stop - start
