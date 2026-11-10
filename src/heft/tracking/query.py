"""User-selectable query-point generation for tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from heft.data import TrackAnnotations

from .features import FeatureVideo

type AnnotationQueryPolicy = Literal["first_frame_visible", "first_visible"]


class QueryGenerator(Protocol):
    """Resolve canonical pixel-space ``(x, y, t)`` tracking queries."""

    def generate(self, video: FeatureVideo) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class ExplicitQueries:
    """Use points supplied directly by the caller."""

    points: Tensor

    def generate(self, video: FeatureVideo) -> Tensor:
        points = self.points.detach().to(device="cpu", dtype=torch.float32).contiguous()
        _validate_queries(points, video)
        return points


@dataclass(frozen=True, slots=True)
class GridQueries:
    """Place queries at regularly spaced patch centres."""

    stride: int
    frame: int = 0

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.frame < 0:
            raise ValueError("frame must be non-negative")

    def generate(self, video: FeatureVideo) -> Tensor:
        if self.frame >= video.num_frames:
            raise ValueError("query frame is outside the video")
        height, width = video.frame_size
        offset = self.stride // 2
        x = torch.arange(offset, width, self.stride, dtype=torch.float32)
        y = torch.arange(offset, height, self.stride, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        frames = torch.full((xy.shape[0], 1), float(self.frame))
        return torch.cat((xy, frames), dim=-1).contiguous()


@dataclass(frozen=True, slots=True)
class MaskGridQueries:
    """Filter a regular grid using a binary mask."""

    mask: Tensor
    stride: int
    frame: int = 0

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("mask must have shape [height, width]")
        GridQueries(stride=self.stride, frame=self.frame)

    def generate(self, video: FeatureVideo) -> Tensor:
        queries = GridQueries(stride=self.stride, frame=self.frame).generate(video)
        mask = self.mask.detach().to(device="cpu", dtype=torch.float32)[None, None]
        if tuple(mask.shape[-2:]) != video.frame_size:
            mask = F.interpolate(mask, size=video.frame_size, mode="nearest")
        mask = mask[0, 0].bool()
        x = queries[:, 0].long()
        y = queries[:, 1].long()
        return queries[mask[y, x]].contiguous()


@dataclass(frozen=True, slots=True)
class AnnotationQueries:
    """Generate queries from resolution-independent dataset annotations."""

    annotations: TrackAnnotations
    policy: AnnotationQueryPolicy = "first_visible"

    def indices(self, num_frames: int | None = None) -> tuple[Tensor, Tensor]:
        """Return selected annotation rows and their query frames."""

        visibility = self.annotations.visibility
        if num_frames is not None:
            if not 0 < num_frames <= visibility.shape[1]:
                raise ValueError("num_frames is outside the annotations")
            visibility = visibility[:, :num_frames]
        if self.policy == "first_frame_visible":
            selected = visibility[:, 0].nonzero(as_tuple=False).flatten()
            frames = torch.zeros(selected.shape[0], dtype=torch.long)
        elif self.policy == "first_visible":
            selected = visibility.any(dim=1).nonzero(as_tuple=False).flatten()
            frames = visibility[selected].to(torch.uint8).argmax(dim=1)
        else:
            raise ValueError(f"unsupported annotation query policy: {self.policy}")
        return selected, frames

    def generate(self, video: FeatureVideo) -> Tensor:
        tracks = self.annotations.tracks
        if tracks.shape[1] < video.num_frames:
            raise ValueError("annotations are shorter than the feature video")
        selected, frames = self.indices(video.num_frames)

        xy = tracks[selected, frames].float().clone()
        height, width = video.frame_size
        xy[:, 0].mul_(width - 1)
        xy[:, 1].mul_(height - 1)
        return torch.cat((xy, frames[:, None].float()), dim=1).contiguous()


def _validate_queries(points: Tensor, video: FeatureVideo) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("query points must have shape [point, 3]")
    if not torch.isfinite(points).all():
        raise ValueError("query points must be finite")
    if not torch.equal(points[:, 2], points[:, 2].round()):
        raise ValueError("query frame indices must be integers")
    height, width = video.frame_size
    if points.numel() and (
        (points[:, 0] < 0).any()
        or (points[:, 0] > width - 1).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] > height - 1).any()
        or (points[:, 2] < 0).any()
        or (points[:, 2] >= video.num_frames).any()
    ):
        raise ValueError("query points are outside the feature video")
