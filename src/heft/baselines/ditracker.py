"""DiTracker's multi-scale local 4D costs and iterative training objective."""

from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def local_qk_cost(query: Tensor, key: Tensor) -> Tensor:
    """Compute row-softmax patch correspondences and average the head costs."""
    if (
        query.ndim < 3
        or query.shape[-3] != key.shape[-3]
        or query.shape[-1] != key.shape[-1]
    ):
        raise ValueError(
            "Expected matching [heads, patch_positions, channels] dimensions"
        )
    logits = (query.float() @ key.float().transpose(-1, -2)) / sqrt(query.shape[-1])
    return logits.softmax(dim=-1).mean(dim=-3)


def _resize(maps: Tensor, size: tuple[int, int]) -> Tensor:
    if maps.shape[-2:] == size:
        return maps
    batch, time, heads, channels, _, _ = maps.shape
    flat = maps.reshape(batch * time, heads * channels, *maps.shape[-2:])
    return F.interpolate(flat, size=size, mode="bilinear", align_corners=False).reshape(
        batch, time, heads, channels, *size
    )


def _xy_grid(centers: Tensor, offsets: Tensor, shape: tuple[int, int]) -> Tensor:
    scale = centers.new_tensor((shape[1], shape[0]))
    return (centers[..., None, :] + offsets / scale) * 2.0 - 1.0


def _anchor_patches(maps: Tensor, xy: Tensor, times: Tensor, offsets: Tensor) -> Tensor:
    batch, frames, heads, channels, h, w = maps.shape
    grid = _xy_grid(xy, offsets, (h, w))
    z = (times.to(grid.dtype) + 0.5) * (2.0 / frames) - 1.0
    z = z[:, :, None, None].expand(*grid.shape[:-1], 1)
    coordinates = torch.cat((grid, z), dim=-1).unsqueeze(-2)
    volumes = maps.flatten(2, 3).transpose(1, 2)
    values = F.grid_sample(
        volumes.float(), coordinates, align_corners=False, padding_mode="border"
    )
    return values.reshape(batch, heads, channels, xy.shape[1], len(offsets)).permute(
        0, 3, 1, 4, 2
    )


def _target_patches(maps: Tensor, xy: Tensor, offsets: Tensor) -> Tensor:
    batch, frames, heads, channels, h, w = maps.shape
    coordinates = _xy_grid(xy, offsets, (h, w)).flatten(0, 1)
    values = F.grid_sample(
        maps.reshape(batch * frames, heads * channels, h, w).float(),
        coordinates,
        align_corners=False,
        padding_mode="border",
    )
    return values.reshape(
        batch, frames, heads, channels, xy.shape[2], len(offsets)
    ).permute(0, 1, 4, 2, 5, 3)


class DiTrackerCost(nn.Module):
    """Sample a DiT and ResNet patch pyramid and project the matching costs."""

    def __init__(
        self,
        *,
        scales: int = 4,
        radius: int = 3,
        embedding_width: int = 256,
        hidden_width: int = 384,
    ) -> None:
        super().__init__()
        if scales < 1 or radius < 0 or min(embedding_width, hidden_width) < 1:
            raise ValueError("Invalid matching pyramid dimensions")
        self.scales = scales
        self.embedding_width = embedding_width
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((x, y), dim=-1).reshape(-1, 2)
        self.register_buffer("offsets", offsets, persistent=False)
        self.project = nn.Sequential(
            nn.Linear(2 * scales * len(offsets) ** 2, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, embedding_width),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        resnet: Tensor,
        anchors: Tensor,
        query_times: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Take normalized coordinates and return [batch, time, point, embedding]."""
        if (
            query.ndim != 6
            or query.shape != key.shape
            or resnet.shape[:2] != query.shape[:2]
        ):
            raise ValueError(
                "Expected DiT and ResNet [batch, time, heads, channels, height, width]"
            )
        if resnet.ndim != 6 or resnet.shape[2] != 1:
            raise ValueError(
                "The ResNet branch has a single unpartitioned feature group"
            )
        if anchors.shape != (*query_times.shape, 2) or positions.shape != (
            query.shape[0],
            query.shape[1],
            anchors.shape[1],
            2,
        ):
            raise ValueError("Inconsistent query times and point shapes")
        costs = []
        for scale in range(self.scales):
            size = tuple(max(1, dim // 2**scale) for dim in resnet.shape[-2:])
            for source, target in ((query, key), (resnet, resnet)):
                first = _anchor_patches(
                    _resize(source, size), anchors.float(), query_times, self.offsets
                )
                later = _target_patches(
                    _resize(target, size), positions.float(), self.offsets
                )
                costs.append(local_qk_cost(first[:, None], later).flatten(-2))
        return self.project(torch.cat(costs, dim=-1).to(self.project[0].weight.dtype))


class FactorizedUpdate(nn.Module):
    """Predict position, visibility and confidence updates from point-time tokens."""

    def __init__(
        self, input_width: int, width: int = 384, heads: int = 8, depth: int = 3
    ):
        super().__init__()
        if min(input_width, width, heads, depth) < 1 or width % 2 or width % heads:
            raise ValueError("Invalid refinement width, attention heads or depth")
        self.input = nn.Linear(input_width, width)
        self.temporal = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                heads,
                width * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )
        self.points = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                heads,
                width * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))
        nn.init.normal_(self.output[1].weight, std=0.01)
        nn.init.zeros_(self.output[1].bias)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, time, points, _ = tokens.shape
        values = self.input(tokens)
        width = values.shape[-1]
        phase = torch.arange(time, device=values.device, dtype=torch.float32)[:, None]
        phase = phase * 10000.0 ** (
            -torch.arange(width // 2, device=values.device) / (width // 2)
        )
        values = (
            values
            + torch.cat((phase.sin(), phase.cos()), dim=-1).to(values.dtype)[
                None, :, None
            ]
        )
        for temporal, across_points in zip(self.temporal, self.points, strict=True):
            sequence = values.transpose(1, 2).reshape(batch * points, time, width)
            values = (
                temporal(sequence).reshape(batch, points, time, width).transpose(1, 2)
            )
            values = across_points(values.reshape(batch * time, points, width)).reshape(
                batch, time, points, width
            )
        return self.output(values)


class DiTrackerRefiner(nn.Module):
    """Resample fused costs and iteratively refine point trajectories."""

    def __init__(self, cost: DiTrackerCost, update: nn.Module | None = None):
        super().__init__()
        self.cost = cost
        self.update = (
            update
            if update is not None
            else FactorizedUpdate(cost.embedding_width + 86)
        )
        self.register_buffer("frequencies", 2.0 ** torch.arange(10), persistent=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        resnet: Tensor,
        anchors: Tensor,
        query_times: Tensor,
        *,
        iterations: int = 6,
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        if iterations < 1:
            raise ValueError("At least one refinement iteration is required")
        frames = query.shape[1]
        coordinates = anchors[:, None].expand(-1, frames, -1, -1).float()
        visibility = coordinates.new_zeros(coordinates.shape[:-1])
        confidence = torch.zeros_like(visibility)
        query_mask = (
            torch.arange(frames, device=query.device)[None, :, None]
            == query_times[:, None]
        )
        scale = coordinates.new_tensor((resnet.shape[-1], resnet.shape[-2]))
        outputs = []
        for _ in range(iterations):
            centers = coordinates.detach()
            cost = self.cost(query, key, resnet, anchors, query_times, centers)
            diff = centers[:, 1:] - centers[:, :-1]
            forward = F.pad(diff, (0, 0, 0, 0, 0, 1))
            backward = F.pad(-diff, (0, 0, 0, 0, 1, 0))
            displacement = torch.cat((forward, backward), dim=-1)
            phase = (displacement[..., None] * self.frequencies).flatten(-2)
            motion = torch.cat((displacement, phase.sin(), phase.cos()), dim=-1)
            tokens = torch.cat(
                (visibility[..., None], confidence[..., None], cost, motion), dim=-1
            )
            update = self.update(tokens.to(cost.dtype)).float()
            coordinates = torch.where(
                query_mask[..., None],
                anchors[:, None],
                centers + update[..., :2] / scale,
            )
            visibility = visibility + update[..., 2]
            confidence = confidence + update[..., 3]
            outputs.append((coordinates, visibility, confidence))
        return outputs


def refinement_loss(
    predictions: list[tuple[Tensor, Tensor, Tensor]],
    points: Tensor,
    visible: Tensor,
    valid: Tensor,
    *,
    decay: float = 0.8,
) -> dict[str, Tensor]:
    """Compute the three iterative objectives with positions in evaluation pixels."""
    if not predictions or not 0 < decay <= 1:
        raise ValueError("Expected predictions and a valid iteration decay")
    count = valid.sum().clamp_min(1)
    weights = torch.where(visible, 1.0, 0.2)
    losses = {
        key: points.new_zeros(()) for key in ("position", "visibility", "confidence")
    }
    for index, (xy, vis, conf) in enumerate(predictions):
        weight = decay ** (len(predictions) - index - 1)
        error = torch.where(valid[..., None], xy - points, 0.0)
        huber = F.huber_loss(
            error, torch.zeros_like(error), delta=6.0, reduction="none"
        ).sum(-1)
        target_vis = visible.to(vis.dtype)
        target_conf = (error.detach().norm(dim=-1) < 12.0).to(conf.dtype)
        vis_error = F.binary_cross_entropy_with_logits(
            torch.where(valid, vis, 0.0), target_vis, reduction="none"
        )
        conf_error = F.binary_cross_entropy_with_logits(
            torch.where(valid, conf, 0.0), target_conf, reduction="none"
        )
        losses["position"] += weight * (huber * weights).sum() / count
        losses["visibility"] += (
            weight * torch.where(valid, vis_error, 0.0).sum() / count
        )
        losses["confidence"] += (
            weight * torch.where(valid, conf_error, 0.0).sum() / count
        )
    return losses
