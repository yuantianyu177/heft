"""The RGB pseudo-track clock can recover the sign of a known lag."""

import pytest
import torch

from heft.calibration.motion import MotionTraces, estimate_trace_lag, rgb_point_traces
from heft.calibration.temporal import retime_feature_volume


@pytest.mark.parametrize("lag", [-1.75, 0., 1.25])
def test_trace_clock_projects_onto_known_motion(lag: float) -> None:
    t = torch.arange(30, dtype=torch.float32)
    x = torch.stack((.29 + .10*torch.sin(.19*t), .66 + .10*torch.cos(.22*t),
                     .54 + .14*torch.sin(.26*t)), dim=0)
    y = torch.stack((.27 + .08*torch.cos(.24*t), .63 + .07*torch.sin(.18*t),
                     .48 + .08*torch.cos(.23*t)), dim=0)
    points = torch.stack((x, y), dim=-1)
    yy, xx = torch.meshgrid((torch.arange(38)+.5)/38, (torch.arange(56)+.5)/56, indexing="ij")
    blob = (-((xx-x.T[..., None, None]).square() + (yy-y.T[..., None, None]).square())/.003).exp()
    rgb_features = torch.cat((blob, torch.full_like(blob[:, :1], .4)), dim=1)
    observed = retime_feature_volume(rgb_features, -lag)
    traces = MotionTraces(points, torch.ones_like(x, dtype=torch.bool))
    result = estimate_trace_lag((observed,), (traces,), min_support=10)
    assert result.reliable
    assert result.advance_frames == pytest.approx(lag, abs=.25)


def test_optical_flow_tracks_known_horizontal_translation() -> None:
    texture = torch.randint(0, 256, (1, 90, 112), dtype=torch.uint8,
                            generator=torch.Generator().manual_seed(432))
    rgb = torch.stack([texture.roll(i, dims=-1).expand(3, -1, -1) for i in range(9)])
    traces = rgb_point_traces(rgb, raster=(90, 112), max_points=80)
    selected = traces.visible[:, -1]
    assert int(selected.sum()) >= 25
    delta = traces.points[selected, -1] - traces.points[selected, 0]
    torch.testing.assert_close(delta.median(dim=0).values, torch.tensor([8/111, 0.]), atol=.003, rtol=0)
