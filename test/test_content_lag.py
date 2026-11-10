"""Signed feature clocks and chunk-bounded feature retiming."""

import pytest
import torch

from heft.calibration.temporal import estimate_content_lag, retime_feature_volume


def moving_image(frames: int = 50) -> torch.Tensor:
    t = torch.arange(frames, dtype=torch.float32)[:, None, None]
    y, x = torch.meshgrid(torch.linspace(-1, 1, 18), torch.linspace(-1, 1, 25), indexing="ij")
    horizontal = .49 * torch.sin(t * .23) + .18 * torch.cos(t * .55)
    vertical = .42 * torch.cos(t * .16)
    object_a = torch.exp(-((x-horizontal).square() + (y-vertical).square()) / .024)
    object_b = torch.exp(-((x+.42).square() + (y-.39*torch.sin(.3*t)).square()) / .056)
    texture = (.5 + .5 * torch.sin(11*x + 3*y)) * (.5 + .5 * torch.cos(7*y - x))
    return torch.stack((.16 + .62*object_a + .05*texture,
                        .26 + .47*object_b + .05*texture,
                        .18 + .26*object_a + .35*object_b + .08*texture), dim=1)


@pytest.mark.parametrize("lag", [-2.25, 0., 1.50, 2.25])
def test_identifies_injected_signed_delays_across_chunks(lag: float) -> None:
    rgb = moving_image()
    clips = ((0, 25), (25, 50))
    # Change the descriptor coordinate basis. Give each chunk an independent clock.
    basis = torch.linalg.qr(torch.randn((3, 3), generator=torch.Generator().manual_seed(7)))[0]
    projected = torch.einsum("ab,tbhw->tahw", basis, rgb)
    observed = torch.cat([retime_feature_volume(projected[a:b], -lag) for a, b in clips])
    estimate = estimate_content_lag(rgb, observed, chunks=clips)
    assert estimate.reliable
    assert estimate.advance_frames == pytest.approx(lag, abs=.25)
    assert estimate.support >= 16


def test_static_video_has_no_identifiable_absolute_clock() -> None:
    rgb = moving_image(1).expand(25, -1, -1, -1)
    features = moving_image(25)
    result = estimate_content_lag(rgb, features)
    assert result.advance_frames == 0
    assert not result.reliable
    assert result.support == 0


def test_retiming_preserves_anchor_and_clamps_inside_each_chunk() -> None:
    value = torch.arange(5, dtype=torch.bfloat16).reshape(-1, 1, 1, 1)
    torch.testing.assert_close(retime_feature_volume(value, 1.25).flatten(),
                               torch.tensor([0, 2.25, 3.25, 4, 4], dtype=value.dtype))
    torch.testing.assert_close(retime_feature_volume(value, -1.25).flatten(),
                               torch.tensor([0, 0, .75, 1.75, 2.75], dtype=value.dtype))
    assert retime_feature_volume(value, 0) is value


def test_rejects_incomplete_chunk_coverage() -> None:
    rgb = moving_image()
    with pytest.raises(ValueError, match="partition"):
        estimate_content_lag(rgb, rgb, chunks=((0, 25), (26, 50)))
