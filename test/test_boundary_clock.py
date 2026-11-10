"""Checks for feature-time interpolation at known chunk anchors."""

import torch

from heft.calibration.boundary import retime_with_anchors


def test_next_anchor_supplies_exact_content_time_for_known_linear_delay():
    maps = torch.arange(10).float() - 1.25
    maps[0] = 0.0
    maps[5] = 5.0
    actual = retime_with_anchors(maps, ((0, 5), (5, 10)), 1.25)
    torch.testing.assert_close(actual[:5], torch.arange(5).float())
    assert actual[5] == 5.0
    assert actual[-1] == maps[-1]


def test_negative_delay_uses_unshifted_first_anchor():
    maps = torch.arange(6).float() + 1.0
    maps[0] = 0.0
    actual = retime_with_anchors(maps, ((0, 6),), -1.0)
    torch.testing.assert_close(actual, torch.arange(6).float())


def test_zero_advance_returns_original_maps():
    maps = torch.randn(3, 2, 4, 5)
    assert retime_with_anchors(maps, ((0, 3),), 0.0) is maps
