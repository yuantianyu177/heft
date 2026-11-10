"""Tests for the Point Prompting detector and generation policy."""

import numpy as np
import pytest
import torch

from heft.baselines.point_prompting import (
    balance_red,
    counterfactual_denoise,
    marker_disk,
    trace_marker,
)


def test_balance_red_removes_competing_hue_and_preserves_green() -> None:
    rgb = np.array([[[[255, 0, 0], [0, 255, 0], [120, 0, 0]]]], dtype=np.uint8)
    balanced = balance_red(rgb)
    assert balanced[0, 0, 0, 1] >= 175
    np.testing.assert_array_equal(balanced[0, 0, 1:], rgb[0, 0, 1:])
    assert rgb[0, 0, 0, 1] == 0


def test_marker_detection_expands_radius_and_resets_after_reappearance() -> None:
    rgb = np.zeros((5, 41, 41, 3), dtype=np.uint8)
    rgb[2, 10, 20:23, 0] = 255
    rgb[2, 30, 8, 0] = 255
    rgb[3, 10, 31, 0] = 255
    rgb[4, 10, 31, 0] = 255
    tracks, visibility = trace_marker(
        rgb, (10.0, 10.0), radius=9.0, max_radius=20.0, expansion=1.5, blob_radius=3.0
    )
    np.testing.assert_allclose(
        tracks, [[10, 10], [10, 10], [21, 10], [21, 10], [31, 10]]
    )
    np.testing.assert_array_equal(visibility, [True, False, True, False, True])


def test_marker_disk_uses_xy_order_and_original_shape() -> None:
    mask = marker_disk(np.array([[6.0, 1.0]]), (4, 8), 1.0)
    assert mask.shape == (1, 4, 8)
    assert mask[0, 1, 6]
    assert mask.sum() == 5


def test_counterfactual_generation_refreshes_clean_region_at_each_noise_level() -> None:
    clean = torch.tensor([0.0, 4.0])
    noise = torch.tensor([1.0, 2.0])
    mask = torch.tensor([True, False])
    calls = []

    def denoise(x, t, marked):
        calls.append((float(t), marked, x.tolist()))
        return torch.full_like(x, 3.0 if marked else 1.0)

    def add_noise(x, n, t):
        return x + n * t

    def step(x, predicted, i):
        assert predicted.tolist() == pytest.approx([7.0, 7.0])
        return x - predicted

    result = counterfactual_denoise(
        clean,
        noise,
        torch.tensor([2.0, 1.0]),
        denoise,
        step,
        add_noise,
        guidance=2.0,
        mask=mask,
    )
    assert calls == [
        (2.0, True, [2.0, 8.0]),
        (2.0, False, [2.0, 8.0]),
        (1.0, True, [-5.0, 6.0]),
        (1.0, False, [-5.0, 6.0]),
    ]
    torch.testing.assert_close(result, torch.tensor([-12.0, 4.0]))
