"""Reconstruction of the DiffTrack readout from its published equations."""

import math

import pytest
import torch

from heft.baselines.difftrack import (
    bidirectional_cost,
    ddim_inverse_step,
    interleaved_windows,
    track_difftrack,
)


@pytest.mark.parametrize("frames,anchor", [(1, 0), (13, 0), (25, 0), (26, 0), (103, 6), (200, 97)])
def test_interleaved_windows_cover_every_later_frame(frames: int, anchor: int) -> None:
    chunks = interleaved_windows(frames, anchor)
    assert all(window[0] == anchor and 2 <= len(window) <= 13 for window in chunks)
    assert {target for window in chunks for target in window[1:]} == set(range(anchor + 1, frames))
    if frames == 25:
        assert chunks == ((0, *range(1, 25, 2)), (0, *range(2, 25, 2)))
    if frames == 26:
        assert sum(len(window) - 1 for window in chunks) > 25


def test_bidirectional_cost_samples_the_forward_and_reverse_probability_maps() -> None:
    generator = torch.Generator().manual_seed(93)
    q0, k0, qt, kt = [torch.randn(6, 5, generator=generator) for _ in range(4)]
    indices = torch.tensor([[1, 3, 4, 5], [0, 1, 2, 3]])
    weights = torch.tensor([[.3, .4, .1, .2], [.6, .0, .3, .1]])
    scale = math.sqrt(q0.shape[1])
    dense = ((q0 @ kt.T / scale).softmax(dim=-1) + (qt @ k0.T / scale).softmax(dim=-1).T) * .5
    expected = (dense[indices] * weights[..., None]).sum(1)
    torch.testing.assert_close(bidirectional_cost(q0, k0, qt, kt, indices, weights), expected)


def test_tracker_keeps_late_query_anchors_and_overlapping_targets() -> None:
    tokens = 6
    xy = torch.tensor([[.5 / tokens, .5, 0], [3.5 / tokens, .5, 2]])

    def capture(frames):
        # Token identity i moves right one cell per frame.
        rows = torch.stack([torch.eye(tokens).roll(t, dims=0) for t in frames]) * 8
        return rows, rows

    tracks = track_difftrack(8, xy, (1, tokens), capture, chunk_size=4)
    torch.testing.assert_close(tracks[0, :, 0], (torch.arange(8) % tokens + .5) / tokens)
    torch.testing.assert_close(tracks[1, :2], torch.zeros(2, 2))
    torch.testing.assert_close(tracks[1, 2:, 0], ((torch.arange(2, 8) + 1) % tokens + .5) / tokens)
    torch.testing.assert_close(tracks[1, 2:, 1], torch.full((6,), .5))


@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
def test_ddim_inverse_uses_native_alpha_and_prediction_type(prediction: str) -> None:
    alpha, next_alpha = .88, .64
    clean, noise = torch.tensor([-.6, 1.2]), torch.tensor([.45, -.8])
    sample = math.sqrt(alpha) * clean + math.sqrt(1-alpha) * noise
    output = noise if prediction == "epsilon" else math.sqrt(alpha) * noise - math.sqrt(1-alpha) * clean
    expected = math.sqrt(next_alpha) * clean + math.sqrt(1-next_alpha) * noise
    torch.testing.assert_close(ddim_inverse_step(sample, output, alpha, next_alpha, prediction), expected)
