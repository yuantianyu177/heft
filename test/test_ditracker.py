"""Check the local correlation and supervision used by DiTracker."""

import pytest
import torch

from heft.baselines.ditracker import DiTrackerCost, local_qk_cost, refinement_loss


def test_local_cost_normalizes_each_query_over_target_positions() -> None:
    # Two heads each favor a different target patch element.
    query = torch.ones(1, 2, 1, 1)
    key = torch.tensor([[[[2.0], [0.0]], [[0.0], [2.0]]]])
    cost = local_qk_cost(query, key)
    assert cost.shape == (1, 1, 2)
    torch.testing.assert_close(cost, torch.full_like(cost, 0.5))


def test_cost_fusion_resamples_and_backpropagates_through_both_branches() -> None:
    module = DiTrackerCost(scales=2, radius=1, embedding_width=16)
    q = torch.randn(1, 3, 2, 4, 6, 7, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    rgb = torch.randn(1, 3, 1, 8, 12, 14, requires_grad=True)
    starts = torch.tensor([[[0.5, 0.5], [0.4, 0.65]]])
    times = torch.tensor([[0, 1]])
    current = starts[:, None].repeat(1, 3, 1, 1).requires_grad_()
    result = module(q, k, rgb, starts, times, current)
    assert result.shape == (1, 3, 2, 16)
    result.square().sum().backward()
    for array in (q, k, rgb, current):
        assert array.grad.abs().sum() > 0


def test_training_weights_occlusions_and_ignores_invalid_frames() -> None:
    positions = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1000.0, 1000.0]]]])
    points = torch.zeros_like(positions)
    visible = torch.tensor([[[True, False, False]]])
    valid = torch.tensor([[[True, True, False]]])
    logits = torch.zeros_like(visible, dtype=torch.float32)
    result = refinement_loss([(positions, logits, logits)], points, visible, valid)
    assert result["position"] == pytest.approx(0.5 * 1.2 / 2)
    assert result["visibility"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert result["confidence"] == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_refiner_resamples_costs_at_updated_locations_and_pins_queries() -> None:
    from torch import nn

    from heft.baselines.ditracker import DiTrackerRefiner

    class Cost(nn.Module):
        embedding_width = 4

        def __init__(self):
            super().__init__()
            self.positions = []

        def forward(self, query, key, resnet, anchors, times, positions):
            self.positions.append(positions.clone())
            return positions.new_zeros((*positions.shape[:-1], self.embedding_width))

    class Shift(nn.Module):
        def forward(self, tokens):
            update = tokens.new_zeros((*tokens.shape[:-1], 4))
            update[..., 0] = 1.0
            update[..., 2:] = 0.5
            return update

    cost = Cost()
    refiner = DiTrackerRefiner(cost, Shift())
    dummy = torch.zeros((1, 3, 1, 4, 8, 10))
    anchors = torch.tensor([[[0.5, 0.5], [0.7, 0.4]]])
    times = torch.tensor([[0, 1]])
    outputs = refiner(dummy, dummy, dummy, anchors, times, iterations=2)
    for time, index in ((0, 0), (1, 1)):
        torch.testing.assert_close(outputs[-1][0][0, time, index], anchors[0, index])
    torch.testing.assert_close(cost.positions[1][0, 2, 0], torch.tensor([0.6, 0.5]))
    torch.testing.assert_close(outputs[-1][0][0, 2, 0], torch.tensor([0.7, 0.5]))
    torch.testing.assert_close(outputs[-1][1][0, 2], torch.ones(2))


def test_lora_linear_starts_at_frozen_function_and_learns_residual() -> None:
    from heft.baselines.ditracker_backbone import LoRALinear

    base = torch.nn.Linear(4, 6)
    layer = LoRALinear(base, rank=2)
    inputs = torch.randn(5, 4)
    output = layer(inputs)
    torch.testing.assert_close(output, base(inputs), rtol=0.0, atol=0.0)
    output.square().mean().backward()
    assert base.weight.grad is None
    assert layer.up.weight.grad.abs().sum() > 0
