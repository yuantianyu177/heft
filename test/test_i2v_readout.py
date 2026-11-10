"""Check native reference-conditioned temporal patch alignment."""

import torch

from heft.baselines.matched_i2v_features import stitch_reference_phases
from heft.extraction.temporal import four_phases


def test_front_padded_cog_temporal_tokens_read_at_their_own_centers() -> None:
    phases = four_phases(9, 4)
    entries = []
    for phase in phases:
        times = phase.latent_times
        padded = (0,) * ((-len(times)) % 2) + times
        centers = torch.tensor(
            [sum(padded[i : i + 2]) / 2 for i in range(0, len(padded), 2)]
        )
        entries.append(centers[None, :, None])
    dense = stitch_reference_phases(entries, phases, frames=9, spatial=1, patch=2)
    assert dense.shape == (1, 9, 1)
    torch.testing.assert_close(dense[0, 1, 0], torch.tensor(1.0))
    torch.testing.assert_close(dense[0, 8, 0], torch.tensor(6.0))
    torch.testing.assert_close(dense[0, 4, 0], torch.tensor(4.0))


def test_wan_phase_stitch_recovers_each_frame_without_time_interpolation() -> None:
    phases = four_phases(13, 4)
    values = [
        torch.tensor(phase.latent_times)[None, :, None].float() for phase in phases
    ]
    assert torch.equal(
        stitch_reference_phases(values, phases, frames=13, spatial=1, patch=1)[0, :, 0],
        torch.arange(13).float(),
    )
