"""Norm-preserving RoPE readout interventions for correspondence experiments.

These operations change saved descriptors, not the pretrained model's attention.
All phases are calculated on the original token grid, before interpolation.
"""
from __future__ import annotations

import torch
from torch import Tensor


def wan_angles(frames: int, height: int, width: int, *, device="cpu") -> Tensor:
    """Return the exact Wan2.1 128-D axial phases, [T,H,W,64]."""
    positions = torch.meshgrid(
        torch.arange(frames, device=device, dtype=torch.float64),
        torch.arange(height, device=device, dtype=torch.float64),
        torch.arange(width, device=device, dtype=torch.float64),
        indexing="ij",
    )
    return torch.cat([
        p[..., None] * 10000.0 ** (-torch.arange(n, device=device, dtype=torch.float64) / n)
        for p, n in zip(positions, (22, 21, 21), strict=True)
    ], dim=-1)


def phase_readout(
    feature: Tensor,
    angles: Tensor,
    *,
    temporal: float = 1.0,
    spatial: float = 1.0,
    pairing: str = "adjacent",
    temporal_pairs: int = 22,
    high_fraction: float | None = None,
) -> Tensor:
    """Set readout phase to R(lambda*p), starting from post-RoPE R(p)f.

    Input ends in [T,H,W,D]; optional leading dimensions are heads/batches.
    lambda=1 is unchanged and lambda=0 removes the explicit RoPE rotation.
    With high_fraction, only the highest axis-wise fraction is rerotated.
    """
    if feature.shape[-1] != 2 * angles.shape[-1]:
        raise ValueError("feature/angle pair dimensions differ")
    if tuple(feature.shape[-4:-1]) != tuple(angles.shape[:3]):
        raise ValueError("angles must use the original feature token grid")
    n = angles.shape[-1]
    scales = torch.full((n,), spatial - 1.0, dtype=torch.float64, device=angles.device)
    scales[:temporal_pairs] = temporal - 1.0
    if high_fraction is not None:
        keep = torch.zeros(n, device=angles.device, dtype=torch.bool)
        counts = (temporal_pairs, (n - temporal_pairs) // 2, (n - temporal_pairs) // 2)
        start = 0
        for count in counts:
            keep[start : start + int(count * high_fraction)] = True
            start += count
        scales *= keep
    delta = angles * scales
    # Float32 operations retain BF16 capture precision without quantizing again.
    cosine, sine = delta.cos().float(), delta.sin().float()
    f = feature.float()
    if pairing == "adjacent":
        u, v = f[..., 0::2], f[..., 1::2]
        return torch.stack((u * cosine - v * sine, v * cosine + u * sine), dim=-1).flatten(-2)
    if pairing == "split_half":
        u, v = f[..., :n], f[..., n:]
        return torch.cat((u * cosine - v * sine, v * cosine + u * sine), dim=-1)
    raise ValueError(f"unknown pairing: {pairing}")


def frequency_pairs(lower: float = 0.0, upper: float = 1.0, *, counts=(22, 21, 21)) -> Tensor:
    if not 0 <= lower < upper <= 1:
        raise ValueError("expected 0 <= lower < upper <= 1")
    result, start = [], 0
    for count in counts:
        result.extend(range(start + int(count * lower), start + int(count * upper)))
        start += count
    return torch.tensor(result, dtype=torch.long)


def pair_channels(pairs: Tensor, *, pairing="adjacent", total_pairs=64) -> Tensor:
    if pairing == "adjacent":
        return torch.stack((2 * pairs, 2 * pairs + 1), dim=-1).flatten()
    if pairing == "split_half":
        return torch.cat((pairs, pairs + total_pairs))
    raise ValueError(pairing)


def sanity_check() -> dict[str, float]:
    generator = torch.Generator().manual_seed(1234)
    f = torch.randn(2, 3, 4, 5, 128, generator=generator)
    angles = wan_angles(3, 4, 5)
    rotated = phase_readout(f, angles, temporal=2.0, spatial=2.0)
    recovered = phase_readout(rotated, angles, temporal=0.0, spatial=0.0)
    norm_error = (rotated.norm(dim=-1) - f.norm(dim=-1)).abs().max().item()
    inverse_error = (recovered - f).abs().max().item()
    assert norm_error < 1e-5 and inverse_error < 1e-5
    # The helper must match the exact frequencies used in the vendored model.
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
    actual = rope(torch.empty(1, 16, 3, 8, 10)).reshape(3, 4, 5, 64)
    expected = torch.polar(torch.ones_like(angles), angles)
    phase_error = (actual - expected).abs().max().item()
    assert phase_error < 1e-10
    return {"norm_error": norm_error, "inverse_error": inverse_error, "model_phase_error": phase_error}


if __name__ == "__main__":
    import json
    print(json.dumps(sanity_check(), indent=2))
