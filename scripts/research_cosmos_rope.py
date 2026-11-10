"""Exact Cosmos axial phases and a lossless split-half storage permutation."""
import torch


def cosmos_angles(frames, height, width, *, device="cpu", fps=16):
    positions = torch.meshgrid(torch.arange(frames, device=device, dtype=torch.float32) / fps * 24,
                               torch.arange(height, device=device, dtype=torch.float32),
                               torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    angles = []
    for p, n, scale in zip(positions, (22, 21, 21), (1., 3., 3.), strict=True):
        dim = 2*n
        base = 10000.0 * scale ** (dim / (dim-2))
        exponent = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        frequency = 1.0 / (base ** exponent)
        angles.append(p[..., None] * frequency)
    return torch.cat(angles, dim=-1)


def adjacent_storage(feature):
    return torch.stack((feature[..., :64], feature[..., 64:]), dim=-1).flatten(-2)


if __name__ == "__main__":
    import json
    from diffusers.models.transformers.transformer_cosmos import CosmosRotaryPosEmbed
    from diffusers.models.embeddings import apply_rotary_emb
    from research_rope import phase_readout
    a = cosmos_angles(3, 4, 5)
    module = CosmosRotaryPosEmbed(128, rope_scale=(1., 3., 3.))
    cosine, sine = module(torch.empty(1, 16, 3, 8, 10), fps=16)
    error = max((cosine[:, :64].reshape_as(a)-a.cos()).abs().max().item(),
                (sine[:, :64].reshape_as(a)-a.sin()).abs().max().item())
    x = torch.randn(1, 1, 60, 128)
    rotated = apply_rotary_emb(x, (cosine, sine), use_real=True, use_real_unbind_dim=-2)
    expected = adjacent_storage(rotated).reshape(1, 3, 4, 5, 128)
    obtained = phase_readout(adjacent_storage(x).reshape_as(expected), a, temporal=2, spatial=2)
    rotation_error = (expected-obtained).abs().max().item()
    assert error < 1e-6 and rotation_error < 1e-5
    print(json.dumps({"model_phase_error": error, "storage_permutation_rotation_error": rotation_error}, indent=2))
