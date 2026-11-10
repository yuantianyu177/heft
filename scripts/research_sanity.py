"""Check scientific probe invariants and known correspondences before model runs."""
import json
import torch

from research_probe import VARIANTS, norm_matched_indices, probe, readout
from research_rope import phase_readout, sanity_check, wan_angles


def main():
    result = sanity_check()
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(193)
    h, w, nt = 12, 16, 3
    image = torch.randn(1, 1, h, w, 128, generator=generator, device=device)
    content = torch.cat([torch.roll(image, i, dims=3) for i in range(nt)], dim=1)
    angle = wan_angles(nt, h, w, device=device)
    post = phase_readout(content, angle, temporal=2, spatial=2)
    recovered = phase_readout(post, angle, temporal=0, spatial=0)
    yy, xx = torch.meshgrid(torch.arange(3, h-3, device=device), torch.arange(3, w-5, device=device), indexing="ij")
    points = torch.stack(((xx.flatten()+.5)/w, (yy.flatten()+.5)/h), -1)
    truth = points[None] + torch.tensor([[1/w, 0], [2/w, 0]], device=device)[:, None]
    result_probe = probe(recovered, points, truth, torch.ones(nt-1, len(points), dtype=torch.bool, device=device), points)
    result["known_translation_max_error_256px"] = float(result_probe["errors"].max())
    assert result["known_translation_max_error_256px"] < 1e-4
    # Exact mean + residual dot-product decomposition, including both cross terms.
    mean = recovered.mean((1, 2, 3), keepdim=True)
    repeated_mean = mean.expand_as(recovered)
    carrier = phase_readout(repeated_mean, angle, temporal=2, spatial=2)
    residual = post - carrier
    a, b = post[:, 0, 2, 3], post[:, 2, 4, 6]
    ma, mb = carrier[:, 0, 2, 3], carrier[:, 2, 4, 6]
    ra, rb = residual[:, 0, 2, 3], residual[:, 2, 4, 6]
    decomposed = (ma*mb + ma*rb + ra*mb + ra*rb).sum(-1)
    result["dot_decomposition_error"] = float((decomposed - (a*b).sum(-1)).abs().max())
    assert result["dot_decomposition_error"] < 1e-4
    idx, _ = norm_matched_indices(post)
    for variant in VARIANTS:
        descriptor = readout(post, angle, variant, idx)
        assert descriptor.shape[:-1] == post.shape[:-1]
        assert bool(torch.isfinite(descriptor).all())
    result["variants_checked"] = len(VARIANTS)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
