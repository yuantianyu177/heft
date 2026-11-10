"""Fixed, tracker-independent correspondence probes on saved Wan features.

Reports token-grid nearest-neighbour PCK in the 256x256 coordinate system.
These are representation diagnostics, NOT full TAP-Vid benchmark scores.
Synthetic calibration reads pixels and known transforms only, not annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from research_rope import frequency_pairs, pair_channels, phase_readout, wan_angles


VARIANTS = [
    ("post_full", 1, 1, 0, None),
    ("post_low15", 1, 1, .15, None),
    ("content_full", 0, 0, 0, None),
    ("time_removed", 0, 1, 0, None),
    ("space_removed", 1, 0, 0, None),
    ("phase025", .25, .25, 0, None),
    ("phase050", .5, .5, 0, None),
    ("time0_space025", 0, .25, 0, None),
    ("time0_space050", 0, .5, 0, None),
    ("time0_space010", 0, .1, 0, None),
    ("high15_phase_removed", 0, 0, 0, .15),
    ("content_low15", 0, 0, .15, None),
    *[(f"random_{seed}", 1, 1, 0, None) for seed in range(5)],
    ("norm_matched", 1, 1, 0, None),
    ("content_centered", 0, 0, 0, None),
    ("post_centered", 1, 1, 0, None),
    ("content_whitened", 0, 0, 0, None),
    ("rope_centered", 1, 1, 0, None),
    ("rope_high15_centered", 1, 1, 0, None),
    ("rope_centered_whitened", 1, 1, 0, None),
]


def sample(feature, points):
    """Sample [head,frame,H,W,D] at normalized [frame,point,2]."""
    heads, frames, h, w, d = feature.shape
    grid = points[None].expand(heads, -1, -1, -1).reshape(heads * frames, 1, -1, 2) * 2 - 1
    return F.grid_sample(feature.permute(0, 1, 4, 2, 3).reshape(heads * frames, d, h, w),
                         grid, align_corners=False, padding_mode="border").squeeze(2).transpose(1, 2).reshape(heads, frames, -1, d)


def random_kept(seed, device):
    generator = torch.Generator().manual_seed(seed)
    result, start = [], 0
    for n in (22, 21, 21):
        selected = torch.randperm(n, generator=generator)[int(n * .15):].sort().values + start
        result.append(selected)
        start += n
    return pair_channels(torch.cat(result)).to(device)


def norm_matched_indices(feature):
    """Keep dimension and removed energy matched to low15, independently per head."""
    energy = feature.float().reshape(*feature.shape[:-1], 64, 2).square().sum(-1).mean((1, 2, 3))
    removed = frequency_pairs(0, .15).to(feature.device)
    target = energy[:, removed].sum(-1)
    generator = torch.Generator().manual_seed(93847)
    candidates = []
    for _ in range(2048):
        r, start = [], 0
        for n in (22, 21, 21):
            r.append(torch.randperm(n, generator=generator)[:int(n * .15)] + start)
            start += n
        candidates.append(torch.cat(r))
    candidates = torch.stack(candidates).to(feature.device)
    candidate_energy = energy[:, candidates].sum(-1)
    discrepancy = (candidate_energy - target[:, None]).abs() / target[:, None].clamp_min(1e-10)
    best = discrepancy.argmin(-1)
    keeps = []
    for head in range(len(energy)):
        mask = torch.ones(64, dtype=torch.bool, device=feature.device)
        mask[candidates[best[head]]] = False
        keeps.append(pair_channels(mask.nonzero().flatten()))
    return torch.stack(keeps), discrepancy[torch.arange(len(energy), device=feature.device), best]


def readout(feature, angles, variant, norm_indices):
    name, temporal, spatial, cutoff, high = variant
    if name.startswith("rope_"):
        content = phase_readout(feature, angles, temporal=0, spatial=0)
        mean = content.mean((1, 2, 3), keepdim=True)
        if name == "rope_high15_centered":
            mask = torch.zeros(content.shape[-1], device=content.device)
            mask[pair_channels(frequency_pairs(0, .15)).to(content.device)] = 1
            mean = mean * mask
        content = content - mean
        if name == "rope_centered_whitened":
            content = content.reshape(*content.shape[:-1], 64, 2)
            content = content / content.square().mean((1, 2, 3, 5), keepdim=True).sqrt().clamp_min(1e-5)
            content = content.flatten(-2)
        return phase_readout(content, angles, temporal=2, spatial=2)
    f = phase_readout(feature, angles, temporal=temporal, spatial=spatial, high_fraction=high)
    if name.endswith("centered"):
        f = f - f.mean((1, 2, 3), keepdim=True)
    if name.endswith("whitened"):
        f = f.reshape(*f.shape[:-1], 64, 2)
        f = f / f.square().mean((1, 2, 3, 5), keepdim=True).sqrt().clamp_min(1e-5)
        f = f.flatten(-2)
    if cutoff:
        f = f[..., pair_channels(frequency_pairs(cutoff, 1)).to(f.device)]
    if name.startswith("random_"):
        f = f[..., random_kept(int(name.rsplit("_", 1)[1]), f.device)]
    if name == "norm_matched":
        indices = norm_indices[:, None, None, None, :].expand(*f.shape[:-1], -1)
        f = torch.gather(f, -1, indices)
    return f


def grid_points(device, n=8):
    yy, xx = torch.meshgrid(torch.linspace(.2, .8, n, device=device),
                            torch.linspace(.2, .8, n, device=device), indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), -1)


@torch.inference_mode()
def probe(feature, points, truth, valid, cycle_points):
    """All points are normalized; visibility is used only for diagnostic metrics."""
    nh, nt, h, w, d = feature.shape
    source = F.normalize(sample(feature[:, :1], points[None])[:, 0], dim=-1)
    target = F.normalize(feature[:, 1:].flatten(2, 3), dim=-1)
    score = torch.einsum("hpd,htsd->htps", source, target)
    index = score.argmax(-1)
    prediction = torch.stack(((index % w + .5) / w, (index // w + .5) / h), -1)
    errors = ((prediction - truth[None]) * 256).norm(dim=-1)
    gt_descriptors = F.normalize(sample(feature[:, 1:], truth), dim=-1)
    true_similarity = torch.einsum("hpd,htpd->htp", source, gt_descriptors)
    stationary = points[None].expand(nt - 1, -1, -1)
    stationary_descriptors = F.normalize(sample(feature[:, 1:], stationary), dim=-1)
    static_similarity = torch.einsum("hpd,htpd->htp", source, stationary_descriptors)
    # Cycle-consistency diagnostic uses an independent, annotation-free grid.
    cyc_source = F.normalize(sample(feature[:, :1], cycle_points[None])[:, 0], dim=-1)
    cyc_score = torch.einsum("hpd,htsd->htps", cyc_source, target)
    cyc_index = cyc_score.argmax(-1)
    sampled_target = target.gather(2, cyc_index[..., None].expand(-1, -1, -1, d))
    source_map = F.normalize(feature[:, 0].flatten(1, 2), dim=-1)
    backward = torch.einsum("htpd,hsd->htps", sampled_target, source_map).argmax(-1)
    recovered = torch.stack(((backward % w + .5) / w, (backward // w + .5) / h), -1)
    cycle_error = ((recovered - cycle_points) * 256).norm(dim=-1).mean((1, 2))
    return {
        "errors": errors.cpu().numpy(),
        "prediction": prediction.cpu().numpy(),
        "true_similarity": true_similarity.cpu().numpy(),
        "stationary_similarity": static_similarity.cpu().numpy(),
        "cycle_error": cycle_error.cpu().numpy(),
    }


def annotations(root, name, metadata, device):
    frames = metadata["sparse_frames"]
    if metadata["transform"] == "real":
        with np.load(root / f"data/davis_records/{name}.annotations.npz") as a:
            keep = ~a["occluded"][:, 0]
            points = a["points"][keep][:, frames].transpose(1, 0, 2).astype(np.float32)
            visible = ~a["occluded"][keep][:, frames].T.copy()
        return (torch.tensor(points[0], device=device), torch.tensor(points[1:], device=device),
                torch.tensor(visible[1:], device=device))
    # Do not open the human annotation file during synthetic calibration.
    source = grid_points(device)
    shifts = torch.tensor(metadata["shifts"], dtype=torch.float32, device=device)[frames]
    sh, sw = metadata["source_size"]
    shifts /= torch.tensor([sw, sh], device=device)
    truth = source[None] + shifts[:, None]
    valid = ((truth > .03) & (truth < .97)).all(-1)
    return source, truth[1:], valid[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--features", default="wan_real_t25")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--names")
    parser.add_argument("--layers")
    parser.add_argument("--variants")
    parser.add_argument('--anchor-features',help='Replace only frame-zero descriptors from this feature folder')
    parser.add_argument('--output-tag')
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{args.gpu}"
    variants = [v for v in VARIANTS if args.variants is None or v[0] in args.variants.split(",")]
    feature_root = args.root / "features" / args.features
    output_root = args.root / "results" / (args.output_tag or f"probe_{args.features}")
    output_root.mkdir(exist_ok=True, parents=True)
    paths = sorted(feature_root.glob("*.safetensors"))
    if args.names:
        paths = [p for p in paths if p.stem in args.names.split(",")]
    for path in paths[args.shard::args.shards]:
        name = path.stem
        output = output_root / f"{name}.npz"
        if output.exists() and not args.overwrite:
            continue
        started = time.monotonic()
        metadata = json.loads(path.with_suffix(".json").read_text())
        points, truth, valid = annotations(args.root, name, metadata, device)
        h, w = metadata["feature_size"]
        angles = wan_angles(metadata["frames"], h, w, device=device)[metadata["sparse_frames"]]
        angles[...,:22]*=metadata.get('rope_temporal_scale',1.)
        layers = list(range(metadata["layers"])) if not args.layers else list(map(int, args.layers.split(",")))
        output_arrays = {key: [] for key in ("errors", "prediction", "true_similarity", "stationary_similarity", "cycle_error")}
        norm_discrepancy = []
        content_means, pair_energies, pair_variances = [], [], []
        cycle_points = grid_points(device, 6)
        with safe_open(path, framework="pt", device="cpu") as features:
            for layer in layers:
                feature = features.get_tensor(f"l{layer:02d}_key").to(device)
                if args.anchor_features:
                    with safe_open(args.root/'features'/args.anchor_features/path.name,framework='pt',device='cpu') as anchor:
                        feature[:,0]=anchor.get_tensor(f'l{layer:02d}_key')[:,0].to(device)
                content = phase_readout(feature, angles, temporal=0, spatial=0)
                mean = content.mean((1, 2, 3))
                content_means.append(mean.cpu().numpy())
                pair_energies.append(content.reshape(*content.shape[:-1], 64, 2).square().sum(-1).mean((1, 2, 3)).cpu().numpy())
                residual = content - mean[:, None, None, None]
                pair_variances.append(residual.reshape(*residual.shape[:-1], 64, 2).square().sum(-1).mean((1, 2, 3)).cpu().numpy())
                del content, residual
                norm_indices, discrepancy = norm_matched_indices(feature)
                norm_discrepancy.append(discrepancy.cpu().numpy())
                layer_results = {key: [] for key in output_arrays}
                for variant in variants:
                    descriptors = readout(feature, angles, variant, norm_indices)
                    results = probe(descriptors, points, truth, valid, cycle_points)
                    for key, value in results.items():
                        layer_results[key].append(value)
                    del descriptors
                for key in output_arrays:
                    # [layer, head, variant, ...]
                    output_arrays[key].append(np.stack(layer_results[key], axis=1))
                del feature
        arrays = {key: np.stack(values) for key, values in output_arrays.items()}
        arrays.update(valid=valid.cpu().numpy(), truth=truth.cpu().numpy(), points=points.cpu().numpy(),
                      norm_discrepancy=np.stack(norm_discrepancy),
                      content_means=np.stack(content_means), pair_energies=np.stack(pair_energies),
                      pair_variances=np.stack(pair_variances),
                      variant_names=np.array([v[0] for v in variants]), layers=np.array(layers),
                      target_frames=np.array(metadata["sparse_frames"][1:]))
        np.savez_compressed(output, **arrays)
        (output.with_suffix(".json")).write_text(json.dumps({
            "video": name, "features": str(path), "feature_metadata": metadata,
            "variants": variants, "layers": layers, "seconds": time.monotonic()-started,
            "metric": "token-grid direct NN; visible points; 256x256 coordinates; no tracker or visibility prediction",
            "selection_annotation_access": metadata["transform"] == "real",
            "probe_source_sha256": source_hash,
            'anchor_features_override':args.anchor_features,
        }, indent=2))
        print(json.dumps({"video": name, "seconds": time.monotonic()-started,
                          "shape": list(arrays["errors"].shape),
                          "norm_match_max_error": float(np.max(arrays["norm_discrepancy"]))}), flush=True)


if __name__ == "__main__":
    main()
