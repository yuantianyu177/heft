"""Test future dependence under fixed prefix pixels and encoder weights."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research_temporal_response import load_encoder

SCENES = (
    ("bike-packing", "goat"), ("blackswan", "judo"), ("bmx-trees", "libby"),
    ("breakdance", "loading"), ("camel", "mbike-trick"), ("car-roundabout", "motocross-jump"),
)
PERTURB = (3, 6, 10, 14, 18)
NOMINAL = np.arange(0, 25, 4)


def per_frame_norm(module, args, original):
    x = args[0]
    if x.ndim != 5:
        return original
    b, c, t, h, w = x.shape
    flat = x.transpose(1, 2).reshape(b * t, c, h, w)
    normalized = F.group_norm(flat, module.num_groups, module.weight, module.bias, module.eps)
    return normalized.reshape(b, t, c, h, w).transpose(1, 2).contiguous()


def image(root, name, width, height, device):
    first = np.load(root / f"data/davis_records/{name}.video.npy", mmap_mode="r")[0]
    pixels = cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(pixels).permute(2, 0, 1).to(device).float() / 127.5 - 1


def compare(left, right):
    difference = (left.float() - right.float()).abs()[0].transpose(0, 1).flatten(1)
    reference = left.float()[0].transpose(0, 1).flatten(1)
    return {
        "rms": difference.square().mean(1).sqrt().cpu().tolist(),
        "max_abs": difference.amax(1).cpu().tolist(),
        "relative_rms": (difference.square().mean(1).sqrt() / reference.std(1).clamp_min(1e-10)).cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--encoder", choices=("cogvideox", "wan"), required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--fp16", action="store_true")
    a = p.parse_args()
    device = f"cuda:{a.gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.float16 if a.fp16 else torch.float32
    vae, _, vae_hash, model_id = load_encoder(a.root, a.encoder, False, device, dtype)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    gn_modules = {name: layer for name, layer in vae.encoder.named_modules() if isinstance(layer, nn.GroupNorm)}
    if a.encoder == "cogvideox":
        assert gn_modules, "Expected GroupNorm layers in encoder"
        conditions = [("original", size) for size in (4, 8, 12)] + [("per_frame", size) for size in (4, 8, 12)]
    else:
        conditions = [("original", None)]

    def encode(x):
        return vae.encode(x.to(dtype).contiguous()).latent_dist.mode()

    a.out.mkdir(parents=True, exist_ok=True)
    for i in a.indices:
        base, other = SCENES[i]
        target = a.out / f"{a.encoder}_{base}.json"
        if target.exists():
            continue
        prefix = image(a.root, base, a.width, a.height, device)
        future = image(a.root, other, a.width, a.height, device)
        data = prefix[None, :, None].expand(1, -1, 25, -1, -1).clone()
        anchor, cases = {}, {}
        for mode, size in conditions:
            hooks = [layer.register_forward_hook(per_frame_norm) for layer in gn_modules.values()] if mode == "per_frame" else []
            label = f"{mode}_{size}" if size else mode
            if size:
                vae.num_sample_frames_batch_size = size
            try:
                static = encode(data)
                assert static.shape[2] == len(NOMINAL)
                anchor[label] = static
                changes = {}
                for t in PERTURB:
                    modified = data.clone()
                    modified[:, :, t] = future
                    latent = encode(modified)
                    changes[t] = compare(static, latent)
                # Independent matched-prefix check on a natural suffix switch.
                modified = data.clone()
                modified[:, :, 5:] = future[None, :, None]
                changed_suffix = encode(modified)
                short_prefix = encode(data[:, :, :5])
                cases[label] = {
                    "single_frame": changes,
                    "suffix_from_frame5": compare(static, changed_suffix),
                    "prefix_length5_vs_25": compare(static[:, :, :2], short_prefix),
                }
                anchor[label + "_suffix"] = changed_suffix
            finally:
                for hook in hooks:
                    hook.remove()
        chunks = {
            f"{mode}_{size}_vs_8_{clip}": compare(
                anchor[f"{mode}_{size}" + ("_suffix" if clip == "suffix" else "")],
                anchor[f"{mode}_8" + ("_suffix" if clip == "suffix" else "")],
            )
            for mode, size in conditions if size in (4, 12) for clip in ("static", "suffix")
        }
        record = {
            "encoder": a.encoder, "model_id": model_id, "scene": base, "replacement": other,
            "group_norm_modules": list(gn_modules), "nominal": NOMINAL.tolist(),
            "dtype": str(dtype), "size": [a.height, a.width], "tf32_allowed": True,
            "source_sha256": code_hash, "vae_code_sha256": vae_hash,
            "encoder_implementation": inspect.getfile(vae.__class__), "human_annotations_opened": False,
            "future_cases": cases, "chunk_switch": chunks,
        }
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
        print("DONE", target, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
