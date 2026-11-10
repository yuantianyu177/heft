"""Read out identity in counterfactual clips with pixel-identical endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from research_causal_denoising import native_vae_class
from research_rope import frequency_pairs, pair_channels, phase_readout, wan_angles
from research_visibility_history import PATCHES, SCENES, choose_patch

WIDTH, HEIGHT, FRAMES = 640, 384, 33
FH, FW = HEIGHT // 16, WIDTH // 16
NOMINAL = np.arange(0, FRAMES, 4)
BACKGROUND = SCENES + ("drift-straight", "goat", "horsejump-high", "india")
FOREGROUND = PATCHES + ("car-shadow", "bmx-trees", "blackswan", "dogs-jump")
VARIANTS = ("kk_full", "kk_low15", "kk_content", "qk_cos", "kq_cos", "qk_logit", "kq_logit")
NOISE_SETTINGS = ("last", "sigma035", "sigma065")


def render(root, i):
    def first(name):
        return np.load(root / f"data/davis_records/{name}.video.npy", mmap_mode="r")[0]

    canvas = cv2.resize(first(BACKGROUND[i]), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    patch_image = cv2.resize(first(FOREGROUND[i]), (832, 480), interpolation=cv2.INTER_AREA)
    px, py, patch = choose_patch(patch_image)
    patch = cv2.resize(patch, (72, 72), interpolation=cv2.INTER_AREA)
    size = len(patch)
    r = np.arange(size)
    edge = np.minimum(r + 1, size - r).clip(0, 8) / 8
    alpha = np.minimum(edge[:, None], edge[None, :])[..., None]
    progression = np.clip((np.arange(FRAMES) - 4) / 20, 0, 1)
    orientation = (i % 4) * np.pi / 4
    center = np.array([WIDTH // 2 + (i % 3 - 1) * 16, HEIGHT // 2])
    video, paths = [], []
    for turn in (1, -1):
        angle = orientation + turn * progression * np.pi / 2
        offset = np.rint(108 * np.stack((np.cos(angle), np.sin(angle)), -1)).astype(int)
        positions = center[None, None] + offset[:, None] * np.array([1, -1])[None, :, None]
        clip = np.repeat(canvas[None], FRAMES, 0)
        for t, centers in enumerate(positions):
            for x, y in centers:
                target = clip[t, y-size//2:y+size//2, x-size//2:x+size//2]
                assert target.shape == patch.shape
                target[:] = np.rint(target * (1 - alpha) + patch * alpha).astype(np.uint8)
        video.append(clip)
        paths.append(positions)
    assert np.array_equal(video[0][:5], video[1][:5])
    assert np.array_equal(video[0][24:], video[1][24:])
    assert np.array_equal(paths[0][0], paths[1][0])
    assert np.array_equal(paths[0][-1], paths[1][-1, ::-1])
    # At every time, point 0 and point 1 correspond to different objects.
    return np.stack(video), np.stack(paths), [px, py]


def sample(feature, grid):
    # feature [world, head, latent time, H, W, channel], grid [world, time, object, point, xy]
    b, h, t, _, _, d = feature.shape
    _, _, objects, points, _ = grid.shape
    maps = feature.permute(0, 1, 2, 5, 3, 4).reshape(b*h*t, d, FH, FW)
    coords = grid[:, None].expand(-1, h, -1, -1, -1, -1).reshape(b*h*t, objects, points, 2)
    values = F.grid_sample(maps, coords, align_corners=False, padding_mode="border")
    return values.reshape(b, h, t, d, objects, points).permute(0, 1, 2, 4, 5, 3)


def correlations(source, target):
    endpoint = torch.einsum("bhapd,bhcpd->bhacp", source[:, :, 0], target[:, :, -1])
    adj = torch.einsum("bhtapd,bhtcpd->bhtacp", source[:, :, :-1], target[:, :, 1:])
    # Map endpoint preferences for features taken from different worlds.
    mixed = torch.einsum("xhapd,yhcpd->xyhacp", source[:, :, 0], target[:, :, -1])
    return endpoint.cpu().numpy(), adj.mean(-1).cpu().numpy(), mixed.mean(-1).cpu().numpy()


class Reader:
    def __init__(self, root, device):
        from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

        self.device = device
        path = json.loads((root / "models/Wan2.1-T2V-1.3B-Diffusers.json").read_text())["path"]
        cls, self.vae_hash = native_vae_class()
        self.vae = cls.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        self.model = WanTransformer3DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device).eval()
        self.prompt = load_file(str(root / "models/wan_empty_prompt.safetensors"))["prompt_embeds"].to(device)
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(1, 16, 1, 1, 1)
        scheduler = UniPCMultistepScheduler.from_pretrained(path, subfolder="scheduler", local_files_only=True)
        scheduler.set_timesteps(50, device=device)
        self.noises = {
            "last": (float(scheduler.sigmas[49]), float(scheduler.timesteps[49])),
            "sigma035": (.35, 350.),
            "sigma065": (.65, 650.),
        }
        self.angles = wan_angles(len(NOMINAL), FH, FW, device=device)
        self.keep = pair_channels(frequency_pairs(.15, 1.)).to(device)
        self.grid = None
        self.record = []
        for block in self.model.blocks:
            block.attn1._heft_capture = self.capture

    def capture(self, *, query, key, **kwargs):
        b, h, n, d = key.shape
        shape = (b, h, len(NOMINAL), FH, FW, d)
        key = key.reshape(shape)
        query = query.reshape(shape)
        k = sample(key.float(), self.grid)
        q = sample(query.float(), self.grid)
        keep = F.normalize(k[..., self.keep], dim=-1)
        content = F.normalize(sample(phase_readout(key, self.angles, temporal=0, spatial=0), self.grid), dim=-1)
        kn, qn = F.normalize(k, dim=-1), F.normalize(q, dim=-1)
        self.record.append([
            correlations(a, b) for a, b in (
                (kn, kn), (keep, keep), (content, content), (qn, kn), (kn, qn), (q, k), (k, q),
            )
        ])

    def encode(self, video):
        frames = torch.from_numpy(video).to(self.device, dtype=torch.bfloat16).permute(0, 4, 1, 2, 3) / 127.5 - 1
        result = torch.cat([self.vae.encode(clip[None]).latent_dist.mode() for clip in frames])
        return (result.float() - self.mean) / self.std

    def run(self, latent, coords, level, seed):
        sigma, step = self.noises[level]
        noise = torch.randn(latent[:1].shape, generator=torch.Generator().manual_seed(seed)).to(self.device)
        noisy = ((1 - sigma) * latent + sigma * noise).to(torch.bfloat16)
        self.grid = torch.as_tensor(coords, dtype=torch.float32, device=self.device) * 2 - 1
        self.record = []
        self.model(noisy, timestep=torch.full((2,), step, device=self.device),
                   encoder_hidden_states=self.prompt.expand(2, -1, -1).contiguous(), return_dict=False)
        return tuple(np.stack([[result[field] for result in layer] for layer in self.record])
                     for field in range(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    reader = Reader(args.root, device)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for i in args.indices:
        target = args.out / f"{i:02d}_{BACKGROUND[i]}.json"
        if target.exists():
            continue
        began = time.monotonic()
        video, centers, patch_xy = render(args.root, i)
        offsets = np.array([(x, y) for y in (-16, 0, 16) for x in (-16, 0, 16)])
        positions = (centers[:, NOMINAL, :, None] + offsets + .5) / np.array([WIDTH, HEIGHT])
        latent = reader.encode(video)
        assert latent.shape[2] == len(NOMINAL)
        assert torch.equal(latent[0, :, :2], latent[1, :, :2])
        fixed = latent.clone()
        fixed[:, :, 6:] = latent[:, :, 6:].mean(0)
        assert torch.equal(fixed[0, :, 6:], fixed[1, :, 6:])
        named = {}
        for tag, value in [("natural", latent), ("common_endpoints", fixed)]:
            for level in NOISE_SETTINGS:
                fields = reader.run(value, positions, level, seed=9000 + i)
                for field, array in zip(("endpoint", "adjacent", "mixed"), fields, strict=True):
                    named[f"{tag}_{level}_{field}"] = array
                print(i, tag, level, f"{time.monotonic() - began:.1f}s", flush=True)
        np.savez_compressed(target.with_suffix(".npz"), **named, centers=centers)
        row = {
            "index": i, "background": BACKGROUND[i], "patch": FOREGROUND[i], "patch_xy": patch_xy,
            "input_size": [HEIGHT, WIDTH], "n_frames": FRAMES, "nominal": NOMINAL.tolist(),
            "variants": VARIANTS, "noise": reader.noises, "seed": 9000 + i,
            "first5_and_last9_pixels_identical_between_worlds": True,
            "native_last_latent_relative_rms_difference": float(
                (latent[0, :, -1] - latent[1, :, -1]).square().mean().sqrt() / latent[0, :, -1].std()
            ),
            "source_sha256": source_hash, "vae_sha256": reader.vae_hash, "human_annotations_opened": False,
        }
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(row, indent=2) + "\n")
        temp.replace(target)
        print("DONE", target.name, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
