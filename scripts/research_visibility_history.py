"""Controlled appearance/disappearance with identical visible images and paths."""
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
from research_rope import wan_angles

SCENES = (
    "bike-packing", "blackswan", "bmx-trees", "breakdance", "camel",
    "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog", "dogs-jump", "drift-chicane",
)
PATCHES = (
    "goat", "judo", "libby", "loading", "mbike-trick", "motocross-jump",
    "parkour", "pigs", "scooter-black", "shooting", "soapbox", "gold-fish",
)
WIDTH, HEIGHT, FRAMES, QUERY = 832, 480, 25, 8
PHASES = [[0, 4, 8, 12, 16, 20, 24]] + [[0] + list(range(k, 25, 4)) for k in [1, 2, 3]]
DELAYS = {"nominal": 0., "quarter": .25, "integer": 1., "temporal": 1.25, "reverse": -.75}


class CaptureDone(Exception):
    pass


class Extractor:
    def __init__(self, root, device):
        from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

        self.device = device
        config = json.loads((root / "models/Wan2.1-T2V-1.3B-Diffusers.json").read_text())
        path = config["path"]
        cls, self.vae_hash = native_vae_class()
        self.vae = cls.from_pretrained(
            path, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device).eval()
        self.model = WanTransformer3DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device).eval()
        self.prompt = load_file(str(root / "models/wan_empty_prompt.safetensors"))["prompt_embeds"].to(device)
        scheduler = UniPCMultistepScheduler.from_pretrained(path, subfolder="scheduler", local_files_only=True)
        scheduler.set_timesteps(50, device=device)
        self.sigma = float(scheduler.sigmas[49])
        self.step = scheduler.timesteps[49]
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(1, 16, 1, 1, 1)
        angle = wan_angles(7, 30, 52, device=device)[None].expand(4, -1, -1, -1, -1).clone()
        frequency = 10000.**(-torch.arange(22, device=device, dtype=torch.float64) / 22)
        timestamps = torch.tensor(PHASES, dtype=torch.float64, device=device) / 4
        angle[..., :22] = timestamps[:, :, None, None, None] * frequency
        rotation = torch.polar(torch.ones_like(angle), angle).reshape(4, 1, -1, 64)
        self.model.rope.forward = lambda hidden_states: rotation
        self.model.blocks[18].attn1._heft_capture = self.capture
        noise = torch.randn((1, 16, FRAMES, 60, 104), generator=torch.Generator().manual_seed(42)).to(device)
        self.noise = torch.cat([noise[:, :, indices] for indices in PHASES])
        self.keys = None

    def capture(self, *, key, **kwargs):
        self.keys = key[:, 5].reshape(4, 7, 30, 52, 128).permute(0, 1, 4, 2, 3).float()
        raise CaptureDone

    @staticmethod
    def interleave(features):
        # features [phase, latent time, channel, height, width]
        dense = torch.empty((FRAMES, *features.shape[2:]), device=features.device, dtype=features.dtype)
        for phase, indices in enumerate(PHASES):
            dense[indices] = features[phase]
        dense[0] = features[:, 0].mean(0)
        return dense

    def __call__(self, frames):
        inputs = torch.from_numpy(frames).to(self.device, dtype=torch.bfloat16).permute(3, 0, 1, 2)[None] / 127.5 - 1
        latent = []
        for phase in range(4):
            pad = 0 if phase == 0 else 4 - phase
            x = torch.cat((inputs[:, :, :1].expand(-1, -1, pad, -1, -1), inputs), dim=2)
            x = torch.cat((x, x[:, :, -1:].expand(-1, -1, 29 - x.shape[2], -1, -1)), dim=2)
            latent.append((self.vae.encode(x).latent_dist.mode()[:, :, :7].float() - self.mean) / self.std)
        latent = torch.cat(latent)
        assert torch.equal(latent[:, :, 0], latent[:1, :, 0].expand_as(latent[:, :, 0]))
        noisy = ((1 - self.sigma) * latent + self.sigma * self.noise).to(torch.bfloat16)
        self.keys = None
        try:
            self.model(
                noisy, timestep=self.step.expand(4),
                encoder_hidden_states=self.prompt.expand(4, -1, -1).contiguous(), return_dict=False,
            )
        except CaptureDone:
            pass
        assert self.keys is not None
        return {
            "latent": self.interleave(latent.permute(0, 2, 1, 3, 4)),
            "key": self.interleave(self.keys),
        }


def read_image(root, name):
    first = np.load(root / f"data/davis_records/{name}.video.npy", mmap_mode="r")[0]
    return cv2.resize(first, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def choose_patch(image):
    size = 96
    candidates = [
        (x, y, image[y:y+size, x:x+size])
        for y in (96, 168, 240, 312)
        for x in (128, 248, 368, 488, 608)
    ]

    def contrast(item):
        value = item[2].astype(np.float32)
        return float(np.abs(np.diff(value, axis=0)).mean() + np.abs(np.diff(value, axis=1)).mean())

    return max(candidates, key=contrast)


def scene(background, patch, speed):
    size = patch.shape[0]
    time = np.arange(FRAMES)
    centers = np.array([352, 224])[None] + time[:, None] * np.array(speed)[None]
    offsets = np.linspace(-27, 27, 4)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    points = centers[:, None] + np.stack((xx, yy), -1).reshape(1, -1, 2)
    points = (points + .5) / np.array([WIDTH, HEIGHT])
    r = np.arange(size)
    ramp = np.minimum(r + 1, size - r).clip(0, 6) / 6
    alpha = np.minimum(ramp[:, None], ramp[None, :])[..., None]
    background_video = np.repeat(background[None], FRAMES, axis=0)
    visible = background_video.copy()
    for i, (x, y) in enumerate(centers):
        x, y = int(x - size // 2), int(y - size // 2)
        area = visible[i, y:y+size, x:x+size]
        assert area.shape == patch.shape
        area[:] = np.rint(area * (1 - alpha) + patch * alpha).astype(np.uint8)
    masks = {
        "always": time >= 0, "absent": time < 0, "reveal8": time >= 8,
        "reveal6": time >= 6, "hide12": time < 12,
    }
    return {
        name: np.where(mask[:, None, None, None], visible, background_video)
        for name, mask in masks.items()
    }, points


def at(feature, time):
    time = max(0, min(time, len(feature) - 1))
    low = int(time)
    high = min(low + 1, len(feature) - 1)
    return (1 - (time-low)) * feature[low] + (time-low) * feature[high]


def retime(feature, delay):
    return torch.stack([feature[0]] + [at(feature, t + delay) for t in range(1, len(feature))])


def sample(feature, points):
    grid = torch.as_tensor(points, device=feature.device, dtype=torch.float32)[:, None] * 2 - 1
    return F.grid_sample(feature, grid, align_corners=False, padding_mode="border")[:, :, 0].transpose(1, 2)


def project_visibility(features, points, kind, delay):
    vectors = {name: sample(retime(maps[kind], delay), points) for name, maps in features.items()}
    if kind == "key":
        vectors = {name: F.normalize(value, dim=-1) for name, value in vectors.items()}
    background = vectors["absent"]
    direction = vectors["always"] - background
    energy = direction.square().sum(-1).clamp_min(1e-12)
    return {
        name: (((value - background) * direction).sum(-1) / energy).mean(1).cpu().tolist()
        for name, value in vectors.items() if name not in ("always", "absent")
    }


def match(features, points):
    keys = {name: maps["key"] for name, maps in features.items()}
    gt = points[QUERY+1:QUERY+9]
    previous = points[QUERY-2] * .25 + points[QUERY-1] * .75
    result = {}
    for name in ("always", "reveal6", "reveal8"):
        descriptors = {
            mode: sample(at(keys[name], QUERY + delay)[None], points[QUERY:QUERY+1])[0]
            for mode, delay in DELAYS.items()
        }
        descriptors["spatial_oracle"] = sample(keys[name][QUERY:QUERY+1], previous[None])[0]
        modes = list(descriptors)
        query = F.normalize(torch.stack(list(descriptors.values())), dim=-1)
        pck = {}
        for target in ("always", name):
            volume = retime(keys[target], 1.25)[QUERY+1:QUERY+9]
            volume = F.normalize(F.interpolate(volume, size=(60, 104), mode="bilinear", align_corners=False), dim=1)
            index = torch.einsum("spc,tchw->stphw", query, volume).flatten(-2).argmax(-1).cpu().numpy()
            pred = np.stack(((index % 104 + .5) / 104, (index // 104 + .5) / 60), -1)
            distance = np.linalg.norm((pred - gt) * 256, axis=-1)
            pck[target] = dict(zip(modes, ((distance[..., None] < [1, 2, 4, 8, 16]).mean((1, 2, 3)) * 100).tolist()))
        result[name] = pck
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    extractor = Extractor(args.root, device)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for i in args.indices:
        target = args.out / f"{SCENES[i]}.json"
        if target.exists():
            print("EXISTS", target.name, flush=True)
            continue
        started = time.monotonic()
        background = read_image(args.root, SCENES[i])
        patch_x, patch_y, patch = choose_patch(read_image(args.root, PATCHES[i]))
        row = {
            "background": SCENES[i], "patch_source": PATCHES[i], "patch_xy": [patch_x, patch_y],
            "source_sha256": code_hash, "vae_sha256": extractor.vae_hash, "sigma": extractor.sigma,
            "human_annotations_opened": False, "records": {},
        }
        for motion, speed in [("stationary", (0, 0)), ("moving", [(4, 0), (-4, 0), (0, 4), (0, -4)][i % 4])]:
            clips, points = scene(background, patch, speed)
            assert np.array_equal(clips["always"][8:], clips["reveal8"][8:])
            assert np.array_equal(clips["always"][:12], clips["hide12"][:12])
            features = {}
            for name, frames in clips.items():
                features[name] = extractor(frames)
                print(SCENES[i], motion, name, f"{time.monotonic() - started:.1f}s", flush=True)
            row["records"][motion] = {
                "pck": match(features, points),
                "visibility_projection": {
                    kind: {label: project_visibility(features, points, kind, DELAYS[label])
                           for label in ("nominal", "quarter", "temporal")}
                    for kind in ("latent", "key")
                },
                "speed": speed,
            }
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(row, indent=2) + "\n")
        temp.replace(target)
        print("DONE", target.name, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
