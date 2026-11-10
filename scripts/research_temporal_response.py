"""Estimate frozen VAE motion-response kernels and test unseen trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from research_causal_denoising import native_vae_class

FRAMES = 25
CALIBRATION = (
    "drift-straight", "gold-fish", "horsejump-high", "india",
    "kite-surf", "lab-coat", "paragliding-launch",
)
VALIDATION = (
    "bike-packing", "blackswan", "bmx-trees", "breakdance",
    "car-roundabout", "cows", "dance-twirl", "parkour",
)


def design_trajectories():
    """Positions are in input pixels, normalized to unit maximal excursion."""
    t = np.arange(FRAMES, dtype=np.float64)
    zero = np.zeros_like(t)
    curves = {
        "constant_x": np.stack((t, zero), -1),
        "constant_y": np.stack((zero, t), -1),
        "accelerate": np.stack((t**2, zero), -1),
        "stop14": np.stack((np.minimum(t, 14), zero), -1),
        "reverse14": np.stack((14 - np.abs(t - 14), zero), -1),
        "sine5": np.stack((np.sin(2 * np.pi * t / 5), zero), -1),
        "sine11": np.stack((np.sin(2 * np.pi * t / 11), zero), -1),
    }
    for turn in (11, 14, 15, 18):
        before = np.minimum(t, turn)
        after = np.maximum(t - turn, 0)
        curves[f"corner{turn}_xy"] = np.stack((before, after), -1)
        curves[f"corner{turn}_yx"] = np.stack((after, before), -1)
    for gap in (2, 3, 4, 6):
        pulse = np.clip(t / 3, 0, 1) * (t < 16 - gap)
        curves[f"history_gap{gap}_x"] = np.stack((pulse, zero), -1)
        curves[f"history_gap{gap}_y"] = np.stack((zero, pulse), -1)
    return {name: p / np.linalg.norm(p, axis=1).max() for name, p in curves.items()}


def load_encoder(root, name, aligned, device, dtype):
    from diffusers import AutoencoderKLCogVideoX, AutoencoderKLWan

    if name == "wan":
        cls, code_hash = native_vae_class()
        if aligned:
            cls = AutoencoderKLWan
        config_path = root / "models/wan-vae.json"
        width, kwargs = 832, {}
    else:
        import inspect

        cls = AutoencoderKLCogVideoX
        code_hash = hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
        config_path = root / "models/CogVideoX-2b-vae.json"
        width = 720
        kwargs = {"temporal_compression_ratio": 1} if aligned else {}
    config = json.loads(config_path.read_text())
    vae = cls.from_pretrained(
        config["path"], subfolder="vae", local_files_only=True, torch_dtype=dtype, **kwargs
    ).to(device).eval()
    return vae, width, code_hash, config["model_id"]


class Probe:
    def __init__(self, vae, image, width, device, dtype, aligned, height=480):
        self.vae = vae
        self.width, self.height = width, height
        self.device, self.dtype = device, dtype
        self.nominal = np.arange(0, FRAMES, 1 if aligned else 4)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
        self.image = torch.from_numpy(image).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
        yy, xx = torch.meshgrid(
            (torch.arange(self.height, device=device) + .5) * 2 / self.height - 1,
            (torch.arange(self.width, device=device) + .5) * 2 / self.width - 1,
            indexing="ij",
        )
        self.pixel_grid = torch.stack((xx, yy), -1)[None]
        self.pixel_scale = torch.tensor([2 / self.width, 2 / self.height], device=device)
        yy, xx = torch.meshgrid(
            torch.linspace(.3, .7, 16, device=device),
            torch.linspace(.3, .7, 16, device=device), indexing="ij",
        )
        self.sample_grid = torch.stack((xx, yy), -1).reshape(1, 1, -1, 2) * 2 - 1
        std = getattr(vae.config, "latents_std", None)
        self.scale = torch.tensor(std, device=device)[None, :, None, None] if std else 1.

    def warp(self, offsets):
        offsets = torch.as_tensor(offsets, device=self.device, dtype=torch.float32)
        grid = self.pixel_grid - offsets[:, None, None, :] * self.pixel_scale
        return F.grid_sample(
            self.image.expand(len(offsets), -1, -1, -1), grid,
            padding_mode="reflection", align_corners=False,
        )

    def encode(self, clips):
        b = clips.shape[0]
        x = clips.permute(0, 2, 1, 3, 4).to(self.dtype).contiguous()
        z = self.vae.encode(x).latent_dist.mode().float()
        assert z.shape[2] == len(self.nominal), (z.shape, self.nominal)
        maps = z.permute(0, 2, 1, 3, 4).flatten(0, 1) / self.scale
        features = self.sample(maps).reshape(b, len(self.nominal), -1, maps.shape[1])
        return features, maps.reshape(b, len(self.nominal), *maps.shape[1:])

    def sample(self, maps):
        grid = self.sample_grid.expand(len(maps), -1, -1, -1)
        return F.grid_sample(maps, grid, align_corners=False)[:, :, 0].transpose(1, 2)

    def uniform(self, epsilon):
        # Each direction uses independently encoded uniformly translated clips.
        banks = []
        for axis in np.eye(2):
            images = self.warp(np.stack((axis, -axis)) * epsilon)
            clips = images[:, None].expand(-1, FRAMES, -1, -1, -1)
            values, _ = self.encode(clips)
            banks.append((values[0] - values[1]) / (2 * epsilon))
        return torch.stack(banks, -1).cpu().numpy()

    def impulses(self, epsilon, progress=False):
        axis_grads = []
        for direction, axis in enumerate(np.eye(2)):
            images = self.warp(np.stack((axis, -axis)) * epsilon)
            clips = self.image[:, None].expand(2, FRAMES, -1, -1, -1).clone()
            derivatives = []
            for t in range(FRAMES):
                clips[:, t] = images
                values, _ = self.encode(clips)
                derivatives.append(((values[0] - values[1]) / (2 * epsilon)).cpu())
                clips[:, t] = self.image[0]
                if progress and (t + 1) % 5 == 0:
                    print(f"impulse axis={direction} completed {t + 1}/{FRAMES}", flush=True)
            axis_grads.append(torch.stack(derivatives, 1))
        # [latent time, input time, spatial samples, channels, input direction]
        return torch.stack(axis_grads, -1).numpy()

    def register(self, maps, reference, max_excursion):
        """Direct registration using the original latent-grid objective in 2D."""
        limit = int(np.ceil(max_excursion)) + 2
        y, x = torch.meshgrid(
            torch.arange(-limit, limit + 1, device=self.device, dtype=torch.float32),
            torch.arange(-limit, limit + 1, device=self.device, dtype=torch.float32), indexing="ij",
        )
        coarse = torch.stack((x, y), -1).reshape(1, -1, 2).expand(len(maps), -1, -1)

        def errors(candidates):
            pieces = []
            for offsets in candidates.split(128, dim=1):
                grids = self.sample_grid + offsets[:, :, None, :] * self.pixel_scale
                sampled = F.grid_sample(maps, grids, align_corners=False).permute(0, 2, 3, 1)
                pieces.append((sampled - reference[:, None]).square().mean((-1, -2)))
            return torch.cat(pieces, 1)

        index = torch.arange(len(maps), device=self.device)
        center = coarse[index, errors(coarse).argmin(1)]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 9, device=self.device),
            torch.linspace(-1, 1, 9, device=self.device), indexing="ij",
        )
        fine = center[:, None] + torch.stack((x, y), -1).reshape(1, -1, 2)
        cost = errors(fine)
        best = cost.argmin(1)
        return fine[index, best].cpu().numpy(), cost[index, best].cpu().numpy()


def project(uniform, displacement):
    basis = uniform.reshape(len(uniform), -1, 2).astype(np.float64)
    delta = displacement.reshape(len(uniform), -1).astype(np.float64)
    gram = np.einsum("jdi,jdk->jik", basis, basis)
    rhs = np.einsum("jdi,jd->ji", basis, delta)
    positions = np.linalg.solve(gram, rhs[..., None])[..., 0]
    residual = np.linalg.norm(delta - np.einsum("jdi,ji->jd", basis, positions), axis=1)
    return positions, residual / np.maximum(np.linalg.norm(delta, axis=1), 1e-12)


def validation(probe, uniform, amplitudes, register_amplitude):
    curves = design_trajectories()
    reference, _ = probe.encode(probe.image[:, None].expand(-1, FRAMES, -1, -1, -1))
    records = []
    for name, curve in curves.items():
        for amplitude in amplitudes:
            path = curve * amplitude
            clips = probe.warp(np.concatenate((path, -path))).reshape(
                2, FRAMES, 3, probe.height, probe.width,
            )
            features, maps = probe.encode(clips)
            displacement = ((features[0] - features[1]) / 2).cpu().numpy()
            measured, residual = project(uniform, displacement)
            row = {
                "curve": name, "amplitude": amplitude, "path": path.tolist(),
                "projected_displacement": measured.tolist(),
                "orthogonal_response_fraction": residual.tolist(),
            }
            if amplitude == register_amplitude:
                fits = [probe.register(m, reference[0], amplitude) for m in maps]
                row["registration_displacement"] = ((fits[0][0] - fits[1][0]) / 2).tolist()
                row["registration_plus"] = fits[0][0].tolist()
                row["registration_minus"] = fits[1][0].tolist()
                row["registration_mse"] = [cost.tolist() for _, cost in fits]
            records.append(row)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--encoder", choices=["wan", "cogvideox"], default="wan")
    p.add_argument("--aligned", action="store_true")
    p.add_argument("--mode", choices=["impulse", "validation"], required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--epsilon", type=float, default=.5)
    p.add_argument("--amplitudes", type=float, nargs="+", default=[1., 4., 12.])
    p.add_argument("--register-amplitude", type=float, default=12.)
    precision = p.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{a.gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cudnn.allow_tf32 = a.tf32
    dtype = torch.bfloat16 if a.bf16 else (torch.float16 if a.fp16 else torch.float32)
    vae, width, code_hash, model_id = load_encoder(a.root, a.encoder, a.aligned, device, dtype)
    width = width if a.width is None else a.width
    if a.height % 8 or width % 8:
        p.error("image height and width must be divisible by 8")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    began = time.monotonic()
    for name in a.names.split(","):
        key = f"{a.encoder}_{'aligned' if a.aligned else 'native'}_{a.mode}_{name}"
        path = a.out / f"{key}.json"
        if path.exists():
            print(f"EXISTS {path.name}", flush=True)
            continue
        image = np.load(a.root / f"data/davis_records/{name}.video.npy", mmap_mode="r")[0]
        probe = Probe(vae, image, width, device, dtype, a.aligned, a.height)
        uniform = probe.uniform(a.epsilon)
        record = {
            "image": name, "encoder": a.encoder, "frame_aligned": a.aligned,
            "mode": a.mode, "epsilon_input_pixels": a.epsilon,
            "input_size": [a.height, width], "precision": str(dtype),
            "cudnn_tf32": a.tf32,
            "nominal_times": probe.nominal.tolist(), "model_id": model_id,
            "source_sha256": source_hash, "vae_code_sha256": code_hash,
            "human_annotations_opened": False,
        }
        if a.mode == "impulse":
            grad = probe.impulses(a.epsilon, progress=True)
            gsum = grad.sum(1)
            shape = (len(probe.nominal), -1)
            record["superposition_relative_error"] = (
                np.linalg.norm((gsum - uniform).reshape(*shape), axis=1)
                / np.linalg.norm(uniform.reshape(*shape), axis=1)
            ).tolist()
            is_future = np.arange(FRAMES)[None] > probe.nominal[:, None]
            energy = (grad**2).sum((2, 3, 4))
            record["future_energy_fraction"] = (
                (energy * is_future).sum(1) / energy.sum(1)
            ).tolist()
            np.savez_compressed(a.out / f"{key}.npz", gradient=grad, uniform=uniform)
        else:
            record["amplitudes"] = a.amplitudes
            record["records"] = validation(probe, uniform, a.amplitudes, a.register_amplitude)
        record["elapsed_since_launch_s"] = time.monotonic() - began
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(path)
        print(f"DONE {path.name} {record['elapsed_since_launch_s']:.1f}s", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
