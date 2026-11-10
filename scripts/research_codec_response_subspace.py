"""Intervene on motion components inside a frozen video codec."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from research_temporal_response import FRAMES, Probe, load_encoder, project


def load_codec(root, encoder, device):
    if encoder != "wan":
        return load_encoder(root, encoder, False, device, torch.float32)
    source = root / "provenance/official_autoencoder_kl_wan_v0p35p0.py"
    name = "diffusers.models.autoencoders._heft_official_motion_codec"
    module = types.ModuleType(name)
    module.__package__ = "diffusers.models.autoencoders"
    module.__file__ = str(source)
    sys.modules[name] = module
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    model = json.loads((root / "models/wan-vae.json").read_text())
    vae = module.AutoencoderKLWan.from_pretrained(
        model["path"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    return vae, 832, hashlib.sha256(source.read_bytes()).hexdigest(), model["model_id"]


def trajectories(kernel):
    # Remove the component visible to the calibration-image position operator.
    t = np.arange(FRAMES)
    x = np.zeros((FRAMES, 2))
    x[:, 0] = np.sin(2 * np.pi * t / 5)
    y = np.zeros_like(x)
    y[:, 1] = np.sin(2 * np.pi * t / 7)
    ramp = np.zeros_like(x)
    ramp[:, 0] = t / t[-1]
    a = kernel.transpose(0, 2, 1, 3).reshape(len(kernel) * 2, FRAMES * 2)
    anchor = np.eye(FRAMES * 2)[:2]
    a = np.concatenate((a, anchor), axis=0)
    pinv = np.linalg.pinv(a, rcond=1e-7)

    def null(path):
        q = path.reshape(-1)
        return (q - pinv @ (a @ q)).reshape(FRAMES, 2)

    paths = {"ramp_x": ramp, "sine5_x": x, "null_sine5_x": null(x), "null_sine7_y": null(y)}
    return {name: p / np.linalg.norm(p, axis=-1).max() for name, p in paths.items()}


class CodecExperiment:
    def __init__(self, probe, epsilon):
        self.probe = probe
        self.epsilon = epsilon
        clips = probe.image[:, None].expand(-1, FRAMES, -1, -1, -1)
        _, static = probe.encode(clips)
        self.static_rgb = probe.sample(self.decode(static[0]))
        latent_samples, latent_maps, rgb_samples = [], [], []
        for axis in np.eye(2):
            images = probe.warp(np.stack((axis, -axis)) * epsilon)
            values, z = probe.encode(images[:, None].expand(-1, FRAMES, -1, -1, -1))
            latent_samples.append(((values[0] - values[1]) / (2 * epsilon)).cpu().numpy())
            latent_maps.append((z[0] - z[1]) / (2 * epsilon))
            rgb = [probe.sample(self.decode(latents)) for latents in z]
            rgb_samples.append(((rgb[0] - rgb[1]) / (2 * epsilon)).cpu().numpy())
        self.latent_basis = np.stack(latent_samples, -1)
        self.map_basis = torch.stack(latent_maps, -1)
        self.rgb_basis = np.stack(rgb_samples, -1)

    def decode(self, latents):
        raw = (latents * self.probe.scale).permute(1, 0, 2, 3)[None].contiguous()
        rgb = self.probe.vae.decode(raw).sample[0].float().permute(1, 0, 2, 3)
        assert len(rgb) == FRAMES, rgb.shape
        return rgb

    def read_rgb(self, center, direction, do_register, amplitude):
        samples = []
        registrations = []
        for sign in (1, -1):
            maps = self.decode(center + sign * direction)
            samples.append(self.probe.sample(maps).cpu().numpy())
            if do_register:
                registrations.append(self.probe.register(maps, self.static_rgb, amplitude)[0])
        projected, residual = project(self.rgb_basis, (samples[0] - samples[1]) / 2)
        output = {
            "projected_displacement": projected.tolist(),
            "orthogonal_response_fraction": residual.tolist(),
        }
        if registrations:
            output["registered_displacement"] = ((registrations[0] - registrations[1]) / 2).tolist()
        return output

    def run(self, path, amplitude, register):
        probe = self.probe
        clips = probe.warp(np.concatenate((path, -path))).reshape(
            2, FRAMES, 3, probe.height, probe.width,
        )
        values, z = probe.encode(clips)
        difference = (z[0] - z[1]) / 2
        center = (z[0] + z[1]) / 2
        measured, residual = project(self.latent_basis, ((values[0] - values[1]) / 2).cpu().numpy())
        position = torch.as_tensor(measured, device=probe.device, dtype=self.map_basis.dtype)
        parallel = torch.einsum("jchwk,jk->jchw", self.map_basis, position)
        orthogonal = difference - parallel
        checked, _ = project(self.latent_basis, probe.sample(orthogonal).cpu().numpy())
        return {
            "path": path.tolist(),
            "latent_projected_displacement": measured.tolist(),
            "latent_orthogonal_response_fraction": residual.tolist(),
            "residual_position_projection_max_pixel": float(np.max(np.abs(checked))),
            "full": self.read_rgb(center, difference, register, amplitude),
            "position_component": self.read_rgb(center, parallel, register, amplitude),
            "residual_component": self.read_rgb(center, orthogonal, register, amplitude),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--encoder", choices=["wan", "cogvideox"], default="wan")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--names", required=True)
    p.add_argument("--epsilon", type=float, default=.5)
    p.add_argument("--amplitudes", type=float, nargs="+", default=[2., 6.])
    p.add_argument("--register-amplitude", type=float, default=6.)
    p.add_argument("--no-tf32", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{a.gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cudnn.allow_tf32 = not a.no_tf32
    with np.load(a.kernel) as k:
        assert len(k["per_image"]) == 7, "Freeze the seven-image operator before testing null paths."
        kernel = k["kernel"]
        nominal = k["nominal"]
    curves = trajectories(kernel)
    kernel_hash = hashlib.sha256(a.kernel.read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    vae, width, vae_hash, model_id = load_codec(a.root, a.encoder, device)
    for name in a.names.split(","):
        target = a.out / f"codec_subspace_{a.encoder}_{name}.json"
        if target.exists():
            print("EXISTS", target.name, flush=True)
            continue
        began = time.monotonic()
        image = np.load(a.root / f"data/davis_records/{name}.video.npy", mmap_mode="r")[0]
        experiment = CodecExperiment(
            Probe(vae, image, width, device, torch.float32, False), a.epsilon,
        )
        records = []
        for curve, path in curves.items():
            for amplitude in a.amplitudes:
                p = path * amplitude
                record = experiment.run(p, amplitude, amplitude == a.register_amplitude)
                record.update({
                    "curve": curve, "amplitude": amplitude,
                    "calibration_predicted_latent_positions": np.einsum("jtab,tb->ja", kernel, p).tolist(),
                })
                records.append(record)
                print(name, curve, amplitude, f"{time.monotonic() - began:.1f}s", flush=True)
        payload = {
            "image": name, "encoder": a.encoder, "model_id": model_id, "nominal_times": nominal.tolist(),
            "kernel_sha256": kernel_hash, "script_sha256": code_hash, "vae_source_sha256": vae_hash,
            "input_size": [480, width], "human_annotations_opened": False,
            "epsilon": a.epsilon, "cudnn_tf32": not a.no_tf32, "records": records,
        }
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(target)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
