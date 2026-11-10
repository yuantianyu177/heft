"""Test frozen identity readouts on independent textures and matched frame multisets."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from research_paired_identity import FRAMES, HEIGHT, NOMINAL, VARIANTS, WIDTH, Reader
from research_visibility_history import choose_patch


def angle_paths(kind):
    if kind == "arc":
        time = np.arange(FRAMES)
        phase = np.clip((time - 4) / 20, 0, 1) * np.pi / 2
        return [phase, -phase]
    # Each world visits the same ten angular configurations twice (mod pi).
    # Extra dwell on opposite semicircles equalizes counts, while the final
    # unwrapped angles are +3pi/2 and -3pi/2, which swap the identities.
    m = 10
    paths = []
    for direction in (1, -1):
        states = []
        for step in range(3 * m // 2 + 1):
            states.append(direction * step)
            if m // 2 < step < m:
                states.append(direction * step)
        assert len(states) == 2 * m
        assert sorted(Counter(x % m for x in states).values()) == [2] * m
        padded = [0] * 4 + states + [states[-1]] * 9
        assert len(padded) == FRAMES
        paths.append(np.array(padded) * np.pi / m)
    return paths


def render(background, foreground, index, kind):
    canvas = cv2.resize(background, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    _, _, raw = choose_patch(cv2.resize(foreground, (832, 480), interpolation=cv2.INTER_AREA))
    patch = cv2.resize(raw, (72, 72), interpolation=cv2.INTER_AREA)
    size = len(patch)
    ramp = np.minimum(np.arange(size) + 1, size - np.arange(size)).clip(0, 8) / 8
    alpha = np.minimum(ramp[None], ramp[:, None])[..., None]
    center = np.array([WIDTH // 2 + (index % 3 - 1) * 16, HEIGHT // 2])
    orientation = index % 4 * np.pi / 4
    radius = 108 if kind == "arc" else 96
    video, paths = [], []
    for phase in angle_paths(kind):
        angle = phase + orientation
        offset = np.rint(radius * np.stack((np.cos(angle), np.sin(angle)), -1)).astype(int)
        positions = center[None, None] + offset[:, None] * np.array([1, -1])[None, :, None]
        clip = np.repeat(canvas[None], FRAMES, 0)
        for t, centers in enumerate(positions):
            for x, y in centers:
                target = clip[t, y-size//2:y+size//2, x-size//2:x+size//2]
                target[:] = np.rint(target * (1 - alpha) + patch * alpha).astype(np.uint8)
        video.append(clip)
        paths.append(positions)
    assert np.array_equal(video[0][:5], video[1][:5])
    assert np.array_equal(video[0][24:], video[1][24:])
    assert np.array_equal(paths[0][-1], paths[1][-1, ::-1])
    same_multiset = (
        Counter(hashlib.sha256(frame).digest() for frame in video[0])
        == Counter(hashlib.sha256(frame).digest() for frame in video[1])
    )
    assert same_multiset == (kind == "braid")
    return np.stack(video), np.stack(paths), same_multiset


def geometric_anchors(positions):
    def margin(start, target):
        cost = np.linalg.norm(start[..., :, None, :] - target[..., None, :, :], axis=-1)
        return cost[..., 0, 1] + cost[..., 1, 0] - cost[..., 0, 0] - cost[..., 1, 1]

    return {
        "endpoints": margin(positions[:, 0], positions[:, -1]).tolist(),
        "adjacent_rgb": margin(positions[:, :-1], positions[:, 1:]).tolist(),
        "adjacent_native_times": margin(positions[:, NOMINAL[:-1]], positions[:, NOMINAL[1:]]).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    a = p.parse_args()
    device = f"cuda:{a.gpu}"
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    design = json.loads(a.design.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    reader = Reader(a.root, device)
    data = a.root / "pointodyssey/data/davis_records"
    offsets = np.array([(x, y) for y in (-16, 0, 16) for x in (-16, 0, 16)])
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for i in a.indices:
        bg, fg = design["scenes"][i]
        row = {"scene": bg, "patch": fg, "index": i, "seed": 11000 + i, "cases": {},
               "source_sha256": code_hash, "vae_sha256": reader.vae_hash, "human_annotations_opened": False}
        target = a.out / f"{i:02d}_{bg}.json"
        if target.exists():
            continue
        started = time.monotonic()
        frames = [np.load(data / f"{name}.video.npy", mmap_mode="r")[0] for name in (bg, fg)]
        for kind in ("arc", "braid"):
            video, centers, same_multiset = render(*frames, i, kind)
            grid = (centers[:, NOMINAL, :, None] + offsets + .5) / np.array([WIDTH, HEIGHT])
            latent = reader.encode(video)
            assert torch.equal(latent[0, :, :2], latent[1, :, :2])
            latent[:, :, 6:] = latent[:, :, 6:].mean(0)
            endpoint, adjacent, mixed = reader.run(latent, grid, "last", seed=row["seed"])
            scores = {}
            for rule in design["frozen_readouts"]:
                name, layer, head = rule["readout"], rule["layer"], rule["head"]
                variant = VARIANTS.index(name)
                key = f"l{layer:02d}h{head:02d}_{name}"
                scores[key] = {
                    "endpoint": endpoint[layer, variant, :, head].mean(-1).tolist(),
                    "adjacent": adjacent[layer, variant, :, head].tolist(),
                    "mixed": mixed[layer, variant, :, :, head].tolist(),
                }
            row["cases"][kind] = {
                "identical_image_multiset": same_multiset,
                "shared_endpoint_latents": True,
                "geometry": geometric_anchors(centers),
                "centers": centers.tolist(), "scores": scores,
            }
            print(bg, kind, f"{time.monotonic() - started:.1f}s", flush=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        temp.replace(target)
        print("DONE", target, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
