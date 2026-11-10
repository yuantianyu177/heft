"""Hold clip duration and endpoint pixels fixed while varying path and speed."""
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
from torch.nn import functional as F

from research_paired_identity import FH, FW, HEIGHT, WIDTH, Reader, correlations
from research_visibility_history import choose_patch

FRAMES = 73
NOMINAL = np.arange(0, FRAMES, 4)
CONDITIONS = (
    ("short_fast", 90, 20), ("short_slow", 90, 60),
    ("long_fast", 270, 20), ("long_slow", 270, 60),
)
HEADS = {9: (6,), 18: (1, 5), 27: (0,)}


def matched_loops(index):
    """Two Euler tours with identical orientation counts and angular speeds."""
    m = 10
    rng = np.random.default_rng(21000 + index)
    paths = []
    # Each undirected edge is used three times in the winding tour.
    # Alternating two/four uses preserves every vertex count in the other tour.
    for winding in (0, 1):
        adjacency = [[] for _ in range(m)]
        for i in range(m):
            j = (i + 1) % m
            forward = 2 if winding else 1 + i % 2
            reverse = 1 if winding else forward
            adjacency[i].extend([j] * forward)
            adjacency[j].extend([i] * reverse)
        for edges in adjacency:
            rng.shuffle(edges)
        stack, reverse_route = [0], []
        while stack:
            node = stack[-1]
            if adjacency[node]:
                stack.append(adjacency[node].pop())
            else:
                reverse_route.append(stack.pop())
        route = np.array(reverse_route[::-1])
        assert len(route) == 3 * m + 1
        assert route[0] == route[-1] == 0
        steps = (np.diff(route) + m // 2) % m - m // 2
        assert np.all(abs(steps) == 1) and steps.sum() == winding * m
        unwrapped = np.r_[0, np.cumsum(steps)]
        angles = unwrapped * np.pi / m * (1 if index % 2 else -1)
        paths.append(np.pad(angles, (20, FRAMES - 20 - len(angles)), mode="edge"))
    np.testing.assert_allclose(abs(np.diff(paths[0])), abs(np.diff(paths[1])), atol=1e-12, rtol=0)
    return paths


def render(background, foreground, index, degrees, duration, phase_paths=None):
    canvas = cv2.resize(background, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    _, _, patch = choose_patch(cv2.resize(foreground, (832, 480), interpolation=cv2.INTER_AREA))
    patch = cv2.resize(patch, (72, 72), interpolation=cv2.INTER_AREA)
    size = len(patch)
    r = np.arange(size)
    edge = np.minimum(r + 1, size - r).clip(0, 8) / 8
    alpha = np.minimum(edge[:, None], edge[None, :])[..., None]
    orientation = index % 4 * np.pi / 4
    center = np.array([WIDTH // 2 + (index % 3 - 1) * 16, HEIGHT // 2])
    video, paths = [], []
    if phase_paths is None:
        initial = (64 + 4 - duration) // 2
        progress = np.clip((np.arange(FRAMES) - initial) / duration, 0, 1)
        phase = progress * np.deg2rad(degrees)
        phase_paths = (phase, -phase)
    for phase in phase_paths:
        angle = orientation + phase
        offset = np.rint(108 * np.stack((np.cos(angle), np.sin(angle)), -1)).astype(int)
        positions = center[None, None] + offset[:, None] * np.array([1, -1])[None, :, None]
        clip = np.repeat(canvas[None], FRAMES, 0)
        for t, centers in enumerate(positions):
            for x, y in centers:
                target = clip[t, y-size//2:y+size//2, x-size//2:x+size//2]
                target[:] = np.rint(target * (1 - alpha) + patch * alpha).astype(np.uint8)
        video.append(clip)
        paths.append(positions)
    assert np.array_equal(video[0][:5], video[1][:5])
    assert np.array_equal(video[0][64:], video[1][64:])
    assert np.array_equal(paths[0][-1], paths[1][-1, ::-1])
    return np.stack(video), np.stack(paths)


class SpeedReader(Reader):
    def __init__(self, root, device):
        super().__init__(root, device)
        from research_paired_identity import sample

        self.sample = sample
        for layer, block in enumerate(self.model.blocks):
            block.attn1._heft_capture = self.bind(layer) if layer in HEADS else None

    def bind(self, layer):
        def capture(*, query, key, **kwargs):
            b, _, _, d = key.shape
            heads = HEADS[layer]
            shape = (b, len(heads), len(NOMINAL), FH, FW, d)
            k = self.sample(key[:, heads].reshape(shape).float(), self.grid)
            q = self.sample(query[:, heads].reshape(shape).float(), self.grid)
            kn = F.normalize(k, dim=-1)
            low = F.normalize(k[..., self.keep], dim=-1)
            self.record[layer] = [correlations(a, b) for a, b in (
                (kn, kn), (low, low), (q, k),
            )]
        return capture

    def read(self, latent, coords, seed):
        sigma, step = self.noises["last"]
        noise = torch.randn(latent[:1].shape, generator=torch.Generator().manual_seed(seed)).to(self.device)
        noisy = ((1 - sigma) * latent + sigma * noise).to(torch.bfloat16)
        self.grid = torch.as_tensor(coords, dtype=torch.float32, device=self.device) * 2 - 1
        self.record = {}
        self.model(noisy, timestep=torch.full((2,), step, device=self.device),
                   encoder_hidden_states=self.prompt.expand(2, -1, -1).contiguous(), return_dict=False)
        result = {}
        for layer, variants in self.record.items():
            for kind, (endpoint, adjacent, mixed) in zip(("kk_full", "kk_low15", "qk_logit"), variants, strict=True):
                for idx, head in enumerate(HEADS[layer]):
                    result[f"l{layer:02d}h{head:02d}_{kind}"] = {
                        "endpoint": endpoint[:, idx].mean(-1).tolist(),
                        "adjacent": adjacent[:, idx].tolist(),
                        "mixed": mixed[:, :, idx].tolist(),
                    }
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--matched-loops", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    device = f"cuda:{a.gpu}"
    torch.cuda.set_device(device)
    design = json.loads(a.design.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    reader = SpeedReader(a.root, device)
    data = a.root / "pointodyssey/data/davis_records"
    offsets = np.array([(x, y) for y in (-16, 0, 16) for x in (-16, 0, 16)])
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for i in a.indices:
        bg, fg = design["scenes"][i]
        target = a.out / f"{i:02d}_{bg}.json"
        if target.exists():
            continue
        started = time.monotonic()
        row = {"scene": bg, "patch": fg, "index": i, "seed": 12000 + i, "cases": {},
               "source_sha256": code_hash, "vae_sha256": reader.vae_hash, "human_annotations_opened": False}
        images = [np.load(data / f"{name}.video.npy", mmap_mode="r")[0] for name in (bg, fg)]
        same_pixels, endpoint_anchor = None, None
        conditions = (("matched_loop", 180, 30),) if a.matched_loops else CONDITIONS
        for name, degrees, duration in conditions:
            phase = matched_loops(i) if a.matched_loops else None
            video, centers = render(*images, i, degrees, duration, phase_paths=phase)
            if a.matched_loops:
                counts = [Counter(hashlib.sha256(frame).digest() for frame in clip) for clip in video]
                assert counts[0] == counts[1]
                assert np.array_equal(video[0, :21], video[1, :21])
                assert np.array_equal(video[0, 50:], video[1, 50:])
            endpoints = video[:, [0, -1]]
            if same_pixels is None:
                same_pixels = endpoints.copy()
                static_anchor = np.repeat(video[0, -1:][None], FRAMES, axis=1)
                endpoint_anchor = reader.encode(static_anchor)[:, :, -3:]
            else:
                assert np.array_equal(endpoints, same_pixels)
            latent = reader.encode(video)
            assert latent.shape[2] == len(NOMINAL)
            assert torch.equal(latent[0, :, :2], latent[1, :, :2])
            latent[:, :, -3:] = endpoint_anchor
            grid = (centers[:, NOMINAL, :, None] + offsets + .5) / np.array([WIDTH, HEIGHT])
            row["cases"][name] = {
                "angular_degrees": degrees, "duration": duration, "degrees_per_frame": degrees / duration,
                "identical_image_multiset_and_step_speeds": a.matched_loops,
                "centers": centers.tolist(), "scores": reader.read(latent, grid, row["seed"]),
            }
            if a.matched_loops:
                row["cases"][name]["degrees_per_frame"] = 18.
                row["cases"][name]["world_net_angles_degrees"] = [float(np.rad2deg(p[-1])) for p in phase]
            print(bg, name, f"{time.monotonic() - started:.1f}s", flush=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
        print("DONE", target, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
