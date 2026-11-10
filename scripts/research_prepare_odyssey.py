"""Prepare all available PointOdyssey test prefixes for independent transfer.

Uses 200 frames and 64 randomly sampled frame-zero-valid, visible queries per
sequence. Every query index and all validity masks are saved. This is a
controlled transfer diagnostic, not a claim of the official full benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/share/data/point_odyssey/test"))
    parser.add_argument("--output", type=Path, default=Path("/share/yty/heft/pointodyssey"))
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--queries", type=int, default=64)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    destination = args.output/"data/davis_records"
    destination.mkdir(exist_ok=True, parents=True)
    (args.output/"logs").mkdir(exist_ok=True)
    if not (args.output/"models").exists():
        (args.output/"models").symlink_to(Path("/share/yty/heft/models"))
    (args.output/"scripts").mkdir(exist_ok=True)
    evaluator = args.output/"scripts/tapvid_official_evaluation_datasets.py"
    if not evaluator.exists():
        evaluator.symlink_to(Path("/share/yty/heft/scripts/tapvid_official_evaluation_datasets.py"))
    manifest = []
    for sequence in sorted(args.source.iterdir()):
        if not sequence.is_dir() or not (sequence/"anno.npz").exists():
            continue
        name = f"po_{sequence.name}"
        rgb_paths = sorted((sequence/"rgbs").glob("*.jpg"))
        if not rgb_paths:
            rgb_paths = sorted((sequence/"rgbs").glob("*.png"))
        if len(rgb_paths) < args.frames:
            continue
        first = cv2.imread(str(rgb_paths[0]))
        h, w = first.shape[:2]
        with np.load(sequence/"anno.npz") as annotation:
            trajectories = annotation["trajs_2d"][:args.frames].copy()
            visibility = annotation["visibs"][:args.frames].astype(bool)
            validity = annotation["valids"][:args.frames].astype(bool)
        finite = np.isfinite(trajectories).all(-1)
        in_bounds = ((trajectories[..., 0] >= 0) & (trajectories[..., 0] < w-1) &
                     (trajectories[..., 1] >= 0) & (trajectories[..., 1] < h-1))
        valid = validity & finite
        visible = visibility & valid & in_bounds
        candidates = np.flatnonzero(visible[0])
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        generator = np.random.default_rng(seed)
        indices = np.sort(generator.choice(candidates, min(args.queries, len(candidates)), replace=False))
        tracks = trajectories[:, indices].transpose(1, 0, 2) / np.array([w, h], np.float32)
        # Invalid coordinates are never considered visible; retain the validity mask for masked re-evaluation.
        tracks = np.nan_to_num(tracks, nan=-1, posinf=-1, neginf=-1)
        np.savez_compressed(destination/f"{name}.annotations.npz", points=tracks.astype(np.float32),
                            occluded=~visible[:, indices].T, valid=valid[:, indices].T, query_indices=indices)
        output_video = destination/f"{name}.video.npy"
        if not output_video.exists():
            frames = np.lib.format.open_memmap(output_video, mode="w+", dtype=np.uint8,
                                               shape=(args.frames, h, w, 3))
            for t, path in enumerate(rgb_paths[:args.frames]):
                frames[t] = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            frames.flush()
            del frames
        record = dict(name=name, frames=args.frames, points=len(indices), split="evaluation",
                      dataset="PointOdyssey-test-prefix", source=str(sequence), source_size=[h, w],
                      query_seed=seed, query_indices=indices.tolist(),
                      invalid_point_frames=int((~valid[:, indices]).sum()))
        manifest.append(record)
        print(json.dumps(record), flush=True)
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output/"protocol.json").write_text(json.dumps({
        "source": str(args.source), "sequences": len(manifest), "frames": args.frames,
        "queries": args.queries, "query_policy": "uniform random from valid visible points in frame zero; per-sequence SHA256 seed",
        "representation_calibration": "Frozen on DAVIS synthetic transforms; no PointOdyssey configuration selection",
        "metric_note": "Coordinate accuracy counts only valid visible GT. Visibility/Jaccard should be re-evaluated with saved validity masks before quoting.",
    }, indent=2))


if __name__ == "__main__":
    main()
