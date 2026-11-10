"""Evaluate native I2V Point Prompting on a fixed one-query-per-video split."""

import fcntl
import hashlib
import json
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from research_track import official_metrics

from heft.baselines import point_prompting, point_prompting_i2v
from heft.baselines.point_prompting_i2v import PointPromptingI2V

MODELS = {"wan": (480, 832), "cog": (480, 832)}
METRICS = ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")


def select_query(name: str, visible: np.ndarray) -> int:
    candidates = np.flatnonzero(visible)
    if not len(candidates):
        raise ValueError(f"No frame-zero query in {name}")
    index = int.from_bytes(
        hashlib.sha256(f"point_prompting:42:{name}".encode()).digest()[:8], "little"
    )
    return int(candidates[index % len(candidates)])


def save(path: Path, value) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def ensure_plan(path: Path, plan: dict) -> None:
    """Share one experiment identity across cooperating GPU processes."""
    with path.with_suffix(".lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != json.loads(json.dumps(plan)):
                raise ValueError(f"Mismatched experiment in {path.parent}")
        else:
            save(path, plan)


@contextmanager
def claim_video(path: Path):
    """Allow one writer and leave interrupted work available for a retry."""
    claims = path.parent / ".claims"
    claims.mkdir(exist_ok=True)
    with (claims / f"{path.stem}.lock").open("ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield not path.exists()


def evaluate_video(tracker, evaluator, row, records, frame_limit):
    name = row["name"]
    frames = min(row["frames"], frame_limit or row["frames"])
    rgb = np.load(records / f"{name}.video.npy", mmap_mode="r", allow_pickle=False)
    with np.load(records / f"{name}.annotations.npz", allow_pickle=False) as data:
        valid = data["valid"] if "valid" in data else np.ones_like(data["occluded"])
        query_id = select_query(name, valid[:, 0] & ~data["occluded"][:, 0])
        gt = data["points"][query_id, :frames]
        occ = data["occluded"][query_id, :frames]
        valid = valid[query_id, :frames]
    if len(rgb) != row["frames"] or len(gt) != frames:
        raise ValueError(f"Inconsistent frames in {name}")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    track, visible = tracker.track(rgb[:frames], gt[0])
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    query_yx_256 = np.array(
        [[[0.0, gt[0, 1] * 256.0, gt[0, 0] * 256.0]]], dtype=np.float32
    )
    raw = evaluator(
        query_yx_256,
        occ[None, None],
        gt[None, None] * 256.0,
        ~visible[None, None],
        track[None, None] * 256.0,
        "first",
        valid_mask=valid[None, None],
    )
    return {
        "video": name,
        "frames": frames,
        "query_index": query_id,
        "query_xy": gt[0].tolist(),
        "metrics": {key: float(raw[key][0]) * 100 for key in METRICS},
        "seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--family", required=True, choices=MODELS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--text-encoder-source", type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--metric-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--work-video", help="Claim one video from the declared experiment")
    parser.add_argument("--max-frames", type=int, help="Restrict pilot clips only")
    args = parser.parse_args()
    if args.max_frames is not None and (not args.names or args.max_frames < 2):
        parser.error("Specify pilot names and at least two frames for a frame limit")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    evaluator, evaluator_hash = official_metrics(args.metric_root)
    manifest = args.records / "manifest.json"
    rows = json.loads(manifest.read_text())
    if args.names:
        rows = [row for row in rows if row["name"] in args.names]
        if len(rows) != len(set(args.names)):
            raise ValueError("One or more pilot names are absent")
    if args.work_video is not None and not any(row["name"] == args.work_video for row in rows):
        raise ValueError("The requested work video is outside the declared experiment")
    plan = {
        "family": args.family,
        "resolution": MODELS[args.family],
        "checkpoint": str(args.checkpoint),
        "text_encoder_source": str(args.text_encoder_source)
        if args.text_encoder_source is not None
        else None,
        "source": str(args.records),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "records": {row["name"]: row["frames"] for row in rows},
        "guidance": 8.0,
        "strength": 0.5,
        "steps": args.steps,
        "max_frames": args.max_frames,
        "window_frames": 49,
        "refinement_radius_native_pixels": 20,
        "marker_radius_evaluation_pixels": 2,
        "query_policy": "one frame-zero visible point per video selected by name hash before scoring",
        "evaluator": evaluator_hash,
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in map(
                Path, (__file__, point_prompting.__file__, point_prompting_i2v.__file__)
            )
        },
    }
    args.output.mkdir(exist_ok=True, parents=True)
    ensure_plan(args.output / "plan.json", plan)
    tracker = None
    for row in rows:
        if args.work_video is not None and row["name"] != args.work_video:
            continue
        destination = args.output / f"{row['name']}.json"
        if destination.exists():
            continue
        with claim_video(destination) as claimed:
            if not claimed:
                continue
            if tracker is None:
                tracker = PointPromptingI2V(
                    args.checkpoint,
                    args.family,
                    args.gpu,
                    resolution=MODELS[args.family],
                    steps=args.steps,
                    text_encoder_source=args.text_encoder_source,
                )
            result = evaluate_video(tracker, evaluator, row, args.records, args.max_frames)
            save(destination, result)
            print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
