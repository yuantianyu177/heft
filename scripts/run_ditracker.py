"""Train and evaluate the shared DiTracker reimplementation."""

import hashlib
import json
import time
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from evaluate_difftrack import annotations, official_metrics

from heft.baselines import ditracker, ditracker_backbone
from heft.baselines.ditracker import refinement_loss
from heft.baselines.ditracker_backbone import DiTracker, DiTrackerBackbone
from heft.extraction import COGVIDEOX, WAN_2_1

BACKBONES = {
    "wan": (WAN_2_1, 15, (0, 3, 6, 9)),
    "cog": (COGVIDEOX, 17, (2, 5, 8, 20)),
}
METRICS = ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")


def save_json(path: Path, value) -> None:
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    stage.replace(path)


def source_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in map(
            Path, (__file__, ditracker.__file__, ditracker_backbone.__file__)
        )
    }


def build(args, device, *, training: bool) -> DiTracker:
    profile, layer, heads = BACKBONES[args.model]
    profile = replace(profile, resolution=tuple(args.resolution or profile.resolution))
    readout = DiTrackerBackbone(
        profile,
        args.checkpoint,
        device=device,
        layer=layer,
        heads=heads,
        rank=128,
        chunk=args.chunk,
        gradient_checkpointing=training,
    )
    return DiTracker(readout).to(device).train(training)


def sample(
    root: Path,
    records: list[dict],
    rng: np.random.Generator,
    frames: int,
    tracks: int,
    *,
    training: bool,
):
    """Take a temporal crop and visible query anchors from Kubric."""
    for index in rng.permutation(len(records)):
        row = records[index]
        folder = root / row["name"]
        source = np.load(folder / "video.npy", mmap_mode="r", allow_pickle=False)
        label = np.load(folder / "points.npy", mmap_mode="r", allow_pickle=False)
        occluded = np.load(folder / "occluded.npy", mmap_mode="r", allow_pickle=False)
        valid = np.load(folder / "valid.npy", mmap_mode="r", allow_pickle=False)
        max_stride = min(3, (len(source) - 1) // (frames - 1))
        if max_stride < 1:
            continue
        stride = (
            int(rng.integers(1, max_stride + 1)) if training else min(max_stride, 2)
        )
        start = int(rng.integers(len(source) - (frames - 1) * stride))
        select = slice(start, start + frames * stride, stride)
        valid_crop = valid[:, select]
        visible = ~occluded[:, select] & valid_crop
        eligible = np.flatnonzero(visible[:, 0] & visible[:, 1:].any(1))
        if len(eligible) < tracks:
            continue
        ids = rng.choice(eligible, size=tracks, replace=False)
        points = np.array(label[ids, select])
        visibility = visible[ids]
        chosen = np.zeros(tracks, dtype=np.int64)
        if training:
            for point in np.flatnonzero(rng.random(tracks) < 0.35):
                candidates = np.flatnonzero(visibility[point, :-1])
                chosen[point] = rng.choice(candidates)
        rgb = np.array(source[select])
        if training and rng.random() < 0.5:
            rgb = rgb[:, :, ::-1].copy()
            points[..., 0] = 1.0 - points[..., 0]
        anchor = points[np.arange(tracks), chosen]
        valid_crop = valid_crop[ids] & (np.arange(frames)[None] >= chosen[:, None])
        return rgb, points, visibility, valid_crop, anchor, chosen
    raise ValueError("No Kubric sequence has the requested crop and visible queries")


def tensors(data, device):
    rgb, points, visible, valid, anchors, times = data
    return (
        torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].to(device),
        torch.from_numpy(points.transpose(1, 0, 2))[None].to(device),
        torch.from_numpy(visible.T)[None].to(device),
        torch.from_numpy(valid.T)[None].to(device),
        torch.from_numpy(anchors)[None].to(device),
        torch.from_numpy(times)[None].to(device),
    )


def restore(model, saved):
    expected = set(model.trainable_state())
    if set(saved) != expected:
        raise ValueError(
            f"Unexpected trainable state keys, missing {expected - set(saved)}"
        )
    model.load_state_dict(saved, strict=False)


@torch.inference_mode()
def score(model, data, device, evaluator, iterations, query_batch=None):
    raw, gt, vis, valid_eval, anchors_np, times_np = data
    rgb = torch.from_numpy(raw).permute(0, 3, 1, 2)[None].to(device)
    anchors = torch.from_numpy(anchors_np)[None].to(device)
    times = torch.from_numpy(times_np)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction, logits, _ = model(
            rgb, anchors, times, iterations=iterations, query_batch=query_batch
        )[-1]
    xy = prediction[0].transpose(0, 1).cpu().numpy()
    pred_occ = (logits[0].T <= 0).cpu().numpy()
    queries = np.column_stack(
        (times_np.astype(np.float32), anchors_np[:, ::-1] * 256.0)
    )
    scored = evaluator(
        queries[None],
        (~vis)[None],
        gt[None] * 256.0,
        pred_occ[None],
        xy[None] * 256.0,
        "first",
        valid_mask=valid_eval[None],
    )
    return {key: 100.0 * float(scored[key][0]) for key in METRICS}


def train(args, device):
    if (
        min(
            args.steps,
            args.points,
            args.iterations,
            args.validate_every,
            args.accumulate,
        )
        < 1
        or args.frames < 3
    ):
        raise ValueError("Invalid training schedule")
    manifest = args.records / "manifest.json"
    rows = json.loads(manifest.read_text())
    training = [r for r in rows if r["split"] == "training"]
    validation = [r for r in rows if r["split"] == "validation"]
    if (
        not training
        or not validation
        or {r["name"] for r in training} & {r["name"] for r in validation}
    ):
        raise ValueError(
            "Kubric training and validation splits must be disjoint and nonempty"
        )
    evaluator, evaluator_hash = official_metrics(args.metric_root)
    plan = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "records": str(args.records),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "layer": BACKBONES[args.model][1],
        "heads": list(BACKBONES[args.model][2]),
        "rank": 128,
        "steps": args.steps,
        "accumulate": args.accumulate,
        "frames": args.frames,
        "points": args.points,
        "iterations": args.iterations,
        "chunk": args.chunk,
        "train_resolution": list(
            args.resolution or BACKBONES[args.model][0].resolution
        ),
        "lr": args.lr,
        "seed": args.seed,
        "validate_every": args.validate_every,
        "evaluator": evaluator_hash,
        "sources": source_hashes(),
    }
    plan_file = args.output / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError(f"Training protocol differs from {plan_file}")
    save_json(plan_file, plan)
    best_file = args.output / "best.pt"
    latest_file = args.output / "latest.pt"
    summary_path = args.output / "summary.json"
    if summary_path.is_file():
        print(f"COMPLETE {args.model} {summary_path}", flush=True)
        return
    model = build(args, device, training=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    first, best_aj = 1, -1.0
    if latest_file.exists():
        saved = torch.load(latest_file, map_location="cpu", weights_only=True)
        if saved["plan"] != plan:
            raise ValueError("Training checkpoint plan differs")
        restore(model, saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        first, best_aj = saved["step"] + 1, saved["best_aj"]
    validation_clips = [
        sample(
            args.records,
            [row],
            np.random.default_rng(args.seed + 101 * i + 11),
            args.frames,
            args.points,
            training=False,
        )
        for i, row in enumerate(validation)
    ]
    started = time.monotonic()
    for step in range(first, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.monotonic()
        for micro in range(args.accumulate):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, step, micro])
            )
            data = sample(
                args.records, training, rng, args.frames, args.points, training=True
            )
            rgb, points, visible, valid, anchors, times = tensors(data, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(rgb, anchors, times, iterations=args.iterations)
            loss = refinement_loss(
                [(xy.float() * 256.0, v.float(), c.float()) for xy, v, c in outputs],
                points * 256.0,
                visible,
                valid,
            )
            total = sum(loss.values()) / args.accumulate
            total.backward()
            losses.append(float(total.detach()))
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=1.0, error_if_nonfinite=True
            )
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        if step == first or step % 10 == 0:
            print(
                "TRAIN",
                step,
                "loss",
                round(sum(losses), 4),
                "grad",
                round(norm, 4),
                "seconds",
                round(time.monotonic() - tick, 2),
                "peak_gib",
                round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                flush=True,
            )
        if step % args.validate_every != 0 and step != args.steps:
            continue
        model.eval()
        raw = [
            score(model, data, device, evaluator, args.iterations)
            for data in validation_clips
        ]
        model.train()
        metrics = {key: float(np.mean([row[key] for row in raw])) for key in METRICS}
        print("VALIDATE", step, metrics, flush=True)
        weights = model.trainable_state()
        if metrics[METRICS[0]] > best_aj:
            best_aj = metrics[METRICS[0]]
            stage = best_file.with_suffix(".tmp")
            torch.save(
                {"plan": plan, "weights": weights, "step": step, "metrics": metrics},
                stage,
            )
            stage.replace(best_file)
        stage = latest_file.with_suffix(".tmp")
        torch.save(
            {
                "plan": plan,
                "weights": weights,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_aj": best_aj,
            },
            stage,
        )
        stage.replace(latest_file)
        save_json(
            args.output / f"validation_{step:05d}.json",
            {"step": step, "means": metrics, "videos": raw},
        )
    best = torch.load(best_file, map_location="cpu", weights_only=True)
    save_json(
        summary_path,
        {
            "plan": plan,
            "best_step": best["step"],
            "metrics": best["metrics"],
            "training_seconds_this_run": time.monotonic() - started,
        },
    )


def evaluate(args, device):
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if state["plan"]["model"] != args.model or state["plan"]["checkpoint"] != str(
        args.checkpoint
    ):
        raise ValueError("Checkpoint differs from the training plan")
    evaluator, evaluator_hash = official_metrics(args.metric_root)
    manifest = args.records / "manifest.json"
    rows = json.loads(manifest.read_text())
    if args.names:
        rows = [row for row in rows if row["name"] in args.names]
        if len(rows) != len(set(args.names)):
            raise ValueError("Unknown evaluation clip")
    plan = {
        "training_plan": state["plan"],
        "weights": str(args.weights),
        "weights_step": state["step"],
        "records": str(args.records),
        "names": [r["name"] for r in rows],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "eval_resolution": list(args.resolution or BACKBONES[args.model][0].resolution),
        "chunk": args.chunk,
        "iterations": args.iterations,
        "query_batch": args.query_batch,
        "evaluator": evaluator_hash,
        "source": source_hashes(),
    }
    plan_file = args.output / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError(f"Evaluation protocol differs from {plan_file}")
    save_json(plan_file, plan)
    pending = [
        row for row in rows if not (args.output / f"{row['name']}.json").exists()
    ]
    if pending:
        model = build(args, device, training=False)
        restore(model, state["weights"])
        for row in pending:
            name = row["name"]
            rgb = np.load(
                args.records / f"{name}.video.npy", mmap_mode="r", allow_pickle=False
            )
            gt, occ, valid, query_times = annotations(args.records, name)
            if len(rgb) != row["frames"] or len(query_times) < 1:
                raise ValueError(f"Unexpected evaluation length or queries in {name}")
            xy = gt[np.arange(len(query_times)), query_times]
            data = (np.array(rgb), gt, ~occ, valid, xy, query_times)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.monotonic()
            metrics = score(
                model,
                data,
                device,
                evaluator,
                args.iterations,
                query_batch=args.query_batch,
            )
            torch.cuda.synchronize(device)
            row = {
                "video": name,
                "frames": len(rgb),
                "queries": len(query_times),
                "metrics": metrics,
                "seconds": time.monotonic() - start,
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            save_json(args.output / f"{name}.json", row)
            print("DONE", json.dumps(row), flush=True)
    values = [
        json.loads((args.output / f"{row['name']}.json").read_text()) for row in rows
    ]
    aggregate = {
        key: float(np.mean([row["metrics"][key] for row in values])) for key in METRICS
    }
    save_json(
        args.output / "summary.json", {"videos": len(values), "metrics": aggregate}
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--model", required=True, choices=BACKBONES)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--metric-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=int, nargs=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=412)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--names", nargs="+")
    args = parser.parse_args()
    if args.mode == "evaluate" and args.weights is None:
        parser.error("Evaluation requires a trained --weights checkpoint")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda", args.gpu)
    if args.mode == "train":
        train(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
