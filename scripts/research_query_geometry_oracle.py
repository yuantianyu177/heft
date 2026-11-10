"""Compare temporal query calibration with an oracle spatial correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from research_analyze_temporal_response import interval

DELAY = 1.25
THRESHOLDS = np.array([1, 2, 4, 8, 16])
SOURCES = ("nominal", "temporal", "spatial_oracle", "spatial_reverse")


def sample_time(path, timestamps):
    lo = timestamps.astype(int)
    fraction = timestamps - lo
    return (1 - fraction[..., None]) * path[:, lo] + fraction[..., None] * path[:, lo + 1]


def predict(feature, corrected, points, q):
    targets = np.arange(q + 1, q + 9)
    point_q = points[:, q]
    earlier_q = sample_time(points, np.array(q - DELAY))
    earlier_targets = sample_time(points, targets - DELAY)
    shift = points[:, targets] - earlier_targets
    samples = np.stack((point_q, point_q, earlier_q, 2 * point_q - earlier_q))
    grid = torch.as_tensor(samples, device=feature.device, dtype=torch.float32)[:, None] * 2 - 1
    query_volume = torch.stack((feature[q], corrected[q], feature[q], feature[q]))
    query = F.grid_sample(query_volume, grid, align_corners=False, padding_mode="border")[:, :, 0]
    query = F.normalize(query.transpose(1, 2), dim=-1)
    positions = {}
    for target_label, volume in (("nominal", feature), ("temporal", corrected)):
        target = F.interpolate(volume[targets], size=(60, 104), mode="bilinear", align_corners=False)
        target = F.normalize(target, dim=1).flatten(2)
        logits = torch.einsum("spc,tci->spti", query, target)
        index = logits.argmax(-1).cpu().numpy()
        output = np.stack(((index % 104 + .5) / 104, (index // 104 + .5) / 60), -1)
        for source, tracks in zip(SOURCES, output, strict=True):
            positions[source + "/" + target_label] = tracks
            if target_label == "nominal":
                positions[source + "/spatial_oracle"] = tracks + shift
    return positions, np.linalg.norm((point_q - earlier_q) * 256, axis=-1)


def analyze(rows):
    output = {}
    for q in (8, 12):
        for group in ("all", "small_source_shift", "large_source_shift"):
            selected = [row for row in rows if row["query_frame"] == q and row["group"] == group]
            if not selected:
                continue
            result = {"videos": len(selected), "queries": sum(row["queries"] for row in selected)}
            result["pck_ci95"] = {
                key: interval([r["pck"][key] for r in selected]) for key in selected[0]["pck"]
            }
            pairs = [
                ("temporal/temporal", "spatial_oracle/temporal"),
                ("spatial_oracle/temporal", "nominal/temporal"),
                ("spatial_oracle/temporal", "spatial_reverse/temporal"),
                ("temporal/temporal", "spatial_oracle/spatial_oracle"),
                ("temporal/temporal", "nominal/nominal"),
            ]
            result["paired_gains_ci95"] = {
                a + " minus " + b: interval([r["pck"][a] - r["pck"][b] for r in selected])
                for a, b in pairs
            }
            output[f"q{q}/{group}"] = result
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    design = {
        "head": [18, 5], "delay": DELAY, "targets": 8,
        "primary": "query at frame 8, continuously visible from frame 6 through 16",
        "secondary": "query at frame 12, same 2-frame history and 8-frame future visibility",
        "source_oracle": "sample unretimed map at annotated position q - delay",
        "target_oracle": "add true displacement from t - delay to t after matching unretimed maps",
        "main_comparison": "temporal source vs oracle spatial source on identical calibrated target maps",
        "fully_spatial_control": "oracle spatial source and oracle target displacement",
        "diagnostic_only": "ground truth is supplied at inference solely for the oracle contrast",
        "small_source_shift_256px": "<=2", "large_source_shift_256px": ">=8",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (a.out / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    feature_root = a.root / "features/wan_native_polyphase_chunk25"
    rows = []
    began = time.monotonic()
    for path in sorted(feature_root.glob("*_c000.safetensors")):
        name = path.stem.rsplit("_c", 1)[0]
        output = a.out / (name + ".json")
        if output.exists():
            rows.extend(json.loads(output.read_text()))
            continue
        with np.load(a.root / f"data/davis_records/{name}.annotations.npz") as ann:
            points = ann["points"][:, :25]
            visible = ~ann["occluded"][:, :25]
        with safe_open(path, framework="pt", device=a.device) as file:
            feature = file.get_tensor("l18_h05_dense_key").float().permute(0, 3, 1, 2)
        t = (torch.arange(len(feature), device=a.device) + DELAY).clamp(max=len(feature)-1)
        lo, hi = t.floor().long(), t.ceil().long()
        alpha = (t - lo)[:, None, None, None]
        corrected = feature[lo] * (1 - alpha) + feature[hi] * alpha
        corrected[0] = feature[0]
        per_video = []
        for q in (8, 12):
            if q + 8 >= points.shape[1]:
                continue
            ids = np.flatnonzero(visible[:, q - 2:q + 9].all(1))
            if not len(ids):
                continue
            positions, displacement = predict(feature, corrected, points[ids], q)
            gt = points[ids, q + 1:q + 9]
            errors = {key: np.linalg.norm((pos - gt) * 256, axis=-1) for key, pos in positions.items()}
            for group, take in (
                ("all", np.ones(len(ids), bool)),
                ("small_source_shift", displacement <= 2),
                ("large_source_shift", displacement >= 8),
            ):
                if not take.any():
                    continue
                per_video.append({
                    "video": name, "query_frame": q, "group": group, "queries": int(take.sum()),
                    "pck": {key: float((e[take, :, None] < THRESHOLDS).mean()*100) for key, e in errors.items()},
                    "mean_distance_256px": {key: float(e[take].mean()) for key, e in errors.items()},
                })
            np.savez_compressed(
                a.out / f"{name}_q{q}.npz", query_indices=ids, gt=gt, source_displacement_256px=displacement,
                **{key.replace("/", "__"): pos for key, pos in positions.items()},
            )
        output.write_text(json.dumps(per_video, indent=2) + "\n")
        rows.extend(per_video)
        print(name, len(per_video), f"{time.monotonic()-began:.1f}s", flush=True)
    payload = {"design": design, "groups": analyze(rows), "elapsed_s": time.monotonic()-began}
    (a.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["groups"], indent=2), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
