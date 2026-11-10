"""Aggregate the follow-up probes with paired resampling at the scene level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research_analyze_temporal_response import fit_kernel, interval


def read(path):
    return json.loads(path.read_text())


def records(folder, expected, pattern="*.json"):
    result = [read(path) for path in sorted(folder.glob(pattern))]
    assert len(result) == expected, (folder, len(result), expected)
    return result


def visibility(root):
    design = read(root / "design.json")
    rows = records(root / "artifacts", len(design["scenes"]))
    assert {(r["background"], r["patch_source"]) for r in rows} == set(map(tuple, design["scenes"]))
    assert all(r["source_sha256"] == design["source_sha256"] for r in rows)
    assert all(not r["human_annotations_opened"] for r in rows)
    result = {}
    for motion in ("stationary", "moving"):
        examples = [row["records"][motion] for row in rows]
        pck = {}
        for history in ("always", "reveal6", "reveal8"):
            for target in examples[0]["pck"][history]:
                pck[f"{history}/{target}"] = {
                    mode: [r["pck"][history][target][mode] for r in examples]
                    for mode in examples[0]["pck"][history][target]
                }
        revealed = pck["reveal8/always"]
        always = pck["always/always"]
        gain = np.array(revealed["temporal"]) - revealed["nominal"]
        steady_gain = np.array(always["temporal"]) - always["nominal"]
        curves = {
            stage: {
                mode: {
                    history: [
                        interval(samples) for samples in np.asarray([
                            r["visibility_projection"][stage][mode][history] for r in examples
                        ]).T
                    ] for history in ("reveal8", "reveal6", "hide12")
                } for mode in ("nominal", "quarter", "temporal")
            } for stage in ("latent", "key")
        }
        assert all(r["visibility_projection"]["latent"]["nominal"]["reveal8"][:8] == [0.] * 8 for r in examples)
        assert all(r["visibility_projection"]["latent"]["nominal"]["hide12"][:12] == [1.] * 12 for r in examples)
        if motion == "stationary":
            assert all(groups["spatial_oracle"] == groups["nominal"] for groups in pck.values())
        result[motion] = {
            "scenes": len(examples),
            "pck_ci95": {group: {k: interval(v) for k, v in scores.items()} for group, scores in pck.items()},
            "paired_gains_ci95": {
                "temporal_minus_nominal_reveal8": interval(gain),
                "temporal_minus_nominal_always": interval(steady_gain),
                "reveal8_minus_always_gain": interval(gain - steady_gain),
                "temporal_minus_quarter_reveal8": interval(np.array(revealed["temporal"]) - revealed["quarter"]),
                "temporal_minus_spatial_oracle_reveal8": interval(np.array(revealed["temporal"]) - revealed["spatial_oracle"]),
                "temporal_minus_spatial_oracle_always": interval(np.array(always["temporal"]) - always["spatial_oracle"]),
            },
            "contrast_coordinate_ci95": curves,
        }
    return result


def chunk_end(t, size):
    return min((t + size - 1) // size * size, 24) if t else size


def encoder_prefix(root):
    design = read(root / "design.json")
    result = {}
    for kind, count in (("artifacts", 6), ("native_resolution", 2)):
        group = {}
        for encoder in ("cogvideox", "wan"):
            rows = records(root / kind, count, encoder + "_*.json")
            assert all(row["source_sha256"] == design["source_sha256"] for row in rows)
            assert all(not row["human_annotations_opened"] for row in rows)
            measures = {}
            for condition in rows[0]["future_cases"]:
                cases = [row["future_cases"][condition] for row in rows]
                matrix = {
                    int(frame): np.array([row["single_frame"][frame]["relative_rms"] for row in cases])
                    for frame in cases[0]["single_frame"]
                }
                expected_zeros, inside = [], []
                for t in matrix:
                    for j, label in enumerate(rows[0]["nominal"]):
                        if label >= t:
                            continue
                        if encoder == "wan" or condition.startswith("per_frame"):
                            expected_zeros.extend(matrix[t][:, j])
                            continue
                        size = int(condition.rsplit("_", 1)[1])
                        if t > chunk_end(label, size):
                            expected_zeros.extend(matrix[t][:, j])
                        else:
                            inside.extend(matrix[t][:, j])
                measures[condition] = {
                    "relative_rms_single_frame_ci95": {t: [interval(col) for col in m.T] for t, m in matrix.items()},
                    "relative_rms_suffix_ci95": [interval(col) for col in np.array([
                        row["suffix_from_frame5"]["relative_rms"] for row in cases]).T],
                    "relative_rms_prefix_length_ci95": [interval(col) for col in np.array([
                        row["prefix_length5_vs_25"]["relative_rms"] for row in cases]).T],
                    "max_expected_zero": float(np.max(expected_zeros, initial=0)),
                    "min_inside_future": float(np.min(inside)) if inside else None,
                    "max_inside_future": float(np.max(inside)) if inside else None,
                }
                assert measures[condition]["max_expected_zero"] < 1e-7
            group[encoder] = {
                "scenes": [r["scene"] for r in rows], "size": rows[0]["size"], "dtype": rows[0]["dtype"],
                "nominal": rows[0]["nominal"], "groupnorm_count": len(rows[0]["group_norm_modules"]),
                "cases": measures,
                "chunk_change_max_relative_rms": {
                    key: float(max(max(r["chunk_switch"][key]["relative_rms"]) for r in rows))
                    for key in rows[0]["chunk_switch"]
                },
            }
        result[kind] = group
    return result


def margin(matrix):
    return matrix[..., 0, 0] + matrix[..., 1, 1] - matrix[..., 0, 1] - matrix[..., 1, 0]


def wins(value, threshold=1e-6):
    return (value > threshold) + .5 * (abs(value) <= threshold)


def composition(adjacent):
    order = np.sign(adjacent)
    order[abs(adjacent) <= 1e-6] = 0
    return order.prod(-1)


def identity_metrics(endpoint, adjacent):
    direct, local = margin(endpoint), margin(adjacent)
    chained = composition(local)
    return {
        "scenes": len(direct),
        "direct_correct_worlds": int((direct > 1e-6).sum()),
        "chained_correct_worlds": int((chained > 0).sum()),
        "worlds": int(direct.size),
        "both_worlds_correct": int((direct > 1e-6).all(1).sum()),
        "direct_correct_per_world": (direct > 1e-6).sum(0).tolist(),
        "chained_correct_per_world": (chained > 0).sum(0).tolist(),
        "direct_ci95": interval(wins(direct).mean(1)),
        "chained_ci95": interval(wins(chained).mean(1)),
        "adjacent_ci95": interval(wins(local).mean((1, 2))),
        "chained_minus_direct_ci95": interval((wins(chained) - wins(direct)).mean(1)),
        "scene_direct_margins": direct.tolist(),
        "scene_chained_signs": chained.tolist(),
    }


def identity_original(base):
    root = base / "paired_identity"
    design = read(root / "design.json")
    rows = records(root / "artifacts", len(design["scenes"]))
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert all(r["source_sha256"] == design["source_sha256"] for r in rows)
    arrays = [np.load(p) for p in sorted((root / "artifacts").glob("*.npz"))]
    cases = {}
    real = []
    for path in sorted((base.parent / "results/probe_wan_native_polyphase_t25").glob("*.npz")):
        with np.load(path) as file:
            variant = file["variant_names"].tolist().index("post_full")
            valid = file["valid"]
            e = file["errors"][:, :, variant]
            real.append(((e[..., None] < [1, 2, 4, 8, 16]).mean(-1) * valid).sum((-1, -2)) / valid.sum() * 100)
    real_pck = np.mean(real, axis=0)
    try:
        for kind in ("natural", "common_endpoints"):
            for noise in ("last", "sigma035", "sigma065"):
                label = kind + "_" + noise
                endpoint = np.stack([a[label + "_endpoint"].mean(-1) for a in arrays])
                adjacent = np.stack([a[label + "_adjacent"] for a in arrays])
                m = margin(endpoint)
                if kind == "common_endpoints":
                    # Layer 0 has no contextual update at capture time.
                    np.testing.assert_allclose(m[:, 0, :, 0], -m[:, 0, :, 1], atol=3e-4, rtol=1e-5)
                table = {}
                for v, variant in enumerate(design["feature_variants"]):
                    a = m[:, :, v]
                    calibration = wins(a[:8]).mean((0, 2)).flatten()
                    minimum_margin = a[:8].min(2).mean(0).flatten()
                    pick = np.lexsort((np.arange(calibration.size), -minimum_margin, -calibration))[0]
                    layer, head = map(int, divmod(pick, 12))
                    ranking = wins(a).mean((0, 2))
                    entry = {
                        "l18h05": identity_metrics(endpoint[:, 18, v, :, 5], adjacent[:, 18, v, :, 5]),
                        "l18h01": identity_metrics(endpoint[:, 18, v, :, 1], adjacent[:, 18, v, :, 1]),
                        "head_from_first_eight": [layer, head],
                        "selected_first_eight": identity_metrics(endpoint[:8, layer, v, :, head], adjacent[:8, layer, v, :, head]),
                        "selected_last_eight": identity_metrics(endpoint[8:, layer, v, :, head], adjacent[8:, layer, v, :, head]),
                        "head_accuracy": ranking.tolist(),
                    }
                    if variant == "kk_full":
                        entry["comparison_with_seven_real_prefixes"] = {
                            "spearman_rho_across_360_heads": float(spearmanr(real_pck.flatten(), ranking.flatten()).statistic),
                            "selected_pck": float(real_pck[layer, head]),
                            "l18h05_pck": float(real_pck[18, 5]), "l18h01_pck": float(real_pck[18, 1]),
                            "n_real_videos": len(real),
                        }
                    table[variant] = entry
                cases[label] = table
    finally:
        for array in arrays:
            array.close()
    return cases


def identity_transfer(root):
    design = read(root / "design.json")
    rows = records(root / "artifacts", len(design["scenes"]))
    assert all(r["source_sha256"] == design["source_sha256"] for r in rows)
    assert all(not r["human_annotations_opened"] for r in rows)
    return {
        case: {
            label: identity_metrics(
                np.array([r["cases"][case]["scores"][label]["endpoint"] for r in rows]),
                np.array([r["cases"][case]["scores"][label]["adjacent"] for r in rows]),
            )
            for label in rows[0]["cases"][case]["scores"]
        }
        for case in rows[0]["cases"]
    }


def codec(root):
    design = read(root / "design.json")
    rows = records(root / "artifacts", len(design["test_images"]))
    assert {r["image"] for r in rows} == set(design["test_images"])
    assert all(r["kernel_sha256"] == design["kernel_sha256"] for r in rows)
    assert all(r["script_sha256"] == design["source_sha256"] for r in rows)
    aggregate = {}
    for family in ("null", "ramp_x", "sine5_x"):
        for amplitude in (2, 6):
            collection = []
            for r in rows:
                current = []
                for x in r["records"]:
                    if x["amplitude"] != amplitude or not x["curve"].startswith(family):
                        continue
                    p = np.array(x["path"])
                    latent = np.array(x["latent_projected_displacement"])
                    response = {
                        "actual_path_mean_norm": float(np.linalg.norm(p[1:], axis=1).mean()),
                        "latent_projected_position_mean_norm": float(np.linalg.norm(latent[1:], axis=1).mean()),
                        "residual_position_leak_max": x["residual_position_projection_max_pixel"],
                    }
                    for name in ("full", "position_component", "residual_component"):
                        for readout in ("projected", "registered"):
                            key = readout + "_displacement"
                            if key not in x[name]:
                                continue
                            estimate = np.asarray(x[name][key])
                            error = np.linalg.norm(estimate[1:] - p[1:], axis=1)
                            response[f"{readout}/{name}/mae_pixels"] = float(error.mean())
                            response[f"{readout}/{name}/signed_path_gain"] = float(np.sum(estimate[1:] * p[1:]) / np.sum(p[1:]**2))
                    current.append(response)
                collection.append({key: float(np.mean([item[key] for item in current])) for key in current[0]})
            aggregate[f"{family}/{amplitude}"] = {
                key: interval([x[key] for x in collection]) for key in collection[0]
            }
    return aggregate


def precision(root):
    name = "wan_native_impulse_drift-straight.npz"
    with np.load(root / "full_fp16/artifacts" / name) as fast, np.load(root / "precision_check" / name) as reference:
        a, b = fit_kernel(fast)[0], fit_kernel(reference)[0]
    nominal = np.arange(0, 25, 4)
    lags = nominal[:, None] - np.arange(25)[None]
    return {
        "image": "drift-straight", "comparison": "FP16 vs FP32 with TF32 allowed",
        "max_abs_matrix_coefficient_difference": float(abs(a - b).max()),
        "max_mean_lag_difference_frames": float(abs(np.sum(lags * np.trace(a - b, axis1=-2, axis2=-1) / 2, 1)).max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    base = a.root
    result = {
        "visibility": visibility(base / "visibility_history"),
        "prefix": encoder_prefix(base / "encoder_prefix"),
        "identity": identity_original(base),
        "identity_transfer": identity_transfer(base / "identity_transfer"),
        "identity_speed": identity_transfer(base / "identity_speed"),
        "identity_matched_loops": identity_transfer(base / "identity_matched_loops"),
        "codec": codec(base / "codec_subspace"),
        "precision_check": precision(base),
    }
    path = base / "followup_summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
