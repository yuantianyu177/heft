"""Compare a measured temporal response operator with fixed-delay alternatives."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(31415)
    resample = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return [float(values.mean()), *np.quantile(resample, [.025, .975]).tolist()]


def fit_kernel(arrays):
    grad = arrays["gradient"].astype(np.float64)
    uniform = arrays["uniform"].astype(np.float64)
    basis = uniform.reshape(len(uniform), -1, 2)
    gradient = grad.reshape(*grad.shape[:2], -1, 2)
    gram = np.einsum("jdi,jdk->jik", basis, basis)
    rhs = np.einsum("jdi,jtdk->jtik", basis, gradient)
    raw = np.linalg.solve(gram[:, None], rhs)
    normalized = np.einsum("jtik,jkl->jtil", raw, np.linalg.inv(raw.sum(1)))
    residual = gradient - np.einsum("jdi,jtik->jtdk", basis, raw)
    channel = (
        (uniform[:, None] * grad).sum((2, 4))
        / np.maximum((uniform**2).sum((1, 3))[:, None], 1e-20)
    )
    return normalized, raw, channel, float(np.sum(residual**2) / np.sum(gradient**2))


def closest_path_distance(point, path):
    start = path[:-1]
    direction = np.diff(path, axis=0)
    progress = np.clip(
        np.einsum("td,td->t", point - start, direction)
        / np.maximum(np.sum(direction**2, axis=1), 1e-20), 0, 1,
    )
    return float(np.min(np.linalg.norm(point - (start + progress[:, None] * direction), axis=1)))


def sample_clock(path, times):
    return np.stack(
        [np.interp(times[:, dim], np.arange(len(path)), path[:, dim]) for dim in range(2)],
        axis=-1,
    )


def predict(path, nominal, kernel, delay, stride):
    lags = nominal[:, None] - np.arange(len(path))[None]
    weights = np.diagonal(kernel, axis1=-2, axis2=-1)
    mean_lag = np.einsum("jt,jtd->jd", lags, weights)
    positive = np.maximum(weights, 0)
    positive /= positive.sum(1, keepdims=True)
    boxes = np.array([path[max(t - stride + 1, 0):t + 1].mean(0) for t in nominal])
    return {
        "nominal": path[nominal],
        "frozen_clock": sample_clock(path, np.repeat((nominal - delay * (nominal > 0))[:, None], 2, 1)),
        "moment_matched_clock": sample_clock(path, nominal[:, None] - mean_lag),
        "uniform_window": boxes,
        "positive_kernel": np.einsum("jtd,td->jd", positive, path),
        "diagonal_kernel": np.einsum("jtd,td->jd", weights, path),
        "matrix_kernel": np.einsum("jtab,tb->ja", kernel, path),
    }


def motion_class(name):
    if name.startswith("history"):
        return "matched_history"
    if name.startswith("corner"):
        return "right_angle"
    if name.startswith("constant"):
        return "constant"
    return "other_nonaffine"


def corner_time(name):
    return int(name.split("_")[0].removeprefix("corner"))


def summarize_validation(records, nominal, kernel, frozen_delay, stride):
    errors = defaultdict(lambda: defaultdict(list))
    corners = defaultdict(lambda: defaultdict(list))
    histories = defaultdict(lambda: defaultdict(list))
    leakage = defaultdict(lambda: defaultdict(list))
    for image in records:
        name = image["image"]
        for sample in image["records"]:
            curve = sample["curve"]
            group = motion_class(curve)
            path = np.asarray(sample["path"], dtype=float)
            amplitude = sample["amplitude"]
            predictions = predict(path, nominal, kernel, frozen_delay, stride)
            if group == "matched_history":
                index = np.flatnonzero(nominal == 16)
            elif group == "right_angle":
                index = np.flatnonzero(nominal > corner_time(curve))[:1]
            else:
                index = np.flatnonzero(nominal > 0)
            for measurement in ("projected", "registration"):
                field = measurement + "_displacement"
                if field not in sample:
                    continue
                measured = np.asarray(sample[field], dtype=float)
                base_key = (measurement, amplitude, group)
                for model, estimated in predictions.items():
                    error = np.linalg.norm(measured[index] - estimated[index], axis=-1)
                    errors[base_key + (model,)][name].append(float(error.mean()))
                    if group in ("right_angle", "other_nonaffine"):
                        errors[(measurement, amplitude, "nonaffine", model)][name].append(float(error.mean()))
                if group == "right_angle":
                    first_axis = 0 if curve.endswith("_xy") else 1
                    second_axis = 1 - first_axis
                    vertex = path[corner_time(curve)]
                    j = int(index[0])
                    for model, position in {"measured": measured, **predictions}.items():
                        point = position[j]
                        depth = min(vertex[first_axis] - point[first_axis], point[second_axis])
                        corners[(measurement, amplitude, model, "inside_depth")][name].append(float(depth))
                        corners[(measurement, amplitude, model, "path_distance")][name].append(
                            closest_path_distance(point, path)
                        )
                if group == "matched_history":
                    gap = int(curve.split("_")[1].removeprefix("gap"))
                    axis = 0 if curve.endswith("_x") else 1
                    for model, position in {"measured": measured, **predictions}.items():
                        histories[(measurement, amplitude, gap, model)][name].append(
                            float(position[index[0], axis])
                        )
                if group == "constant":
                    axis = 1 if curve.endswith("_x") else 0
                    leakage[(measurement, amplitude)][name].append(float(np.mean(np.abs(measured[index, axis]))))

    def finish(data):
        return {
            "/".join(map(str, key)): {
                "mean_ci95": interval([np.mean(values) for values in by_image.values()]),
                "per_image": {image: float(np.mean(values)) for image, values in by_image.items()},
            }
            for key, by_image in sorted(data.items())
        }

    stats = finish(errors)
    paired = {}
    for key, values in stats.items():
        if not key.endswith("/matrix_kernel"):
            continue
        for baseline in ("frozen_clock", "moment_matched_clock", "uniform_window", "positive_kernel"):
            control = stats[key.rsplit("/", 1)[0] + "/" + baseline]["per_image"]
            experiment = values["per_image"]
            assert control.keys() == experiment.keys()
            paired[key + "_vs_" + baseline] = interval(
                [control[name] - experiment[name] for name in sorted(control)]
            )
    return {
        "errors_pixels": stats, "kernel_error_reduction_ci95_pixels": paired,
        "corners_pixels": finish(corners), "history_pixels": finish(histories),
        "straight_axis_leakage_pixels": finish(leakage),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    design = json.loads((a.root / "design.json").read_text())
    root = a.root / "artifacts"
    groups = {}
    for name in ("wan_native", "wan_aligned", "cogvideox_native", "cogvideox_aligned"):
        calibration = []
        for path in sorted(root.glob(name + "_impulse_*.json")):
            metadata = json.loads(path.read_text())
            assert metadata["image"] in design["calibration_images"]
            with np.load(path.with_suffix(".npz")) as arrays:
                kernel, raw, channels, orthogonal = fit_kernel(arrays)
            calibration.append((metadata, kernel, raw, channels, orthogonal))
        if not calibration:
            continue
        nominal = np.array(calibration[0][0]["nominal_times"])
        kernels = np.stack([item[1] for item in calibration])
        kernel = kernels.mean(0)
        lags = nominal[:, None] - np.arange(kernel.shape[1])
        weights = np.trace(kernel, axis1=-2, axis2=-1) / 2
        moment = np.sum(lags * weights, 1)
        variance = np.sum((lags - moment[:, None])**2 * weights, 1)
        raw_sums = np.stack([item[2].sum(1) for item in calibration])
        channels = np.stack([item[3] for item in calibration])
        channel_sum = channels.sum(2)
        channel_mean = np.einsum("jt,ijtc->ijc", lags, channels) / channel_sum
        records = [json.loads(f.read_text()) for f in sorted(root.glob(name + "_validation_*.json"))]
        assert all(r["image"] in design["heldout_images"] for r in records)
        assert all(np.array_equal(r["nominal_times"], nominal) for r in records)
        result = {
            "calibration_images": [item[0]["image"] for item in calibration],
            "validation_images": [r["image"] for r in records],
            "nominal_times": nominal.tolist(),
            "first_moment_frames": moment.tolist(), "signed_second_central_moment_frames2": variance.tolist(),
            "scalar_weights_by_input_time": weights.tolist(),
            "negative_weight_mass": np.maximum(-weights, 0).sum(1).tolist(),
            "max_future_energy_fraction": max(max(item[0]["future_energy_fraction"]) for item in calibration),
            "mean_superposition_error": np.mean([item[0]["superposition_relative_error"] for item in calibration], 0).tolist(),
            "mean_raw_sum_deviation_from_identity": float(np.linalg.norm(raw_sums - np.eye(2), axis=(-1, -2)).mean()),
            "orthogonal_impulse_energy_fraction_mean": float(np.mean([item[-1] for item in calibration])),
            "channel_mean_lag": channel_mean.mean(0).tolist(),
        }
        if records:
            result["validation"] = summarize_validation(
                records, nominal, kernel, design["frozen_delays_input_frames"][name],
                int(nominal[1] - nominal[0]),
            )
        groups[name] = result
        np.savez_compressed(a.root / (name + "_motion_kernel.npz"), kernel=kernel, per_image=kernels, nominal=nominal)
        print(name, len(calibration), "calibration", len(records), "validation",
              "mean lag", np.round(moment, 3).tolist(),
              "orthogonal energy", round(result["orthogonal_impulse_energy_fraction_mean"], 3), flush=True)
    (a.root / "summary.json").write_text(json.dumps(groups, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
