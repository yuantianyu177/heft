"""Freeze annotation-free choices, summarize diagnostics, and plot evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_probe(folder: Path):
    records = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as f:
            data = {k: f[k] for k in f.files}
        data["name"] = path.stem
        data["metadata"] = json.loads(path.with_suffix(".json").read_text())
        errors = np.nan_to_num(data["errors"], nan=np.inf)
        valid = data["valid"]
        within = (errors[..., None] < np.array([1, 2, 4, 8, 16])).mean(-1)
        data["score"] = (within * valid).sum((-1, -2)) / valid.sum()
        data["epe_capped"] = (np.minimum(errors, 64) * valid).sum((-1, -2)) / valid.sum()
        data["file_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(data)
    if not records:
        raise ValueError(f"no probes in {folder}")
    return records


def choose(records, fixed_variant=None, layers=None):
    names = records[0]["variant_names"].tolist()
    layer_ids = records[0]["layers"].tolist()
    score = np.stack([r["score"] for r in records]).mean(0)
    epe = np.stack([r["epe_capped"] for r in records]).mean(0)
    allowed = np.ones(score.shape, dtype=bool)
    for i, variant in enumerate(names):
        if fixed_variant:
            allowed[:, :, i] = variant == fixed_variant
        elif variant.startswith("random_") or variant == "norm_matched":
            allowed[:, :, i] = False
    if layers:
        for i, layer in enumerate(layer_ids):
            if layer not in layers:
                allowed[i] = False
    objective = np.where(allowed, score - epe * 1e-7, -np.inf)
    indices = np.unravel_index(objective.argmax(), objective.shape)
    layer_index, head, variant_index = indices
    return dict(layer=int(layer_ids[layer_index]), head=int(head), variant=names[variant_index],
                synthetic_pck_avg=float(score[indices]*100), synthetic_capped_epe=float(epe[indices]),
                candidate_heads=int(allowed.any(-1).sum()), candidate_settings=int(allowed.sum()))


def freeze(args):
    destination = args.root / "results/frozen_annotation_free_choices.json"
    if destination.exists():
        raise FileExistsError(f"already frozen: {destination}")
    pools = {
        "translation": load_probe(args.root / "results/probe_wan_translate_t25"),
        "appearance": load_probe(args.root / "results/probe_wan_translate_appearance_t25"),
    }
    for records in pools.values():
        for r in records:
            assert r["metadata"]["feature_metadata"]["transform"] != "real"
            assert not r["metadata"]["selection_annotation_access"]
    pools["pooled"] = pools["translation"] + pools["appearance"]
    choices = {}
    for pool, records in pools.items():
        choices[pool] = {
            "joint": choose(records),
            "post_full_head": choose(records, "post_full"),
            "post_low15_head": choose(records, "post_low15"),
            "content_full_head": choose(records, "content_full"),
            "joint_layers14_20_sensitivity": choose(records, layers=list(range(14, 21))),
        }
    payload = {
        "primary": "pooled/joint", "choices": choices,
        "selection_uses_human_annotations": False,
        "metric": "macro synthetic direct nearest-neighbour PCK averaged over 1,2,4,8,16 px; capped EPE only breaks numerical ties",
        "source_records": [{"name": r["name"], "transform": r["metadata"]["feature_metadata"]["transform"],
                            "hash": r["file_hash"]} for r in pools["pooled"]],
        "warning": "This file must be frozen before inspecting real-video scores. Synthetic transfer itself may fail; retain negative results.",
    }
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["choices"], indent=2))


def bootstrap_mean(values, seed=42, draws=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(-1)
    return [float(values.mean()), *map(float, np.quantile(means, [.025, .975]))]


def summarize(args):
    records = load_probe(args.root / "results" / f"probe_{args.features}")
    names = records[0]["variant_names"].tolist()
    layers = records[0]["layers"].tolist()
    scores = np.stack([r["score"] for r in records]) * 100
    evaluation = np.array([r["metadata"]["feature_metadata"]["split"] == "evaluation" for r in records])
    frozen_path = args.root / "results/frozen_annotation_free_choices.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    summary = {"videos": [r["name"] for r in records], "frames": args.features,
               "annotation_free_choice_provenance": "Choices were frozen on Wan synthetic probes; applying those indices to Cosmos is a cross-backbone transfer, not Cosmos-specific head selection.",
               "evaluation_videos": int(evaluation.sum()), "calibration_videos": int((~evaluation).sum()),
               "metric": "direct NN PCK average in 256x256 coords; not full benchmark tracking",
               "population_mean": {n: bootstrap_mean(scores[..., i].mean((1, 2))) for i, n in enumerate(names)},
               "fixed_heads": {}, "annotation_free_choices": {}}
    for layer, head in [(18, 1), (15, 2), (18, 5), (20, 8), (14, 6)]:
        if layer not in layers:
            continue
        li = layers.index(layer)
        values = scores[:, li, head]
        summary["fixed_heads"][f"L{layer}H{head}"] = {n: bootstrap_mean(values[:, vi]) for vi, n in enumerate(names)}
        summary["fixed_heads"][f"L{layer}H{head}"]["paired_gain_over_low15"] = {
            n: bootstrap_mean(values[:, vi] - values[:, names.index("post_low15")]) for vi, n in enumerate(names)
        }
    if frozen and evaluation.any():
        for pool, choices in frozen["choices"].items():
            for key, choice in choices.items():
                li, hi, vi = layers.index(choice["layer"]), choice["head"], names.index(choice["variant"])
                summary["annotation_free_choices"][f"{pool}/{key}"] = {
                    **choice, "selection_backbone":"wan", "application_backbone":"cosmos2" if 'cosmos' in args.features else 'wan',
                    "evaluation_direct_pck_avg": bootstrap_mean(scores[evaluation, li, hi, vi]),
                }
    np.savez_compressed(args.root / "results" / f"summary_{args.features}.npz", scores=scores,
                        video_names=np.array(summary["videos"]), variants=np.array(names), evaluation=evaluation,
                        layers=np.array(layers))
    (args.root / "results" / f"summary_{args.features}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"population_mean": summary["population_mean"], "annotation_free_choices": summary["annotation_free_choices"]}, indent=2))
    make_plots(args, records, scores, names, layers)


def make_plots(args, records, scores, names, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    destination = args.root / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    means = scores.mean(0)
    vi = {n: names.index(n) for n in names}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, other, title in zip(axes, ["post_low15", "content_full", "rope_centered"],
                                ["Remove high-frequency channels", "Remove explicit phase", "Remove shared key component"], strict=True):
        x, y = means[..., vi["post_full"]].flatten(), means[..., vi[other]].flatten()
        ax.scatter(x, y, c=np.repeat(layers, means.shape[1]), cmap="viridis", s=13, alpha=.7)
        limit = max(x.max(), y.max()) + 3
        ax.plot([0, limit], [0, limit], "k--", linewidth=.7)
        ax.set(xlabel="Full post-RoPE direct PCK (%)", ylabel="Intervention direct PCK (%)", title=title)
        ax.text(.03, .95, f"{np.mean(y>x)*100:.1f}% of heads improve", transform=ax.transAxes, va="top")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"head_readout_interventions_{args.features}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fixed = [(18, 1), (15, 2), (20, 8)]
    display = ["post_full", "post_low15", "content_full", "time_removed", "space_removed", "high15_phase_removed", "rope_centered", "rope_high15_centered"]
    fig, ax = plt.subplots(figsize=(11, 4))
    width = .25
    for i, (layer, head) in enumerate(fixed):
        if layer not in layers:
            continue
        vals = [means[layers.index(layer), head, vi[n]] for n in display]
        ax.bar(np.arange(len(display)) + (i-1)*width, vals, width, label=f"L{layer} H{head}")
    ax.set_xticks(np.arange(len(display)), display, rotation=25, ha="right")
    ax.set_ylabel("Direct PCK average (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"fixed_head_controls_{args.features}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # Shared mean fraction is invariant to the final RoPE operation only after demodulation.
    mean_energy = np.mean([r["content_means"].reshape(len(layers), -1, 64, 2).__pow__(2).sum(-1) for r in records], axis=0)
    total_energy = np.mean([r["pair_energies"] for r in records], axis=0)
    fraction = mean_energy.sum(-1) / np.maximum(total_energy.sum(-1), 1e-10)
    gain = means[..., vi["rope_centered"]] - means[..., vi["post_full"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(fraction.flatten(), gain.flatten(), c=np.repeat(layers, means.shape[1]), cmap="viridis", s=14)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=.7)
    rho = spearmanr(fraction.flatten(), gain.flatten()).statistic
    axes[0].set(xlabel="Shared key mean energy / total energy", ylabel="Centering gain (PCK points)", title=f"Population diagnostic (Spearman {rho:.2f})")
    for layer, head in fixed:
        if layer in layers:
            li = layers.index(layer)
            axes[1].plot(mean_energy[li, head] / np.maximum(total_energy[li, head], 1e-10), label=f"L{layer} H{head}")
    for boundary in (22, 43):
        axes[1].axvline(boundary-.5, color="gray", linewidth=.6)
    axes[1].set(xlabel="Axial RoPE pair (t / h / w)", ylabel="Shared fraction per pair", ylim=(0, 1.05))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"shared_key_component_{args.features}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "summarize"])
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--features", default="wan_real_t25")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
