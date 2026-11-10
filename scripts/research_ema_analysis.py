"""Analytic and simulated phase mixing under ordinary descriptor EMA.

This is a controlled mathematical sanity experiment, not a model benchmark.
For constant canonical content c and p_t=p_0+v*t, post-RoPE EMA has transfer
H(omega*v)=alpha / (1-(1-alpha)*exp(-i*omega*v)) relative to the current frame.
"""
import csv
import json
from pathlib import Path

import numpy as np


def main():
    root = Path("/share/yty/heft")
    rows = []
    max_error = 0
    alpha = .05
    for velocity in (.0, .25, 1., 2.):
        for frequency in np.geomspace(1e-4, 1, 100):
            step = frequency * velocity
            value = 1 + 0j
            for t in range(1, 2049):
                value = (1-alpha)*value + alpha*np.exp(1j*step*t)
            relative = value*np.exp(-1j*step*2048)
            expected = alpha / (1-(1-alpha)*np.exp(-1j*step))
            max_error = max(max_error, abs(relative-expected))
            rows.append(dict(alpha=alpha, velocity=velocity, frequency=frequency,
                             observed_gain=abs(relative), predicted_gain=abs(expected),
                             predicted_lag=float(np.angle(expected))))
    assert max_error < 1e-10
    with (root / "results/ema_phase_mixing.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"experiment": "constant canonical content; constant token-space velocity; 2048 EMA updates",
              "max_complex_error": max_error,
              "temporal_highest_pair_gain_at_alpha005": float(abs(alpha/(1-(1-alpha)*np.exp(-1j)))),
              "canonical_ema_gain": 1., "status": "analytic prediction verified; real-model relevance must be tested separately"}
    (root / "results/ema_phase_mixing.json").write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for velocity in (.25, 1., 2.):
        r = [v for v in rows if v["velocity"] == velocity]
        axes[0].semilogx([v["frequency"] for v in r], [v["predicted_gain"] for v in r], label=f"v = {velocity:g} tokens/frame")
        axes[1].semilogx([v["frequency"] for v in r], [v["predicted_lag"] for v in r])
    axes[0].axhline(1, color="k", linestyle="--", linewidth=.8, label="Canonical EMA")
    axes[0].set(xlabel="RoPE angular frequency", ylabel="EMA amplitude / true amplitude", ylim=(0, 1.08))
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set(xlabel="RoPE angular frequency", ylabel="Lag relative to current frame (radians)")
    fig.suptitle("Coordinate-frame mixing in descriptor EMA (analytic control, alpha = 0.05)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"figures/ema_phase_mixing.{suffix}", dpi=180, bbox_inches="tight")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
