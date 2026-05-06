"""Panel d — Reliability score vs simulation time on 4 EDG++ MD trajectories.

Story: the frozen reliability estimator (queryable at inference, unique to
EDG++) tracks per-frame distance from the teacher's training distribution.
4 independent 50 ps trajectories show differentiated OOD drift behavior:
one stays inside the training-reliability band, three drift outside to
varying degrees. This signal is decoupled from RMSD-from-initial (panel c)
and not available from a baseline that lacks the teacher.

Plot:
  x = simulation time (ps)
  y = reliability score (higher = more reliable)
  - 4 traj lines (one color per init id)
  - horizontal shaded band = training-set reliability (mean ± 2σ)

Output:
  panel_d_reliability.png  600x400 px @ 200dpi, transparent
  panel_d_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


HERE = Path(__file__).parent
RELIABILITY_NPZ = HERE / "md_reliability.npz"

DT_PS_PER_FRAME = 0.1

TRAJ_COLORS = {
    1571: "#4d8b4a",   # forest green
    3376: "#D4652A",   # paper orange (matches EDG++ identity in panels a/b/c)
    7419: "#d99000",   # amber
    9524: "#4878b0",   # steel blue
}

# Savitzky-Golay smoothing window (frames) — 21 frames = 2.1 ps at 0.1 ps/frame
SMOOTH_WINDOW = 21
SMOOTH_POLY = 3


def main() -> None:
    d = np.load(RELIABILITY_NPZ)
    init_ids = d["init_ids"].tolist()
    rel = d["reliability"]
    n_per = d["n_frames"].tolist()
    train_rel = d["train_reliability"]

    train_mean = float(train_rel.mean())
    train_std = float(train_rel.std())
    band_lo = train_mean - 2 * train_std
    band_hi = train_mean + 2 * train_std

    n_frames_max = rel.shape[1]
    t_ps = np.arange(n_frames_max) * DT_PS_PER_FRAME

    fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=200)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#bbbbbb", alpha=0.7)

    # Training-set band (mean ± 2σ) as legend element
    ax.axhspan(
        band_lo, band_hi,
        color="#888888", alpha=0.18, linewidth=0, zorder=1,
        label="Training set (mean ± 2σ)",
    )

    # 4 trajectory lines: Savitzky-Golay smoothed only
    for i, init_id in enumerate(init_ids):
        n = n_per[i]
        color = TRAJ_COLORS[init_id]

        raw = rel[i, :n]

        # Smoothed line — main visual element
        win = min(SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
        if win >= SMOOTH_POLY + 2:
            smoothed = savgol_filter(raw, win, SMOOTH_POLY)
        else:
            smoothed = raw
        ax.plot(
            t_ps[:n], smoothed,
            color=color, linewidth=1.4, alpha=0.95,
            label=f"Trajectory #{init_id}",
            solid_capstyle="round", solid_joinstyle="round",
            zorder=5,
        )

    ax.set_xlabel("Simulation time (ps)", fontsize=7, labelpad=2)
    ax.set_ylabel("Reliability score", fontsize=7, labelpad=2)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1)
    ax.set_xlim(0, t_ps[-1])

    leg = ax.legend(
        fontsize=5.5, loc="lower left", frameon=False,
        handlelength=1.4, handletextpad=0.5, labelspacing=0.35,
    )
    for line in leg.get_lines():
        line.set_linewidth(1.4)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)

    plt.tight_layout(pad=0.3)
    out_path = HERE / "panel_d_reliability.png"
    plt.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)

    stats = {
        "training_set": {
            "n": int(len(train_rel)),
            "mean": train_mean,
            "std": train_std,
            "band_2sigma": [band_lo, band_hi],
        },
        "trajectories": [
            {
                "init_id": int(init_id),
                "n_frames": int(n_per[i]),
                "duration_ps": float(n_per[i] * DT_PS_PER_FRAME),
                "mean": float(np.nanmean(rel[i, :n_per[i]])),
                "std": float(np.nanstd(rel[i, :n_per[i]])),
                "min": float(np.nanmin(rel[i, :n_per[i]])),
                "max": float(np.nanmax(rel[i, :n_per[i]])),
                "frac_outside_2sigma": float(
                    np.mean(
                        (rel[i, :n_per[i]] < band_lo) |
                        (rel[i, :n_per[i]] > band_hi)
                    )
                ),
                "delta_vs_train_mean": float(
                    np.nanmean(rel[i, :n_per[i]]) - train_mean
                ),
            }
            for i, init_id in enumerate(init_ids)
        ],
    }
    (HERE / "panel_d_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"→ {out_path}")
    for t in stats["trajectories"]:
        print(f"  init {t['init_id']:5d}  mean={t['mean']:+.3f}  "
              f"Δ_train={t['delta_vs_train_mean']:+.3f}  "
              f"frac_OOD={t['frac_outside_2sigma']*100:5.1f}%")


if __name__ == "__main__":
    main()
