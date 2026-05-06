"""Panel (c) — Cα-RMSD vs the initial conformation of each trajectory,
over 50 ps Langevin NVT MD driven by ViSNet (baseline) and ViSNet+EDG++.

10 trajectories per method; mean ± std band over surviving trajectories
at each time t (NaN-aware).

Input:  panel_c_rmsd.npz  (produced by compute_rmsd.py)
Output: panel_c_rmsd.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
NPZ = HERE / "panel_c_rmsd.npz"
OUT = HERE / "panel_c_rmsd.png"

C_BASE = "#7a7a7a"   # baseline grey
C_EPP  = "#c93a3a"   # EDG++ red (matches original figure)


def alive_mean_std(arr):
    n_alive = (~np.isnan(arr)).sum(axis=0)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    return mu, sd, n_alive


def main() -> None:
    d = np.load(NPZ)
    base = d["baseline"]; epp = d["edg_pp"]; t_ps = d["t_ps"]

    mu_b, sd_b, alive_b = alive_mean_std(base)
    mu_e, sd_e, alive_e = alive_mean_std(epp)

    fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=200, facecolor="white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#bbbbbb", alpha=0.7)

    valid_b = alive_b >= 2
    valid_e = alive_e >= 2
    ax.fill_between(t_ps[valid_b], mu_b[valid_b] - sd_b[valid_b],
                    mu_b[valid_b] + sd_b[valid_b],
                    color=C_BASE, alpha=0.20, linewidth=0)
    ax.fill_between(t_ps[valid_e], mu_e[valid_e] - sd_e[valid_e],
                    mu_e[valid_e] + sd_e[valid_e],
                    color=C_EPP, alpha=0.18, linewidth=0)
    ax.plot(t_ps[valid_b], mu_b[valid_b], color=C_BASE, linewidth=1.4,
            label="ViSNet")
    ax.plot(t_ps[valid_e], mu_e[valid_e], color=C_EPP, linewidth=1.4,
            label="ViSNet + EDG++")

    ax.set_xlabel("Simulation time (ps)", fontsize=7, labelpad=2)
    ax.set_ylabel(r"C$\alpha$ RMSD vs. initial (Å)", fontsize=7, labelpad=2)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1)
    ax.set_xlim(0, t_ps[-1])
    ax.set_ylim(0, max(np.nanmax(mu_b + sd_b), np.nanmax(mu_e + sd_e)) * 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)

    leg = ax.legend(fontsize=6, loc="upper left", frameon=False,
                    handlelength=1.6, handletextpad=0.5, labelspacing=0.3)
    for line in leg.get_lines():
        line.set_linewidth(1.4)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT, dpi=200, transparent=True, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
