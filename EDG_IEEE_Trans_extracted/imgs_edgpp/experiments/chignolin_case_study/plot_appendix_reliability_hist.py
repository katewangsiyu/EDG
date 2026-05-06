"""Appendix figure — Chignolin training-set reliability distribution
versus the rMD17 reliability range (cited mean −0.68 to −0.79 in body).

Shows that Chignolin reliability is shifted toward lower values yet
overlaps the most-reliable rMD17 regime — exactly the regime where
naive distillation hurts and selective distillation helps.

Input:  train_reliability.npy
Output: appendix_reliability_hist.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
IN_NPY = HERE / "train_reliability.npy"
OUT_PNG = HERE / "appendix_reliability_hist.png"
OUT_PDF = HERE / "appendix_reliability_hist.pdf"

# Cited rMD17 reliability mean range (paper line 194)
RMD17_MEAN_LO, RMD17_MEAN_HI = -0.79, -0.68
RMD17_STD = 0.055   # cited as ~0.055 (also from line 194)


def main() -> None:
    chig = np.load(IN_NPY)
    chig_mean = float(chig.mean())
    chig_std = float(chig.std())

    fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=200)
    bins = np.linspace(chig.min() - 0.05, RMD17_MEAN_HI + 4*RMD17_STD, 35)
    ax.hist(chig, bins=bins, color="#4878b0", edgecolor="white",
            linewidth=0.4, alpha=0.92, label=f"Chignolin (n={len(chig)})")

    # rMD17 reference band: across the cited per-molecule mean range
    ax.axvspan(RMD17_MEAN_LO - 2*RMD17_STD, RMD17_MEAN_HI + 2*RMD17_STD,
               color="#cccccc", alpha=0.35, linewidth=0,
               label="rMD17 (cited mean ± 2σ range)")
    ax.axvline(chig_mean, color="#c93a3a", linewidth=1.2, linestyle="--",
               label=f"Chignolin mean = {chig_mean:.2f}")
    ax.axvline(RMD17_MEAN_LO, color="#666666", linewidth=0.5)
    ax.axvline(RMD17_MEAN_HI, color="#666666", linewidth=0.5)

    ax.set_xlabel("Reliability score (higher = more reliable)",
                  fontsize=8, labelpad=2)
    ax.set_ylabel("# training conformations", fontsize=8, labelpad=2)
    ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)

    leg = ax.legend(fontsize=6.5, frameon=False, loc="upper left",
                    handlelength=1.6, handletextpad=0.5, labelspacing=0.3)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    print(f"Chignolin reliability: mean={chig_mean:.3f}  std={chig_std:.3f}  "
          f"min={chig.min():.3f}  max={chig.max():.3f}")
    overlap = (chig > RMD17_MEAN_LO - 2*RMD17_STD).sum()
    print(f"Chignolin frames in rMD17-2σ window: {overlap}/{len(chig)}  "
          f"({overlap/len(chig)*100:.1f}%)")


if __name__ == "__main__":
    main()
