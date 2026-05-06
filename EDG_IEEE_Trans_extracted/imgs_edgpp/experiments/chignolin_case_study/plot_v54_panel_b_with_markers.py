"""Panel b center with numbered conformer markers — variant A2 + markers.

Mirrors plot_v54_panel_b_center.py exactly, then overlays 6 numbered circles
at each representative conformer's (d_DG, d_ET) coordinate.

Numbering follows the assembly script's row order:
  Top row (above scatter):    1=659  2=6456  3=5645
  Bottom row (below scatter): 4=569  5=2950  6=5031

Output: v54_panel_b_with_markers.png  (does NOT replace v54_panel_b_center.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).parent
NPZ = Path("/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/AI2BMD/chignolin_data/raw/chignolin.npz")
ANCHOR = HERE / "anchor_atoms_mapping.json"
VISNET_CBAR_NPY = "/tmp/visnet_cbar_colors.npy"
OUT = HERE / "v54_panel_b_with_markers.png"

# (cid, number) — number matches assembly row order
# Top row left→right: 1, 2, 3   Bottom row left→right: 4, 5, 6
MARKERS = [
    (659,  1),   # folded_1 (top row col 0)
    (6456, 2),   # transition_2 (top row col 1)
    (5645, 3),   # unfolded_1 (top row col 2)
    (569,  4),   # folded_2 (bottom row col 0)
    (2950, 5),   # transition_1 (bottom row col 1)
    (5031, 6),   # unfolded_2 (bottom row col 2)
]


def make_visnet_cmap():
    colors = np.load(VISNET_CBAR_NPY)
    trim = max(1, int(len(colors) * 0.005))
    return LinearSegmentedColormap.from_list("visnet_jet_r", colors[trim:-trim], N=256)


def main() -> None:
    anchor = json.load(open(ANCHOR))["anchor_atoms_npz_idx"]
    D3_O, G7_N, E5_O, T8_N = anchor["D3_O"], anchor["G7_N"], anchor["E5_O"], anchor["T8_N"]
    d = np.load(NPZ)
    N_FRAMES, N_ATOMS = len(d["N"]), int(d["N"][0])
    R = d["R"].reshape(N_FRAMES, N_ATOMS, 3)
    d_DG = np.linalg.norm(R[:, D3_O] - R[:, G7_N], axis=-1)
    d_ET = np.linalg.norm(R[:, E5_O] - R[:, T8_N], axis=-1)

    xy = np.vstack([d_DG, d_ET])
    kde = gaussian_kde(xy, bw_method=0.06)
    density = kde(xy)
    ref = np.percentile(density, 90)
    F = -np.log(density / ref + 1e-12)
    F = np.minimum(np.maximum(F, 0), 5.0)
    order = (-F).argsort()

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=200, facecolor="white")
    cmap = make_visnet_cmap()
    sc = ax.scatter(
        d_DG[order], d_ET[order], c=F[order], cmap=cmap,
        vmin=0, vmax=5, s=12, alpha=0.85, marker="o",
        edgecolors="none", rasterized=True,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.7,
                        ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.tick_params(labelsize=8)

    # Strip axis decoration (unchanged from v54)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(d_DG.min() - 0.2, d_DG.max() + 0.2)
    ax.set_ylim(d_ET.min() - 0.15, d_ET.max() + 0.15)

    # === Numbered conformer markers ===
    for cid, num in MARKERS:
        x = float(d_DG[cid])
        y = float(d_ET[cid])
        # White-filled black-outlined circle, large, on top of all scatter points
        ax.scatter(
            x, y, s=240, facecolor="white", edgecolor="black",
            linewidths=1.6, zorder=20,
        )
        # Number label inside
        ax.text(
            x, y, str(num),
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="black", zorder=21,
        )

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT}")
    print("Conformer markers (number → cid → coords):")
    for cid, num in MARKERS:
        x = float(d_DG[cid]); y = float(d_ET[cid])
        print(f"  {num} → cid {cid:4d}  (d_DG={x:.2f}, d_ET={y:.2f})")


if __name__ == "__main__":
    main()
