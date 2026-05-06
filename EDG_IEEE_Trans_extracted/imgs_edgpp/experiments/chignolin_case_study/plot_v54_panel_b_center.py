"""Panel b center — 2D free-energy landscape of 9,543 Chignolin DFT
conformations projected onto ViSNet reaction coordinates
  d_DG = ||O_{D3} - N_{G7}||,  d_ET = ||O_{E5} - N_{T8}||.

Density via Gaussian KDE (bw=0.06), pseudo-free-energy
  F = -log(p / p_{90th-pct})
clipped to [0, 5]. Colormap matches ViSNet Fig.4(b) reference (loaded from
the pre-extracted /tmp/visnet_cbar_colors.npy).

No conformer markers — see plot_v54_panel_b_with_markers.py for the
overlay variant.

Output: v54_panel_b_center.png
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
OUT = HERE / "v54_panel_b_center.png"


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

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, facecolor="white")
    cmap = make_visnet_cmap()
    sc = ax.scatter(
        d_DG[order], d_ET[order], c=F[order], cmap=cmap,
        vmin=0, vmax=5, s=12, alpha=0.85, marker="o",
        edgecolors="none", rasterized=True,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.7,
                        ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(d_DG.min() - 0.2, d_DG.max() + 0.2)
    ax.set_ylim(d_ET.min() - 0.15, d_ET.max() + 0.15)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
