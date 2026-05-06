"""Assemble Figure: Chignolin case study (panels a / b / c / d).

Layout (per paper caption):
  ─────────────────────────────────────────────────────────────────
  │            │   3 thumbnails (top: 659 / 6456 / 5645)          │
  │            │     w/ tiny bar chart underneath each            │
  │            │ ───────────────────────────────────────────      │
  │  panel a   │              v54_panel_b_center                   │
  │  (5AWL +   │           (2D free-energy landscape)              │
  │  anchors)  │ ───────────────────────────────────────────      │
  │            │   3 thumbnails (bot: 569 / 2950 / 5031)          │
  │            │     w/ tiny bar chart underneath each            │
  ─────────────┼─────────────────────────────────────────────────-
                          panel c (RMSD)
  ─────────────┼─────────────────────────────────────────────────-
                          panel d (Reliability)

Output: chignolin_case_study_final.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

HERE = Path(__file__).parent

# ============================================================================
# Inputs
# ============================================================================
PANEL_A = HERE / "panel_a_native.png"
PANEL_B = HERE / "v54_panel_b_center.png"
PANEL_C = HERE / "panel_c_rmsd.png"
PANEL_D = HERE / "panel_d_reliability.png"
THUMB = HERE / "representatives_v4"
BARS = HERE / "mini_bars"

OUT_PNG = HERE / "chignolin_case_study_final.png"
OUT_PDF = HERE / "chignolin_case_study_final.pdf"

# Thumbnail row order: matches plot_v54_panel_b_with_markers.py
TOP_ROW = [659, 6456, 5645]
BOT_ROW = [569, 2950, 5031]


def load(p: Path):
    if not p.exists():
        print(f"  [warn] missing: {p}")
        return None
    return imread(p)


def main() -> None:
    fig = plt.figure(figsize=(15.0, 11.5), dpi=200, facecolor="white")
    outer = GridSpec(2, 1, height_ratios=[1.4, 1.0], hspace=0.18,
                     left=0.02, right=0.99, top=0.97, bottom=0.04)

    # ───────────────────── TOP HALF: a + b ─────────────────────
    top = outer[0].subgridspec(1, 2, width_ratios=[0.85, 1.6], wspace=0.03)

    # Panel (a) — full height
    ax_a = fig.add_subplot(top[0, 0])
    ax_a.imshow(load(PANEL_A))
    ax_a.axis("off")
    ax_a.text(-0.02, 0.98, "(a)", transform=ax_a.transAxes,
              fontsize=15, fontweight="bold", va="top", ha="left")

    # Panel (b) — center scatter sandwiched between thumbnails+bars
    b_outer = top[0, 1].subgridspec(3, 1,
                                    height_ratios=[1.05, 0.95, 1.05],
                                    hspace=0.02)

    def thumbnail_row(spec, conf_ids, place_bars_below: bool):
        # 6 cells: 3 thumbnails + 3 bars (alternating row arrangement)
        row = spec.subgridspec(2, 3, height_ratios=[1.0, 0.55],
                               hspace=0.02, wspace=0.04)
        for col, cid in enumerate(conf_ids):
            tax = fig.add_subplot(row[0, col])
            tax.imshow(load(THUMB / f"conf_{cid:06d}.png"))
            tax.axis("off")
            bax = fig.add_subplot(row[1, col])
            bars_img = load(BARS / f"conf_{cid:06d}_bars.png")
            if bars_img is not None:
                bax.imshow(bars_img)
            bax.axis("off")

    thumbnail_row(b_outer[0, 0], TOP_ROW, place_bars_below=True)

    # Center scatter (panel b core)
    ax_b = fig.add_subplot(b_outer[1, 0])
    ax_b.imshow(load(PANEL_B))
    ax_b.axis("off")
    ax_b.text(-0.01, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=15, fontweight="bold", va="top", ha="left")

    thumbnail_row(b_outer[2, 0], BOT_ROW, place_bars_below=False)

    # ───────────────────── BOTTOM HALF: c + d ─────────────────────
    bot = outer[1].subgridspec(1, 2, wspace=0.06)

    ax_c = fig.add_subplot(bot[0, 0])
    img_c = load(PANEL_C)
    if img_c is not None:
        ax_c.imshow(img_c)
    ax_c.axis("off")
    ax_c.text(-0.02, 1.02, "(c)", transform=ax_c.transAxes,
              fontsize=15, fontweight="bold", va="top", ha="left")

    ax_d = fig.add_subplot(bot[0, 1])
    img_d = load(PANEL_D)
    if img_d is not None:
        ax_d.imshow(img_d)
    ax_d.axis("off")
    ax_d.text(-0.02, 1.02, "(d)", transform=ax_d.transAxes,
              fontsize=15, fontweight="bold", va="top", ha="left")

    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.savefig(OUT_PDF, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
