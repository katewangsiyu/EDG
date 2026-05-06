"""Mini bar charts under each Chignolin representative thumbnail in
panel (b). Three bars per conf: baseline / +EDG / +EDG++ — energy MAE
in kcal/mol. Shared Y-axis across all 6 so they compare visually.

Input:  reps_errors.json  (sourced from
        EDG-for-VisNet/scripts_chignolin/reps_errors.json)
Output: mini_bars/conf_{cid:06d}_bars.png  (6 files, transparent bg)

Color scheme:
  baseline = light grey   #b0b0b0
  +EDG     = amber        #d99000
  +EDG++   = paper orange #D4652A   (EDG++ identity color)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SRC_REPS = Path("/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/scripts_chignolin/reps_errors.json")
LOCAL_REPS = HERE / "reps_errors.json"
OUT_DIR = HERE / "mini_bars"

CONF_ORDER = [659, 6456, 5645, 569, 2950, 5031]  # match assembly row order

COLORS = ["#b0b0b0", "#d99000", "#D4652A"]
MODEL_LABELS = ["Base", "EDG", "EDG++"]

FIGSIZE = (1.55, 1.05)   # inches
DPI = 200


def main() -> None:
    if not LOCAL_REPS.exists():
        shutil.copy(SRC_REPS, LOCAL_REPS)
    reps = json.loads(LOCAL_REPS.read_text())

    # Shared y-limit: rounded ceiling above max bar height
    all_vals = [reps[str(c)][m] for c in CONF_ORDER for m in ("baseline", "edg", "edg_pp")]
    y_max = float(np.ceil(max(all_vals) * 1.08))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cid in CONF_ORDER:
        e = reps[str(cid)]
        vals = [e["baseline"], e["edg"], e["edg_pp"]]

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        x = np.arange(3)
        bars = ax.bar(x, vals, color=COLORS, width=0.72,
                      edgecolor="black", linewidth=0.5)

        # Value labels on top
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + y_max*0.02,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=6.5, color="black")
        ax.tick_params(axis="x", length=0, pad=1)
        ax.set_ylim(0, y_max)
        ax.set_yticks([0, y_max/2, y_max])
        ax.set_yticklabels([f"{v:.0f}" for v in [0, y_max/2, y_max]],
                           fontsize=6, color="black")
        ax.tick_params(axis="y", length=2, pad=1)
        ax.set_ylabel("MAE", fontsize=6.5, labelpad=1.5)

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.5)

        plt.tight_layout(pad=0.15)
        out = OUT_DIR / f"conf_{cid:06d}_bars.png"
        plt.savefig(out, dpi=DPI, transparent=True, bbox_inches="tight",
                    pad_inches=0.02)
        plt.close(fig)
        print(f"  conf {cid:5d}  base={vals[0]:5.2f}  edg={vals[1]:5.2f}  "
              f"edg++={vals[2]:5.2f}  →  {out.name}")

    print(f"\nShared y_max = {y_max} kcal/mol  (6 bars in {OUT_DIR})")


if __name__ == "__main__":
    main()
