"""Render Chignolin (5AWL) for Panel (a) of the case-study figure.

Output: a high-resolution cartoon + side-chain stick rendering of 5AWL mini-protein,
styled to match the ViSNet Fig.4a convention (gray backbone cartoon, colored sidechain sticks,
residue labels near C-alpha).

Run in the pymol_edgpp conda env:
    conda activate pymol_edgpp
    python render_chignolin_panel_a.py --pdb 5AWL.pdb --out chignolin_cartoon.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymol2


def render(pdb_path: str, out_path: str, width: int = 900, height: int = 700) -> None:
    """ViSNet Fig.4a style: gray backbone cartoon + single-color (light purple) side chains.
    Side chains shown as ball-and-stick with CPK atom colouring (C=light purple, O=red, N=blue).
    Labels: one-letter residue code + index at each CA.
    """
    p = pymol2.PyMOL()
    p.start()
    c = p.cmd

    c.load(pdb_path, "chig")
    c.remove("resn HOH")
    c.bg_color("white")
    c.set("ray_opaque_background", 1)
    c.set("ray_trace_mode", 0)            # 0 = smooth shading, no black outline (VisNet Fig.4a style)
    c.set("antialias", 2)
    c.set("specular", 0.2)
    c.set("ambient", 0.35)
    # NO cartoon ribbon — pure ball-and-stick on all heavy atoms.
    # Cartoon ribbon was occluding side-chain bonds passing behind it (especially long Glu5 -COO⁻
    # that wraps through CB-CG-CD around the hairpin), creating visual disconnection.
    # Pure ball-and-stick guarantees every bond is visible.

    c.set_color("visnet_cyan", [0.40, 0.85, 0.88])
    c.h_add("chig")
    c.hide("everything", "chig")

    SPH_HEAVY = 0.18      # small balls so structure isn't crowded
    SPH_H     = 0.10
    STK       = 0.13      # thin sticks but ~= sphere radius for fused look

    # Backbone heavy atoms (N, C, O, CA) — gray ribbon replaced by these
    sel_bb = "chig and (name C+N+O+CA)"
    c.show("sticks", sel_bb)
    c.show("spheres", sel_bb)
    c.set("sphere_scale", SPH_HEAVY, sel_bb)
    # Color backbone the same way as side chain
    c.color("visnet_cyan", f"{sel_bb} and elem C")
    c.color("firebrick",   f"{sel_bb} and elem O")
    c.color("marine",      f"{sel_bb} and elem N")

    # Side-chain heavy atoms (CB onwards)
    sel_sc = "chig and not (name C+N+O+CA) and not elem H"
    c.show("sticks", sel_sc)
    c.show("spheres", sel_sc)
    c.set("sphere_scale", SPH_HEAVY, sel_sc)
    c.color("visnet_cyan", f"{sel_sc} and elem C")
    c.color("firebrick",   f"{sel_sc} and elem O")
    c.color("marine",      f"{sel_sc} and elem N")

    # All hydrogens (small white)
    sel_h = "chig and elem H"
    c.show("sticks", sel_h)
    c.show("spheres", sel_h)
    c.set("sphere_scale", SPH_H, sel_h)
    c.color("white", sel_h)

    c.set("stick_radius", STK)
    c.set("stick_ball", 0)
    c.set("valence", 0)

    # Residue labels at CA — VisNet Fig.4a style: bold black text with white outline,
    # offset slightly off the atom position so the label doesn't overlap the structure.
    c.set("label_size", 22)
    c.set("label_color", "black")
    c.set("label_font_id", 7)
    c.set("label_outline_color", "white")
    c.set("float_labels", 1)
    c.set("label_relative_mode", 0)
    c.set("label_position", (3.5, 3.5, 0))   # bigger offset so labels float OUTSIDE the structure
    c.label("chig and name CA", "oneletter+str(resi)")

    c.orient("chig")
    c.rotate("z", 90)   # vertical hairpin axis — spreads labels along strand direction (cleanest layout)
    c.zoom("chig", buffer=7.0, complete=1)

    c.png(out_path, width=width, height=height, dpi=300, ray=1)
    p.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=700)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render(args.pdb, args.out, args.width, args.height)
    print(f"[panel-a] saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
