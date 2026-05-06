"""Panel (a) — Chignolin folded native (CLN025/5AWL) in ViSNet Fig.4(a) style:

  - grey smooth cartoon tube backbone
  - full ball-and-stick (all heavy atoms + hydrogens)
  - CPK coloring: C cyan, O red, N blue, H white
  - all 10 residues labeled at CA: Y1, Y2, D3, P4, E5, T6, G7, T8, W9, Y10
  - white background, AA, ambient occlusion

Reference: EDG-for-VisNet/fig4（a）.png from Wang et al. ViSNet paper.

Usage:
  python render_chignolin_panel_a_native.py --out-png /path/to/panel_a.png
"""
from __future__ import annotations

import argparse
import sys

import pymol2

DEFAULT_PDB = (
    "/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/"
    "Chignolin/raw/5AWL_H.pdb"
)

# CLN025 sequence YYDPETGTWY → residue labels at PDB seqid 1..10
RESI_LABELS = {
    "1": "Y1", "2": "Y2", "3": "D3", "4": "P4", "5": "E5",
    "6": "T6", "7": "G7", "8": "T8", "9": "W9", "10": "Y10",
}

# Visual constants — tuned to ViSNet Fig.4(a) thumbnail
TUBE_GREY = [0.62, 0.62, 0.64]    # neutral grey, slight blue tint


def render(pdb_path: str, out_png: str, width: int, height: int,
           rot_z: float, rot_y: float, rot_x: float,
           zoom_buffer: float) -> None:
    p = pymol2.PyMOL()
    p.start()
    c = p.cmd
    try:
        c.load(pdb_path, "chig")
        c.remove("chig and resn HOH")

        c.bg_color("white")
        c.set("ray_opaque_background", 0)
        c.set("ray_trace_mode", 0)
        c.set("antialias", 4)
        c.set("specular", 0.22)
        c.set("ambient", 0.48)
        c.set("ray_shadows", 0)

        # Smooth tube spline
        c.set("cartoon_sampling", 20)
        c.set("cartoon_loop_quality", 30)
        c.set("cartoon_tube_quality", 20)
        c.set("cartoon_refine", 5)
        c.set("cartoon_smooth_first", 1)
        c.set("cartoon_smooth_last", 1)
        c.set("cartoon_round_helices", 1)
        c.set("cartoon_smooth_loops", 1)

        # Depth shading
        c.set("ambient_occlusion_mode", 1)
        c.set("ambient_occlusion_scale", 15)

        c.set_color("tube_grey", TUBE_GREY)

        c.hide("everything", "chig")

        # Grey cartoon tube
        c.show("cartoon", "chig")
        c.cartoon("tube", "chig")
        c.set("cartoon_tube_radius", 0.30)
        c.color("tube_grey", "chig")

        # Ball-and-stick for all atoms (heavy + H)
        c.show("sticks", "chig")
        c.show("spheres", "chig")
        c.set("stick_radius", 0.10)
        c.set("sphere_scale", 0.22)
        c.set("sphere_scale", 0.13, "chig and elem H")    # smaller H spheres
        c.set("stick_ball", 0)
        c.set("valence", 0)

        # CPK with cyan C (ViSNet style)
        c.color("0x33D6D6", "chig and elem C")
        c.color("firebrick", "chig and elem O")
        c.color("marine",    "chig and elem N")
        c.color("white",     "chig and elem H")
        # Tryptophan ring N (W9 indole) often blue — already covered by elem N.

        # Residue labels at CA
        c.set("label_size", 22)
        c.set("label_color", "black")
        c.set("label_font_id", 7)
        c.set("label_outline_color", "white")
        c.set("label_position", (0, 0, 4.0))
        for resi, lab in RESI_LABELS.items():
            c.label(f"chig and resi {resi} and name CA", f'"{lab}"')

        c.orient("chig")
        if rot_z: c.rotate("z", rot_z)
        if rot_y: c.rotate("y", rot_y)
        if rot_x: c.rotate("x", rot_x)
        c.zoom("chig", buffer=zoom_buffer, complete=1)

        c.png(out_png, width=width, height=height, dpi=300, ray=1)
    finally:
        p.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=DEFAULT_PDB)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=1300)
    ap.add_argument("--rot-z", type=float, default=0.0)
    ap.add_argument("--rot-y", type=float, default=0.0)
    ap.add_argument("--rot-x", type=float, default=0.0)
    ap.add_argument("--zoom-buffer", type=float, default=4.0)
    args = ap.parse_args()
    render(args.pdb, args.out_png, args.width, args.height,
           args.rot_z, args.rot_y, args.rot_x, args.zoom_buffer)
    print(f"Saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
