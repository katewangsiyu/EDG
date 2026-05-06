"""Render Chignolin (CLN025, PDB 5AWL) panel (a) v24 — two styles for user to pick.

Run in pymol_edgpp conda env:
    conda activate pymol_edgpp
    python render_chignolin_panel_a_v24.py --style full   --out v24_full.png
    python render_chignolin_panel_a_v24.py --style simple --out v24_simple.png

Default PDB: GEOM3D/examples_3D/dataset/Chignolin/raw/5AWL_H.pdb (CLN025 with H, 165 atoms).
This is the same native folded reference cited in experiment.tex line 223.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymol2

DEFAULT_PDB = (
    "/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/"
    "Chignolin/raw/5AWL_H.pdb"
)

# 4 anchor residues from experiment.tex caption (panel b reaction coordinates d_DG and d_ET)
ANCHOR_RESI = ["3", "5", "7", "8"]   # D3, E5, G7, T8
# Plus aromatic side chains worth showing for visual context (only used in 'simple' style)
AROMATIC_RESI = ["1", "9", "10"]     # Y1, W9, Y10
# Label only key residues per user decision: 4 anchors (caption) + W9 (hydrophobic core)
KEY_LABELS_RESI = ["3", "5", "7", "8", "9"]   # D3, E5, G7, T8, W9

# vdW radii (Å) per user spec
VDW = {"C": 1.5, "N": 1.4, "O": 1.4, "H": 0.8}


def setup_common(c) -> None:
    """Background, ray-trace, lighting — shared by both styles."""
    c.bg_color("white")
    c.set("ray_opaque_background", 0)        # transparent for paper insertion
    c.set("ray_trace_mode", 0)               # smooth shading, no outline
    c.set("antialias", 2)
    c.set("specular", 0.25)
    c.set("ambient", 0.4)
    c.set("ray_shadows", 0)                  # cleaner for figures
    c.set_color("cpk_cyan", [0.40, 0.85, 0.88])


def add_residue_labels(c, sel_obj: str) -> None:
    """Place residue labels closer to CA, using screen-space radial direction so labels
    push outward in the camera plane (not behind the cartoon tube). Must be called AFTER
    c.orient/rotate so view matrix reflects the final camera frame.
    """
    import numpy as np
    c.set("label_size", 7)
    c.set("label_color", "black")
    c.set("label_font_id", 7)
    c.set("label_outline_color", "white")
    c.set("float_labels", 1)

    # Camera axes
    view = c.get_view()
    R = np.array(view[:9]).reshape(3, 3)
    right, up = R[0], R[1]

    # Get CA + sidechain-tip coords per residue
    ca_coords, sc_tips = {}, {}
    for resi in KEY_LABELS_RESI:
        ca_arr = c.get_coords(f"{sel_obj} and name CA and resi {resi}")
        if ca_arr is None or len(ca_arr) == 0:
            continue
        ca_coords[resi] = np.array(ca_arr[0])
        # Furthest sidechain heavy atom from CA = "tip"
        sc_arr = c.get_coords(
            f"{sel_obj} and resi {resi} and not (name N+CA+C+O+OXT) and not elem H"
        )
        if sc_arr is not None and len(sc_arr) > 0:
            sc_arr = np.array(sc_arr)
            dists = np.linalg.norm(sc_arr - ca_coords[resi], axis=1)
            sc_tips[resi] = sc_arr[int(np.argmax(dists))]

    if not ca_coords:
        return

    # Screen centroid (used as fallback for Gly which has no sidechain)
    screen_pos = {
        resi: np.array([np.dot(ca, right), np.dot(ca, up)])
        for resi, ca in ca_coords.items()
    }
    screen_centroid = np.mean(list(screen_pos.values()), axis=0)

    # Per-residue absolute screen offsets in Å (overrides sidechain-direction logic).
    # +x = camera right, +y = camera up.
    OFFSET_OVERRIDES = {
        "9": np.array([4.5,  2.0]),   # W9: less right, less up per user (was 5.5,3.0)
        "5": np.array([5.0, -1.5]),   # E5: 5 Å right of CA (out of hairpin corner)
    }

    push = 4.5
    for resi, ca in ca_coords.items():
        if resi in OFFSET_OVERRIDES:
            # Raw Å offset, no normalization
            offset_screen = OFFSET_OVERRIDES[resi]
            offset_world = offset_screen[0] * right + offset_screen[1] * up
        else:
            if resi in sc_tips:
                d3 = sc_tips[resi] - ca
                d_screen = np.array([np.dot(d3, right), np.dot(d3, up)])
                norm = np.linalg.norm(d_screen)
                if norm < 0.1:
                    d_screen = screen_pos[resi] - screen_centroid
                    norm = np.linalg.norm(d_screen) or 1.0
            else:
                d_screen = screen_pos[resi] - screen_centroid
                norm = np.linalg.norm(d_screen) or 1.0
            direction = d_screen / norm
            offset_world = (direction[0] * right + direction[1] * up) * push
        c.set("label_position", tuple(offset_world.tolist()))
        c.label(f"{sel_obj} and name CA and resi {resi}", "oneletter+str(resi)")


def render_full(c, obj: str) -> None:
    """User spec: gray cartoon tube backbone + ALL atoms as CPK sticks + spheres (incl H)."""
    c.h_add(obj)
    c.hide("everything", obj)

    # Gray cartoon tube backbone (user spec: radius ~0.1 nm = 1.0 Å)
    c.show("cartoon", obj)
    c.cartoon("tube", obj)
    c.set("cartoon_tube_radius", 0.5)        # 1.0 Å looks too fat against sticks; 0.5 reads cleaner
    c.set("cartoon_color", "gray60", obj)
    c.set("cartoon_transparency", 0.35)      # let backbone atoms show through

    # All side-chain heavy atoms + side-chain H — sticks + spheres
    sel_heavy = f"{obj} and not elem H"
    sel_h     = f"{obj} and elem H"

    c.show("sticks",  sel_heavy)
    c.show("spheres", sel_heavy)
    c.show("sticks",  sel_h)
    c.show("spheres", sel_h)

    # Sphere radii: user spec is full vdW, but for 166-atom mini-protein this fuses into
    # an opaque blob. Per user feedback shrink another 2× from 0.5 → 0.25.
    SHRINK = 0.25
    c.set("sphere_scale", SHRINK * VDW["C"] / 1.7,  f"{obj} and elem C")
    c.set("sphere_scale", SHRINK * VDW["N"] / 1.55, f"{obj} and elem N")
    c.set("sphere_scale", SHRINK * VDW["O"] / 1.52, f"{obj} and elem O")
    c.set("sphere_scale", SHRINK * VDW["H"] / 1.20, f"{obj} and elem H")

    # CPK coloring per user spec
    c.color("cpk_cyan",  f"{obj} and elem C")
    c.color("firebrick", f"{obj} and elem O")
    c.color("marine",    f"{obj} and elem N")
    c.color("white",     f"{obj} and elem H")

    c.set("stick_radius", 0.13)              # ≈0.07 nm (user spec 0.05-0.07)
    c.set("stick_ball", 0)
    c.set("valence", 0)


def render_simple(c, obj: str) -> None:
    """ViSNet Fig.4a style with strict size relationships per user spec:
       - cartoon tube radius = 0.20 (gray backbone)
       - stick radius = tube/4 = 0.05
       - heavy atom (C/N/O) sphere radius = tube radius = 0.20 (same diameter as tube)
       - H sphere radius = C/2 = 0.10
    """
    TUBE_R = 0.20
    HEAVY_R = 0.30            # diameter 0.60
    H_R = 0.175               # diameter 0.35 (user spec)
    STICK_R = 0.10            # diameter 0.20 (user spec)
    OH_STICK_R = 0.115        # O-H bond diameter 0.23 (user spec)

    c.h_add(obj)
    c.hide("everything", obj)

    # Gray cartoon tube backbone
    c.show("cartoon", obj)
    c.cartoon("tube", obj)
    c.set("cartoon_tube_radius", TUBE_R)
    c.set("cartoon_color", "gray60", obj)
    c.set("cartoon_side_chain_helper", 1)

    # Sidechain ball-and-stick: heavy atoms + SIDECHAIN polar H (bonded to sidechain N/O,
    # NOT backbone N — those H would dangle since backbone N is hidden by cartoon)
    # Also exclude OXT (C-terminal carboxyl O), which only bonds to backbone C.
    sel_sc = (
        f"{obj} and not (name N+C+O+OXT) and "
        f"(not elem H or (neighbor (elem N+O and not name N+C+O+OXT)))"
    )
    c.show("sticks", sel_sc)
    c.show("spheres", sel_sc)

    # CPK colors
    c.color("cpk_cyan",  f"{sel_sc} and elem C")
    c.color("firebrick", f"{sel_sc} and elem O")
    c.color("marine",    f"{sel_sc} and elem N")
    c.color("white",     f"{sel_sc} and elem H")

    # Per-element sphere scales (PyMol default vdW: C=1.7, N=1.55, O=1.52, H=1.20)
    c.set("sphere_scale", HEAVY_R / 1.7,  f"{sel_sc} and elem C")
    c.set("sphere_scale", HEAVY_R / 1.55, f"{sel_sc} and elem N")
    c.set("sphere_scale", HEAVY_R / 1.52, f"{sel_sc} and elem O")
    c.set("sphere_scale", H_R / 1.20,     f"{sel_sc} and elem H")

    c.set("stick_radius", STICK_R)
    c.set("stick_ball", 0)
    c.set("valence", 0)

    # Per-bond override: O-H bonds slightly thinner per user spec
    c.set_bond("stick_radius", OH_STICK_R, f"{obj} and elem O", f"{obj} and elem H")

    # Hide selected residue sidechains to reduce visual overlap (per user spec).
    # Backbone is unaffected (cartoon tube still draws it).
    HIDE_SIDECHAIN_RESI = ["2", "10"]  # Y2 + Y10 (keep Y1 only on N/C-term)
    if HIDE_SIDECHAIN_RESI:
        sel_hide = f"{obj} and resi {'+'.join(HIDE_SIDECHAIN_RESI)} and not (name N+CA+C+O)"
        c.hide("sticks", sel_hide)
        c.hide("spheres", sel_hide)


def render(pdb_path: str, out_path: str, style: str, width: int, height: int,
           rot_z: float, rot_y: float, rot_x: float, zoom_buffer: float) -> None:
    p = pymol2.PyMOL()
    p.start()
    c = p.cmd

    obj = "chig"
    c.load(pdb_path, obj)
    c.remove("resn HOH")
    setup_common(c)

    if style == "full":
        render_full(c, obj)
    elif style == "simple":
        render_simple(c, obj)
    else:
        raise ValueError(f"unknown style: {style!r}")

    c.orient(obj)
    if rot_z: c.rotate("z", rot_z)
    if rot_y: c.rotate("y", rot_y)
    if rot_x: c.rotate("x", rot_x)
    c.zoom(obj, buffer=zoom_buffer, complete=1)

    # Labels added AFTER orient/rotate so screen-space radial uses final camera frame
    add_residue_labels(c, obj)

    c.png(out_path, width=width, height=height, dpi=300, ray=1)
    p.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=DEFAULT_PDB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", choices=["full", "simple"], required=True)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--rot-z", type=float, default=90.0)
    ap.add_argument("--rot-y", type=float, default=0.0)
    ap.add_argument("--rot-x", type=float, default=0.0)
    ap.add_argument("--zoom-buffer", type=float, default=14.0)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render(args.pdb, args.out, args.style, args.width, args.height,
           args.rot_z, args.rot_y, args.rot_x, args.zoom_buffer)
    print(f"[panel-a v24/{args.style} z{args.rot_z:+g} y{args.rot_y:+g} x{args.rot_x:+g}] saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
