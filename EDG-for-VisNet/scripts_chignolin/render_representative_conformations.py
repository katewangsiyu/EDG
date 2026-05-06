"""Render 6 representative Chignolin conformations as cartoon+rainbow thumbnails.

Strategy:
1. Use 5AWL.pdb as topology template (93 heavy atoms with correct residue/atom names).
2. For each target npz conformation index, substitute coords via pdb_to_npz heavy mapping.
3. Render with cartoon + per-residue rainbow side-chain sticks (same style as Panel (a)).

Usage:
    python render_representative_conformations.py \\
        --pdb 5AWL.pdb --mapping-json pdb_to_npz_heavy_mapping.json \\
        --npz chignolin.npz --conf-ids 1234 2345 ... 6789 \\
        --out-dir imgs/representatives
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pymol2


def parse_pdb_heavy(pdb_path: str):
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                lines.append(line.rstrip("\n"))
    return lines


def write_pdb_with_coords(template_lines: List[str], positions: np.ndarray, out_path: str) -> None:
    """Write a new PDB file using template_lines atom records but replacing xyz with positions.

    positions: (93, 3) for heavy atoms, matching template_lines order.
    """
    assert len(template_lines) == positions.shape[0]
    with open(out_path, "w") as f:
        for line, pos in zip(template_lines, positions):
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            # PDB fixed-width format: cols 31-54 for xyz (8.3f each)
            new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            f.write(new_line + "\n")
        f.write("END\n")


def render_cartoon(pdb_path: str, out_png: str, width: int = 500, height: int = 400) -> None:
    """Render in ViSNet Fig.4 style:
    - Gray cartoon ribbon for the backbone
    - Side chains: ball-and-stick with LIGHT PURPLE carbon, red oxygen, blue nitrogen
    """
    p = pymol2.PyMOL()
    p.start()
    c = p.cmd
    c.load(pdb_path, "chig")
    c.bg_color("white")
    c.set("ray_opaque_background", 1)
    c.set("ray_trace_mode", 0)            # smooth shading, no black outline (VisNet Fig.4 style)
    c.set("specular", 0.2)
    c.set("ambient", 0.35)
    c.set("ray_trace_color", "black")
    c.set("antialias", 2)
    # Backbone: thinner ribbon so CA spheres are visible sitting on it
    c.set("cartoon_tube_radius", 0.32)
    c.set("cartoon_fancy_helices", 1)
    c.set("cartoon_smooth_loops", 1)
    c.set("cartoon_loop_radius", 0.28)

    c.set_color("visnet_cyan", [0.40, 0.85, 0.88])

    # Template PDB is heavy-atom only — add hydrogens for the characteristic small white spheres.
    c.h_add("chig")

    c.hide("everything", "chig")
    c.show("cartoon", "chig")
    c.set("cartoon_color", "gray55", "chig")

    # Ball/stick proportion matched to ViSNet Fig.4 (~1.9:1)
    SPH_HEAVY = 0.25
    SPH_H     = 0.14
    STK       = 0.22

    sel_ca = "chig and name CA"
    c.show("spheres", sel_ca)
    c.set("sphere_scale", SPH_HEAVY, sel_ca)
    c.color("visnet_cyan", sel_ca)

    sel_sc_all = "chig and not (name C+N+O) and not elem H"
    c.show("sticks", sel_sc_all)
    sel_sc = "chig and not (name C+N+O+CA) and not elem H"
    c.show("spheres", sel_sc)
    c.set("sphere_scale", SPH_HEAVY, sel_sc)
    c.color("visnet_cyan", f"{sel_sc_all} and elem C")
    c.color("firebrick",   f"{sel_sc} and elem O")
    c.color("marine",      f"{sel_sc} and elem N")

    sel_sc_h = "chig and not (name C+N+O+CA) and elem H"
    c.show("sticks", sel_sc_h)
    c.show("spheres", sel_sc_h)
    c.set("sphere_scale", SPH_H, sel_sc_h)
    c.color("white", sel_sc_h)
    sel_ca_h = "chig and name HA*"
    c.show("sticks", sel_ca_h)
    c.show("spheres", sel_ca_h)
    c.set("sphere_scale", SPH_H, sel_ca_h)
    c.color("white", sel_ca_h)

    c.set("stick_radius", STK)
    c.set("stick_ball", 0)
    c.set("valence", 0)

    c.orient("chig")
    c.zoom("chig", buffer=0.8, complete=1)

    c.png(out_png, width=width, height=height, dpi=150, ray=1)
    p.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="5AWL.pdb template")
    ap.add_argument("--mapping-json", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--conf-ids", type=int, nargs="+", required=True, help="conformation indices to render")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--height", type=int, default=400)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mapping_json) as f:
        mapping = json.load(f)
    pdb_to_npz = mapping["pdb_to_npz"]
    n_heavy = len(pdb_to_npz)

    template_lines = parse_pdb_heavy(args.pdb)
    assert len(template_lines) == n_heavy, f"template lines ({len(template_lines)}) != mapping ({n_heavy})"

    data = np.load(args.npz)
    n_atoms = 166
    R_all = data["R"].reshape(-1, n_atoms, 3).astype(np.float32)

    for conf_id in args.conf_ids:
        pos = R_all[conf_id]
        heavy_pos = np.stack([pos[pdb_to_npz[i]] for i in range(n_heavy)], axis=0)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            tmp_pdb = tf.name
        try:
            write_pdb_with_coords(template_lines, heavy_pos, tmp_pdb)
            out_png = out_dir / f"conf_{conf_id:06d}.png"
            render_cartoon(tmp_pdb, str(out_png), width=args.width, height=args.height)
            print(f"[rep] conf_id={conf_id} -> {out_png}")
        finally:
            if os.path.exists(tmp_pdb):
                os.unlink(tmp_pdb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
