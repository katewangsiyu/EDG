"""Render 6 representative Chignolin conformations matching ViSNet Fig.4
6-thumbnail reference (`EDG-for-VisNet/6 个 thumbnail.png`):

  - cartoon ribbon underneath (light purple, fairly thick)
  - all backbone + side-chain heavy atoms as STICKS only (no spheres)
  - CPK-ish coloring: carbon = light purple, O = red, N = blue
  - hydrogens hidden
  - thick sticks (chunky look)
  - transparent background

All 6 conformers are CA-aligned to 5AWL_H so the camera frame is shared.
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

DEFAULT_TEMPLATE_PDB = (
    "/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/"
    "Chignolin/raw/5AWL.pdb"
)
DEFAULT_REF_PDB = (
    "/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/"
    "Chignolin/raw/5AWL_H.pdb"
)
DEFAULT_MAPPING = (
    "/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/"
    "Chignolin/raw/pdb_to_npz_heavy_mapping.json"
)
DEFAULT_NPZ = (
    "/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/AI2BMD/"
    "chignolin_data/raw/chignolin.npz"
)
N_ATOMS_NPZ = 166

# ViSNet 风格的紫 (取自 6 个 thumbnail.png 的目测)
VISNET_PURPLE = [0.70, 0.65, 0.85]   # 略带蓝调的薰衣草紫
VISNET_CARTOON = [0.78, 0.72, 0.88]  # cartoon 稍浅


def parse_pdb_atoms(pdb_path: str) -> List[str]:
    lines: List[str] = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                lines.append(line.rstrip("\n"))
    return lines


def write_pdb_with_coords(template_lines: List[str],
                          positions: np.ndarray,
                          out_path: str) -> None:
    assert len(template_lines) == positions.shape[0]
    with open(out_path, "w") as f:
        for line, pos in zip(template_lines, positions):
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            f.write(new_line + "\n")
        f.write("END\n")


def render_one(conf_id: int,
               heavy_pos: np.ndarray,
               template_lines: List[str],
               ref_pdb: str,
               out_png: str,
               width: int,
               height: int,
               rot_z: float,
               rot_y: float,
               rot_x: float,
               zoom_buffer: float) -> None:
    p = pymol2.PyMOL()
    p.start()
    c = p.cmd
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            tmp_pdb = tf.name
        write_pdb_with_coords(template_lines, heavy_pos, tmp_pdb)

        c.load(ref_pdb, "ref")
        c.remove("ref and resn HOH")
        c.load(tmp_pdb, "conf")
        c.remove("conf and resn HOH")

        c.bg_color("white")
        c.set("ray_opaque_background", 0)
        c.set("ray_trace_mode", 0)
        c.set("antialias", 4)
        c.set("specular", 0.18)
        c.set("ambient", 0.50)
        c.set("ray_shadows", 0)
        # Smooth tube quality (Plan A): denser spline sampling + finer
        # cross-section + extra refinement passes. Eliminates per-residue
        # kinks in the cartoon tube on a 10-residue peptide.
        c.set("cartoon_sampling", 20)         # default 7
        c.set("cartoon_loop_quality", 30)     # default 12
        c.set("cartoon_tube_quality", 20)     # default 9
        c.set("cartoon_refine", 5)
        c.set("cartoon_smooth_first", 1)
        c.set("cartoon_smooth_last", 1)
        c.set("cartoon_round_helices", 1)
        # Mild ambient occlusion for ViSNet-like depth
        c.set("ambient_occlusion_mode", 1)
        c.set("ambient_occlusion_scale", 15)
        c.set_color("visnet_purple",  VISNET_PURPLE)
        c.set_color("visnet_cartoon", VISNET_CARTOON)

        # CA-align conf to ref
        c.align("conf and name CA", "ref and name CA", cycles=0)

        # Style conf: cartoon + all-heavy-atom sticks (no spheres, no H)
        c.hide("everything", "conf")

        # Cartoon as smooth tube — dominant backbone trace (ViSNet style).
        # Tube mode interpolates a continuous smooth curve through CA atoms,
        # eliminating the zig-zag of stick-by-stick backbone.
        c.show("cartoon", "conf")
        c.cartoon("tube", "conf")
        c.set("cartoon_tube_radius", 0.55)        # 0.45 → 0.55, tube dominates
        c.set("cartoon_smooth_loops", 1)
        c.color("visnet_cartoon", "conf")
        c.set("cartoon_transparency", 0.0)         # fully opaque so it dominates

        # All heavy atoms as sticks (backbone + side chain), but cartoon tube
        # carries the smooth backbone trace — sticks add atomic detail (the
        # carbonyl O / amide N caps + sidechains).
        sel_heavy = "conf and not elem H"
        c.show("sticks", sel_heavy)
        c.set("stick_radius", 0.14)           # 0.18 → 0.14, thinner sidechain
        c.set("stick_ball", 0)
        c.set("valence", 0)

        # Backbone stick bonds — kept thinner so the smooth cartoon tube reads
        # as the dominant backbone trace (ViSNet-style hierarchy).
        sel_bb = "conf and name N+CA+C+O+OXT"
        c.set_bond("stick_radius", 0.16, sel_bb, sel_bb)

        # CPK-ish: C purple, O red, N blue
        c.color("visnet_purple", f"{sel_heavy} and elem C")
        c.color("firebrick",     f"{sel_heavy} and elem O")
        c.color("marine",        f"{sel_heavy} and elem N")

        # Hide Y2 + Y10 sidechains to match panel a v24 (reduces aromatic
        # overlap especially in folded conformations). Backbone N/CA/C/O kept
        # so the cartoon tube has no gaps.
        HIDE_SIDECHAIN_RESI = ["2", "10"]
        sel_hide = (
            f"conf and resi {'+'.join(HIDE_SIDECHAIN_RESI)} "
            f"and not (name N+CA+C+O)"
        )
        c.hide("sticks", sel_hide)

        # Hide ref entirely
        c.hide("everything", "ref")

        # Camera: orient on ref + apply panel-a rotation, zoom on conf
        c.orient("ref")
        if rot_z: c.rotate("z", rot_z)
        if rot_y: c.rotate("y", rot_y)
        if rot_x: c.rotate("x", rot_x)
        c.zoom("conf", buffer=zoom_buffer, complete=1)

        c.png(out_png, width=width, height=height, dpi=300, ray=1)
    finally:
        if os.path.exists(tmp_pdb):
            os.unlink(tmp_pdb)
        p.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-pdb", default=DEFAULT_TEMPLATE_PDB)
    ap.add_argument("--ref-pdb", default=DEFAULT_REF_PDB)
    ap.add_argument("--mapping-json", default=DEFAULT_MAPPING)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--conf-ids", type=int, nargs="+",
                    default=[659, 569, 2950, 6456, 5645, 5031])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--rot-z", type=float, default=275.0)
    ap.add_argument("--rot-y", type=float, default=0.0)
    ap.add_argument("--rot-x", type=float, default=20.0)
    ap.add_argument("--zoom-buffer", type=float, default=4.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mapping_json) as f:
        mapping = json.load(f)
    pdb_to_npz = mapping["pdb_to_npz"]
    n_heavy = len(pdb_to_npz)

    template_lines = parse_pdb_atoms(args.template_pdb)
    assert len(template_lines) == n_heavy

    data = np.load(args.npz)
    R_all = data["R"].reshape(-1, N_ATOMS_NPZ, 3).astype(np.float32)

    for conf_id in args.conf_ids:
        pos = R_all[conf_id]
        heavy_pos = np.stack([pos[pdb_to_npz[i]] for i in range(n_heavy)], axis=0)
        out_png = out_dir / f"conf_{conf_id:06d}.png"
        render_one(conf_id, heavy_pos, template_lines, args.ref_pdb, str(out_png),
                   args.width, args.height,
                   args.rot_z, args.rot_y, args.rot_x, args.zoom_buffer)
        print(f"[v4 visnet-match] conf_id={conf_id} -> {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
