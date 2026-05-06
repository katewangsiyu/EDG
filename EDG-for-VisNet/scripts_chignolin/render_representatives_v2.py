"""Render 6 representative Chignolin conformations as INDIVIDUAL thumbnails for
panel-b周围拼图 (user combines manually).

Style locked to Panel a v24 'simple':
  - cartoon tube radius 0.20 (gray60)
  - sidechain ball-and-stick (CPK colors), polar-H only (sidechain N/O bonded)
  - hide Y2 + Y10 sidechain (matches panel a)
  - transparent background, 600×600 square

All 6 conformations are aligned (CA superposition) to the 5AWL_H folded reference
so the camera frame is identical across thumbnails — folded → unfolded shape
differences read clearly.

Run:
    conda run -n pymol_edgpp python render_representatives_v2.py \\
        --conf-ids 659 569 2950 6456 5645 5031 \\
        --out-dir representatives_v2/
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


def setup_common(c) -> None:
    c.bg_color("white")
    c.set("ray_opaque_background", 0)
    c.set("ray_trace_mode", 0)
    c.set("antialias", 2)
    c.set("specular", 0.25)
    c.set("ambient", 0.4)
    c.set("ray_shadows", 0)
    c.set_color("cpk_cyan", [0.40, 0.85, 0.88])


def apply_simple_style(c, obj: str) -> None:
    """Panel a v24 'simple' style — exact parameters."""
    TUBE_R = 0.20
    HEAVY_R = 0.30
    H_R = 0.175
    STICK_R = 0.10
    OH_STICK_R = 0.115

    c.h_add(obj)
    c.hide("everything", obj)

    c.show("cartoon", obj)
    c.cartoon("tube", obj)
    c.set("cartoon_tube_radius", TUBE_R)
    c.set("cartoon_color", "gray60", obj)
    c.set("cartoon_side_chain_helper", 1)

    sel_sc = (
        f"{obj} and not (name N+C+O+OXT) and "
        f"(not elem H or (neighbor (elem N+O and not name N+C+O+OXT)))"
    )
    c.show("sticks", sel_sc)
    c.show("spheres", sel_sc)

    c.color("cpk_cyan",  f"{sel_sc} and elem C")
    c.color("firebrick", f"{sel_sc} and elem O")
    c.color("marine",    f"{sel_sc} and elem N")
    c.color("white",     f"{sel_sc} and elem H")

    c.set("sphere_scale", HEAVY_R / 1.7,  f"{sel_sc} and elem C")
    c.set("sphere_scale", HEAVY_R / 1.55, f"{sel_sc} and elem N")
    c.set("sphere_scale", HEAVY_R / 1.52, f"{sel_sc} and elem O")
    c.set("sphere_scale", H_R / 1.20,     f"{sel_sc} and elem H")

    c.set("stick_radius", STICK_R)
    c.set("stick_ball", 0)
    c.set("valence", 0)
    c.set_bond("stick_radius", OH_STICK_R, f"{obj} and elem O", f"{obj} and elem H")

    HIDE_RESI = ["2", "10"]
    sel_hide = f"{obj} and resi {'+'.join(HIDE_RESI)} and not (name N+CA+C+O)"
    c.hide("sticks", sel_hide)
    c.hide("spheres", sel_hide)


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

        # Load reference (5AWL_H, with hydrogens) — used for CA alignment + camera frame
        c.load(ref_pdb, "ref")
        c.remove("ref and resn HOH")

        # Load this conformation
        c.load(tmp_pdb, "conf")
        c.remove("conf and resn HOH")

        setup_common(c)

        # Align conf onto ref via CA superposition (no outlier rejection)
        c.align("conf and name CA", "ref and name CA", cycles=0)

        # Style only the conf object
        apply_simple_style(c, "conf")

        # Hide ref entirely (keep only for camera framing baseline)
        c.hide("everything", "ref")

        # Camera frame: orient on ref's principal axes (same for every conf), then apply
        # Panel a's rotation to keep visual continuity with the main figure.
        c.orient("ref")
        if rot_z: c.rotate("z", rot_z)
        if rot_y: c.rotate("y", rot_y)
        if rot_x: c.rotate("x", rot_x)

        # Zoom on conf so unfolded structures still fit; buffer kept generous.
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
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
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
    assert len(template_lines) == n_heavy, (
        f"template lines ({len(template_lines)}) != mapping ({n_heavy})"
    )

    data = np.load(args.npz)
    R_all = data["R"].reshape(-1, N_ATOMS_NPZ, 3).astype(np.float32)

    for conf_id in args.conf_ids:
        pos = R_all[conf_id]
        heavy_pos = np.stack([pos[pdb_to_npz[i]] for i in range(n_heavy)], axis=0)
        out_png = out_dir / f"conf_{conf_id:06d}.png"
        render_one(conf_id, heavy_pos, template_lines, args.ref_pdb, str(out_png),
                   args.width, args.height,
                   args.rot_z, args.rot_y, args.rot_x, args.zoom_buffer)
        print(f"[v2] conf_id={conf_id} -> {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
