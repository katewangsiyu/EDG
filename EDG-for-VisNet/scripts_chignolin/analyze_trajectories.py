"""Post-process MD trajectories for the Chignolin case study.

For every trajectory (.traj file) in the given directory, compute:
  1. RMSD vs the 5AWL folded reference (heavy atoms only, via the cached pdb_to_npz mapping).
  2. Per-frame reliability score: render 4 structural views, feed through the pre-trained
     ResNet18 teacher -> EDEvaluator -> reliability score (-evaluator output).
  3. Radius of gyration (R_g) per frame (all-atom).

Outputs one npz per trajectory with fields: {times_ps, rmsd_A, rg_A, reliability, energy_eV}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ase.io.trajectory import Trajectory

Z_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
VIEWS: List[Tuple[str, int, str]] = [
    ("x", 0, "x0"),
    ("x", 180, "x180"),
    ("y", 180, "y180"),
    ("z", 180, "z180"),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


def build_evaluator() -> nn.Module:
    return nn.Sequential(OrderedDict([
        ("linear1", nn.Linear(512, 256)),
        ("Softplus", nn.Softplus()),
        ("linear2", nn.Linear(256, 1)),
    ]))


def load_pretrained(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    teacher = VideoTeacher()
    teacher.load_state_dict(ckpt["video_teacher"], strict=True)
    teacher.eval().to(device)

    evaluator = build_evaluator()
    state = ckpt["EDEvaluator"]
    new_state = OrderedDict((k.replace("network.", ""), v) for k, v in state.items())
    evaluator.load_state_dict(new_state, strict=True)
    evaluator.eval().to(device)
    return teacher, evaluator


def compute_rg(pos: np.ndarray) -> float:
    cog = pos.mean(axis=0)
    return float(np.sqrt(((pos - cog) ** 2).sum(axis=1).mean()))


def compute_rmsd_heavy(pos: np.ndarray, ref_heavy: np.ndarray, heavy_idx_npz: np.ndarray) -> float:
    """RMSD (Å) of heavy atoms vs ref_heavy, after optimal rotation (Kabsch)."""
    p = pos[heavy_idx_npz]
    q = ref_heavy
    p = p - p.mean(0)
    q = q - q.mean(0)
    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    p_aligned = p @ R.T
    return float(np.sqrt(((p_aligned - q) ** 2).sum(axis=1).mean()))


def write_xyz(path: Path, z: np.ndarray, r: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(f"{len(z)}\nChignolin frame\n")
        for zi, (x, y, zc) in zip(z, r):
            f.write(f"{Z_TO_SYM.get(int(zi), 'X')} {x:.6f} {y:.6f} {zc:.6f}\n")


def render_views_with_pymol_env(
    z: np.ndarray,
    positions_list: List[np.ndarray],
    tmp_root: Path,
    pymol_python: str,
) -> List[List[Image.Image]]:
    """Render many frames in one PyMol subprocess (much faster than launching per-frame)."""
    scene_dir = tmp_root / "frames"
    scene_dir.mkdir(parents=True, exist_ok=True)
    # Dump xyz files
    for i, pos in enumerate(positions_list):
        write_xyz(scene_dir / f"frame_{i:06d}.xyz", z, pos)
    # Write a small PyMol-driver script
    driver_py = tmp_root / "render_driver.py"
    driver_py.write_text(f"""
import os
import sys
import pymol2
from pathlib import Path

scene_dir = Path(r"{scene_dir}")
out_dir = Path(r"{tmp_root / 'png'}")
out_dir.mkdir(parents=True, exist_ok=True)

p = pymol2.PyMOL()
p.start()
c = p.cmd
c.bg_color("white")
c.set("ray_opaque_background", 1)
c.set("antialias", 2)

for xyz in sorted(scene_dir.glob("frame_*.xyz")):
    c.delete("all")
    c.load(str(xyz), "mol")
    c.hide("everything", "mol")
    c.show("sticks", "mol")
    c.show("spheres", "mol")
    c.set("sphere_scale", 0.25)
    c.set("stick_radius", 0.12)
    c.color("gray70", "elem C")
    c.color("white", "elem H")
    c.color("blue", "elem N")
    c.color("red", "elem O")
    c.orient("mol")
    c.zoom("mol", buffer=2.0, complete=1)
    stem = xyz.stem
    for axis, angle, tag in [("x", 0, "x0"), ("x", 180, "x180"), ("y", 180, "y180"), ("z", 180, "z180")]:
        c.orient("mol")
        c.zoom("mol", buffer=2.0, complete=1)
        if angle != 0:
            c.turn(axis, angle)
        c.png(str(out_dir / f"{{stem}}_{{tag}}.png"), width=224, height=224, ray=0)

p.stop()
""")
    cmd = [pymol_python, str(driver_py)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pymol driver failed: {res.stderr[-500:]}")

    # Load the PNGs back
    png_dir = tmp_root / "png"
    images_per_frame = []
    for i in range(len(positions_list)):
        stem = f"frame_{i:06d}"
        views = [Image.open(png_dir / f"{stem}_{tag}.png").convert("RGB") for _, _, tag in VIEWS]
        images_per_frame.append(views)
    return images_per_frame


def images_to_teacher_features(images_per_frame, teacher, device: str) -> np.ndarray:
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    feats_list = []
    with torch.no_grad():
        for views in images_per_frame:
            tens = torch.stack([trans(im) for im in views], dim=0).to(device)
            feat = teacher(tens).mean(dim=0).cpu().numpy()
            feats_list.append(feat)
    return np.asarray(feats_list, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="pre-trained teacher+evaluator ckpt (.pth)")
    ap.add_argument("--pdb-mapping", required=True, help="pdb_to_npz_heavy_mapping.json")
    ap.add_argument("--ref-pdb", required=True, help="5AWL.pdb")
    ap.add_argument("--stride-ps", type=float, default=1.0, help="subsample frames to every N ps")
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--pymol-python",
        default="/home/lzeng/miniconda3/envs/pymol_edgpp/bin/python",
    )
    ap.add_argument("--skip-reliability", action="store_true", help="skip PyMol render + reliability; only compute RMSD/Rg (baseline model)")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference
    with open(args.pdb_mapping) as f:
        mapping = json.load(f)
    heavy_idx_npz = np.array(mapping["pdb_to_npz"], dtype=np.int64)

    heavy_pdb = []
    with open(args.ref_pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            heavy_pdb.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
    ref_heavy = np.asarray(heavy_pdb, dtype=np.float64)

    if not args.skip_reliability:
        teacher, evaluator = load_pretrained(args.ckpt, args.device)

    # Determine frame stride
    frame_dt_ps = (args.dt_fs * args.log_every) / 1000.0
    stride = max(1, int(round(args.stride_ps / frame_dt_ps)))
    print(f"[analyze] frame dt = {frame_dt_ps:.3f} ps; stride = {stride} frames (= {stride*frame_dt_ps:.2f} ps / sample)")

    traj_files = sorted(traj_dir.glob("traj_*.traj"))
    if not traj_files:
        print(f"[analyze] no trajectories found under {traj_dir}")
        return 1

    for traj_path in traj_files:
        stem = traj_path.stem
        print(f"\n[analyze] {stem}")
        traj = Trajectory(str(traj_path), "r")
        # Collect frames (subsampled)
        positions_list = []
        times_ps = []
        for i, atoms in enumerate(traj):
            if i % stride != 0:
                continue
            positions_list.append(atoms.get_positions().astype(np.float32))
            times_ps.append(i * frame_dt_ps)
        z_sample = traj[0].numbers
        traj.close()

        rmsd = np.asarray([compute_rmsd_heavy(p, ref_heavy, heavy_idx_npz) for p in positions_list], dtype=np.float32)
        rg = np.asarray([compute_rg(p) for p in positions_list], dtype=np.float32)

        # energies
        eng_path = traj_dir / f"{stem}_energies.npy"
        energies = np.load(eng_path) if eng_path.exists() else None

        saved = {"times_ps": np.asarray(times_ps, dtype=np.float32), "rmsd_A": rmsd, "rg_A": rg}
        if energies is not None:
            saved["energy_eV"] = energies[:: stride] if len(energies) >= len(times_ps) else energies

        if not args.skip_reliability:
            with tempfile.TemporaryDirectory(prefix=f"{stem}_render_") as td:
                td = Path(td)
                images = render_views_with_pymol_env(z_sample, positions_list, td, args.pymol_python)
                feats = images_to_teacher_features(images, teacher, args.device)
            with torch.no_grad():
                scores = (-1.0 * evaluator(torch.from_numpy(feats).to(args.device))).squeeze(-1).cpu().numpy()
            saved["reliability"] = scores
            saved["teacher_feats"] = feats

        out_path = out_dir / f"{stem}_analysis.npz"
        np.savez(out_path, **saved)
        print(
            f"  saved {out_path}: n_frames={len(times_ps)}, "
            f"rmsd_mean={rmsd.mean():.2f} rg_mean={rg.mean():.2f}"
            + (f" rel_mean={saved['reliability'].mean():.3f}" if 'reliability' in saved else "")
        )

    print("\n[analyze] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
