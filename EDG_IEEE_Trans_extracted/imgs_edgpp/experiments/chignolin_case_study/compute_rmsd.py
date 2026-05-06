"""Compute Cα-RMSD vs the **initial conformation of each trajectory** for
every Chignolin MD trajectory (baseline + EDG++, 10 each, 50 ps).

For each traj: align every frame to that traj's frame 0 (Kabsch over Cα),
then RMSD. Per-traj reference (not the 5AWL folded crystal) — matches the
final figure y-axis "Cα RMSD vs. initial".

Output: panel_c_rmsd.npz   keys = baseline (n_traj, n_frames),
                                   edg_pp   (n_traj, n_frames),
                                   t_ps     (n_frames,)
        panel_c_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import Trajectory

HERE = Path(__file__).parent
EXP = Path("/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/experiments/run_chignolin_md")
PDB = Path("/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/Chignolin/raw/5AWL.pdb")
MAPPING = Path("/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/Chignolin/raw/pdb_to_npz_heavy_mapping.json")

OUT_NPZ = HERE / "panel_c_rmsd.npz"
OUT_JSON = HERE / "panel_c_stats.json"


def read_pdb_ca_indices(pdb_path: Path):
    pdb_idx_running = -1
    ca_pdb_idx = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        pdb_idx_running += 1
        if line[12:16].strip() == "CA":
            ca_pdb_idx.append(pdb_idx_running)
    return ca_pdb_idx


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D) @ U.T
    P_aligned = Pc @ R.T
    diff = P_aligned - Qc
    return float(np.sqrt((diff * diff).sum() / len(P)))


def compute_group(group_dir: Path, ca_npz_idx: np.ndarray):
    traj_files = sorted(group_dir.glob("traj_*.traj"))
    rmsd_all = []
    for tf in traj_files:
        rmsds = []
        with Trajectory(str(tf)) as traj:
            ref_ca = None
            for atoms in traj:
                pos = atoms.get_positions()
                ca = pos[ca_npz_idx]
                if ref_ca is None:
                    ref_ca = ca.copy()
                rmsds.append(kabsch_rmsd(ca, ref_ca))
        rmsd_all.append(np.array(rmsds, dtype=np.float32))
        print(f"  {tf.name}  n={len(rmsds)}  mean={np.mean(rmsds):.2f} Å  "
              f"final={rmsds[-1]:.2f}")
    n_max = max(len(r) for r in rmsd_all)
    arr = np.full((len(rmsd_all), n_max), np.nan, dtype=np.float32)
    for i, r in enumerate(rmsd_all):
        arr[i, :len(r)] = r
    return arr


def main() -> None:
    mapping = json.loads(MAPPING.read_text())["pdb_to_npz"]
    ca_pdb_idx = read_pdb_ca_indices(PDB)
    ca_npz_idx = np.array([mapping[i] for i in ca_pdb_idx], dtype=np.int64)
    print(f"CA atom NPZ indices: {ca_npz_idx.tolist()}")

    print("\n[baseline]")
    baseline = compute_group(EXP / "baseline", ca_npz_idx)
    print("\n[edg_pp]")
    edg_pp = compute_group(EXP / "edg_pp", ca_npz_idx)

    n_frames = max(baseline.shape[1], edg_pp.shape[1])
    DT_PS = 400 * 0.25 / 1000.0   # log_every=400, dt_fs=0.25
    t_ps = np.arange(n_frames) * DT_PS

    np.savez(OUT_NPZ, baseline=baseline, edg_pp=edg_pp, t_ps=t_ps,
             dt_ps=DT_PS, ca_npz_idx=ca_npz_idx)
    print(f"\nSaved {OUT_NPZ}")

    def summarize(arr):
        per_traj_mean = np.nanmean(arr, axis=1)
        finals = []
        for row in arr:
            valid = row[~np.isnan(row)]
            if len(valid):
                finals.append(valid[-1])
        finals = np.array(finals)
        return {
            "per_traj_mean_RMSD_mean": float(np.nanmean(per_traj_mean)),
            "per_traj_mean_RMSD_std":  float(np.nanstd(per_traj_mean)),
            "final_RMSD_mean":         float(finals.mean()),
            "final_RMSD_std":          float(finals.std()),
            "n_traj":                  int(arr.shape[0]),
        }

    stats = {"reference": "per-trajectory frame 0 (initial)",
             "baseline": summarize(baseline),
             "edg_pp":   summarize(edg_pp)}
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    for k in ("baseline", "edg_pp"):
        v = stats[k]
        print(f"  [{k}]  ⟨RMSD⟩ = {v['per_traj_mean_RMSD_mean']:.2f} ± "
              f"{v['per_traj_mean_RMSD_std']:.2f} Å  (final "
              f"{v['final_RMSD_mean']:.2f} ± {v['final_RMSD_std']:.2f})")


if __name__ == "__main__":
    main()
