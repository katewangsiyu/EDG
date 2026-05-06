"""Run 100ps Langevin NVT simulations of Chignolin driven by a trained ViSNet (EDG++ or baseline).

Follows ViSNet Fig.4 protocol: 10 initial conformations x 100 ps, 0.5 fs timestep, T=300K.

Usage:
    python run_chignolin_md.py \\
        --ckpt /path/to/baseline_or_edgpp.ckpt \\
        --out-dir experiments/run_chignolin_md/edg_pp \\
        --chignolin-npz /path/to/chignolin.npz \\
        --n-init 10 --dt-fs 0.5 --total-ps 100 --temp-k 300 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_chignolin.visnet_edgpp_calculator import ViSNetEDGppCalculator  # noqa: E402


def make_atoms_from_conformation(z: np.ndarray, r: np.ndarray) -> Atoms:
    """Build an ASE Atoms object from atomic numbers + positions."""
    return Atoms(numbers=z.tolist(), positions=r, pbc=False)


def run_one(
    atoms: Atoms,
    calc: ViSNetEDGppCalculator,
    out_dir: Path,
    traj_name: str,
    dt_fs: float,
    total_ps: float,
    temp_k: float,
    log_every: int,
    friction_inv_ps: float = 1.0,
) -> dict:
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k)
    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temp_k,
        friction=friction_inv_ps / (1000 * units.fs),  # convert 1/ps to 1/(ASE time unit)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"{traj_name}.traj"
    eng_path = out_dir / f"{traj_name}_energies.npy"

    traj = Trajectory(str(traj_path), "w", atoms)
    energies = []
    temperatures = []
    aborted = {"flag": False, "reason": None}

    n_steps = int(total_ps * 1000 / dt_fs)

    class _NaNAbort(Exception):
        pass

    def on_step():
        e = atoms.get_potential_energy()
        T = atoms.get_temperature()
        energies.append(e)
        temperatures.append(T)
        sim_ps = len(energies) * dt_fs * log_every / 1000.0 if len(energies) > 1 else 0.0
        print(
            f"  [{traj_name}] step~{(len(energies)-1)*log_every:6d}  t={sim_ps:7.3f} ps  E={e:.3f} eV  T={T:.1f} K",
            flush=True,
        )
        # Graceful early-abort on NaN/Inf: raise a sentinel exception that the main loop catches,
        # so this trajectory ends here but the next trajectory still launches.
        import math
        if math.isnan(T) or math.isinf(T) or math.isnan(e) or math.isinf(e):
            aborted["flag"] = True
            aborted["reason"] = f"NaN/Inf at step {len(energies)-1}: E={e}, T={T}"
            print(f"  [{traj_name}] *** ABORTING THIS TRAJECTORY: {aborted['reason']} ***", flush=True)
            raise _NaNAbort(aborted["reason"])

    # Capture initial state
    on_step()
    traj.write()
    dyn.attach(lambda: (on_step(), traj.write()), interval=log_every)

    try:
        dyn.run(n_steps)
    except _NaNAbort:
        pass  # graceful abort: data already saved up to last valid step
    traj.close()
    # Drop the final NaN entry (if any) so downstream analysis ignores it
    if aborted["flag"] and len(energies) > 1:
        energies = energies[:-1]
        temperatures = temperatures[:-1]
    np.save(eng_path, np.asarray(energies, dtype=np.float32))
    np.save(out_dir / f"{traj_name}_temperatures.npy", np.asarray(temperatures, dtype=np.float32))
    return {
        "n_steps": n_steps,
        "total_ps": total_ps,
        "dt_fs": dt_fs,
        "temp_k": temp_k,
        "friction_inv_ps": friction_inv_ps,
        "aborted": aborted["flag"],
        "abort_reason": aborted["reason"],
        "n_frames_valid": len(energies),
        "log_every": log_every,
        "n_frames_saved": len(energies),
        "traj": str(traj_path),
        "energies": str(eng_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="visnet_for_EDG LNNP checkpoint (pl .ckpt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chignolin-npz", required=True, help="path to chignolin.npz")
    ap.add_argument("--n-init", type=int, default=10, help="number of starting conformations")
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--total-ps", type=float, default=100.0)
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--friction-inv-ps", type=float, default=1.0, help="Langevin friction in 1/ps")
    ap.add_argument("--log-every", type=int, default=100, help="save every N steps (0.5 fs * 100 = 50 fs)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-idx-path", default=None, help="optional npz with idx_test from training splits.npz; ensure init confs are from test set only")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.chignolin_npz)
    n_atoms = 166
    n_total = len(data["Z"]) // n_atoms
    z0 = data["Z"][:n_atoms]
    R_all = data["R"].reshape(n_total, n_atoms, 3).astype(np.float32)

    # pick initial conformations (preferably from test set)
    rng = np.random.RandomState(args.init_seed)
    if args.test_idx_path and os.path.exists(args.test_idx_path):
        splits = np.load(args.test_idx_path)
        test_idx = splits["idx_test"]
        cand = rng.choice(test_idx, size=args.n_init, replace=False)
        print(f"[md] sampling {args.n_init} initial conformations from test set (n_test={len(test_idx)})")
    else:
        cand = rng.choice(n_total, size=args.n_init, replace=False)
        print(f"[md] sampling {args.n_init} initial conformations from full dataset (no split file)")
    cand = sorted(int(i) for i in cand)
    print(f"[md] init indices: {cand}")

    calc = ViSNetEDGppCalculator(ckpt_path=args.ckpt, device=args.device)

    summaries = []
    for i, conf_id in enumerate(cand):
        print(f"\n=== trajectory {i+1}/{args.n_init}: init conf_id={conf_id} ===")
        atoms = make_atoms_from_conformation(z0, R_all[conf_id])
        s = run_one(
            atoms=atoms,
            calc=calc,
            out_dir=out_dir,
            traj_name=f"traj_{i:02d}_init{conf_id:06d}",
            dt_fs=args.dt_fs,
            total_ps=args.total_ps,
            temp_k=args.temp_k,
            log_every=args.log_every,
            friction_inv_ps=args.friction_inv_ps,
        )
        s["init_conf_id"] = conf_id
        summaries.append(s)

    with open(out_dir / "md_summary.json", "w") as f:
        json.dump({"args": vars(args), "trajectories": summaries}, f, indent=2, default=str)
    print(f"\n[md] all {args.n_init} trajectories done. Summary: {out_dir / 'md_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
