"""Aggregate per-trajectory _analysis.npz outputs (from analyze_trajectories.py)
plus the training-set reliability subsample (from compute_train_reliability.py)
into the single md_reliability.npz that plot_panel_d_reliability.py expects.

Selected 4 trajectories (paper convention): init_ids 1571, 3376, 7419, 9524 —
all four survive the full 50 ps with 502 frames each.

Schema written:
  init_ids           (n_traj,)   dtype=int
  reliability        (n_traj, n_frames_max)  NaN-padded
  n_frames           (n_traj,)   per-traj valid frame count
  train_reliability  (n_train,)  reliability scores from training set sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_INIT_IDS = [1571, 3376, 7419, 9524]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses-dir", required=True,
                    help="directory containing traj_NN_initXXXXXX_analysis.npz")
    ap.add_argument("--train-rel", required=True,
                    help="train_reliability.npy")
    ap.add_argument("--init-ids", type=int, nargs="+", default=DEFAULT_INIT_IDS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    analyses_dir = Path(args.analyses_dir)
    train_rel = np.load(args.train_rel).astype(np.float32)

    series = []
    for init_id in args.init_ids:
        cands = list(analyses_dir.glob(f"traj_*_init{init_id:06d}_analysis.npz"))
        if not cands:
            print(f"[error] no analysis npz for init {init_id} in {analyses_dir}")
            return 1
        d = np.load(cands[0])
        rel = d["reliability"].astype(np.float32)
        series.append((init_id, rel))
        print(f"  init {init_id:5d}: n={len(rel)}  mean={rel.mean():+.3f}  "
              f"min={rel.min():+.3f}  max={rel.max():+.3f}")

    n_max = max(len(r) for _, r in series)
    arr = np.full((len(series), n_max), np.nan, dtype=np.float32)
    n_frames = np.zeros(len(series), dtype=np.int64)
    init_ids = np.zeros(len(series), dtype=np.int64)
    for i, (init_id, r) in enumerate(series):
        arr[i, :len(r)] = r
        n_frames[i] = len(r)
        init_ids[i] = init_id

    np.savez(args.out,
             init_ids=init_ids,
             reliability=arr,
             n_frames=n_frames,
             train_reliability=train_rel)
    print(f"\nSaved {args.out}")
    print(f"  init_ids         : {init_ids.tolist()}")
    print(f"  reliability shape: {arr.shape}")
    print(f"  n_frames         : {n_frames.tolist()}")
    print(f"  train_rel        : n={len(train_rel)}  "
          f"mean={train_rel.mean():.3f}  std={train_rel.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
