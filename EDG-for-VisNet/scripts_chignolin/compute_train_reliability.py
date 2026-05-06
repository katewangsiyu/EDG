"""Compute reliability scores on a subsample of the Chignolin training set
to give the gray "Training set (mean ± 2σ)" band in panel (d).

Reuses the rendering + scoring machinery from analyze_trajectories.py so
the resulting scores are directly comparable to MD-frame reliability.

Output: train_reliability.npy   (1D float array, length = n_samples)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from analyze_trajectories import (
    load_pretrained, render_views_with_pymol_env, images_to_teacher_features,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/AI2BMD/chignolin_data/raw/chignolin.npz")
    ap.add_argument("--splits", default="/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/scripts_chignolin/chignolin_splits_seed42.npz")
    ap.add_argument("--ckpt", default="/home/lzeng/workspace/EDG_for_PR/pretrain/pretrained_models/ED_Evaluator/ckpts/best_epoch=18_loss=0.30.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pymol-python", default="/home/lzeng/miniconda3/envs/pymol_edgpp/bin/python")
    args = ap.parse_args()

    d = np.load(args.npz)
    R = d["R"].reshape(len(d["N"]), int(d["N"][0]), 3).astype(np.float32)
    Z = d["Z"].reshape(len(d["N"]), int(d["N"][0]))[0].astype(np.int64)

    splits = np.load(args.splits)
    train_idx = splits["idx_train"] if "idx_train" in splits.files else splits["train_idx"]
    print(f"train split size: {len(train_idx)}; sampling {args.n_samples}")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(train_idx, size=args.n_samples, replace=False)
    pick.sort()

    teacher, evaluator = load_pretrained(args.ckpt, args.device)

    positions_list = [R[i] for i in pick]
    print(f"rendering {len(positions_list)} frames via PyMOL ...")
    with tempfile.TemporaryDirectory(prefix="train_render_") as td:
        td = Path(td)
        images = render_views_with_pymol_env(Z, positions_list, td, args.pymol_python)
        feats = images_to_teacher_features(images, teacher, args.device)

    with torch.no_grad():
        scores = (-1.0 * evaluator(torch.from_numpy(feats).to(args.device))).squeeze(-1).cpu().numpy()
    print(f"reliability: mean={scores.mean():.3f}  std={scores.std():.3f}  "
          f"min={scores.min():.3f}  max={scores.max():.3f}")

    np.save(args.out, scores.astype(np.float32))
    print(f"Saved: {args.out}  ({len(scores)} scores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
