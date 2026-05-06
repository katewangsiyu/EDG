"""Compute per-conformation force MAE over the FULL Chignolin test set
(955 conformations under seed=42, 8:1:1 split) for 3 models.

Mirrors compute_test_errors.py but computes forces via autograd and
compares against DFT forces stored in chignolin.npz["F"].

Output: test_force_errors.json
        {conf_id, baseline, edg, edg_pp, split_seed, n_test, unit:"kcal/mol/Å"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch_geometric.data.data as pgd
torch.serialization.add_safe_globals(
    [pgd.DataEdgeAttr, pgd.DataTensorAttr, pgd.GlobalStorage]
)
from torch_geometric.data import Batch, Data
from torch_scatter import scatter

from visnet_for_EDG.module import LNNP
from visnet_for_EDG.datasets.chignolin import Chignolin

EV_PER_A_TO_KCAL_PER_A = 23.0605


def load_lnnp(ckpt: str, device: torch.device) -> LNNP:
    m = LNNP.load_from_checkpoint(ckpt, map_location=device, strict=False)
    m.eval()
    m.hparams.weight_ED = 0.0
    return m.to(device)


def predict_forces(model: LNNP, data: Data, device: torch.device) -> np.ndarray:
    """Forces (N,3) in eV/Å, computed as -dE/dpos via autograd."""
    d = data.clone().to(device)
    d.pos = d.pos.detach().requires_grad_(True)
    batch = Batch.from_data_list([d])
    batch.batch = torch.zeros(len(d.z), dtype=torch.long, device=device)
    with torch.enable_grad():
        x, v = model.model.representation_model(batch)
        x = model.model.output_model.pre_reduce(x, v, batch.z, batch.pos, batch.batch)
        x = x * model.model.std
        if model.model.prior_model is not None:
            x = model.model.prior_model(x, batch.z)
        energy = scatter(x, batch.batch, dim=0, reduce=model.model.reduce_op)
        energy = model.model.output_model.post_reduce(energy) + model.model.mean
        # Forces = -dE/dpos
        grad, = torch.autograd.grad(energy.sum(), d.pos, create_graph=False)
    return (-grad.detach().cpu().numpy())


def get_test_indices(n: int, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[n_train + n_val:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/AI2BMD/chignolin_data")
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--edg-ckpt", required=True)
    ap.add_argument("--edgpp-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    dataset = Chignolin(root=args.dataset_root)
    n = len(dataset)
    test_ids = get_test_indices(n, seed=args.seed)
    print(f"[force_errors] dataset n={n}, test set size={len(test_ids)} (seed={args.seed})")

    device = torch.device(args.device)
    ckpts = {"baseline": args.baseline_ckpt, "edg": args.edg_ckpt, "edg_pp": args.edgpp_ckpt}

    errors = {m: [] for m in ckpts}
    for method, ckpt_path in ckpts.items():
        print(f"[force_errors] loading {method}: {ckpt_path}")
        model = load_lnnp(ckpt_path, device)
        for k, cid in enumerate(test_ids):
            cid = int(cid)
            data = dataset[cid]
            f_true = data.dy.cpu().numpy()        # (N,3) in eV/Å
            f_pred = predict_forces(model, data, device)
            # MAE per conformation in kcal/mol/Å
            mae_kcal = float(np.abs(f_pred - f_true).mean()) * EV_PER_A_TO_KCAL_PER_A
            errors[method].append(mae_kcal)
            if (k + 1) % args.progress_every == 0:
                rm = float(np.mean(errors[method]))
                print(f"  [{method}] {k+1}/{len(test_ids)}  running mean={rm:.4f} kcal/mol/Å")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    payload = {
        "conf_id": [int(c) for c in test_ids],
        "baseline": errors["baseline"],
        "edg":      errors["edg"],
        "edg_pp":   errors["edg_pp"],
        "split_seed": args.seed,
        "n_test": len(test_ids),
        "unit": "kcal/mol/Å",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[force_errors] saved {out_path}")

    arrs = {m: np.asarray(errors[m]) for m in ckpts}
    print("\n=== Force MAE summary (kcal/mol/Å) ===")
    print(f"{'method':10s} {'mean':>8s} {'median':>8s} {'std':>8s} {'p95':>8s}")
    for m in ["baseline", "edg", "edg_pp"]:
        a = arrs[m]
        print(f"{m:10s} {a.mean():8.4f} {np.median(a):8.4f} {a.std():8.4f} {np.percentile(a,95):8.4f}")
    b, e, p = arrs["baseline"], arrs["edg"], arrs["edg_pp"]
    print(f"\nmean reduction edg_pp vs baseline: {(1 - p.mean()/b.mean())*100:+.2f}%")
    print(f"mean reduction edg    vs baseline: {(1 - e.mean()/b.mean())*100:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
