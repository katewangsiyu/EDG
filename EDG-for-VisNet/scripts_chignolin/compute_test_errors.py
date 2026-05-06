"""Compute per-conformation energy absolute error over the FULL Chignolin test set
(955 conformations under seed=42, 8:1:1 split) for 3 models: baseline / edg / edg_pp.

Output: test_errors.json with structure
    {
      "conf_id": [list of test set conformation IDs (npz indices)],
      "baseline": [list of |dE| in kcal/mol],
      "edg":      [list of |dE| in kcal/mol],
      "edg_pp":   [list of |dE| in kcal/mol],
      "split_seed": 42,
      "n_test": 955
    }

Also prints summary statistics: mean/median/quantiles per method and strict-monotonicity rate.
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

EV_TO_KCAL = 23.0605


def load_lnnp(ckpt: str, device: torch.device) -> LNNP:
    m = LNNP.load_from_checkpoint(ckpt, map_location=device, strict=False)
    m.eval()
    m.hparams.weight_ED = 0.0
    return m.to(device)


@torch.no_grad()
def _predict_energy(model: LNNP, data: Data, device: torch.device) -> float:
    d = data.clone().to(device)
    if d.pos.requires_grad is False:
        d.pos = d.pos.detach()
    d.pos = d.pos.requires_grad_(True)
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
    return float(energy.detach().cpu().numpy().squeeze())


def get_test_indices(n: int, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1) -> np.ndarray:
    """Replicate the deterministic 8:1:1 split used by training (numpy RandomState shuffle)."""
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
    print(f"[test_errors] dataset n={n}, test set size={len(test_ids)} (seed={args.seed})")

    device = torch.device(args.device)
    ckpts = {"baseline": args.baseline_ckpt, "edg": args.edg_ckpt, "edg_pp": args.edgpp_ckpt}

    errors: dict[str, list[float]] = {m: [] for m in ckpts.keys()}
    e_true_list: list[float] = []

    for method, ckpt_path in ckpts.items():
        print(f"[test_errors] loading {method}: {ckpt_path}")
        model = load_lnnp(ckpt_path, device)
        method_errs: list[float] = []
        if not e_true_list:
            for k, cid in enumerate(test_ids):
                e_true_list.append(float(dataset[int(cid)].y.squeeze().item()))
        for k, cid in enumerate(test_ids):
            cid = int(cid)
            data = dataset[cid]
            e_true = e_true_list[k]
            e_pred = _predict_energy(model, data, device)
            err_kcal = abs(e_pred - e_true) * EV_TO_KCAL
            method_errs.append(err_kcal)
            if (k + 1) % args.progress_every == 0:
                running_mean = float(np.mean(method_errs))
                print(f"  [{method}] {k+1}/{len(test_ids)}  running mean={running_mean:.3f} kcal/mol")
        errors[method] = method_errs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conf_id": [int(c) for c in test_ids],
        "baseline": errors["baseline"],
        "edg": errors["edg"],
        "edg_pp": errors["edg_pp"],
        "split_seed": args.seed,
        "n_test": len(test_ids),
        "unit": "kcal/mol",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[test_errors] saved {out_path}")

    # Summary
    print("\n=== Summary (kcal/mol) ===")
    arrs = {m: np.asarray(errors[m]) for m in ckpts}
    print(f"{'method':10s} {'mean':>8s} {'median':>8s} {'std':>8s} {'p90':>8s} {'p95':>8s} {'max':>8s}")
    for m in ["baseline", "edg", "edg_pp"]:
        a = arrs[m]
        print(f"{m:10s} {a.mean():8.4f} {np.median(a):8.4f} {a.std():8.4f} {np.percentile(a,90):8.4f} {np.percentile(a,95):8.4f} {a.max():8.4f}")

    # Strict monotonicity rate: edg_pp < edg < baseline (per-conf)
    b, e, p = arrs["baseline"], arrs["edg"], arrs["edg_pp"]
    strict_mono = (p < e) & (e < b)
    pp_better_than_b = p < b
    e_better_than_b = e < b
    pp_better_than_e = p < e
    print(f"\nstrict monotone (edg_pp < edg < baseline): {strict_mono.sum()}/{len(b)} = {strict_mono.mean()*100:.1f}%")
    print(f"edg_pp < baseline:                          {pp_better_than_b.sum()}/{len(b)} = {pp_better_than_b.mean()*100:.1f}%")
    print(f"edg    < baseline:                          {e_better_than_b.sum()}/{len(b)} = {e_better_than_b.mean()*100:.1f}%")
    print(f"edg_pp < edg:                               {pp_better_than_e.sum()}/{len(b)} = {pp_better_than_e.mean()*100:.1f}%")

    # Mean improvements
    print(f"\nmean reduction edg_pp vs baseline: {(1 - p.mean()/b.mean())*100:+.2f}%")
    print(f"mean reduction edg    vs baseline: {(1 - e.mean()/b.mean())*100:+.2f}%")
    print(f"mean reduction edg_pp vs edg     : {(1 - p.mean()/e.mean())*100:+.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
