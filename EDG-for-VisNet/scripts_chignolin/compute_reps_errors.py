"""Compute per-conformation energy absolute error for the 6 representative Chignolin conformations,
evaluated with each of the 3 trained ViSNet models (baseline / +EDG / +EDG++).

Output: reps_errors.json compatible with plot_chignolin_case_study.py --reps-errors-json.
Format: {conf_id (str): {"baseline": float, "edg": float, "edg_pp": float}, ...}

Usage:
    python compute_reps_errors.py \\
        --npz chignolin.npz \\
        --reps-json representative_conformations.json \\
        --baseline-ckpt <...> --edg-ckpt <...> --edgpp-ckpt <...> \\
        --out reps_errors.json

Energy absolute error = |E_pred - E_DFT| in kcal/mol (converted from model's internal eV if needed).
The npz stores total potential energies in Hartree, the processed PyG dataset stores (E - self_energies) in eV.
Our models predict energy in eV per the training standardization; for display we convert to kcal/mol
(1 eV = 23.0605 kcal/mol).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/AI2BMD/chignolin_data")
    ap.add_argument("--reps-json", required=True, help="representative_conformations.json")
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--edg-ckpt", required=True)
    ap.add_argument("--edgpp-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--unit", choices=["eV", "kcal/mol"], default="kcal/mol")
    args = ap.parse_args()

    reps = json.loads(Path(args.reps_json).read_text())
    conf_ids: List[int] = reps["conf_ids"]

    dataset = Chignolin(root=args.dataset_root)
    device = torch.device(args.device)

    ckpts = {"baseline": args.baseline_ckpt, "edg": args.edg_ckpt, "edg_pp": args.edgpp_ckpt}

    errors: Dict[int, Dict[str, float]] = {cid: {} for cid in conf_ids}

    for method, ckpt_path in ckpts.items():
        print(f"[reps_errors] loading {method}: {ckpt_path}")
        model = load_lnnp(ckpt_path, device)
        for cid in conf_ids:
            data = dataset[cid]
            e_true = float(data.y.squeeze().item())  # eV, reference-subtracted
            e_pred = _predict_energy(model, data, device)
            err_eV = abs(e_pred - e_true)
            err_out = err_eV * (EV_TO_KCAL if args.unit == "kcal/mol" else 1.0)
            errors[cid][method] = float(err_out)
            print(f"  conf {cid:>6d}: true={e_true:+.4f} eV  pred={e_pred:+.4f} eV  err={err_out:.3f} {args.unit}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Store with conformation IDs as string keys (JSON-friendly)
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in errors.items()}, f, indent=2)
    print(f"[reps_errors] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
