"""ASE Calculator wrapping a trained visnet_for_EDG LNNP checkpoint.

Designed for single-molecule MD (no fragmentation, no PBC), matching the Chignolin case study.
Works with any ASE MD driver (Langevin, VelocityVerlet, NVTBerendsen).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

# Make sure visnet_for_EDG is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch_geometric.data.data as pgd
from torch_geometric.data import Batch, Data

# Allow-list torch_geometric globals for torch>=2.6 weights_only default.
torch.serialization.add_safe_globals(
    [pgd.DataEdgeAttr, pgd.DataTensorAttr, pgd.GlobalStorage]
)

from visnet_for_EDG.module import LNNP  # noqa: E402


def _load_lnnp(ckpt_path: str, map_location: str = "cpu") -> LNNP:
    model = LNNP.load_from_checkpoint(ckpt_path, map_location=map_location, strict=False)
    model.eval()
    return model


class ViSNetEDGppCalculator(Calculator):
    """ASE Calculator for a visnet_for_EDG-trained ViSNet student.

    Energy units: the LNNP returns energy in the same units as training labels. For Chignolin the
    AI2BMD npz is in eV (Hartree converted). Forces are eV/A.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, ckpt_path: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        print(f"[ViSNetEDGppCalculator] loading {ckpt_path} on {device}")
        self.model = _load_lnnp(ckpt_path, map_location=device).to(self.device)
        # disable distillation path during inference
        self.model.hparams.weight_ED = 0.0

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        z = torch.as_tensor(atoms.numbers, dtype=torch.long, device=self.device)
        pos = torch.as_tensor(atoms.positions, dtype=torch.float32, device=self.device)
        pos.requires_grad_(True)

        data = Data(z=z, pos=pos, batch=torch.zeros(len(atoms), dtype=torch.long, device=self.device))
        batch = Batch.from_data_list([data])
        batch.batch = torch.zeros(len(atoms), dtype=torch.long, device=self.device)

        # Use the representation model + output head; compute forces via autograd
        x, v = self.model.model.representation_model(batch)
        x = self.model.model.output_model.pre_reduce(x, v, batch.z, batch.pos, batch.batch)
        x = x * self.model.model.std
        if self.model.model.prior_model is not None:
            x = self.model.model.prior_model(x, batch.z)
        from torch_scatter import scatter
        energy = scatter(x, batch.batch, dim=0, reduce=self.model.model.reduce_op)
        energy = self.model.model.output_model.post_reduce(energy) + self.model.model.mean

        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=False, retain_graph=False)[0]

        self.results["energy"] = float(energy.detach().cpu().numpy().squeeze())
        self.results["forces"] = forces.detach().cpu().numpy()
