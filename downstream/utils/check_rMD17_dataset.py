

"""检查 rMD 17数据集
"""

import numpy as np
import torch
from rdkit import Chem


def check(data_root, datasets):
    pt = Chem.GetPeriodicTable()  # be used for atom_id to symbol

    for dataset in datasets:
        npz_path = f"{data_root}/npz_data/rmd17_{dataset}.npz"
        pt_path = f"{data_root}/{dataset}/processed/geometric_data_processed.pt"

        data = np.load(npz_path, allow_pickle=True)
        pt_data = torch.load(pt_path)

        atoms_id = data['nuclear_charges']  # 核电荷数=原子序数
        energies = data['energies']
        coords = data['coords']
        forces = data['forces']

        pt_x, pt_y, pt_pos, pt_force = pt_data[0].x, pt_data[0].y, pt_data[0].positions, pt_data[0].force

        assert len(energies) == len(coords) == len(forces)

        assert (torch.from_numpy(forces).float() == pt_force.view(forces.shape)).all()  # forces 完全一样
        assert (pt_pos == pt_pos.view(pt_pos.shape)).all()  # position 完全一样
        assert (pt_y == torch.from_numpy(energies).float()).all()  # position 完全一样

        print(f"[{dataset}] 检查完毕，没有问题")

if __name__ == '__main__':
    data_root = "/data/xianghongxin/datasets/Geom3D/rMD17"
    datasets = ['ethanol', 'azobenzene', 'naphthalene', 'salicylic', 'toluene', 'aspirin', 'uracil', 'paracetamol', 'malonaldehyde', 'benzene']

    check(data_root, datasets)
