# Dataset Preparation

## QM9 Dataset

**Description**: 134k molecules with 12 quantum properties

**Download**:
```bash
bash scripts/download_datasets.sh qm9
```

**Manual Download**:
1. Download from [QM9 official](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
2. Place in `downstream/datasets/QM9/raw/`
3. Run preprocessing:
```bash
cd downstream
python datasets/preprocess_qm9.py
```

**Preprocessed ED Features**:
- File: `QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz`
- Size: ~261MB
- Contains: ED predictions + reliability scores from pretrained teacher

## rMD17 Dataset

**Description**: 10 molecules with MD trajectories (energy + forces)

**Molecules**: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic_acid, toluene, uracil

**Download**:
```bash
bash scripts/download_datasets.sh rmd17
```

**Preprocessed ED Features**:
- Per molecule: `rMD17/{molecule}/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz`
- Size: ~19MB per molecule

## Dataset Structure

```
downstream/datasets/
├── QM9/
│   ├── raw/
│   └── processed/
│       └── teacher_feats/
└── rMD17/
    ├── aspirin/
    │   └── processed/
    │       └── teacher_feats/
    └── ...
```
