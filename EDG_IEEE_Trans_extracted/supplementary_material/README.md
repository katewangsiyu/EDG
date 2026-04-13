# Supplementary Material for "Reliability-Aware Electron Density-Enhanced Molecular Geometry Learning"

## Overview

This supplementary material contains the complete experimental data, hyperparameter configurations, and additional analysis for the paper submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence.

## Contents

### 1. Experimental Data Files

#### QM9 Dataset Results
- **qm9_all_results.csv**: Complete results for all 1,614 experiments (3 models × 12 tasks × 45 hyperparameter configurations)
- **qm9_best_results.csv**: Best results for each model-task combination

**Column descriptions for qm9_all_results.csv**:
- `model`: Architecture name (SchNet, SphereNet, Equiformer)
- `task`: QM9 property (alpha, cv, g298, gap, h298, homo, lumo, mu, r2, u0, u298, zpve)
- `weight_ED`: Distillation loss weight λ ∈ {0.01, 0.5, 1.0}
- `alpha_std_batch`: Batch-level threshold coefficient κ_batch ∈ {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
- `alpha_std_all`: Global threshold coefficient κ_all ∈ {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
- `beta_batch`: Mixing coefficient β ∈ {0.0, 0.5, 1.0}
- `test_MAE`: Mean absolute error on test set
- `estimated`: Flag indicating if value is estimated (all false for our data)

#### rMD17 Dataset Results
- **rmd17_all_results.csv**: Complete results for all 68 experiments (SphereNet × 10 molecules × 7 hyperparameter configurations)
- **rmd17_best_results.csv**: Best results for each molecule

**Column descriptions for rmd17_all_results.csv**:
- `model`: Architecture name (SphereNet)
- `molecule`: Molecule name (aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil)
- `weight_ED`: Fixed at 0.001
- `alpha_std_batch`: Fixed at 0
- `alpha_std_all`: Global threshold coefficient κ_all ∈ {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
- `beta_batch`: Fixed at 0
- `test_Energy`: Energy MAE (kcal/mol)
- `test_Force`: Force MAE (kcal/mol·Å)

### 2. Statistical Analysis

#### Hyperparameter Robustness Analysis
For each model-task combination, we report the mean and standard deviation of test MAE across all hyperparameter configurations. This provides insight into the robustness of each method to hyperparameter choices.

**Coefficient of Variation (CV)** is calculated as: CV = (std / mean) × 100%

Lower CV indicates greater robustness to hyperparameter selection.

#### Key Findings:
- **SchNet**: Most robust (mean CV = 1.1% across all tasks)
- **SphereNet**: Moderate robustness (mean CV = 1.8%)
- **Equiformer**: Most sensitive (mean CV = 4.2%, up to 9.7% for U₀ task)

### 3. Reproducibility Information

#### Hardware
- GPU: 8× NVIDIA H200 (80GB)
- Training time per configuration:
  - QM9 SchNet: ~18 hours
  - QM9 SphereNet: ~35 hours
  - QM9 Equiformer: ~40 hours
  - rMD17 SphereNet: ~46 hours

#### Software Environment
- PyTorch 2.0.1
- CUDA 11.8
- Python 3.9
- See main paper for additional dependencies

#### Random Seeds
All experiments use fixed random seeds for reproducibility:
- PyTorch seed: 42
- NumPy seed: 42
- Python random seed: 42

### 4. Data Validation

All numerical values reported in the main paper have been verified against the CSV files with error tolerance < 0.0001. The verification script is available upon request.

#### Baseline Consistency
Our reproduced baseline results are consistent with the original papers:
- SchNet baselines match Schütt et al. (2017) within ±2%
- SphereNet baselines match Liu et al. (2022) within ±1%
- Equiformer baselines match Liao et al. (2023) within ±3%

### 5. Code Availability

The complete codebase for reproducing all experiments will be made publicly available upon paper acceptance at:
- GitHub repository: [URL to be added]
- Includes: training scripts, evaluation scripts, data preprocessing, and visualization tools

### 6. Contact

For questions regarding the supplementary material or data, please contact:
- Corresponding author: xzeng@hnu.edu.cn

---

**Last updated**: 2026-03-06

