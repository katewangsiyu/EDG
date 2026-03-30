# Distillation Training Guide

## Overview

EDG++ uses a three-stage pipeline. This guide focuses on **Stage 3: Downstream Distillation**.

## Quick Start

### QM9 Training

```bash
cd downstream

# EDG++ (selective distillation)
python scripts/run_QM9_distillation.py \
    --model_3d SchNet \
    --task alpha \
    --use_ED \
    --use_evaluator \
    --weight_ED 0.5 \
    --alpha_std_all 0.5 \
    --beta_batch 0.5 \
    --epochs 100

# EDG (naive distillation, for comparison)
python scripts/run_QM9_distillation.py \
    --model_3d SchNet \
    --task alpha \
    --use_ED \
    --weight_ED 0.5 \
    --alpha_std_all 0 \
    --epochs 100
```

### rMD17 Training

```bash
# EDG++ on aspirin
python scripts/run_rMD17_distillation.py \
    --model_3d SphereNet \
    --task aspirin \
    --use_ED \
    --use_evaluator \
    --weight_ED 0.001 \
    --alpha_std_all 0.5 \
    --epochs 500
```

## Key Hyperparameters

### Distillation Control

- `--use_ED`: Enable ED distillation (required)
- `--use_evaluator`: Enable reliability estimator (EDG++ vs EDG)
- `--weight_ED`: Weight for ED loss (0.5 for QM9, 0.001 for rMD17)

### Selective Distillation (EDG++)

- `--alpha_std_all`: Global threshold offset (0=EDG, 0.5=EDG++)
- `--alpha_std_batch`: Batch-level threshold offset (default: 0)
- `--beta_batch`: Mixing ratio for global/local (default: 0.5)

### Model Selection

- `--model_3d`: SchNet | Equiformer | SphereNet | ViSNet
- `--task`: For QM9: alpha, gap, homo, lumo, mu, Cv, G, H, r2, U, U0, zpve
- `--task`: For rMD17: aspirin, benzene, ethanol, etc.

## Understanding Parameters

### EDG vs EDG++

| Setting | alpha_std_all | use_evaluator | Description |
|---------|---------------|---------------|-------------|
| Baseline | N/A | False | No distillation |
| EDG | 0 | True | Naive distillation |
| EDG++ | >0 (e.g., 0.5) | True | Selective distillation |

### Adaptive Threshold Formula

```
threshold = mean(reliability) + alpha_std_all * std(all_reliability) + alpha_std_batch * std(batch_reliability)
final_threshold = beta_batch * threshold_batch + (1 - beta_batch) * threshold_global
```

Only samples with `reliability > final_threshold` are used for distillation.

## Recommended Settings

### QM9
- weight_ED: 0.5
- alpha_std_all: 0.5
- beta_batch: 0.5
- epochs: 100
- batch_size: 128

### rMD17
- weight_ED: 0.001
- alpha_std_all: 0.5
- beta_batch: 0.5
- epochs: 500
- batch_size: 1

## Output

Results saved to `experiments/run_{dataset}_distillation/{model}_{task}/`:
- `model_best.pth`: Best model checkpoint
- `evaluation_best.pth.npz`: Predictions on test set
- `train.log`: Training logs

## Troubleshooting

**Issue**: Loss becomes NaN
- Reduce learning rate or weight_ED

**Issue**: No improvement over baseline
- Check if ED features are loaded correctly
- Try different alpha_std_all values (0.3-0.7)
