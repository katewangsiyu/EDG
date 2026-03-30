# Experiment Results

This directory stores experimental results from distillation training.

## Structure

```
experiments/
├── run_QM9_distillation/
│   ├── SchNet_alpha/
│   │   ├── model_best.pth
│   │   ├── evaluation_best.pth.npz
│   │   └── train.log
│   └── ...
└── run_rMD17_distillation/
    ├── SphereNet_aspirin/
    └── ...
```

## Result Files

- `model_best.pth`: Best model checkpoint (excluded from git)
- `evaluation_best.pth.npz`: Test set predictions and ground truth
- `train.log`: Training logs with loss curves

## Parsing Results

```python
import numpy as np

# Load evaluation results
data = np.load('evaluation_best.pth.npz')
predictions = data['predictions']
targets = data['targets']

# Calculate MAE
mae = np.mean(np.abs(predictions - targets))
```

## Hyperparameter Mapping

### EDG vs EDG++

- **Baseline**: No `--use_ED` flag
- **EDG (naive)**: `--use_ED --alpha_std_all 0`
- **EDG++ (selective)**: `--use_ED --use_evaluator --alpha_std_all 0.5`

### Key Parameters

- `weight_ED`: Distillation loss weight
- `alpha_std_all`: Global threshold offset
- `alpha_std_batch`: Batch threshold offset
- `beta_batch`: Global/local mixing ratio
