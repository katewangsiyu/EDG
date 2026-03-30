# Pretraining Guide

## Overview

EDG++ uses a two-stage pretraining pipeline:
1. **Stage 1**: ImageED pretraining (ViT-Large MAE on 2M ED images)
2. **Stage 2**: ED-aware Teacher + Reliability Estimator pretraining

**Note**: We provide pretrained weights. You only need to run this if you want to reproduce from scratch.

## Stage 1: ImageED Pretraining

### Dataset
- 2M electron density images (224×224)
- Generated from molecular structures

### Training
```bash
cd pretrain/stage1_ImageED_pretrain
python pretrain_mae.py \
    --model vit_large_patch16 \
    --epochs 200 \
    --batch_size 256 \
    --mask_ratio 0.75
```

### Output
- Pretrained ViT-Large encoder
- Used to initialize Stage 2 teacher

## Stage 2: ED-aware Teacher Pretraining

### Dataset
- PCQM4Mv2 (3.8M molecules)
- Structure images + ED images

### Training
```bash
cd pretrain/stage2_teacher_pretrain
python pretrain/pretrain_teachers_single_GPU.py \
    --dataset pcqm4m-v2 \
    --model resnet18 \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-3
```

### Output
- `ED_Evaluator_from_10_epoch/best_epoch=18_loss=0.30.pth`
- Contains: ResNet18 teacher + reliability estimator head

### Architecture
- Backbone: ResNet18
- Input: Structure images (224×224)
- Output: ED features (512-dim) + reliability score (1-dim)

## Using Pretrained Models

Download our pretrained models:
```bash
bash scripts/download_pretrained_models.sh
```

Models will be placed in:
- `pretrain/pretrained_models/ED_Evaluator/`

## Generating ED Features for New Datasets

```bash
cd pretrain/stage2_teacher_pretrain
python generate_ed_features.py \
    --dataset_path /path/to/dataset \
    --model_path pretrained_models/ED_Evaluator/best_epoch=18_loss=0.30.pth \
    --output_path /path/to/output.npz
```
