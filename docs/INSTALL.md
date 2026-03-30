# Installation Guide

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.1 (for GPU support)

## Step 1: Create Environment

```bash
conda create -n edgpp python=3.8
conda activate edgpp
```

## Step 2: Install PyTorch

```bash
# For CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# For CPU only
pip install torch==1.12.0 torchvision==0.13.0
```

## Step 3: Install PyTorch Geometric

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric
```

## Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Install EDG++

```bash
pip install -e .
```

## Verify Installation

```bash
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

## Troubleshooting

**Issue**: CUDA out of memory
- Solution: Reduce batch size in config files

**Issue**: PyG installation fails
- Solution: Check CUDA version compatibility at https://pytorch-geometric.readthedocs.io/
