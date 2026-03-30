# EDG++: Selective Knowledge Distillation with Electron Density for Molecular Representation Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IJCAI%202025-blue)](https://www.ijcai.org/proceedings/2025/872)

Official PyTorch implementation of **EDG++**, an extension of our IJCAI 2025 paper that introduces **selective knowledge distillation** using electron density as privileged information for enhanced molecular property prediction.

- 📄 [**EDG Paper (IJCAI 2025)**](https://www.ijcai.org/proceedings/2025/872) – Original EDG paper
- 📄 **EDG++ Paper (IEEE Transactions)** – Extended version with selective distillation (Coming soon)

## 🔥 What's New in EDG++

**EDG** (IJCAI 2025) introduced cross-modal knowledge distillation from electron density (ED) to geometric models.

**EDG++** (IEEE Transactions) extends this with:
- **Pretrained Reliability Estimator**: Learns to assess ED prediction quality during pretraining
- **Adaptive Threshold Mechanism**: Global + local hybrid filtering for selective distillation
- **Negative Transfer Mitigation**: Filters unreliable ED predictions to prevent performance degradation

## ⚙️ Installation

### Requirements
- CUDA >= 11.1
- Python >= 3.8

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/EDG_plusplus.git
cd EDG_plusplus

# Create environment
conda create -n edgpp python=3.8
conda activate edgpp

# Install PyTorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed instructions.

## 🚀 Pipeline

### Pretrained Models

We provide pretrained models for all stages:

| Stage | Model | Path | Description |
|-------|-------|------|-------------|
| Stage 1 | ImageED (ViT-Large) | `pretrain/pretrained_models/` | ED representation learning |
| Stage 2 | ED-aware Teacher + Reliability Estimator | `pretrain/pretrained_models/ED_Evaluator/ckpts/best_epoch=18_loss=0.30.pth` | ResNet18 teacher (90MB) |

### Stage 1: ImageED Pretraining (Optional)

```bash
cd pretrain/stage1_ImageED
python pretrain_ImageED.py \
    --model mae_vit_base_patch16 \
    --batch_size 256 \
    --epochs 800 \
    --data_path /path/to/ED_images
```

### Stage 2: ED-aware Teacher Pretraining (Optional)

```bash
cd pretrain/stage2_teacher_pretrain
python pretrain/pretrain_teachers_single_GPU.py \
    --model_name resnet18 \
    --dataset pcqm4m-v2 \
    --epochs 50 \
    --batch_size 128 \
    --use_ED
```

### Stage 3: Downstream Distillation (Main Usage)

**QM9 Dataset:**
```bash
cd downstream
python scripts/run_QM9_distillation.py \
    --model_3d SchNet \
    --task alpha \
    --dataroot datasets \
    --use_ED \
    --use_evaluator \
    --weight_ED 0.5 \
    --alpha_std_all 0.5 \
    --beta_batch 0.5 \
    --epochs 1000 \
    --batch_size 128
```

**rMD17 Dataset:**
```bash
python scripts/run_rMD17_distillation.py \
    --model_3d SphereNet \
    --task aspirin \
    --dataroot datasets \
    --use_ED \
    --use_evaluator \
    --weight_ED 0.001 \
    --alpha_std_all 0.5 \
    --epochs 1000
```

### Key Hyperparameters

- `--use_ED`: Enable ED distillation
- `--use_evaluator`: Enable selective distillation (EDG++)
- `--weight_ED`: Weight for ED distillation loss
- `--alpha_std_all`: Global threshold offset (0 = EDG naive, >0 = EDG++)
- `--alpha_std_batch`: Batch-level threshold offset
- `--beta_batch`: Mixing ratio for global/local thresholds

See [docs/DISTILLATION.md](docs/DISTILLATION.md) for detailed parameter tuning.

## 📁 Project Structure

## 📚 Citation

If you use this code, please cite:

```bibtex
@inproceedings{ijcai2025p872,
  title     = {Electron Density-enhanced Molecular Geometry Learning},
  author    = {Xiang, Hongxin and Xia, Jun and Jin, Xin and Du, Wenjie and Zeng, Li and Zeng, Xiangxiang},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {7840--7848},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/872},
  url       = {https://doi.org/10.24963/ijcai.2025/872}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Based on [Geom3D](https://github.com/chao1224/Geom3D) framework
- EDG baseline from [EDG (IJCAI 2025)](https://github.com/HongxinXiang/EDG)

## 📧 Contact

For questions, please open an issue or contact: [wangsiyu2030@gmail.com]
