#!/bin/bash
# Launch 3 ViSNet training runs on Chignolin:
#   (1) baseline ViSNet (no distillation)
#   (2) ViSNet + EDG (naive distillation)
#   (3) ViSNet + EDG++ (reliability-aware selective distillation, kappa=0 default)
#
# Usage:
#   cd /home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet
#   bash scripts_chignolin/run_chignolin_training.sh [baseline|edg|edg_pp|all]
#
# Hardware: 4x Tesla T4 via DDP (per-GPU batch_size=2 -> effective batch=8)

set -euo pipefail

REPO_ROOT="/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet"
CFG="${REPO_ROOT}/examples/ViSNet-Chignolin-T4.yml"
DATA_ROOT="/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet/AI2BMD/chignolin_data"
IMG_FEAT="/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/Chignolin/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz"
PRETRAINED="/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth"
LOG_BASE="${REPO_ROOT}/experiments/run_chignolin"

mkdir -p "${LOG_BASE}"

# Note: pl.Trainer picks up all 4 T4s automatically via ngpus=-1 in YAML.
# Override with CUDA_VISIBLE_DEVICES if you want to restrict.

MODE="${1:-all}"
cd "${REPO_ROOT}"
source /home/lzeng/miniconda3/etc/profile.d/conda.sh
conda activate visnet

run_baseline() {
  echo "========== [1/3] ViSNet baseline on Chignolin =========="
  python train_with_EDG.py \
    --conf "${CFG}" \
    --dataset-root "${DATA_ROOT}" \
    --log-dir "${LOG_BASE}/baseline" \
    --seed 42 \
    --weight_ED 0.0
}

run_edg() {
  echo "========== [2/3] ViSNet + EDG (naive) on Chignolin =========="
  python train_with_EDG.py \
    --conf "${CFG}" \
    --dataset-root "${DATA_ROOT}" \
    --log-dir "${LOG_BASE}/edg" \
    --seed 42 \
    --img_feat_path "${IMG_FEAT}" \
    --pretrained_pth "${PRETRAINED}" \
    --use_ED \
    --weight_ED 0.001
}

run_edg_pp() {
  echo "========== [3/3] ViSNet + EDG++ (kappa=0) on Chignolin =========="
  python train_with_EDG.py \
    --conf "${CFG}" \
    --dataset-root "${DATA_ROOT}" \
    --log-dir "${LOG_BASE}/edg_pp" \
    --seed 42 \
    --img_feat_path "${IMG_FEAT}" \
    --pretrained_pth "${PRETRAINED}" \
    --use_ED \
    --weight_ED 0.001 \
    --use_evaluator \
    --evaluator_name mean_std \
    --alpha_std_batch 0.0 \
    --alpha_std_all 0.0 \
    --beta_batch 0.0
}

case "${MODE}" in
  baseline) run_baseline ;;
  edg)      run_edg ;;
  edg_pp)   run_edg_pp ;;
  all)
    run_baseline
    run_edg
    run_edg_pp
    ;;
  *) echo "unknown mode: ${MODE}"; exit 1 ;;
esac
