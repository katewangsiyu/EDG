#!/bin/bash
# Launch 3 ViSNet-Chignolin trainings in PARALLEL, one per T4 GPU.
# GPU 1 = baseline, GPU 2 = +EDG, GPU 3 = +EDG++
# 1000 epochs each, early_stopping_patience=100.
# Wall-clock ~10.8 days on single T4 each, finishing roughly at the same time.

set -u
REPO_ROOT=/home/lzeng/workspace/EDG_for_PR/EDG-for-VisNet
CFG="${REPO_ROOT}/examples/ViSNet-Chignolin-T4.yml"
DATA_ROOT="${REPO_ROOT}/AI2BMD/chignolin_data"
IMG_FEAT=/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/dataset/Chignolin/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz
PRETRAINED=/home/lzeng/workspace/EDG_for_PR/GEOM3D/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth
LOG_BASE="${REPO_ROOT}/experiments/run_chignolin"
mkdir -p "${LOG_BASE}"
cd "${REPO_ROOT}"
source /home/lzeng/miniconda3/etc/profile.d/conda.sh
conda activate visnet

# --- Baseline on GPU 1 ---
nohup env CUDA_VISIBLE_DEVICES=1 python train_with_EDG.py \
  --conf "${CFG}" \
  --dataset-root "${DATA_ROOT}" \
  --log-dir "${LOG_BASE}/baseline" \
  --seed 42 \
  --weight_ED 0.0 \
  > "${LOG_BASE}/baseline.log" 2>&1 &
echo "baseline pid: $!"

# --- ViSNet + EDG (naive) on GPU 2 ---
nohup env CUDA_VISIBLE_DEVICES=2 python train_with_EDG.py \
  --conf "${CFG}" \
  --dataset-root "${DATA_ROOT}" \
  --log-dir "${LOG_BASE}/edg" \
  --seed 42 \
  --img_feat_path "${IMG_FEAT}" \
  --pretrained_pth "${PRETRAINED}" \
  --use_ED \
  --weight_ED 0.001 \
  > "${LOG_BASE}/edg.log" 2>&1 &
echo "edg pid: $!"

# --- ViSNet + EDG++ (kappa=0) on GPU 3 ---
nohup env CUDA_VISIBLE_DEVICES=3 python train_with_EDG.py \
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
  --beta_batch 0.0 \
  > "${LOG_BASE}/edg_pp.log" 2>&1 &
echo "edg_pp pid: $!"

echo ""
echo "3 trainings launched. Logs: ${LOG_BASE}/{baseline,edg,edg_pp}.log"
echo "To follow progress: tail -f ${LOG_BASE}/edg_pp.log"
echo "To monitor metrics: cat ${LOG_BASE}/*/output_*/metrics.csv | tail"
