#!/bin/bash
# SchNet EDG++ on rMD17 (κ=0 默认配置)
# 用法：
#   bash run_SchNet_rMD17_EDGpp.sh 0,1,2,3     # 指定 GPU
#   bash run_SchNet_rMD17_EDGpp.sh 3            # 单卡（当前机器 GPU3 空闲）

set -e
cd /home/lzeng/workspace/GEOM3D

PYTHON_BIN=/home/lzeng/miniconda3/envs/visnet/bin/python
GPUS=${1:-"0,1,2,3"}
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

MOLECULES=(ethanol azobenzene naphthalene salicylic toluene aspirin uracil paracetamol malonaldehyde benzene)

DATAROOT=/home/lzeng/workspace/GEOM3D/examples_3D/dataset
PRETRAINED_PTH=/home/lzeng/workspace/GEOM3D/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth
OUTPUT_BASE=./experiments/run_rMD17_distillation/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.001_E@asb0_asa0_bb0/rs42

echo "=== SchNet EDG++ rMD17 (κ=0, λ=0.001) ==="
echo "GPUs: ${GPUS} (${NUM_GPUS} cards)"
echo "Molecules: ${MOLECULES[*]}"
echo ""

idx=0
for mol in "${MOLECULES[@]}"; do
    GPU_ID=${GPU_LIST[$((idx % NUM_GPUS))]}
    OUTPUT_DIR=${OUTPUT_BASE}/${mol}
    IMG_FEAT=${DATAROOT}/rMD17/${mol}/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz
    LOG_FILE=${OUTPUT_DIR}/logs.log

    # 跳过已完成的实验
    if [ -f "$LOG_FILE" ]; then
        if tail -1 "$LOG_FILE" | grep -q "best Force"; then
            echo "[SKIP] ${mol} — already completed"
            continue
        else
            echo "[CLEAN] ${mol} — incomplete, removing old results"
            rm -rf "$OUTPUT_DIR"
        fi
    fi

    echo "[START] ${mol} → GPU ${GPU_ID}"
    nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} examples_3D/finetune_MD17_distillation.py \
        --verbose --model_3d SchNet \
        --dataroot ${DATAROOT} \
        --dataset rMD17 --task ${mol} --rMD17_split_id 01 --seed 42 \
        --epochs 1000 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR \
        --no_eval_train --print_every_epoch 1 --energy_force_with_normalization \
        --img_feat_path ${IMG_FEAT} \
        --num_workers 8 \
        --pretrained_pth ${PRETRAINED_PTH} \
        --output_model_dir ${OUTPUT_DIR} \
        --use_ED --weight_ED 0.001 \
        --use_evaluator --evaluator_name mean_std --alpha_std_batch 0 --alpha_std_all 0 --beta_batch 0" \
        > /home/lzeng/workspace/GEOM3D/nohup_SchNet_${mol}.out 2>&1 &

    idx=$((idx + 1))

    # 每轮 GPU 分配满后等 5 秒，让进程稳定启动
    if [ $((idx % NUM_GPUS)) -eq 0 ]; then
        sleep 5
    fi
done

echo ""
echo "=== Launched ${idx} experiments ==="
echo ""
echo "Monitor commands:"
echo "  # 查看进度"
echo "  for mol in ${MOLECULES[*]}; do echo \"=== \$mol ===\"; tail -3 ${OUTPUT_BASE}/\$mol/logs.log 2>/dev/null || echo 'not started'; done"
echo ""
echo "  # 查看 GPU"
echo "  nvidia-smi"
echo ""
echo "  # 查看完成状态"
echo "  for mol in ${MOLECULES[*]}; do f=${OUTPUT_BASE}/\$mol/logs.log; if [ -f \"\$f\" ] && tail -1 \"\$f\" | grep -q 'best Force'; then echo \"\$mol: DONE\"; else echo \"\$mol: RUNNING\"; fi; done"
