#!/bin/bash
# Baseline实验：SphereNet on rMD17（无蒸馏）
# 用于绘制Baseline vs EDG++的可视化图

model_3d=SphereNet
EDG_ROOT=/home/ubuntu/wsy/GEOM3D
dataroot=$EDG_ROOT/examples_3D/dataset
output_model_dir=./experiments/run_rMD17_baseline
gpu=0

cd $EDG_ROOT
source activate geom3d
export PYTHONPATH=$EDG_ROOT:$PYTHONPATH

# 创建输出目录
mkdir -p $output_model_dir/naphthalene
mkdir -p $output_model_dir/uracil
mkdir -p $output_model_dir/malonaldehyde

# 运行Naphthalene baseline
echo "=== Running Naphthalene Baseline ==="
CUDA_VISIBLE_DEVICES=$gpu python examples_3D/finetune_MD17_distillation.py \
    --verbose --model_3d $model_3d \
    --dataroot $dataroot \
    --dataset rMD17 \
    --task naphthalene \
    --rMD17_split_id 01 \
    --seed 42 \
    --epochs 1000 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR \
    --no_eval_train --print_every_epoch 1 \
    --energy_force_with_normalization \
    --num_workers 8 \
    --output_model_dir $output_model_dir/naphthalene

# 运行Uracil baseline
echo "=== Running Uracil Baseline ==="
CUDA_VISIBLE_DEVICES=$gpu python examples_3D/finetune_MD17_distillation.py \
    --verbose --model_3d $model_3d \
    --dataroot $dataroot \
    --dataset rMD17 \
    --task uracil \
    --rMD17_split_id 01 \
    --seed 42 \
    --epochs 1000 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR \
    --no_eval_train --print_every_epoch 1 \
    --energy_force_with_normalization \
    --num_workers 8 \
    --output_model_dir $output_model_dir/uracil

# 运行Malonaldehyde baseline
echo "=== Running Malonaldehyde Baseline ==="
CUDA_VISIBLE_DEVICES=$gpu python examples_3D/finetune_MD17_distillation.py \
    --verbose --model_3d $model_3d \
    --dataroot $dataroot \
    --dataset rMD17 \
    --task malonaldehyde \
    --rMD17_split_id 01 \
    --seed 42 \
    --epochs 1000 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR \
    --no_eval_train --print_every_epoch 1 \
    --energy_force_with_normalization \
    --num_workers 8 \
    --output_model_dir $output_model_dir/malonaldehyde

echo "=== Baseline实验完成 ==="
echo "结果保存在: $output_model_dir"
