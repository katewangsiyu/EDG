# QM9 数据集

```bash
########### SchNet-QM9 【H200】
screen -R run_QM9
model_3d=SchNet  # 跑这三个：SchNet, SphereNet, Equiformer
EDG_ROOT=/home/ubuntu/wsy/GEOM3D
dataroot=$EDG_ROOT/examples_3D/dataset
output_model_dir=./experiments/run_QM9_distillation
img_feat_path=$dataroot/QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz
pretrained_pth=$EDG_ROOT/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth
gpus=0,1,2,3,4,5,6,7  # 选择用哪几个 GPU（H200,8卡）
m=2  # 选择每个 GPU 跑几个程序，这个需要根据内存消耗调整
cd $EDG_ROOT
source activate Geom3D

# global
python examples_3D/run_QM9_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_path $img_feat_path --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.01,0.5,1.0 --use_evaluator --evaluator_name mean_std --alpha_std_alls '\-1.5,0,1.5' --beta_batchs 0
# global+local
python examples_3D/run_QM9_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_path $img_feat_path --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.01,0.5,1.0 --use_evaluator --evaluator_name mean_std --alpha_std_batchs '\-1.5,0,1.5' --alpha_std_alls '\-1.5,0,1.5' --beta_batchs 0.5
# local
python examples_3D/run_QM9_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_path $img_feat_path --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.01,0.5,1.0 --use_evaluator --evaluator_name mean_std --alpha_std_alls '\-1.5,0,1.5' --beta_batchs 1
```



# rMD17 数据集

```bash
########### SphereNet-rMD17 H200 上跑
screen -R IEMv2_mean_std
model_3d=SphereNet
EDG_ROOT=/home/ubuntu/wsy/GEOM3D
dataroot=$EDG_ROOT/examples_3D/dataset
output_model_dir=./experiments/run_rMD17_distillation
img_feat_root=$dataroot/rMD17
pretrained_pth=$EDG_ROOT/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth
gpus=0,1,2,3,4,5,6,7  # 选择用哪几个 GPU（H200,8卡）
m=4  # 选择每个 GPU 跑几个程序，这个需要根据内存消耗调整

cd $EDG_ROOT
source activate Geom3D

python examples_3D/run_rMD17_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_root $img_feat_root --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.001 --use_evaluator --evaluator_name mean_std --alpha_std_alls '\-1.5,\-1,\-0.5,0,0.5,1.0,1.5' --beta_batchs 0
```

