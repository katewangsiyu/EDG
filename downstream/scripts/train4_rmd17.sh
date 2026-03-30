# rMD17 数据集


########### SphereNet-rMD17 4090 上跑
cd /mnt/shared-storage-user/qiaojingyang-p/code1/sy/GEOM3D/
model_3d=SphereNet
EDG_ROOT=/mnt/shared-storage-user/qiaojingyang-p/code1/sy/GEOM3D
dataroot=$EDG_ROOT/examples_3D/dataset
output_model_dir=./experiments/run_rMD17_distillation
img_feat_root=$dataroot/rMD17
pretrained_pth=$EDG_ROOT/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth
gpus=0,1  # 选择用哪几个 GPU
m=4  # 选择每个 GPU 跑几个程序，这个需要根据内存消耗调整
source /mnt/shared-storage-user/qiaojingyang-p/miniconda3/bin/activate
source activate Geom3D2

python examples_3D/run_rMD17_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_root $img_feat_root --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.001 --use_evaluator --evaluator_name mean_std --alpha_std_alls '\-1.5,\-1,\-0.5,0,0.5,1.0,1.5' --beta_batchs 0
#python examples_3D/run_rMD17_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_root $img_feat_root --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.001 --use_evaluator --evaluator_name mean_std --alpha_std_alls '1.5' --beta_batchs 0