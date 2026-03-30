set -ex
cd /mnt/shared-storage-user/qiaojingyang-p/code1/sy/GEOM3D
model_3d=SchNet
EDG_ROOT=/mnt/shared-storage-user/qiaojingyang-p/code1/sy/GEOM3D
dataroot=$EDG_ROOT/examples_3D/dataset
output_model_dir=./experiments/run_QM9_distillation
img_feat_path=$dataroot/QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz
pretrained_pth=$EDG_ROOT/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth
gpus=0,1,2,3  # 选择用哪几个 GPU
m=8  # 选择每个 GPU 跑几个程序，这个需要根据内存消耗调整
source /mnt/shared-storage-user/qiaojingyang-p/miniconda3/bin/activate
source activate Geom3D2

# global
python examples_3D/run_QM9_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_path $img_feat_path --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.01,0.5,1.0 --use_evaluator --evaluator_name mean_std --alpha_std_alls '\-1.5,0,1.5' --beta_batchs 0
# global+localzon
python examples_3D/run_QM9_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_path $img_feat_path --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.01,0.5,1.0 --use_evaluator --evaluator_name mean_std --alpha_std_batchs '\-1.5,0,1.5' --alpha_std_alls '\-1.5,0,1.5' --beta_batchs 0.5
# local
python examples_3D/run_QM9_distillation.py --model_3d $model_3d --gpus $gpus --m $m --output_model_dir $output_model_dir --dataroot $dataroot --img_feat_path $img_feat_path --pretrained_pth $pretrained_pth --use_ED --weight_EDs 0.01,0.5,1.0 --use_evaluator --evaluator_name mean_std --alpha_std_alls '\-1.5,0,1.5' --beta_batchs 1

