#!/usr/bin/env python3
"""
绘制rMD17 Baseline vs EDG++可视化图
模仿IJCAI论文中的rMD17_vis.png
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
from PIL import Image

# 设置绘图风格
sns.set_style("white")  # 改为white，移除底纹
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

# 生成分子结构图（透明背景）
def generate_molecule_image(smiles, size=(200, 200)):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    # 转换为RGBA，设置白色背景为透明
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:  # 白色背景
            newData.append((255, 255, 255, 0))  # 设为透明
        else:
            newData.append(item)
    img.putdata(newData)
    return img

# 分子SMILES
naphthalene_smiles = "c1ccc2ccccc2c1"
uracil_smiles = "O=C1NC=CC(=O)N1"

# 生成分子结构图
mol_img_nap = generate_molecule_image(naphthalene_smiles)
mol_img_ura = generate_molecule_image(uracil_smiles)

# 数据路径
baseline_naphthalene = "/home/ubuntu/wsy/GEOM3D/experiments/run_rMD17_baseline/naphthalene/evaluation_best.pth.npz"
baseline_uracil = "/home/ubuntu/wsy/GEOM3D/experiments/run_rMD17_baseline/uracil/evaluation_best.pth.npz"
edgpp_naphthalene = "/home/ubuntu/wsy/GEOM3D/experiments/run_rMD17_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.001_E@asb0_asa1.5_bb0/rs42/naphthalene/evaluation_best.pth.npz"
edgpp_uracil = "/home/ubuntu/wsy/GEOM3D/experiments/run_rMD17_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.001_E@asb0_asa1.5_bb0/rs42/uracil/evaluation_best.pth.npz"

# 加载数据
baseline_nap = np.load(baseline_naphthalene, allow_pickle=True)
baseline_ura = np.load(baseline_uracil, allow_pickle=True)
edgpp_nap = np.load(edgpp_naphthalene, allow_pickle=True)
edgpp_ura = np.load(edgpp_uracil, allow_pickle=True)

# 提取能量预测和真实值
y_true_nap = baseline_nap['test_eval_dict'].item()['y_energy_true']
y_pred_baseline_nap = baseline_nap['test_eval_dict'].item()['y_energy_pred']
y_pred_edgpp_nap = edgpp_nap['test_eval_dict'].item()['y_energy_pred']

y_true_ura = baseline_ura['test_eval_dict'].item()['y_energy_true']
y_pred_baseline_ura = baseline_ura['test_eval_dict'].item()['y_energy_pred']
y_pred_edgpp_ura = edgpp_ura['test_eval_dict'].item()['y_energy_pred']

# 计算误差
error_baseline_nap = np.abs(y_true_nap - y_pred_baseline_nap)
error_edgpp_nap = np.abs(y_true_nap - y_pred_edgpp_nap)
error_baseline_ura = np.abs(y_true_ura - y_pred_baseline_ura)
error_edgpp_ura = np.abs(y_true_ura - y_pred_edgpp_ura)

# 平滑处理（使用Savitzky-Golay滤波器）
window_length = 51  # 窗口长度，必须是奇数
polyorder = 3  # 多项式阶数

error_baseline_nap_smooth = savgol_filter(error_baseline_nap, window_length, polyorder)
error_edgpp_nap_smooth = savgol_filter(error_edgpp_nap, window_length, polyorder)
error_baseline_ura_smooth = savgol_filter(error_baseline_ura, window_length, polyorder)
error_edgpp_ura_smooth = savgol_filter(error_edgpp_ura, window_length, polyorder)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Naphthalene
axes[0].plot(error_baseline_nap_smooth, color='#E57373', linewidth=1.5, alpha=0.8, label='Baseline')
axes[0].plot(error_edgpp_nap_smooth, color='#64B5F6', linewidth=1.5, alpha=0.8, label='EDG++')
axes[0].set_xlabel('Dynamic trajectories')
axes[0].set_ylabel(r'abs($y_{true} - y_{pred}$)')
axes[0].set_title('SphereNet on Naphthalene')
# 设置y轴上限为数据最大值的1.3倍，为分子结构图预留空间
y_max_nap = max(error_baseline_nap_smooth.max(), error_edgpp_nap_smooth.max())
axes[0].set_ylim(0, y_max_nap * 1.3)
axes[0].legend(loc='upper right', framealpha=0.9)
# 移除grid
axes[0].grid(False)
# 添加分子结构图到左上角
imagebox = OffsetImage(mol_img_nap, zoom=0.3)
ab = AnnotationBbox(imagebox, (100, y_max_nap * 1.15), frameon=False, xycoords='data')
axes[0].add_artist(ab)

# Uracil
axes[1].plot(error_baseline_ura_smooth, color='#E57373', linewidth=1.5, alpha=0.8, label='Baseline')
axes[1].plot(error_edgpp_ura_smooth, color='#64B5F6', linewidth=1.5, alpha=0.8, label='EDG++')
axes[1].set_xlabel('Dynamic trajectories')
axes[1].set_ylabel(r'abs($y_{true} - y_{pred}$)')
axes[1].set_title('SphereNet on Uracil')
# 设置y轴上限为数据最大值的1.3倍，为分子结构图预留空间
y_max_ura = max(error_baseline_ura_smooth.max(), error_edgpp_ura_smooth.max())
axes[1].set_ylim(0, y_max_ura * 1.3)
axes[1].legend(loc='upper right', framealpha=0.9)
# 移除grid
axes[1].grid(False)
# 添加分子结构图到左上角
imagebox = OffsetImage(mol_img_ura, zoom=0.3)
ab = AnnotationBbox(imagebox, (100, y_max_ura * 1.15), frameon=False, xycoords='data')
axes[1].add_artist(ab)

plt.tight_layout()
plt.savefig('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments/rMD17_baseline_vs_edgpp.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments/rMD17_baseline_vs_edgpp.png',
            dpi=300, bbox_inches='tight')
print("图片已保存到:")
print("  - rMD17_baseline_vs_edgpp.pdf")
print("  - rMD17_baseline_vs_edgpp.png")
