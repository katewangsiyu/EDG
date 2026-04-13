"""
精选4个任务的最终可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'figure.dpi': 300
})

COLOR_EDG = '#8FAADC'
COLOR_EDGPP = '#F4B183'

path = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results/error_distribution_data.npz')
data = np.load(path, allow_pickle=True)
results = data['results']

selected_keys = ['SchNet-r2', 'Equiformer-u0', 'Equiformer-g298', 'SphereNet-h298']

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

plot_idx = 0
for result in results:
    key = f"{result['stats']['model']}-{result['stats']['task']}"
    if key not in selected_keys:
        continue

    ax = axes[plot_idx]
    stats = result['stats']
    edg_error = result['edg_error']
    edgpp_error = result['edgpp_error']

    # 展示前500个样本
    edg_sorted = np.sort(edg_error)[:500]
    edgpp_sorted = np.sort(edgpp_error)[:500]
    x = np.arange(len(edg_sorted))

    ax.plot(x, edg_sorted, color=COLOR_EDG, label='EDG', linewidth=1.2, alpha=0.8)
    ax.plot(x, edgpp_sorted, color=COLOR_EDGPP, label='EDG++', linewidth=1.2, alpha=0.8)

    model = stats['model']
    task = stats['task'].upper()
    std_imp = (1 - stats['edgpp_std']/stats['edg_std']) * 100
    max_imp = (1 - stats['edgpp_max']/stats['edg_max']) * 100

    ax.set_title(f'{model} - {task}\\nStd ↓{std_imp:.1f}%, Max ↓{max_imp:.1f}%',
                fontweight='bold', fontsize=9)
    ax.set_xlabel('Sample Index (sorted by error)', fontsize=8)
    ax.set_ylabel('Absolute Error', fontsize=8)
    ax.legend(loc='upper left', fontsize=7, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.2, linestyle=':')

    plot_idx += 1

plt.tight_layout()

# 保存PNG版本
output_png = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/figures/final_4tasks_comparison.png')
plt.savefig(output_png, bbox_inches='tight', pad_inches=0.05, dpi=300)
print(f"✓ PNG version: {output_png}")

# 保存PDF版本（IEEE Trans格式）
output_pdf = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/figures/final_4tasks_comparison.pdf')
plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.05, format='pdf')
print(f"✓ PDF version: {output_pdf}")
