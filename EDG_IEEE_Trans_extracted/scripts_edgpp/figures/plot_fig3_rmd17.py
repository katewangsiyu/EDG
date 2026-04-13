"""Figure 3: rMD17 threshold sensitivity curves"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 600, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

rmd17 = pd.read_csv('/home/lzeng/workspace/GEOM3D/结果与分析/rmd17_all_results.csv')

# rMD17 baselines (no distillation)
baselines_energy = {
    'aspirin': 0.18091, 'azobenzene': 0.09794, 'benzene': 0.00647,
    'ethanol': 0.03784, 'malonaldehyde': 0.06005, 'naphthalene': 0.03823,
    'paracetamol': 0.10425, 'salicylic': 0.14119, 'toluene': 0.03452,
    'uracil': 0.08088
}

molecules = ['aspirin', 'azobenzene', 'benzene', 'ethanol', 'malonaldehyde',
             'naphthalene', 'paracetamol', 'salicylic', 'toluene', 'uracil']

mol_display = {
    'aspirin': 'Aspirin', 'azobenzene': 'Azobenzene', 'benzene': 'Benzene',
    'ethanol': 'Ethanol', 'malonaldehyde': 'Malona.', 'naphthalene': 'Naphth.',
    'paracetamol': 'Paracet.', 'salicylic': 'Salicylic', 'toluene': 'Toluene',
    'uracil': 'Uracil'
}

# EDG++ color palette (3 base colors + variations)
colors = [
    '#5B9BD5',  # Blue (circle)
    '#F4B183',  # Orange (square) - EDG++ SchNet
    '#A9D18E',  # Green (triangle) - EDG++ SphereNet
    '#8FAADC',  # Light blue (diamond) - EDG++ Equiformer
    '#C55A5A',  # Red (down triangle)
    '#7BC8A4',  # Teal (left triangle)
    '#D4A5D6',  # Purple (right triangle)
    '#E8C468',  # Yellow (pentagon)
    '#6BAED6',  # Sky blue (hexagon)
    '#999999',  # Gray (star)
]
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

fig, ax = plt.subplots(figsize=(3.5, 3.0))  # IEEE single-column

for i, mol in enumerate(molecules):
    subset = rmd17[rmd17['molecule'] == mol].sort_values('alpha_std_all')
    x = subset['alpha_std_all'].values
    baseline = baselines_energy[mol]
    y = (1 - subset['test_Energy'].values / baseline) * 100  # relative improvement %

    ax.plot(x, y, color=colors[i], marker=markers[i], markersize=4,
            linewidth=1.2, label=mol_display[mol])

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel(r'$\kappa_{all}$')
ax.set_ylabel('Relative improvement (%)')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, edgecolor='gray', ncol=1, fontsize=6.5)
# 移除grid，保持白色背景
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'
plt.savefig(f'{out}/rmd17_kappa_sensitivity.png')
plt.savefig(f'{out}/rmd17_kappa_sensitivity.pdf')
print("rMD17 sensitivity curves saved.")
