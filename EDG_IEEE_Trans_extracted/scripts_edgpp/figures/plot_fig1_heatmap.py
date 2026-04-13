"""
Hyperparameter interaction heatmap for QM9 (Appendix A.10).

2 models × 4 tasks = 8 subplots.  Each cell shows relative improvement (%)
over no-distillation baseline as a function of κ_batch and κ_all with β=0.5.
Shared colour scale across all subplots for fair comparison.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.linewidth':     0.5,
    'xtick.major.width':  0.4,
    'ytick.major.width':  0.4,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

OUTPUT_DIR = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'

qm9 = pd.read_csv('/home/lzeng/workspace/GEOM3D/结果与分析/qm9_all_results.csv')

# Baselines (no distillation)
baselines = {
    'SchNet':    {'alpha': 0.07021, 'g298': 0.014678, 'r2': 0.13455, 'mu': 0.03013},
    'SphereNet': {'alpha': 0.04670, 'g298': 0.007875, 'r2': 0.25821, 'mu': 0.02689},
}

task_display = {
    'alpha': r'$\alpha$', 'g298': r'$G$', 'r2': r'$R^2$', 'mu': r'$\mu$'
}

tasks = ['alpha', 'g298', 'r2', 'mu']
models = ['SchNet', 'SphereNet']

# ── First pass: compute all heatmap data and find global vlim ──
all_heatmaps = {}
global_vmax = 0

for model in models:
    for task in tasks:
        baseline = baselines[model][task]
        subset = qm9[(qm9['model'] == model) & (qm9['task'] == task) &
                      (qm9['beta_batch'] == 0.5)]
        pivot_data = subset.groupby(
            ['alpha_std_batch', 'alpha_std_all'])['test_MAE'].min().reset_index()
        pivot_data['improvement'] = (1 - pivot_data['test_MAE'] / baseline) * 100
        heatmap_data = pivot_data.pivot(
            index='alpha_std_batch', columns='alpha_std_all', values='improvement')
        heatmap_data = heatmap_data.sort_index(ascending=False)
        all_heatmaps[(model, task)] = heatmap_data
        global_vmax = max(global_vmax, abs(heatmap_data.values).max())

# Symmetric limits for shared colour scale
global_vmax = np.ceil(global_vmax)
norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0, vmax=global_vmax)
cmap = matplotlib.cm.RdBu

# ── Build figure ──
fig = plt.figure(figsize=(7.16, 3.0))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05],
                      hspace=0.35, wspace=0.25)

axes_grid = {}
for i, model in enumerate(models):
    for j, task in enumerate(tasks):
        ax = fig.add_subplot(gs[i, j])
        axes_grid[(i, j)] = ax
        heatmap_data = all_heatmaps[(model, task)]
        n_rows, n_cols = heatmap_data.shape

        im = ax.imshow(heatmap_data.values, cmap=cmap, norm=norm, aspect='equal')

        # ── Cell grid ──
        for r in range(n_rows + 1):
            ax.axhline(r - 0.5, color='white', lw=1.2, zorder=1)
        for c in range(n_cols + 1):
            ax.axvline(c - 0.5, color='white', lw=1.2, zorder=1)

        # ── Cell text ──
        for ri in range(n_rows):
            for ci in range(n_cols):
                v = heatmap_data.iloc[ri, ci]
                txt_color = 'white' if abs(v) > 4 else '#222222'
                weight = 'bold' if abs(v) > 5 else 'normal'
                ax.text(ci, ri, f'{v:.1f}', ha='center', va='center',
                        fontsize=6.5, color=txt_color, fontweight=weight)

        # ── Task title (top row only) ──
        if i == 0:
            ax.set_title(task_display[task], fontsize=9.5, pad=4)

        # ── x-axis: κ_all (bottom row only) ──
        ax.set_xticks(range(n_cols))
        if i == 1:
            ax.set_xticklabels([f'{x:.1f}' for x in sorted(heatmap_data.columns)],
                               fontsize=6.5)
            ax.set_xlabel(r'$\kappa_{all}$', fontsize=8, labelpad=1)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        # ── y-axis: κ_batch (leftmost column only) ──
        ax.set_yticks(range(n_rows))
        if j == 0:
            ax.set_yticklabels(
                [f'{y:.1f}' for y in sorted(heatmap_data.index, reverse=True)],
                fontsize=6.5)
            ax.set_ylabel(r'$\kappa_{batch}$', fontsize=8, labelpad=1)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

        ax.tick_params(length=0, pad=2)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

# ── Row (model) labels on the leftmost column axes ──
for i, model in enumerate(models):
    axes_grid[(i, 0)].text(
        -0.5, 0.5, model, fontsize=9, fontweight='bold',
        va='center', ha='right', rotation=90,
        transform=axes_grid[(i, 0)].transAxes)

# ── Shared colour bar ──
cax = fig.add_subplot(gs[:, 4])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'$\Delta$ (%)', fontsize=8, labelpad=3)
cbar.ax.tick_params(labelsize=6, width=0.4, length=2.5)
cbar.outline.set_linewidth(0.4)

fig.subplots_adjust(left=0.09, right=0.93, top=0.92, bottom=0.12)
for ext in ('pdf', 'png'):
    plt.savefig(f'{OUTPUT_DIR}/heatmap_hyperparam.{ext}',
                bbox_inches='tight', dpi=300)
plt.close()
print('Saved heatmap_hyperparam.pdf / .png')
