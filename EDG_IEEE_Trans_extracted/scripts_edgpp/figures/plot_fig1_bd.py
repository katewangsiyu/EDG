"""
Redraw Fig 1 components (b) and (d) for EDG++ journal paper.

(b): Comparison of ED representations (point cloud, voxel, image)
     — RMSE, GPU memory, training time
(d): Average relative improvement on QM9 (SchNet, Equiformer)
     — EDG vs EDG++

Style: matches other paper figures (serif, IEEE Trans, blue/orange palette).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

OUTPUT_DIR = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/introduction'

# ── Paper colour palette ──────────────────────────────────────────────
C_POINT  = '#B0B0B0'    # light gray — point cloud
C_VOXEL  = '#8CAAC6'    # muted blue-gray — voxel
C_IMAGE  = '#3D6FA5'    # paper blue — our image (best)

C_BASE   = '#8FBC8F'    # sage green — baseline
C_EDG    = '#3D6FA5'    # paper blue — EDG
C_EDGPP  = '#D4652A'    # paper orange — EDG++


# ======================================================================
#  (b) ED representation comparison  — 3 grouped bar charts
# ======================================================================

def plot_fig1b():
    labels = ['Point\ncloud', 'Voxel', 'Our\nimage']
    colors = [C_POINT, C_VOXEL, C_IMAGE]

    metrics = [
        {
            'title': 'Prediction ability on energy',
            'ylabel': 'RMSE',
            'vals': [288.9, 202.0, 124.5],
            'imp_pct': 38.4,
        },
        {
            'title': 'GPU memory requirement',
            'ylabel': 'GPU (GiB)',
            'vals': [23.0, 5.0, 2.9],
            'imp_pct': 42.1,
        },
        {
            'title': 'Training time cost',
            'ylabel': 'Time (min)',
            'vals': [5.8, 5.0, 4.76],
            'imp_pct': 4.8,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.55),
                             gridspec_kw={'wspace': 0.5})

    for ax, m in zip(axes, metrics):
        vals = m['vals']
        x = np.arange(len(labels))
        ax.bar(x, vals, width=0.6, color=colors, edgecolor='white',
               linewidth=0.5, zorder=3)

        y_max = max(vals) * 1.30
        ax.set_ylim(0, y_max)

        # Value labels on top of bars
        for xi, v in zip(x, vals):
            fmt = f'{v:.0f}' if v >= 10 else f'{v:.1f}'
            ax.text(xi, v + y_max * 0.015, fmt,
                    ha='center', va='bottom', fontsize=5.5, color='#444444')

        # Improvement annotation: text only, placed to the right
        # Use a clean "↓X%" label next to the "Our image" bar
        v_to = vals[2]
        pct  = m['imp_pct']
        ax.text(2, v_to * 0.45, f'$\\downarrow${pct:.1f}%',
                fontsize=6.5, color=C_EDGPP, fontweight='bold',
                ha='center', va='center', zorder=6)

        # Axes styling
        ax.set_title(m['title'], fontsize=6.5, pad=5)
        ax.set_ylabel(m['ylabel'], fontsize=6.5, labelpad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5, linespacing=1.1)
        ax.tick_params(axis='y', labelsize=5.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.96, top=0.84, bottom=0.15)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUTPUT_DIR}/Fig1b_ed_representation.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('Saved Fig1b_ed_representation.pdf / .png')


# ======================================================================
#  (d) Average normalised MAE on QM9  — 3 horizontal bars per model
#      Matches the original rMD17 (d) style: Baseline / EDG / EDG++
# ======================================================================

def plot_fig1d():
    models = ['SchNet', 'Equiformer']

    # Normalised average MAE: Baseline = 100% for each model
    # SchNet:     EDG → 2.22% avg improvement,  EDG++ → 3.20%
    # Equiformer: EDG → 6.42% avg improvement,  EDG++ → 9.04%
    baseline  = [100.0, 100.0]
    edg_val   = [100 - 2.22,  100 - 6.42]
    edgpp_val = [100 - 3.20,  100 - 9.04]

    edg_pct   = [2.22,  6.42]
    edgpp_pct = [3.20,  9.04]

    y = np.arange(len(models))
    bar_h = 0.22

    fig, ax = plt.subplots(figsize=(3.8, 1.5))

    # 3 bars per model, top→bottom: Baseline, EDG, EDG++ (matching original style)
    ax.barh(y + bar_h,  baseline,  height=bar_h, color=C_BASE,
            edgecolor='white', linewidth=0.5, zorder=3)
    ax.barh(y,          edg_val,   height=bar_h, color=C_EDG,
            edgecolor='white', linewidth=0.5, zorder=3)
    ax.barh(y - bar_h,  edgpp_val, height=bar_h, color=C_EDGPP,
            edgecolor='white', linewidth=0.5, zorder=3)

    # Improvement annotations at bar tips
    for yi in range(len(models)):
        ax.text(edg_val[yi] + 0.3, yi,
                f'$\\downarrow${edg_pct[yi]:.2f}%',
                va='center', ha='left', fontsize=6, color=C_EDG,
                fontweight='bold')
        ax.text(edgpp_val[yi] + 0.3, yi - bar_h,
                f'$\\downarrow${edgpp_pct[yi]:.1f}%',
                va='center', ha='left', fontsize=6, color=C_EDGPP,
                fontweight='bold')

    # Axes
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.5, fontweight='medium')
    ax.set_xlabel('Normalised avg. MAE on QM9 (%)', fontsize=6.5, labelpad=3)
    ax.set_xlim(0, 112)
    ax.tick_params(axis='x', labelsize=5.5)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend: horizontal row below x-axis
    handles = [
        mpatches.Patch(facecolor=C_BASE,  edgecolor='#BBBBBB', label='Baseline'),
        mpatches.Patch(facecolor=C_EDG,   edgecolor='white',   label='EDG'),
        mpatches.Patch(facecolor=C_EDGPP, edgecolor='white',   label='EDG++'),
    ]
    ax.legend(handles=handles, fontsize=5.5, ncol=3,
              frameon=True, fancybox=False, edgecolor='#D0D0D0',
              framealpha=0.92, borderpad=0.3, handlelength=1.0,
              handletextpad=0.35, labelspacing=0.2, columnspacing=0.8,
              loc='lower center', bbox_to_anchor=(0.5, -0.52))

    fig.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.32)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUTPUT_DIR}/Fig1d_qm9_improvement.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('Saved Fig1d_qm9_improvement.pdf / .png')


if __name__ == '__main__':
    plot_fig1b()
    plot_fig1d()
