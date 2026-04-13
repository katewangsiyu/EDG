"""
Property-group improvement heatmap for EDG++ on QM9.

3 models (rows) × 12 properties (columns), transposed for compact layout.
Colour = relative improvement %. Properties grouped by Hohenberg-Kohn relevance.
Data source: Table 4 (tab:qm9_edgpp).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory

matplotlib.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.linewidth':     0.6,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

OUTPUT_DIR = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'

# ── Data ─────────────────────────────────────────────────────────────
MODELS = ['SchNet', 'SphereNet', 'Equiformer']

# Ordered: Thermodynamic (5) → Electronic (5) → Other (2)
PROPERTIES = [
    r'$C_v$', r'$G$', r'$H$', r'$U$', r'$U_0$',
    r'$\alpha$', r'$\Delta\varepsilon$', 'HOMO', 'LUMO', r'$R^2$',
    r'$\mu$', 'ZPVE',
]

# Original DATA shape: [12 properties, 3 models] → transpose to [3 models, 12 props]
DATA_ORIG = np.array([
    [ 2.5,  1.5,  3.5],   # Cv
    [ 6.5,  2.8, 34.6],   # G
    [ 2.2, 11.6, 10.6],   # H
    [ 3.7,  8.4,  3.5],   # U
    [ 3.6,  3.3, 29.3],   # U0
    [ 3.0,  2.2,  8.7],   # α
    [ 2.8,  1.6,  2.6],   # Δε
    [ 1.8,  1.1,  2.9],   # HOMO
    [ 2.6,  3.5,  5.9],   # LUMO
    [ 6.4,  1.9, 11.9],   # R²
    [ 1.0,  1.4, -1.9],   # μ
    [ 2.3,  4.4, -3.2],   # ZPVE
])
DATA = DATA_ORIG.T   # shape: [3 models, 12 properties]

# Group boundaries along the property (column) axis
GROUP_BOUNDS = [0, 5, 10, 12]
GROUP_LABELS = ['Thermodynamic', 'Electronic structure', 'Other']


def main():
    n_models, n_props = DATA.shape   # 3 × 12

    fig = plt.figure(figsize=(7.16, 1.75))     # IEEE double-col width, short height
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.015], wspace=0.015)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    # ── Diverging colour map ──
    vmax = 35
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=vmax)
    cmap = matplotlib.cm.RdBu

    im = ax.imshow(DATA, cmap=cmap, norm=norm, aspect='auto')

    # ── Subtle cell grid ──
    for i in range(n_models + 1):
        ax.axhline(i - 0.5, color='#E0E0E0', lw=0.3, zorder=1)
    for j in range(n_props + 1):
        ax.axvline(j - 0.5, color='#E0E0E0', lw=0.3, zorder=1)

    # ── Cell text ──
    for i in range(n_models):
        for j in range(n_props):
            val = DATA[i, j]
            txt_color = 'white' if abs(val) > 18 else '#222222'
            weight = 'bold' if (val < 0 or abs(val) > 10) else 'normal'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=6.5, color=txt_color, fontweight=weight)

    # ── Group separators (vertical white lines between property groups) ──
    for b in GROUP_BOUNDS[1:-1]:
        ax.axvline(b - 0.5, color='white', lw=3, zorder=3)

    # ── Group brackets and labels (below x-axis) ──
    trans_bracket = blended_transform_factory(ax.transData, ax.transAxes)

    by = -0.22           # bracket y in axes fraction (below axis)
    tick_len = 0.06      # vertical tick length in axes fraction

    for lab, start, end in zip(GROUP_LABELS, GROUP_BOUNDS[:-1], GROUP_BOUNDS[1:]):
        xmid = (start + end - 1) / 2
        x_left  = start - 0.35
        x_right = end - 0.65

        # Horizontal bracket line
        ax.plot([x_left, x_right], [by, by],
                color='#777777', lw=0.8, clip_on=False,
                transform=trans_bracket, zorder=5)
        # Left tick
        ax.plot([x_left, x_left], [by, by + tick_len],
                color='#777777', lw=0.8, clip_on=False,
                transform=trans_bracket, zorder=5)
        # Right tick
        ax.plot([x_right, x_right], [by, by + tick_len],
                color='#777777', lw=0.8, clip_on=False,
                transform=trans_bracket, zorder=5)

        # Group label below bracket
        ax.text(xmid, by - 0.04, lab,
                fontsize=6.5, va='top', ha='center',
                fontstyle='italic', color='#444444',
                clip_on=False,
                transform=trans_bracket)

    # ── Axes: properties on top (x), models on left (y) ──
    ax.set_xticks(range(n_props))
    ax.set_xticklabels(PROPERTIES, fontsize=7.5)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(MODELS, fontsize=8, fontweight='medium')

    ax.tick_params(axis='both', length=0, pad=3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Vertical colour bar on the right ──
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r'$\Delta$ (%)', fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    cbar.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.22)
    for ext in ('pdf', 'png'):
        plt.savefig(f'{OUTPUT_DIR}/property_group_improvement.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved property_group_improvement.pdf / .png')

    # ── Print group averages ──
    print('\n=== Group averages ===')
    for lab, start, end in zip(GROUP_LABELS, GROUP_BOUNDS[:-1], GROUP_BOUNDS[1:]):
        group = DATA_ORIG[start:end, :]
        print(f'{lab}:  '
              + '  '.join(f'{m}: {group[:, j].mean():+.1f}%'
                          for j, m in enumerate(MODELS)))


if __name__ == '__main__':
    main()
