"""
Fig. S1: Difficulty Stratification — Single Panel, Relative % Change

All 4 tasks on one panel using per-bin relative % change.
P0-25 values are clipped and annotated since they are far off-scale (~330-374%).
Shows the full crossover: degradation (easy) → improvement (hard).
IEEE Trans format, vector PDF output with embedded fonts.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

OUTPUT_DIR = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'

BASE = '/home/lzeng/workspace/GEOM3D/experiments/run_QM9_distillation'

TASKS = [
    dict(display=r'SchNet-$R^2$',
         k0=f'{BASE}/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED1.0_E@asb0_asa0_bb0/rs42/r2/evaluation_best.pth.npz',
         best=f'{BASE}/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.01_E@asb0_asa1.5_bb0.5/rs42/r2/evaluation_best.pth.npz'),
    dict(display=r'Equiformer-$U_0$',
         k0=f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa0_bb0/rs42/u0/evaluation_best.pth.npz',
         best=f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa-1.5_bb0/rs42/u0/evaluation_best.pth.npz'),
    dict(display=r'Equiformer-$G$',
         k0=f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa0_bb0/rs42/g298/evaluation_best.pth.npz',
         best=f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.01_E@asb0_asa1.5_bb0/rs42/g298/evaluation_best.pth.npz'),
    dict(display=r'SphereNet-$H$',
         k0=f'{BASE}/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.01_E@asb0_asa0_bb0/rs42/h298/evaluation_best.pth.npz',
         best=f'{BASE}/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb1.5_asa1.5_bb0.5/rs42/h298/evaluation_best.pth.npz'),
]

STYLES = [
    dict(color='#004080', marker='o'),    # 深蓝
    dict(color='#FD9F02', marker='s'),    # 暖橙
    dict(color='#A0A0A4', marker='^'),    # 灰色
    dict(color='#2E8B57', marker='D'),    # 海绿
]

BINS = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 100)]
BIN_LABELS = ['P0\u201325\n(Easy)', 'P25\u201350', 'P50\u201375',
              'P75\u201390', 'P90\u201395', 'P95\u2013100\n(Hard)']

Y_LO, Y_HI = -48, 78   # visible y-range (%)
CLIP_Y = 73             # clip P0-25 values here and annotate


def load_errors(t):
    k0   = np.load(t['k0'])
    best = np.load(t['best'])
    tgt  = k0['test_target']
    return np.abs(tgt - k0['test_pred']), np.abs(tgt - best['test_pred'])


def bin_relative_change(k0_err, best_err):
    """Per-bin relative change (%).  Negative = improvement."""
    vals = []
    for lo, hi in BINS:
        lo_t = np.percentile(k0_err, lo)
        hi_t = np.percentile(k0_err, hi)
        mask = (k0_err <= hi_t) if lo == 0 else ((k0_err > lo_t) & (k0_err <= hi_t))
        edg_mean   = k0_err[mask].mean()
        edgpp_mean = best_err[mask].mean()
        vals.append((edgpp_mean - edg_mean) / edg_mean * 100)
    return np.array(vals)


def plot_stratification(all_data):
    all_rel = [bin_relative_change(k, b) for k, b in all_data]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    x = np.arange(len(BINS))

    # ── Subtle horizontal grid for readability ──
    ax.axhline(0, color='#555555', lw=0.7, zorder=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E8E8E8', lw=0.4, zorder=0)

    # ── Plot lines with P0-25 clipped ──
    for sty, rel in zip(STYLES, all_rel):
        rel_plot = rel.copy()
        rel_plot[0] = min(rel[0], CLIP_Y)  # clip P0-25
        ax.plot(x, rel_plot, color=sty['color'], marker=sty['marker'],
                markersize=6.5, markeredgecolor='white', markeredgewidth=0.6,
                linewidth=1.8, clip_on=True, zorder=3)

    # ── P0-25: annotate actual values since they are far off-scale ──
    p0_vals = [rel[0] for rel in all_rel]
    p0_min, p0_max = min(p0_vals), max(p0_vals)
    ax.annotate(f'+{p0_min:.0f}\u2013{p0_max:.0f}%',
                xy=(0, CLIP_Y), xytext=(0.45, CLIP_Y + 2),
                fontsize=7, fontweight='bold', color='#666666',
                ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#999999',
                                lw=0.8, connectionstyle='arc3,rad=0.15'),
                zorder=6)

    # ── End-value labels at P95-100: stacked to avoid overlap ──
    end_items = sorted(
        [(rel[-1], sty['color']) for sty, rel in zip(STYLES, all_rel)],
        key=lambda v: -v[0])
    placed = []
    min_gap = 3.5
    for val, col in end_items:
        y = val
        for py in placed:
            if abs(y - py) < min_gap:
                y = py - min_gap
        ax.text(x[-1] + 0.18, y, f'{val:.1f}%',
                fontsize=7, color=col, fontweight='bold',
                va='center', ha='left', zorder=6)
        placed.append(y)

    # ── Region labels near zero line ──
    ax.text(5.15, 6, r'Worse $\uparrow$', fontsize=7.5, color='#B06060',
            fontstyle='italic', va='bottom', ha='left', zorder=6)
    ax.text(5.15, -6, r'Better $\downarrow$', fontsize=7.5, color='#508060',
            fontstyle='italic', va='top', ha='left', zorder=6)

    # ── Axes ──
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=8.5)
    ax.set_ylabel('Relative error change (%)', fontsize=9.5)
    ax.set_xlabel('Sample difficulty percentile', fontsize=9.5)
    ax.set_xlim(-0.4, 5.8)
    ax.set_ylim(Y_LO, Y_HI)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Legend ──
    handles = [
        Line2D([0], [0], color=s['color'], marker=s['marker'], markersize=6,
               markeredgecolor='white', markeredgewidth=0.5, linewidth=1.8,
               label=t['display'])
        for s, t in zip(STYLES, TASKS)
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper right',
              frameon=True, fancybox=False, edgecolor='#CCCCCC',
              framealpha=0.95, borderpad=0.4, handlelength=1.6,
              handletextpad=0.3, bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUTPUT_DIR}/difficulty_stratification_lines.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('Saved difficulty_stratification_lines.pdf / .png')


def print_table(all_data):
    print('\n=== Relative Error Change by Difficulty Bin (%) ===')
    bl = ['P0-25', 'P25-50', 'P50-75', 'P75-90', 'P90-95', 'P95-100']
    print(f'{"Task":>20s}  ' + '  '.join(f'{b:>8s}' for b in bl))
    for t, (k, b) in zip(TASKS, all_data):
        rel = bin_relative_change(k, b)
        print(f'{t["display"]:>20s}  ' + '  '.join(f'{v:+8.1f}' for v in rel))


def main():
    print('Loading data ...')
    all_data = [load_errors(t) for t in TASKS]
    print_table(all_data)
    print('\nGenerating figure ...')
    plot_stratification(all_data)
    print('Done.')


if __name__ == '__main__':
    main()
