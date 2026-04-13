"""
Improved Fig. 3: Tail Error Analysis — Cumulative Mean Error Curves
IEEE Trans format: serif font, white background, clean styling.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

# ── IEEE Trans global rcParams ──────────────────────────────────────
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

OUTPUT_DIR = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'

BASE = '/home/lzeng/workspace/GEOM3D/experiments/run_QM9_distillation'

TASKS = [
    {
        'model': 'SchNet', 'task': 'r2',
        'display': r'SchNet-$R^2$',
        'k0': f'{BASE}/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED1.0_E@asb0_asa0_bb0/rs42/r2/evaluation_best.pth.npz',
        'best': f'{BASE}/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.01_E@asb0_asa1.5_bb0.5/rs42/r2/evaluation_best.pth.npz',
    },
    {
        'model': 'Equiformer', 'task': 'u0',
        'display': r'Equiformer-$U_0$',
        'k0': f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa0_bb0/rs42/u0/evaluation_best.pth.npz',
        'best': f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa-1.5_bb0/rs42/u0/evaluation_best.pth.npz',
    },
    {
        'model': 'Equiformer', 'task': 'g298',
        'display': r'Equiformer-$G$',
        'k0': f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa0_bb0/rs42/g298/evaluation_best.pth.npz',
        'best': f'{BASE}/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.01_E@asb0_asa1.5_bb0/rs42/g298/evaluation_best.pth.npz',
    },
    {
        'model': 'SphereNet', 'task': 'h298',
        'display': r'SphereNet-$H$',
        'k0': f'{BASE}/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.01_E@asb0_asa0_bb0/rs42/h298/evaluation_best.pth.npz',
        'best': f'{BASE}/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb1.5_asa1.5_bb0.5/rs42/h298/evaluation_best.pth.npz',
    },
]

# Refined palette
C_EDG   = '#3D6FA5'   # deeper, more saturated blue
C_EDGPP = '#D4652A'   # warmer, richer orange
C_FILL_EDG  = '#B8CCE4'  # light blue tint for gap shading
C_FILL_EDGPP = '#F5D4C0'  # light orange tint


def load_errors(task_cfg):
    k0   = np.load(task_cfg['k0'])
    best = np.load(task_cfg['best'])
    target = k0['test_target']
    return np.abs(target - k0['test_pred']), np.abs(target - best['test_pred'])


def plot_tail_curves(all_data):
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0),
                             gridspec_kw={'wspace': 0.08})

    for idx, (ax, task_cfg, (k0_err, best_err)) in enumerate(
            zip(axes, TASKS, all_data)):
        n = len(k0_err)

        # Sort by EDG error desc (hardest first) — matched samples
        order       = np.argsort(k0_err)[::-1]
        k0_sorted   = k0_err[order]
        best_sorted = best_err[order]

        # Cumulative mean from hardest outward
        k0_cm   = np.cumsum(k0_sorted)   / np.arange(1, n + 1)
        best_cm = np.cumsum(best_sorted) / np.arange(1, n + 1)
        top_k   = np.arange(1, n + 1) / n * 100

        # Visible range: Top 0.5 % → 25 %
        mask = (top_k >= 0.5) & (top_k <= 25)
        x      = top_k[mask]
        y_edg  = k0_cm[mask]
        y_pp   = best_cm[mask]

        # Light Gaussian smoothing for silk-smooth curves
        sigma   = max(1, int(0.002 * mask.sum()))
        y_edg_s = gaussian_filter1d(y_edg, sigma=sigma)
        y_pp_s  = gaussian_filter1d(y_pp,  sigma=sigma)

        # ── Shaded gap between curves ──
        ax.fill_between(x, y_edg_s, y_pp_s,
                        where=(y_pp_s < y_edg_s),
                        interpolate=True, alpha=0.22,
                        color=C_FILL_EDG, edgecolor='none', zorder=2)

        # ── Main curves ──
        ax.plot(x, y_edg_s, color=C_EDG,   linewidth=1.5, zorder=3,
                label=r'EDG ($\kappa\!=\!0$)')
        ax.plot(x, y_pp_s,  color=C_EDGPP, linewidth=1.5, zorder=3,
                label=r'EDG++ (tuned $\kappa$)')

        # ── Subtitle ──
        ax.set_title(task_cfg['display'],
                     fontsize=8, fontweight='bold', pad=5)

        # ── Key-percentile annotations ──
        y_range = y_edg.max() - y_pp.min()

        for pct in [1, 5]:
            idx_k = max(0, int(pct / 100 * n) - 1)
            yk    = k0_cm[idx_k]
            yb    = best_cm[idx_k]
            imp   = (1 - yb / yk) * 100

            # Dots on both curves
            dot_kw = dict(markersize=3.5, markeredgecolor='white',
                          markeredgewidth=0.5, zorder=5)
            ax.plot(pct, yk, 'o', color=C_EDG,   **dot_kw)
            ax.plot(pct, yb, 'o', color=C_EDGPP, **dot_kw)

            # Annotation with background for readability
            y_pad = y_range * 0.06
            ax.annotate(f'{imp:.1f}%',
                        xy=(pct, yk + y_pad),
                        fontsize=6, fontweight='bold', color='#333333',
                        ha='center', va='bottom', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.75))

        # ── Small dots at 10 % and 20 % (no text) ──
        for pct in [10, 20]:
            idx_k = max(0, int(pct / 100 * n) - 1)
            dot_kw = dict(markersize=2, markeredgecolor='white',
                          markeredgewidth=0.3, zorder=5, alpha=0.5)
            ax.plot(pct, k0_cm[idx_k],   'o', color=C_EDG,   **dot_kw)
            ax.plot(pct, best_cm[idx_k], 'o', color=C_EDGPP, **dot_kw)

        # ── Axes ──
        ax.set_xscale('log')
        ax.set_xlim(0.45, 28)
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda v, _: f'{v:g}%'))
        ax.minorticks_off()

        ax.set_xlabel('Top K %', fontsize=7, labelpad=2)

        # Only leftmost subplot gets y-axis label
        if idx == 0:
            ax.set_ylabel('Cumul. mean abs. error', fontsize=7, labelpad=3)
        else:
            ax.set_yticklabels([])

        ax.tick_params(labelsize=6, pad=2)


        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Shared legend at top-right of first subplot ──
    axes[0].legend(fontsize=5.5, loc='upper right',
                   frameon=True, fancybox=False,
                   edgecolor='#D0D0D0', framealpha=0.92,
                   borderpad=0.35, handlelength=1.4, handletextpad=0.4,
                   labelspacing=0.3)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22, wspace=0.08)
    for ext in ('pdf', 'png'):
        out = f'{OUTPUT_DIR}/tail_error_ecdf.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved tail_error_ecdf.pdf / .png')


def print_stats(all_data):
    print('\n=== Tail Improvement Statistics ===')
    for task_cfg, (k0_err, best_err) in zip(TASKS, all_data):
        n = len(k0_err)
        order      = np.argsort(k0_err)[::-1]
        k0_cm      = np.cumsum(k0_err[order])   / np.arange(1, n + 1)
        best_cm    = np.cumsum(best_err[order]) / np.arange(1, n + 1)
        parts = []
        for pct in [1, 5, 10, 20, 100]:
            idx = max(0, int(pct / 100 * n) - 1)
            parts.append(f'Top{pct}%={((1 - best_cm[idx]/k0_cm[idx])*100):+.1f}%')
        print(f'  {task_cfg["display"]:>20s}: {", ".join(parts)}')


def main():
    print('Loading per-sample prediction data ...')
    all_data = [load_errors(t) for t in TASKS]
    print_stats(all_data)
    print('\nGenerating tail error curves ...')
    plot_tail_curves(all_data)
    print('Done.')


if __name__ == '__main__':
    main()
