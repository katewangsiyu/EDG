"""
Error Distribution Analysis: κ=0 (default threshold) vs tuned κ (best EDG++)
Generates:
  1. tail_error_improvement.pdf - tail percentile bar chart
  2. difficulty_improvement_ieee.pdf - difficulty stratification analysis
  3. Updated error_distribution_data.npz
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = '/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'
DATA_DIR = '/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results'

# 4 tasks: (model, task, display_name, kappa0_path, best_path)
TASKS = [
    {
        'model': 'SchNet', 'task': 'r2',
        'display': r'SchNet-$R^2$',
        'k0': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED1.0_E@asb0_asa0_bb0/rs42/r2/evaluation_best.pth.npz',
        'best': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.01_E@asb0_asa1.5_bb0.5/rs42/r2/evaluation_best.pth.npz',
    },
    {
        'model': 'Equiformer', 'task': 'u0',
        'display': r'Equiformer-$U_0$',
        'k0': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa0_bb0/rs42/u0/evaluation_best.pth.npz',
        'best': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa-1.5_bb0/rs42/u0/evaluation_best.pth.npz',
    },
    {
        'model': 'Equiformer', 'task': 'g298',
        'display': r'Equiformer-$G$',
        'k0': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.5_E@asb0_asa0_bb0/rs42/g298/evaluation_best.pth.npz',
        'best': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/Equiformer/e300_b128_lr5e-4_ed128_lsCosine_Eh1/ED_E@mean_std/ED0.01_E@asb0_asa1.5_bb0/rs42/g298/evaluation_best.pth.npz',
    },
    {
        'model': 'SphereNet', 'task': 'h298',
        'display': r'SphereNet-$H$',
        'k0': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.01_E@asb0_asa0_bb0/rs42/h298/evaluation_best.pth.npz',
        'best': '/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb1.5_asa1.5_bb0.5/rs42/h298/evaluation_best.pth.npz',
    },
]


def load_errors(task_cfg):
    """Load kappa=0 and best errors for a task."""
    k0 = np.load(task_cfg['k0'])
    best = np.load(task_cfg['best'])
    target = k0['test_target']
    k0_err = np.abs(target - k0['test_pred'])
    best_err = np.abs(target - best['test_pred'])
    return target, k0_err, best_err


def plot_tail_error(all_data):
    """Fig 1: Tail error improvement bar chart."""
    percentiles = [1, 5, 10, 20]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

    for ax, task_cfg, (target, k0_err, best_err) in zip(axes, TASKS, all_data):
        # Rank by k0 error (difficulty)
        order = np.argsort(k0_err)[::-1]
        k0_sorted = k0_err[order]
        best_sorted = best_err[order]

        x = np.arange(len(percentiles))
        k0_vals, best_vals, imps = [], [], []
        for p in percentiles:
            n = max(1, int(len(k0_sorted) * p / 100))
            k0_mean = k0_sorted[:n].mean()
            best_mean = best_sorted[:n].mean()
            imp = (1 - best_mean / k0_mean) * 100
            k0_vals.append(k0_mean)
            best_vals.append(best_mean)
            imps.append(imp)

        w = 0.35
        bars1 = ax.bar(x - w/2, k0_vals, w, label=r'Default $\kappa$ ($\kappa$=0)',
                       color='#5B9BD5', edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + w/2, best_vals, w, label=r'Tuned $\kappa$',
                       color='#ED7D31', edgecolor='white', linewidth=0.5)

        for i, imp in enumerate(imps):
            y_max = max(k0_vals[i], best_vals[i])
            ax.text(x[i], y_max * 1.02, f'{imp:.1f}%',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Top {p}%' for p in percentiles], fontsize=8)
        ax.set_title(task_cfg['display'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=8)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(fontsize=7.5, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tail_error_improvement.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved tail_error_improvement.pdf')


def plot_difficulty_stratification(all_data):
    """Fig 2: Difficulty stratification (error change by percentile)."""
    bins = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 100)]
    bin_labels = ['P0-25', 'P25-50', 'P50-75', 'P75-90', 'P90-95', 'P95-100']

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

    for ax, task_cfg, (target, k0_err, best_err) in zip(axes, TASKS, all_data):
        # Stratify by k0 error magnitude
        diff = best_err - k0_err  # negative = improvement

        vals = []
        for lo, hi in bins:
            lo_thresh = np.percentile(k0_err, lo)
            hi_thresh = np.percentile(k0_err, hi)
            if lo == 0:
                mask = k0_err <= hi_thresh
            else:
                mask = (k0_err > lo_thresh) & (k0_err <= hi_thresh)
            vals.append(diff[mask].mean())

        colors = ['#E74C3C' if v > 0 else '#27AE60' for v in vals]
        x = np.arange(len(bins))
        ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=30, ha='right')
        ax.set_title(task_cfg['display'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Error Change', fontsize=8)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate values
        for i, v in enumerate(vals):
            va = 'bottom' if v >= 0 else 'top'
            ax.text(i, v, f'{v:+.4f}', ha='center', va=va, fontsize=6.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/difficulty_improvement_ieee.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved difficulty_improvement_ieee.pdf')


def save_data(all_data):
    """Save updated analysis data."""
    results = []
    for task_cfg, (target, k0_err, best_err) in zip(TASKS, all_data):
        stats = {
            'model': task_cfg['model'],
            'task': task_cfg['task'],
            'n_samples': len(target),
            'edg_mae': float(k0_err.mean()),
            'edgpp_mae': float(best_err.mean()),
            'edg_std': float(k0_err.std()),
            'edgpp_std': float(best_err.std()),
            'edg_max': float(k0_err.max()),
            'edgpp_max': float(best_err.max()),
        }
        results.append({
            'stats': stats,
            'edg_error': k0_err,
            'edgpp_error': best_err,
            'target': target,
        })
        print(f"{task_cfg['display']:>20s}: MAE {stats['edg_mae']:.6f} -> {stats['edgpp_mae']:.6f} "
              f"({(1-stats['edgpp_mae']/stats['edg_mae'])*100:+.2f}%), "
              f"Std {(1-stats['edgpp_std']/stats['edg_std'])*100:+.1f}%, "
              f"Max {(1-stats['edgpp_max']/stats['edg_max'])*100:+.1f}%")

    np.savez(f'{DATA_DIR}/error_distribution_data.npz',
             results=results, allow_pickle=True)
    print(f'\nSaved error_distribution_data.npz')


def main():
    print('Loading data...')
    all_data = [load_errors(t) for t in TASKS]

    print('\n=== Statistics (κ=0 vs tuned κ) ===')
    save_data(all_data)

    print('\nGenerating figures...')
    plot_tail_error(all_data)
    plot_difficulty_stratification(all_data)
    print('\nDone.')


if __name__ == '__main__':
    main()
