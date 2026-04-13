"""
QM9误差分布与稳定性分析
比较EDG vs EDG++的误差分布，分析极端误差和稳定性
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# 设置绘图样式
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

def load_best_configs():
    """加载最优配置"""
    return pd.read_csv('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results/best_configs.csv')

def format_param(val):
    """格式化参数"""
    if val == int(val):
        return str(int(val))
    return str(val)

def find_experiment_file(model, task, weight_ed, asb, asa, bb):
    """查找实验文件"""
    base = Path('/home/ubuntu/wsy/GEOM3D/experiments/run_QM9_distillation')

    weight_str = f"ED{format_param(weight_ed)}"
    asb_str = f"asb{format_param(asb)}"
    asa_str = f"asa{format_param(asa)}"
    bb_str = f"bb{format_param(bb)}"
    pattern = f"{weight_str}_E@{asb_str}_{asa_str}_{bb_str}"

    for exp_dir in (base / model).rglob('*'):
        if pattern in str(exp_dir):
            npz_file = exp_dir / 'rs42' / task / 'evaluation_best.pth.npz'
            if npz_file.exists():
                return npz_file
    return None

def analyze_error_distribution(model, task, edg_config, edgpp_config):
    """分析误差分布"""
    print(f"\n{'='*70}")
    print(f"{model} - {task}")
    print(f"{'='*70}")

    # 查找文件
    edg_file = find_experiment_file(model, task, *edg_config)
    edgpp_file = find_experiment_file(model, task, *edgpp_config)

    if edg_file is None or edgpp_file is None:
        print("⚠ Files not found")
        return None

    # 加载测试集结果
    edg_data = np.load(edg_file)
    edgpp_data = np.load(edgpp_file)

    target = edg_data['test_target']
    edg_pred = edg_data['test_pred']
    edgpp_pred = edgpp_data['test_pred']

    edg_error = np.abs(target - edg_pred)
    edgpp_error = np.abs(target - edgpp_pred)

    # 统计分析
    stats_dict = {
        'model': model,
        'task': task,
        'n_samples': len(target),
        'edg_mae': edg_error.mean(),
        'edgpp_mae': edgpp_error.mean(),
        'edg_std': edg_error.std(),
        'edgpp_std': edgpp_error.std(),
        'edg_median': np.median(edg_error),
        'edgpp_median': np.median(edgpp_error),
        'edg_q95': np.percentile(edg_error, 95),
        'edgpp_q95': np.percentile(edgpp_error, 95),
        'edg_max': edg_error.max(),
        'edgpp_max': edgpp_error.max()
    }

    print(f"Samples: {len(target)}")
    print(f"\n{'Metric':<15} {'EDG':<12} {'EDG++':<12} {'Δ%':<8}")
    print(f"{'-'*50}")
    print(f"{'MAE':<15} {stats_dict['edg_mae']:<12.6f} {stats_dict['edgpp_mae']:<12.6f} "
          f"{(1-stats_dict['edgpp_mae']/stats_dict['edg_mae'])*100:>6.2f}%")
    print(f"{'Std':<15} {stats_dict['edg_std']:<12.6f} {stats_dict['edgpp_std']:<12.6f} "
          f"{(1-stats_dict['edgpp_std']/stats_dict['edg_std'])*100:>6.2f}%")
    print(f"{'Median':<15} {stats_dict['edg_median']:<12.6f} {stats_dict['edgpp_median']:<12.6f} "
          f"{(1-stats_dict['edgpp_median']/stats_dict['edg_median'])*100:>6.2f}%")
    print(f"{'95th %ile':<15} {stats_dict['edg_q95']:<12.6f} {stats_dict['edgpp_q95']:<12.6f} "
          f"{(1-stats_dict['edgpp_q95']/stats_dict['edg_q95'])*100:>6.2f}%")
    print(f"{'Max':<15} {stats_dict['edg_max']:<12.6f} {stats_dict['edgpp_max']:<12.6f} "
          f"{(1-stats_dict['edgpp_max']/stats_dict['edg_max'])*100:>6.2f}%")

    return {
        'stats': stats_dict,
        'edg_error': edg_error,
        'edgpp_error': edgpp_error,
        'target': target
    }

def main():
    print("=== QM9 Error Distribution Analysis ===")

    best_configs = load_best_configs()

    # 选择提升大的任务
    selected_tasks = ['g298', 'u0', 'h298', 'r2']

    all_results = []
    all_stats = []

    for _, row in best_configs.iterrows():
        if row['task'] in selected_tasks:
            edg_config = (row['edg_weight'], row['edg_asb'], row['edg_asa'], row['edg_bb'])
            edgpp_config = (row['edgpp_weight'], row['edgpp_asb'], row['edgpp_asa'], row['edgpp_bb'])

            result = analyze_error_distribution(row['model'], row['task'], edg_config, edgpp_config)
            if result:
                all_results.append(result)
                all_stats.append(result['stats'])

    # 保存结果
    output_dir = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results')

    # 保存统计数据
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(output_dir / 'error_distribution_stats.csv', index=False)

    # 保存详细数据
    np.savez(output_dir / 'error_distribution_data.npz',
             results=all_results, allow_pickle=True)

    print(f"\n✓ Analyzed {len(all_results)} task-model combinations")
    print(f"✓ Saved to: {output_dir}")

if __name__ == '__main__':
    main()
