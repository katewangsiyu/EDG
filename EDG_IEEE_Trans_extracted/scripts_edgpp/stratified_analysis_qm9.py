"""
分层分析：按可靠性分数分组，比较EDG vs EDG++
核心假设：低可靠性样本在EDG中表现差，EDG++通过过滤改进
"""
import numpy as np
import pandas as pd
from pathlib import Path

def load_data():
    """加载分析结果"""
    path = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results/qm9_reliability_analysis.npz')
    data = np.load(path, allow_pickle=True)
    return data['results']

def stratified_analysis(result):
    """分层分析单个任务"""
    model = result['model']
    task = result['task']
    reliability = result['reliability']
    edg_error = result['edg_error']
    edgpp_error = result['edgpp_error']

    # 按可靠性分成3组：低(0-33%), 中(33-67%), 高(67-100%)
    q33 = np.percentile(reliability, 33.3)
    q67 = np.percentile(reliability, 66.7)

    low_mask = reliability <= q33
    mid_mask = (reliability > q33) & (reliability <= q67)
    high_mask = reliability > q67

    groups = {
        'Low': low_mask,
        'Mid': mid_mask,
        'High': high_mask
    }

    print(f"\n{'='*60}")
    print(f"{model} - {task}")
    print(f"{'='*60}")
    print(f"{'Group':<8} {'N':<6} {'EDG MAE':<12} {'EDG++ MAE':<12} {'Δ%':<8}")
    print(f"{'-'*60}")

    results = []
    for group_name, mask in groups.items():
        n = mask.sum()
        edg_mae = edg_error[mask].mean()
        edgpp_mae = edgpp_error[mask].mean()
        improvement = (1 - edgpp_mae/edg_mae) * 100

        print(f"{group_name:<8} {n:<6} {edg_mae:<12.6f} {edgpp_mae:<12.6f} {improvement:>6.2f}%")

        results.append({
            'model': model,
            'task': task,
            'group': group_name,
            'n': n,
            'edg_mae': edg_mae,
            'edgpp_mae': edgpp_mae,
            'improvement': improvement
        })

    return results

def main():
    print("=== Stratified Analysis: EDG vs EDG++ ===")

    results_list = load_data()

    all_results = []
    for result in results_list:
        group_results = stratified_analysis(result)
        all_results.extend(group_results)

    # 保存结果
    df = pd.DataFrame(all_results)
    output_path = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results/stratified_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")

    # 汇总统计
    print(f"\n{'='*60}")
    print("Summary: Average improvement by reliability group")
    print(f"{'='*60}")
    summary = df.groupby('group')['improvement'].agg(['mean', 'std', 'count'])
    print(summary)

if __name__ == '__main__':
    main()
