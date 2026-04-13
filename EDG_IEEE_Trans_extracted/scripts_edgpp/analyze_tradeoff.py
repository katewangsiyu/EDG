"""
深入分析：EDG++在不同误差区间的表现
"""
import numpy as np
from pathlib import Path

path = Path('/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/scripts_edgpp/analysis_results/error_distribution_data.npz')
data = np.load(path, allow_pickle=True)
results = data['results']

selected_keys = ['SchNet-r2', 'Equiformer-u0', 'Equiformer-g298', 'SphereNet-h298']

for result in results:
    key = f"{result['stats']['model']}-{result['stats']['task']}"
    if key not in selected_keys:
        continue

    stats = result['stats']
    edg_error = result['edg_error']
    edgpp_error = result['edgpp_error']

    print(f"\n{'='*70}")
    print(f"{key}")
    print(f"{'='*70}")

    # 分析不同区间
    edg_sorted = np.sort(edg_error)
    edgpp_sorted = np.sort(edgpp_error)

    # 前500个（低误差）
    edg_low = edg_sorted[:500]
    edgpp_low = edgpp_sorted[:500]

    # 后500个（高误差）
    edg_high = edg_sorted[-500:]
    edgpp_high = edgpp_sorted[-500:]

    print(f"\n前500个样本（低误差区域）:")
    print(f"  EDG  平均: {edg_low.mean():.6f}")
    print(f"  EDG++ 平均: {edgpp_low.mean():.6f}")
    print(f"  差异: {((edgpp_low.mean() - edg_low.mean()) / edg_low.mean() * 100):+.2f}%")

    print(f"\n后500个样本（高误差区域）:")
    print(f"  EDG  平均: {edg_high.mean():.6f}")
    print(f"  EDG++ 平均: {edgpp_high.mean():.6f}")
    print(f"  差异: {((edgpp_high.mean() - edg_high.mean()) / edg_high.mean() * 100):+.2f}%")

    print(f"\n整体统计:")
    print(f"  Std改进: {(1 - stats['edgpp_std']/stats['edg_std']) * 100:.2f}%")
    print(f"  Max改进: {(1 - stats['edgpp_max']/stats['edg_max']) * 100:.2f}%")
