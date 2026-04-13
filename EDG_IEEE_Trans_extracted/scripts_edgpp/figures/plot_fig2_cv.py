"""Figure 2: Model robustness comparison - CV bar chart"""
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
    'legend.fontsize': 8,
    'figure.dpi': 600, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

qm9 = pd.read_csv('/home/ubuntu/wsy/GEOM3D/结果与分析/qm9_all_results.csv')

tasks_order = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'cv',
               'g298', 'h298', 'u298', 'u0', 'zpve', 'r2']
task_display = {
    'alpha': r'$\alpha$', 'gap': r'$\Delta\varepsilon$',
    'homo': 'HOMO', 'lumo': 'LUMO',
    'mu': r'$\mu$', 'cv': r'$C_v$',
    'g298': r'$G$', 'h298': r'$H$',
    'r2': r'$R^2$', 'u298': r'$U$', 'u0': r'$U_0$', 'zpve': 'ZPVE'
}

models = ['SchNet', 'SphereNet', 'Equiformer']
# EDG++ standard colors
colors = ['#F4B183', '#A9D18E', '#8FAADC']  # SchNet, SphereNet, Equiformer

# Compute CV for each model x task
cv_data = {}
for model in models:
    cv_data[model] = []
    for task in tasks_order:
        subset = qm9[(qm9['model'] == model) & (qm9['task'] == task)]
        mae_values = subset['test_MAE'].values
        cv = np.std(mae_values) / np.mean(mae_values) * 100
        cv_data[model].append(cv)

# Plot
fig, ax = plt.subplots(figsize=(7.16, 2.5))
x = np.arange(len(tasks_order))
width = 0.25

for i, model in enumerate(models):
    bars = ax.bar(x + i * width, cv_data[model], width,
                  label=model, color=colors[i], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Coefficient of Variation (%)')
ax.set_xticks(x + width)
ax.set_xticklabels([task_display[t] for t in tasks_order])
ax.legend(loc='upper left', frameon=True, edgecolor='gray')
ax.set_ylim(0, 12)
ax.axhline(y=3, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
# 移除grid，保持白色背景
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate the high-CV cases
for i, model in enumerate(models):
    for j, cv_val in enumerate(cv_data[model]):
        if cv_val > 5:
            ax.annotate(f'{cv_val:.1f}%',
                       xy=(j + i * width, cv_val),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=6, color=colors[i])

plt.tight_layout()
out = '/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'
plt.savefig(f'{out}/model_robustness_cv.png')
plt.savefig(f'{out}/model_robustness_cv.pdf')
print("CV bar chart saved.")

# Print summary
print("\nCV Summary:")
for model in models:
    vals = cv_data[model]
    print(f"  {model}: mean CV={np.mean(vals):.1f}%, "
          f"max CV={np.max(vals):.1f}% ({tasks_order[np.argmax(vals)]})")
