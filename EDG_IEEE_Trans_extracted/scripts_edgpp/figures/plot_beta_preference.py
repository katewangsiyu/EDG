"""
Beta preference analysis: single-panel grouped bar chart
Style: consistent with project (CV bar chart, heatmap)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 600, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

OUTPUT = '/home/ubuntu/wsy/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'

# Project standard colors
colors = {
    'SchNet': '#F4B183',
    'SphereNet': '#A9D18E',
    'Equiformer': '#8FAADC',
}

# Data: avg improvement (%) and std for each (model, beta)
data = {
    'SchNet':     {'avg': [2.12, 2.64, 2.31], 'std': [1.48, 1.80, 2.02]},
    'SphereNet':  {'avg': [1.27, 1.75, 0.50], 'std': [0.92, 1.06, 1.14]},
    'Equiformer': {'avg': [2.73, -0.61, 1.14], 'std': [4.09, 4.28, 2.66]},
}

# Win counts per (model, beta)
wins = {
    'SchNet':     [2, 5, 5],
    'SphereNet':  [3, 7, 2],
    'Equiformer': [7, 5, 0],
}

models = ['SchNet', 'SphereNet', 'Equiformer']
beta_labels = [r'$\beta$=0' + '\n(Global)', r'$\beta$=0.5' + '\n(Hybrid)', r'$\beta$=1' + '\n(Local)']

fig, ax = plt.subplots(figsize=(7.16, 2.8))

x = np.arange(3)  # 3 beta values
width = 0.22
offsets = [-width, 0, width]

for i, model in enumerate(models):
    avg = data[model]['avg']
    std = data[model]['std']
    # Use std/sqrt(12) as standard error for error bars
    se = [s / np.sqrt(12) for s in std]

    bars = ax.bar(x + offsets[i], avg, width,
                  yerr=se, capsize=2,
                  label=model, color=colors[model],
                  edgecolor='white', linewidth=0.5,
                  error_kw={'linewidth': 0.8, 'color': '#555555'})

    # Annotate win counts inside bars
    for j, (a, w) in enumerate(zip(avg, wins[model])):
        y_pos = max(a, 0) + se[j] + 0.15
        ax.text(x[j] + offsets[i], y_pos, f'{w}/12',
                ha='center', va='bottom', fontsize=7, color='#333333')

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Avg. improvement over baseline (%)')
ax.set_xticks(x)
ax.set_xticklabels(beta_labels)
ax.set_ylim(-2.5, 5.5)
ax.legend(loc='upper right', frameon=True, edgecolor='#cccccc', fancybox=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a subtle annotation for the key finding
ax.annotate('', xy=(2.22, -0.61 - 0.3), xytext=(2.22, -1.8),
            arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))
ax.text(2.22, -2.1, 'Negative\ntransfer', ha='center', va='top',
        fontsize=6.5, color='#777777', style='italic')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/beta_preference_analysis.pdf')
plt.savefig(f'{OUTPUT}/beta_preference_analysis.png')
plt.close()
print('Saved beta_preference_analysis.pdf')
