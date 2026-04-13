#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Confidence-MAE Analysis Script
Analyzes correlation between EDEvaluator confidence and prediction errors.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../examples_3D'))

from torch.utils.data import DataLoader


class EDEvaluator(nn.Module):
    """EDEvaluator network (512 -> 256 -> 1)"""
    def __init__(self):
        super(EDEvaluator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)


def load_evaluator(evaluator_path, device):
    """Load pre-trained EDEvaluator"""
    evaluator = EDEvaluator()
    checkpoint = torch.load(evaluator_path, map_location=device, weights_only=False)

    # Map keys from checkpoint format to our model format
    state_dict = {}
    for k, v in checkpoint['EDEvaluator'].items():
        new_key = k.replace('network.linear1', 'network.0').replace('network.linear2', 'network.2')
        state_dict[new_key] = v

    evaluator.load_state_dict(state_dict)
    evaluator = evaluator.to(device)
    evaluator.eval()
    return evaluator


def main():
    # Configuration
    EDG_ROOT = '/home/ubuntu/wsy/GEOM3D'
    dataroot = f'{EDG_ROOT}/examples_3D/dataset'
    img_feat_path = f'{dataroot}/QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz'
    evaluator_path = f'{EDG_ROOT}/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth'

    # Model configs to analyze
    configs = [
        {
            'name': 'SchNet_alpha',
            'model_path': f'{EDG_ROOT}/experiments/run_QM9_distillation/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb1.5_asa0_bb0.5/rs42/alpha/model.pth',
            'model_3d': 'SchNet',
            'task': 'alpha'
        },
        {
            'name': 'SphereNet_alpha',
            'model_path': f'{EDG_ROOT}/experiments/run_QM9_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb-1.5_asa0_bb0.5/rs42/alpha/model.pth',
            'model_3d': 'SphereNet',
            'task': 'alpha'
        }
    ]

    output_dir = f'{EDG_ROOT}/结果与分析/confidence_analysis'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load EDEvaluator
    print(f"Loading EDEvaluator from: {evaluator_path}")
    evaluator = load_evaluator(evaluator_path, device)

    # Load img_feat
    print(f"Loading img_feat from: {img_feat_path}")
    img_feat_data = np.load(img_feat_path, allow_pickle=True)
    all_img_feats = img_feat_data['feats']
    print(f"img_feat shape: {all_img_feats.shape}")

    # Import dataset directly (avoid __init__.py dependency issues)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dataset_QM9_distillation",
        f"{EDG_ROOT}/Geom3D/datasets/dataset_QM9_distillation.py"
    )
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    MoleculeDatasetQM9 = dataset_module.MoleculeDatasetQM9

    spec2 = importlib.util.spec_from_file_location(
        "splitters",
        f"{EDG_ROOT}/examples_3D/splitters.py"
    )
    splitters_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(splitters_module)
    qm9_random_customized_01 = splitters_module.qm9_random_customized_01

    # Process each config
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {config['name']}")
        print(f"{'='*60}")

        task = config['task']

        # Load dataset
        dataset = MoleculeDatasetQM9(
            root=os.path.join(dataroot, 'QM9'),
            dataset='QM9',
            task=task,
            img_feat_path=img_feat_path
        )

        # Split dataset
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_01(
            dataset, null_value=0, seed=42
        )

        print(f"Test set size: {len(test_dataset)}")

        # Get task index
        target_field = ["mu", "alpha", "homo", "lumo", "gap", "r2",
                        "zpve", "u0", "u298", "h298", "g298", "cv"]
        task_id = target_field.index(task)

        # Compute train stats for denormalization
        train_y = [data.y[task_id].item() for data in train_dataset]
        TRAIN_mean = np.mean(train_y)
        TRAIN_std = np.std(train_y)
        print(f"Train stats: mean={TRAIN_mean:.6f}, std={TRAIN_std:.6f}")

        # Load model
        print(f"Loading model from: {config['model_path']}")
        checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)

        if config['model_3d'] == 'SchNet':
            from Geom3D.models import SchNet
            model = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=51,
                cutoff=10,
                readout='mean',
                node_class=119
            )
        elif config['model_3d'] == 'SphereNet':
            from Geom3D.models import SphereNet
            model = SphereNet(
                hidden_channels=128,
                out_channels=1,
                cutoff=5.0,
                num_layers=4,
                int_emb_size=64,
                basis_emb_size_dist=8,
                basis_emb_size_angle=8,
                basis_emb_size_torsion=8,
                out_emb_channels=256,
                num_spherical=3,
                num_radial=6,
            )

        model.load_state_dict(checkpoint['model'])
        graph_pred_linear = torch.nn.Linear(128, 1)
        graph_pred_linear.load_state_dict(checkpoint['graph_pred_linear'])

        model = model.to(device)
        graph_pred_linear = graph_pred_linear.to(device)
        model.eval()
        graph_pred_linear.eval()

        # Create dataloader
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        # Inference
        print("Running inference...")
        results = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch_size = batch.y.size(0)

                # Forward pass
                if config['model_3d'] == 'SchNet':
                    molecule_repr = model(batch.x, batch.positions, batch.batch)
                elif config['model_3d'] == 'SphereNet':
                    molecule_repr = model(batch.x, batch.positions, batch.batch)

                pred_normalized = graph_pred_linear(molecule_repr).squeeze()
                pred = pred_normalized * TRAIN_std + TRAIN_mean

                # Get ground truth
                y_true = batch.y.view(batch_size, -1)[:, task_id]

                # Compute confidence
                confidence = -1 * evaluator(batch.img_feat).squeeze()

                # Compute error
                abs_error = torch.abs(pred - y_true)

                # Store results
                for i in range(batch_size):
                    results.append({
                        'y_true': y_true[i].item(),
                        'y_pred': pred[i].item() if pred.dim() > 0 else pred.item(),
                        'abs_error': abs_error[i].item() if abs_error.dim() > 0 else abs_error.item(),
                        'confidence': confidence[i].item() if confidence.dim() > 0 else confidence.item()
                    })

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save per-sample results
        csv_path = os.path.join(output_dir, f'per_sample_results_{config["name"]}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Per-sample results saved to: {csv_path}")

        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(df['confidence'], df['abs_error'])
        spearman_r, spearman_p = stats.spearmanr(df['confidence'], df['abs_error'])

        print(f"\nCorrelation Analysis:")
        print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})")
        print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

        # Bucket analysis
        n_buckets = 5
        df_sorted = df.sort_values('confidence').reset_index(drop=True)
        bucket_size = len(df_sorted) // n_buckets

        bucket_stats = []
        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = len(df_sorted) if i == n_buckets - 1 else (i + 1) * bucket_size
            bucket_df = df_sorted.iloc[start_idx:end_idx]

            bucket_stats.append({
                'bucket_id': i + 1,
                'confidence_min': bucket_df['confidence'].min(),
                'confidence_max': bucket_df['confidence'].max(),
                'confidence_mean': bucket_df['confidence'].mean(),
                'n_samples': len(bucket_df),
                'mae_mean': bucket_df['abs_error'].mean(),
                'mae_std': bucket_df['abs_error'].std()
            })

        df_stats = pd.DataFrame(bucket_stats)

        # Save bucket stats
        stats_path = os.path.join(output_dir, f'bucket_stats_{config["name"]}.csv')
        df_stats.to_csv(stats_path, index=False)
        print(f"\nBucket statistics saved to: {stats_path}")
        print(df_stats.to_string(index=False))

        # Generate plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        x = df_stats['bucket_id']
        y = df_stats['mae_mean']
        yerr = df_stats['mae_std']

        bars = ax.bar(x, y, yerr=yerr, capsize=5, color='#2E86AB',
                      edgecolor='#1a5276', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Confidence Bucket (Low → High)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        ax.set_title(f'Confidence-MAE Correlation: {config["name"]}\n'
                     f'Pearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}',
                     fontsize=14, fontweight='bold')

        # Add value labels
        for bar, row in zip(bars, df_stats.itertuples()):
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.annotate(f'n={row.n_samples}',
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, -15), textcoords="offset points",
                        ha='center', va='top', fontsize=8, color='gray')

        # X-axis labels
        xlabels = [f'B{int(row["bucket_id"])}\n[{row["confidence_min"]:.2f},\n{row["confidence_max"]:.2f}]'
                   for _, row in df_stats.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'confidence_mae_buckets_{config["name"]}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")

        # Save correlation summary
        summary_path = os.path.join(output_dir, f'correlation_summary_{config["name"]}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Confidence-MAE Correlation Analysis: {config['name']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {config['model_3d']}\n")
            f.write(f"Task: {config['task']}\n")
            f.write(f"Test samples: {len(df)}\n\n")
            f.write("Correlation Coefficients:\n")
            f.write(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})\n")
            f.write(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})\n\n")
            f.write("MAE Statistics:\n")
            f.write(f"  Overall MAE: {df['abs_error'].mean():.6f} ± {df['abs_error'].std():.6f}\n\n")
            f.write("Bucket Analysis:\n")
            f.write(df_stats.to_string(index=False))

        print(f"Summary saved to: {summary_path}")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
