#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Confidence-MAE Analysis Script
No complex dependencies - directly loads data and computes analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Add paths
EDG_ROOT = '/home/ubuntu/wsy/GEOM3D'
sys.path.insert(0, EDG_ROOT)


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

    state_dict = {}
    for k, v in checkpoint['EDEvaluator'].items():
        new_key = k.replace('network.linear1', 'network.0').replace('network.linear2', 'network.2')
        state_dict[new_key] = v

    evaluator.load_state_dict(state_dict)
    evaluator = evaluator.to(device)
    evaluator.eval()
    return evaluator


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    dataroot = f'{EDG_ROOT}/examples_3D/dataset'
    img_feat_path = f'{dataroot}/QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz'
    evaluator_path = f'{EDG_ROOT}/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth'
    output_dir = f'{EDG_ROOT}/结果与分析/confidence_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Load EDEvaluator
    print("Loading EDEvaluator...")
    evaluator = load_evaluator(evaluator_path, device)

    # Load img_feat
    print("Loading img_feat...")
    img_feat_data = np.load(img_feat_path, allow_pickle=True)
    all_img_feats = torch.tensor(img_feat_data['feats'], dtype=torch.float32)
    print(f"img_feat shape: {all_img_feats.shape}")

    # Load QM9 processed data
    print("Loading QM9 dataset...")
    qm9_processed_path = f'{dataroot}/QM9/processed/geometric_data_processed.pt'
    qm9_data = torch.load(qm9_processed_path, weights_only=False)
    data_list = qm9_data[0]  # List of Data objects
    slices = qm9_data[1]
    print(f"Total molecules: {len(data_list) if isinstance(data_list, list) else 'using slices'}")

    # Get target field index for alpha
    target_field = ["mu", "alpha", "homo", "lumo", "gap", "r2",
                    "zpve", "u0", "u298", "h298", "g298", "cv"]
    task = 'alpha'
    task_id = target_field.index(task)
    print(f"Task: {task} (index: {task_id})")

    # Create test split indices (same as qm9_random_customized_01)
    np.random.seed(42)
    n_total = 130831
    perm = np.random.permutation(n_total)
    train_idx = perm[:110000]
    valid_idx = perm[110000:120000]
    test_idx = perm[120000:]
    print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

    # Compute train mean/std for denormalization
    # Load y values from processed data
    y_data = torch.load(f'{dataroot}/QM9/processed/geometric_data_processed.pt', weights_only=False)

    # Get all y values
    if 'y' in slices:
        all_y = data_list.y if hasattr(data_list, 'y') else None

    # Alternative: load from raw CSV
    print("Loading QM9 properties from CSV...")
    qm9_csv_path = f'{dataroot}/QM9/raw/qm9.csv'
    df_qm9 = pd.read_csv(qm9_csv_path)
    all_y_alpha = df_qm9['alpha'].values

    train_y = all_y_alpha[train_idx]
    TRAIN_mean = train_y.mean()
    TRAIN_std = train_y.std()
    print(f"Train stats for {task}: mean={TRAIN_mean:.6f}, std={TRAIN_std:.6f}")

    # Configs to analyze
    configs = [
        {
            'name': 'SchNet_alpha',
            'model_path': f'{EDG_ROOT}/experiments/run_QM9_distillation/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb1.5_asa0_bb0.5/rs42/alpha/model.pth',
            'model_3d': 'SchNet',
        },
        {
            'name': 'SphereNet_alpha',
            'model_path': f'{EDG_ROOT}/experiments/run_QM9_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb-1.5_asa0_bb0.5/rs42/alpha/model.pth',
            'model_3d': 'SphereNet',
        }
    ]

    # Compute confidence for all test samples using EDEvaluator
    print("\nComputing confidence scores for test set...")
    test_img_feats = all_img_feats[test_idx].to(device)

    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 1024
        all_confidence = []
        for i in range(0, len(test_img_feats), batch_size):
            batch_feats = test_img_feats[i:i+batch_size]
            conf = -1 * evaluator(batch_feats).squeeze()
            all_confidence.append(conf.cpu())
        all_confidence = torch.cat(all_confidence).numpy()

    print(f"Confidence range: [{all_confidence.min():.4f}, {all_confidence.max():.4f}]")

    # Get ground truth for test set
    test_y_true = all_y_alpha[test_idx]

    # For each model config, load predictions and compute analysis
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {config['name']}")
        print(f"{'='*60}")

        # Load model and make predictions
        print(f"Loading model: {config['model_path']}")
        checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)

        # Import model
        if config['model_3d'] == 'SchNet':
            from Geom3D.models import SchNet
            model = SchNet(
                hidden_channels=128, num_filters=128, num_interactions=6,
                num_gaussians=51, cutoff=10, readout='mean', node_class=119
            )
        else:
            from Geom3D.models import SphereNet
            model = SphereNet(
                hidden_channels=128, out_channels=1, cutoff=5.0,
                num_layers=4, int_emb_size=64, basis_emb_size_dist=8,
                basis_emb_size_angle=8, basis_emb_size_torsion=8,
                out_emb_channels=256, num_spherical=3, num_radial=6
            )

        model.load_state_dict(checkpoint['model'])
        graph_pred_linear = nn.Linear(128, 1)
        graph_pred_linear.load_state_dict(checkpoint['graph_pred_linear'])

        model = model.to(device)
        graph_pred_linear = graph_pred_linear.to(device)
        model.eval()
        graph_pred_linear.eval()

        # Load test data and make predictions
        # We need to load the actual geometric data for predictions
        print("Loading geometric data for predictions...")

        # Use the dataset class properly
        sys.path.insert(0, f'{EDG_ROOT}/examples_3D')
        from Geom3D.datasets.dataset_QM9_distillation import MoleculeDatasetQM9

        dataset = MoleculeDatasetQM9(
            root=f'{dataroot}/QM9',
            dataset='QM9',
            task=task,
            img_feat_path=img_feat_path
        )

        # Get test subset
        test_dataset = dataset[torch.tensor(test_idx)]
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Make predictions
        print("Running inference...")
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                if config['model_3d'] == 'SchNet':
                    repr = model(batch.x, batch.positions, batch.batch)
                else:
                    repr = model(batch.x, batch.positions, batch.batch)
                pred = graph_pred_linear(repr).squeeze()
                pred = pred * TRAIN_std + TRAIN_mean
                all_preds.append(pred.cpu())

        all_preds = torch.cat(all_preds).numpy()

        # Compute errors
        abs_errors = np.abs(all_preds - test_y_true)

        # Create results DataFrame
        df = pd.DataFrame({
            'y_true': test_y_true,
            'y_pred': all_preds,
            'abs_error': abs_errors,
            'confidence': all_confidence
        })

        # Save per-sample results
        csv_path = os.path.join(output_dir, f'per_sample_{config["name"]}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(df['confidence'], df['abs_error'])
        spearman_r, spearman_p = stats.spearmanr(df['confidence'], df['abs_error'])

        print(f"\nCorrelations:")
        print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})")
        print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

        # Bucket analysis
        n_buckets = 5
        df_sorted = df.sort_values('confidence').reset_index(drop=True)
        bucket_size = len(df_sorted) // n_buckets

        bucket_stats = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = len(df_sorted) if i == n_buckets - 1 else (i + 1) * bucket_size
            bucket_df = df_sorted.iloc[start:end]
            bucket_stats.append({
                'bucket': i + 1,
                'conf_min': bucket_df['confidence'].min(),
                'conf_max': bucket_df['confidence'].max(),
                'n': len(bucket_df),
                'mae_mean': bucket_df['abs_error'].mean(),
                'mae_std': bucket_df['abs_error'].std()
            })

        df_stats = pd.DataFrame(bucket_stats)
        stats_path = os.path.join(output_dir, f'bucket_stats_{config["name"]}.csv')
        df_stats.to_csv(stats_path, index=False)
        print(f"\nBucket stats:\n{df_stats.to_string(index=False)}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df_stats['bucket'], df_stats['mae_mean'],
                      yerr=df_stats['mae_std'], capsize=5,
                      color='#2E86AB', edgecolor='#1a5276', alpha=0.8)

        ax.set_xlabel('Confidence Bucket (Low → High)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title(f'{config["name"]}: Confidence-MAE Analysis\n'
                     f'Pearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}',
                     fontsize=14, fontweight='bold')

        for bar, row in zip(bars, df_stats.itertuples()):
            ax.annotate(f'{row.mae_mean:.4f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'confidence_mae_{config["name"]}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")

        # Summary
        summary_path = os.path.join(output_dir, f'summary_{config["name"]}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Confidence-MAE Analysis: {config['name']}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Test samples: {len(df)}\n")
            f.write(f"Overall MAE: {df['abs_error'].mean():.6f}\n\n")
            f.write(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})\n")
            f.write(f"Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})\n")
        print(f"Saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results in: {output_dir}")


if __name__ == '__main__':
    main()
