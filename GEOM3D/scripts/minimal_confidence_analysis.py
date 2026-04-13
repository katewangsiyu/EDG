#!/usr/bin/env python
"""
Minimal Confidence-MAE Analysis - No Geom3D dependencies
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
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

EDG_ROOT = '/home/ubuntu/wsy/GEOM3D'

class EDEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.network(x)

# Minimal SchNet implementation
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return nn.functional.softplus(x) - self.shift

class SchNetSimple(nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6,
                 num_gaussians=51, cutoff=10.0, node_class=119):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embedding = nn.Embedding(node_class, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            )

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, z, pos, batch):
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # Mean pooling over nodes to get graph-level representation
        out = scatter(h, batch, dim=0, reduce='mean')
        return out  # Returns [batch_size, hidden_channels]

class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class CFConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, nn_module, cutoff):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = nn_module  # Changed from mlp to nn to match checkpoint
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * np.pi / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        row, col = edge_index
        out = scatter(x[col] * W, row, dim=0, dim_size=x.size(0), reduce='add')
        return self.lin2(out)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths
    dataroot = f'{EDG_ROOT}/examples_3D/dataset'
    img_feat_path = f'{dataroot}/QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz'
    evaluator_path = f'{EDG_ROOT}/examples_3D/pretrained_IEMv2_models/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth'
    output_dir = f'{EDG_ROOT}/结果与分析/confidence_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Load EDEvaluator
    print("Loading EDEvaluator...")
    evaluator = EDEvaluator()
    ckpt = torch.load(evaluator_path, map_location=device, weights_only=False)
    state_dict = {k.replace('network.linear1', 'network.0').replace('network.linear2', 'network.2'): v
                  for k, v in ckpt['EDEvaluator'].items()}
    evaluator.load_state_dict(state_dict)
    evaluator = evaluator.to(device).eval()

    # Load img_feat
    print("Loading img_feat...")
    img_data = np.load(img_feat_path, allow_pickle=True)
    all_img_feats = torch.tensor(img_data['feats'], dtype=torch.float32)
    print(f"img_feat shape: {all_img_feats.shape}")

    # Load QM9 data
    print("Loading QM9 processed data...")
    qm9_path = f'{dataroot}/QM9/processed/geometric_data_processed.pt'
    data, slices = torch.load(qm9_path, weights_only=False)

    # Load CSV for labels
    csv_path = f'{dataroot}/QM9/raw/qm9.csv'
    df = pd.read_csv(csv_path)
    all_alpha = df['alpha'].values
    print(f"Total samples in CSV: {len(all_alpha)}")

    # Use img_feat size as the reference (130831 samples)
    n_total = all_img_feats.shape[0]
    print(f"Total samples in img_feat: {n_total}")

    # Split indices - use n_total (130831) not CSV length (133885)
    np.random.seed(42)
    perm = np.random.permutation(n_total)
    train_idx, test_idx = perm[:110000], perm[120000:]
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Train stats
    MEAN = all_alpha[train_idx].mean()
    STD = all_alpha[train_idx].std()
    print(f"Train mean={MEAN:.4f}, std={STD:.4f}")

    # Compute confidence for test set
    print("Computing confidence...")
    test_feats = all_img_feats[test_idx].to(device)
    with torch.no_grad():
        conf_list = []
        for i in range(0, len(test_feats), 1024):
            c = -evaluator(test_feats[i:i+1024]).squeeze()
            conf_list.append(c.cpu())
        confidence = torch.cat(conf_list).numpy()
    print(f"Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

    # Ground truth
    y_true = all_alpha[test_idx]

    # Load SchNet model
    print("\nLoading SchNet model...")
    model_path = f'{EDG_ROOT}/experiments/run_QM9_distillation/SchNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std/ED0.5_E@asb1.5_asa0_bb0.5/rs42/alpha/model.pth'
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = SchNetSimple(hidden_channels=128, num_filters=128, num_interactions=6,
                         num_gaussians=51, cutoff=10.0, node_class=119)

    # Load weights (strict=False to ignore atomic_mass buffer)
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device).eval()

    # Load graph_pred_linear (128 -> 1)
    graph_pred_linear = nn.Linear(128, 1)
    graph_pred_linear.load_state_dict(ckpt['graph_pred_linear'])
    graph_pred_linear = graph_pred_linear.to(device).eval()

    # Build test dataset from processed data
    print("Building test dataset...")

    def get_data(idx):
        # Extract single data point from slices
        d = Data()
        for key in slices.keys():
            if key == 'y':
                continue
            s = slices[key]
            start, end = s[idx].item(), s[idx + 1].item()
            d[key] = data[key][start:end]
        d.y = torch.tensor([all_alpha[idx]], dtype=torch.float32)
        d.img_feat = all_img_feats[idx]
        return d

    # Make predictions
    print("Running inference...")
    preds = []
    batch_size = 64

    for batch_start in range(0, len(test_idx), batch_size):
        batch_end = min(batch_start + batch_size, len(test_idx))
        batch_indices = test_idx[batch_start:batch_end]

        batch_list = [get_data(i) for i in batch_indices]
        batch = Batch.from_data_list(batch_list)
        batch = batch.to(device)

        with torch.no_grad():
            repr = model(batch.x.squeeze(-1), batch.positions, batch.batch)
            out = graph_pred_linear(repr).squeeze()
            pred = out * STD + MEAN
            preds.extend(pred.cpu().numpy().tolist())

        if batch_start % 1000 == 0:
            print(f"  Processed {batch_start}/{len(test_idx)}")

    preds = np.array(preds)
    abs_error = np.abs(preds - y_true)

    print(f"\nMAE: {abs_error.mean():.6f}")

    # Create DataFrame
    results = pd.DataFrame({
        'y_true': y_true,
        'y_pred': preds,
        'abs_error': abs_error,
        'confidence': confidence
    })

    # Save results
    results.to_csv(f'{output_dir}/per_sample_SchNet_alpha.csv', index=False)

    # Correlations
    r_p, p_p = stats.pearsonr(confidence, abs_error)
    r_s, p_s = stats.spearmanr(confidence, abs_error)
    print(f"\nPearson r = {r_p:.4f} (p={p_p:.2e})")
    print(f"Spearman ρ = {r_s:.4f} (p={p_s:.2e})")

    # Bucket analysis
    n_buckets = 5
    df_sorted = results.sort_values('confidence')
    bucket_size = len(df_sorted) // n_buckets

    stats_list = []
    for i in range(n_buckets):
        start = i * bucket_size
        end = len(df_sorted) if i == n_buckets - 1 else (i + 1) * bucket_size
        bucket = df_sorted.iloc[start:end]
        stats_list.append({
            'bucket': i + 1,
            'conf_min': bucket['confidence'].min(),
            'conf_max': bucket['confidence'].max(),
            'n': len(bucket),
            'mae_mean': bucket['abs_error'].mean(),
            'mae_std': bucket['abs_error'].std()
        })

    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv(f'{output_dir}/bucket_stats_SchNet_alpha.csv', index=False)
    print(f"\nBucket stats:\n{df_stats}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_stats['bucket'], df_stats['mae_mean'], yerr=df_stats['mae_std'],
                  capsize=5, color='#2E86AB', alpha=0.8)
    ax.set_xlabel('Confidence Bucket (Low → High)', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title(f'SchNet Alpha: Confidence-MAE\nPearson r={r_p:.4f}, Spearman ρ={r_s:.4f}', fontsize=14)

    for bar, row in zip(bars, df_stats.itertuples()):
        ax.annotate(f'{row.mae_mean:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_mae_SchNet_alpha.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()
