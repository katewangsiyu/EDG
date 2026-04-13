#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export Per-Sample Results Script
=================================
Exports per-sample predictions, errors, and confidence scores for analysis.

Usage:
    python scripts/export_per_sample_results.py \
        --model_path /path/to/model.pth \
        --evaluator_path /path/to/EDEvaluator.pth \
        --img_feat_path /path/to/img_feat.npz \
        --dataroot /path/to/data \
        --task alpha \
        --model_3d SchNet \
        --split test \
        --output_path ./per_sample_results.csv

Author: [Your Name]
Date: 2025
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../examples_3D'))

from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool


def parse_args():
    parser = argparse.ArgumentParser(description='Export Per-Sample Results')

    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained student model checkpoint')
    parser.add_argument('--evaluator_path', type=str, required=True,
                        help='Path to EDEvaluator checkpoint')
    parser.add_argument('--img_feat_path', type=str, required=True,
                        help='Path to img_feat npz file')

    # Data config
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--task', type=str, default='alpha',
                        help='QM9 task name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Which split to export')

    # Model config
    parser.add_argument('--model_3d', type=str, default='SchNet',
                        choices=['SchNet', 'SphereNet', 'Equiformer', 'EGNN'],
                        help='Model architecture')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')

    # SchNet specific
    parser.add_argument('--SchNet_num_filters', type=int, default=128)
    parser.add_argument('--SchNet_num_interactions', type=int, default=6)
    parser.add_argument('--SchNet_num_gaussians', type=int, default=51)
    parser.add_argument('--SchNet_cutoff', type=float, default=10)
    parser.add_argument('--SchNet_readout', type=str, default='mean')

    # Output
    parser.add_argument('--output_path', type=str, default='./per_sample_results.csv',
                        help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


class Predictor(nn.Module):
    """EDEvaluator network structure"""
    def __init__(self, in_features=512, out_features=1):
        super(Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Softplus(),
            nn.Linear(in_features // 2, out_features)
        )

    def forward(self, x):
        return self.network(x)


def load_evaluator(evaluator_path, device):
    """Load pre-trained EDEvaluator"""
    evaluator = Predictor(in_features=512, out_features=1)

    checkpoint = torch.load(evaluator_path, map_location=device)
    if 'EDEvaluator' in checkpoint:
        evaluator.load_state_dict(checkpoint['EDEvaluator'])
    elif 'model' in checkpoint:
        evaluator.load_state_dict(checkpoint['model'])
    else:
        evaluator.load_state_dict(checkpoint)

    evaluator = evaluator.to(device)
    evaluator.eval()
    return evaluator


def get_task_id(task):
    """Get task index for QM9"""
    target_field = ["mu", "alpha", "homo", "lumo", "gap", "r2",
                    "zpve", "u0", "u298", "h298", "g298", "cv"]
    return target_field.index(task)


def main():
    args = parse_args()

    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import dataset and model classes
    from Geom3D.datasets.dataset_QM9_distillation import MoleculeDatasetQM9
    from Geom3D.models import SchNet
    from examples_3D.splitters import qm9_random_customized_01

    # Load dataset
    print(f"Loading dataset from: {args.dataroot}")
    dataset = MoleculeDatasetQM9(
        root=os.path.join(args.dataroot, 'QM9'),
        dataset='QM9',
        task=args.task,
        img_feat_path=args.img_feat_path
    )

    # Get node_class
    node_class = 119  # Default for QM9

    # Split dataset
    train_dataset, valid_dataset, test_dataset = qm9_random_customized_01(
        dataset, null_value=0, seed=args.seed
    )

    # Select split
    if args.split == 'train':
        eval_dataset = train_dataset
    elif args.split == 'valid':
        eval_dataset = valid_dataset
    else:
        eval_dataset = test_dataset

    print(f"Evaluating on {args.split} split: {len(eval_dataset)} samples")

    # Create dataloader
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Load model
    print(f"Loading model from: {args.model_path}")
    if args.model_3d == 'SchNet':
        model = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=node_class,
        )
        graph_pred_linear = torch.nn.Linear(args.emb_dim, 1)
    else:
        raise NotImplementedError(f"Model {args.model_3d} not implemented yet")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if 'graph_pred_linear' in checkpoint:
        graph_pred_linear.load_state_dict(checkpoint['graph_pred_linear'])

    model = model.to(device)
    graph_pred_linear = graph_pred_linear.to(device)
    model.eval()
    graph_pred_linear.eval()

    # Load EDEvaluator
    print(f"Loading EDEvaluator from: {args.evaluator_path}")
    evaluator = load_evaluator(args.evaluator_path, device)

    # Get normalization stats (need to compute from training set)
    task_id = get_task_id(args.task)
    train_y = []
    for data in train_dataset:
        train_y.append(data.y[task_id].item())
    train_y = np.array(train_y)
    TRAIN_mean = train_y.mean()
    TRAIN_std = train_y.std()
    print(f"Train stats: mean={TRAIN_mean:.6f}, std={TRAIN_std:.6f}")

    # Inference
    print("Running inference...")
    results = []
    sample_idx = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            batch_size = batch.y.size(0)

            # Forward pass
            if args.model_3d == 'SchNet':
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
                    'sample_idx': sample_idx,
                    'y_true': y_true[i].item(),
                    'y_pred': pred[i].item(),
                    'abs_error': abs_error[i].item(),
                    'confidence': confidence[i].item()
                })
                sample_idx += 1

    # Save to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to: {args.output_path}")
    print(f"Total samples: {len(df)}")
    print(f"\nSummary statistics:")
    print(f"  MAE: {df['abs_error'].mean():.6f} ± {df['abs_error'].std():.6f}")
    print(f"  Confidence range: [{df['confidence'].min():.4f}, {df['confidence'].max():.4f}]")

    # Quick correlation check
    corr = df['confidence'].corr(df['abs_error'])
    print(f"  Pearson correlation (confidence vs error): {corr:.4f}")


if __name__ == '__main__':
    main()
