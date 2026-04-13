#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Confidence-MAE Bucket Analysis Script (CSV Input Version)
==========================================================
Analyzes correlation between confidence scores and prediction errors.

Usage:
    python scripts/analyze_confidence_mae.py \
        --input_csv ./per_sample_results.csv \
        --output_dir ./analysis_results \
        --task alpha \
        --n_buckets 5

Input CSV format:
    sample_idx, y_true, y_pred, abs_error, confidence

Output:
    - confidence_mae_buckets_{task}.png
    - confidence_mae_stats_{task}.csv
    - correlation_summary_{task}.txt

Author: [Your Name]
Date: 2025
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description='Confidence-MAE Bucket Analysis')

    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to per-sample results CSV')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Output directory')
    parser.add_argument('--task', type=str, default='',
                        help='Task name (for output file naming)')
    parser.add_argument('--n_buckets', type=int, default=5,
                        help='Number of buckets for analysis')

    return parser.parse_args()


def bucket_analysis(df, n_buckets=5):
    """Perform bucket analysis on confidence vs error"""
    # Sort by confidence (ascending, so bucket 1 = lowest confidence)
    df_sorted = df.sort_values('confidence').reset_index(drop=True)
    bucket_size = len(df_sorted) // n_buckets

    results = []
    for i in range(n_buckets):
        start_idx = i * bucket_size
        if i == n_buckets - 1:
            end_idx = len(df_sorted)
        else:
            end_idx = (i + 1) * bucket_size

        bucket_df = df_sorted.iloc[start_idx:end_idx]

        results.append({
            'bucket_id': i + 1,
            'confidence_min': bucket_df['confidence'].min(),
            'confidence_max': bucket_df['confidence'].max(),
            'confidence_mean': bucket_df['confidence'].mean(),
            'n_samples': len(bucket_df),
            'mae_mean': bucket_df['abs_error'].mean(),
            'mae_std': bucket_df['abs_error'].std(),
            'mae_median': bucket_df['abs_error'].median(),
            'mae_min': bucket_df['abs_error'].min(),
            'mae_max': bucket_df['abs_error'].max()
        })

    return pd.DataFrame(results)


def plot_bucket_analysis(df_stats, output_path, task_name=''):
    """Generate bucket analysis plot"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = df_stats['bucket_id']
    y = df_stats['mae_mean']
    yerr = df_stats['mae_std']

    # Bar plot with error bars
    bars = ax.bar(x, y, yerr=yerr, capsize=5, color='#2E86AB',
                  edgecolor='#1a5276', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Confidence Bucket (Low → High)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    title = 'Confidence-MAE Correlation Analysis'
    if task_name:
        title += f' ({task_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, df_stats.itertuples())):
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.annotate(f'n={row.n_samples}',
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, -15), textcoords="offset points",
                    ha='center', va='top', fontsize=8, color='gray')

    # X-axis labels with confidence ranges
    xlabels = []
    for _, row in df_stats.iterrows():
        xlabels.append(f'B{int(row["bucket_id"])}\n[{row["confidence_min"]:.2f},\n{row["confidence_max"]:.2f}]')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)

    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def compute_correlations(df):
    """Compute Pearson and Spearman correlations"""
    pearson_r, pearson_p = stats.pearsonr(df['confidence'], df['abs_error'])
    spearman_r, spearman_p = stats.spearmanr(df['confidence'], df['abs_error'])

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    task_suffix = f'_{args.task}' if args.task else ''

    # Load data
    print(f"Loading data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")

    # Validate columns
    required_cols = ['abs_error', 'confidence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Basic statistics
    print(f"\nData summary:")
    print(f"  MAE: {df['abs_error'].mean():.6f} ± {df['abs_error'].std():.6f}")
    print(f"  Confidence: {df['confidence'].mean():.4f} ± {df['confidence'].std():.4f}")

    # Compute correlations
    print(f"\nComputing correlations...")
    corr = compute_correlations(df)
    print(f"  Pearson r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.2e})")
    print(f"  Spearman ρ = {corr['spearman_r']:.4f} (p = {corr['spearman_p']:.2e})")

    # Bucket analysis
    print(f"\nPerforming bucket analysis with {args.n_buckets} buckets...")
    df_stats = bucket_analysis(df, args.n_buckets)

    # Save statistics
    stats_path = os.path.join(args.output_dir, f'confidence_mae_stats{task_suffix}.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\nBucket statistics saved to: {stats_path}")
    print("\nBucket Statistics:")
    print(df_stats.to_string(index=False))

    # Generate plot
    plot_path = os.path.join(args.output_dir, f'confidence_mae_buckets{task_suffix}.png')
    plot_bucket_analysis(df_stats, plot_path, args.task)

    # Save correlation summary
    summary_path = os.path.join(args.output_dir, f'correlation_summary{task_suffix}.txt')
    with open(summary_path, 'w') as f:
        f.write("Confidence-MAE Correlation Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Task: {args.task if args.task else 'N/A'}\n")
        f.write(f"Number of samples: {len(df)}\n")
        f.write(f"Number of buckets: {args.n_buckets}\n\n")
        f.write("Correlation Coefficients:\n")
        f.write(f"  Pearson r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.2e})\n")
        f.write(f"  Spearman ρ = {corr['spearman_r']:.4f} (p = {corr['spearman_p']:.2e})\n\n")
        f.write("Interpretation:\n")
        if corr['pearson_r'] < -0.3:
            f.write("  Strong negative correlation: Higher confidence → Lower error\n")
            f.write("  EDEvaluator effectively predicts reliability.\n")
        elif corr['pearson_r'] < -0.1:
            f.write("  Weak negative correlation: Some predictive ability.\n")
            f.write("  Consider discussing limitations in SI.\n")
        elif corr['pearson_r'] < 0.1:
            f.write("  No clear correlation.\n")
            f.write("  Use cautious language in SI.\n")
        else:
            f.write("  Unexpected positive correlation.\n")
            f.write("  Investigate potential issues.\n")

    print(f"\nCorrelation summary saved to: {summary_path}")

    # Print interpretation
    print("\n" + "=" * 50)
    print("INTERPRETATION:")
    print("=" * 50)
    if corr['pearson_r'] < -0.3:
        print("✓ Strong negative correlation detected.")
        print("  EDEvaluator effectively predicts reliability.")
    elif corr['pearson_r'] < -0.1:
        print("△ Weak negative correlation detected.")
        print("  EDEvaluator shows some predictive ability.")
    else:
        print("✗ No clear negative correlation.")
        print("  Consider using cautious language in SI.")


if __name__ == '__main__':
    main()
