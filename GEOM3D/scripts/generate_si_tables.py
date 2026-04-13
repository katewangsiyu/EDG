#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate SI LaTeX Tables from Experiment Results
=================================================
Converts qm9_all_results.csv to LaTeX tables for Supplementary Information.

Usage:
    python scripts/generate_si_tables.py \
        --input_csv ./结果与分析/qm9_all_results.csv \
        --output_dir ./结果与分析/si_tables \
        --format compact

Output:
    - best_results_table.tex: Best MAE per model-task
    - schnet_full_results.tex: Full hyperparameter search for SchNet
    - spherenet_full_results.tex: Full hyperparameter search for SphereNet
    - equiformer_full_results.tex: Full hyperparameter search for Equiformer

Author: Auto-generated
Date: 2025
"""

import argparse
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SI LaTeX Tables')
    parser.add_argument('--input_csv', type=str,
                        default='./结果与分析/qm9_all_results.csv',
                        help='Path to qm9_all_results.csv')
    parser.add_argument('--output_dir', type=str,
                        default='./结果与分析/si_tables',
                        help='Output directory for LaTeX tables')
    parser.add_argument('--format', type=str, default='compact',
                        choices=['compact', 'full'],
                        help='Table format: compact (best only) or full')
    return parser.parse_args()


def generate_best_results_table(df, output_path):
    """Generate table with best MAE per model-task combination."""

    # Find best MAE for each model-task combination
    best_results = df.groupby(['model', 'task'])['test_MAE'].min().reset_index()
    pivot = best_results.pivot(index='task', columns='model', values='test_MAE')

    # Reorder columns
    col_order = ['Equiformer', 'SchNet', 'SphereNet']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    # Task display names
    task_names = {
        'alpha': r'$\alpha$',
        'cv': 'cv',
        'g298': r'$G_{298}$',
        'gap': 'gap',
        'h298': r'$H_{298}$',
        'homo': 'HOMO',
        'lumo': 'LUMO',
        'mu': r'$\mu$',
        'r2': r'$\langle R^2 \rangle$',
        'u0': r'$U_0$',
        'u298': r'$U_{298}$',
        'zpve': 'ZPVE'
    }

    with open(output_path, 'w') as f:
        f.write("% Auto-generated best results table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Best test MAE for each model-task combination on QM9.}\n")
        f.write("\\label{tab:best_configs_full}\n")
        f.write("\\begin{tabular}{l" + "c" * len(pivot.columns) + "}\n")
        f.write("\\toprule\n")
        f.write("Task & " + " & ".join(pivot.columns) + " \\\\\n")
        f.write("\\midrule\n")

        for task in sorted(pivot.index):
            display_name = task_names.get(task, task)
            values = []
            for col in pivot.columns:
                if task in pivot.index and col in pivot.columns:
                    val = pivot.loc[task, col]
                    if pd.notna(val):
                        values.append(f"{val:.4f}")
                    else:
                        values.append("--")
                else:
                    values.append("--")
            f.write(f"{display_name} & " + " & ".join(values) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Best results table saved to: {output_path}")


def generate_model_full_table(df, model_name, output_path, max_rows=50):
    """Generate full hyperparameter search table for a specific model."""

    model_df = df[df['model'] == model_name].copy()
    if len(model_df) == 0:
        print(f"No data for model: {model_name}")
        return

    # Sort by task and MAE
    model_df = model_df.sort_values(['task', 'test_MAE'])

    # Get top N results per task
    top_per_task = model_df.groupby('task').head(5)

    with open(output_path, 'w') as f:
        f.write(f"% Auto-generated full results table for {model_name}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(f"\\caption{{Top 5 hyperparameter configurations for {model_name} on QM9.}}\n")
        f.write(f"\\label{{tab:{model_name.lower()}_full}}\n")
        f.write("\\begin{tabular}{lccccr}\n")
        f.write("\\toprule\n")
        f.write("Task & $\\lambda_{\\text{ED}}$ & $\\alpha_{\\text{batch}}$ & ")
        f.write("$\\alpha_{\\text{all}}$ & $\\beta$ & MAE \\\\\n")
        f.write("\\midrule\n")

        current_task = None
        for _, row in top_per_task.iterrows():
            task = row['task']
            if task != current_task:
                if current_task is not None:
                    f.write("\\midrule\n")
                current_task = task

            f.write(f"{task} & {row['weight_ED']} & {row['alpha_std_batch']} & ")
            f.write(f"{row['alpha_std_all']} & {row['beta_batch']} & ")
            f.write(f"{row['test_MAE']:.4f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"{model_name} full table saved to: {output_path}")


def generate_compact_summary(df, output_path):
    """Generate compact summary showing best config per task."""

    # Find best config for each model-task
    idx = df.groupby(['model', 'task'])['test_MAE'].idxmin()
    best_configs = df.loc[idx]

    with open(output_path, 'w') as f:
        f.write("% Auto-generated compact summary\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\footnotesize\n")
        f.write("\\caption{Best hyperparameter configurations per model-task.}\n")
        f.write("\\label{tab:best_configs_detail}\n")
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Task & $\\lambda_{\\text{ED}}$ & ")
        f.write("$\\alpha_{\\text{batch}}$ & $\\alpha_{\\text{all}}$ & ")
        f.write("$\\beta$ & MAE \\\\\n")
        f.write("\\midrule\n")

        for model in ['SchNet', 'SphereNet', 'Equiformer']:
            model_data = best_configs[best_configs['model'] == model]
            model_data = model_data.sort_values('task')

            first_row = True
            for _, row in model_data.iterrows():
                model_col = model if first_row else ""
                f.write(f"{model_col} & {row['task']} & {row['weight_ED']} & ")
                f.write(f"{row['alpha_std_batch']} & {row['alpha_std_all']} & ")
                f.write(f"{row['beta_batch']} & {row['test_MAE']:.4f} \\\\\n")
                first_row = False

            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Compact summary saved to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    # Generate tables
    generate_best_results_table(
        df,
        os.path.join(args.output_dir, 'best_results_table.tex')
    )

    generate_compact_summary(
        df,
        os.path.join(args.output_dir, 'best_configs_detail.tex')
    )

    # Generate per-model tables
    for model in ['SchNet', 'SphereNet', 'Equiformer']:
        generate_model_full_table(
            df, model,
            os.path.join(args.output_dir, f'{model.lower()}_top5_results.tex')
        )

    print(f"\nAll tables generated in: {args.output_dir}")


if __name__ == '__main__':
    main()
