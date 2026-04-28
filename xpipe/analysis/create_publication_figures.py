#!/usr/bin/env python3
"""
Generate updated publication figures with new experimental data.
Includes: 5 domains, 3 models (GPT-2, DistilGPT2, Qwen-3B), ablation studies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_palette("Set2")


def load_data():
    """Load experimental results."""
    exp1 = pd.read_csv('output/exp1_multidomain/metrics/all_configs.csv', comment='#')
    exp2 = pd.read_csv('output/exp2_attribution/metrics/shapley_values.csv', comment='#')
    return exp1, exp2


def figure1_domain_performance(exp1, output_dir):
    """Figure 1: Multi-domain performance comparison with 5 domains"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by domain and model
    grouped = exp1.groupby(['dataset_name', 'synth_model'])['relevance'].agg(['mean', 'std', 'count'])
    grouped = grouped.reset_index()

    # Clean model names
    grouped['model_clean'] = grouped['synth_model'].replace({
        'gpt2': 'GPT-2',
        'distilgpt2': 'DistilGPT-2',
        'Qwen/Qwen2.5-3B-Instruct': 'Qwen2.5-3B'
    })

    # Sort domains by mean performance
    domain_order = exp1.groupby('dataset_name')['relevance'].mean().sort_values(ascending=False).index

    # Create grouped bar plot
    x = np.arange(len(domain_order))
    width = 0.25

    models = ['GPT-2', 'DistilGPT-2', 'Qwen2.5-3B']
    colors = ['#66B2FF', '#FFA366', '#90EE90']

    for i, model in enumerate(models):
        model_data = grouped[grouped['model_clean'] == model]
        means = []
        stds = []
        for domain in domain_order:
            domain_data = model_data[model_data['dataset_name'] == domain]
            if len(domain_data) > 0:
                means.append(domain_data['mean'].values[0])
                stds.append(domain_data['std'].values[0])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i*width, means, width, yerr=stds, label=model,
               color=colors[i], alpha=0.8, capsize=3)

    # Formatting
    ax.set_xlabel('Domain', fontweight='bold', fontsize=11)
    ax.set_ylabel('Grounding Score (Relevance)', fontweight='bold', fontsize=11)
    ax.set_title('Multi-Domain Performance Comparison (5 Telecommunications Domains)',
                 fontweight='bold', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('Network Troubleshooting', 'Network')
                         .replace('Customer Support', 'Customer')
                         .replace('ITU Standards', 'ITU')
                         .replace('TBMP Legal', 'Legal')
                         .replace('Energy/Electricity', 'Energy')
                        for d in domain_order], rotation=0)
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    # Add sample size annotation
    ax.text(0.02, 0.98, f'n = 93 queries, 1,116 measurements',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_domain_performance.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig1_domain_performance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f' Generated: {output_dir}/fig1_domain_performance.pdf')


def figure2_model_comparison(exp1, output_dir):
    """Figure 2: Model comparison with significance markers"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Clean model names
    exp1['model_clean'] = exp1['synth_model'].replace({
        'gpt2': 'GPT-2\n(124M, 2019)',
        'distilgpt2': 'DistilGPT-2\n(82M, 2019)',
        'Qwen/Qwen2.5-3B-Instruct': 'Qwen2.5-3B\n(3B, 2024)'
    })

    # Left panel: Quality comparison
    model_quality = exp1.groupby('model_clean')['relevance'].agg(['mean', 'std', 'count'])
    model_quality = model_quality.reindex(['GPT-2\n(124M, 2019)',
                                           'DistilGPT-2\n(82M, 2019)',
                                           'Qwen2.5-3B\n(3B, 2024)'])

    bars1 = ax1.bar(range(len(model_quality)), model_quality['mean'],
                    yerr=model_quality['std'], capsize=5,
                    color=['#66B2FF', '#FFA366', '#90EE90'], alpha=0.8)

    ax1.set_ylabel('Grounding Score (Relevance)', fontweight='bold')
    ax1.set_title('(a) Quality Comparison', fontweight='bold')
    ax1.set_xticks(range(len(model_quality)))
    ax1.set_xticklabels(model_quality.index, fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add significance markers
    ax1.text(0, model_quality.iloc[0]['mean'] + 0.05, 'ns', ha='center', fontsize=10)
    ax1.text(2, model_quality.iloc[2]['mean'] + 0.05, '***', ha='center', fontsize=12, fontweight='bold')

    # Add line showing significant difference
    ax1.plot([0, 2], [0.92, 0.92], 'k-', linewidth=1)
    ax1.text(1, 0.94, 'p < 0.001', ha='center', fontsize=8, style='italic')

    # Right panel: Latency comparison
    model_latency = exp1.groupby('model_clean')['latency_ms'].agg(['mean', 'std'])
    model_latency = model_latency.reindex(['GPT-2\n(124M, 2019)',
                                           'DistilGPT-2\n(82M, 2019)',
                                           'Qwen2.5-3B\n(3B, 2024)'])

    bars2 = ax2.bar(range(len(model_latency)), model_latency['mean'],
                    yerr=model_latency['std'], capsize=5,
                    color=['#66B2FF', '#FFA366', '#90EE90'], alpha=0.8)

    ax2.set_ylabel('Latency (ms)', fontweight='bold')
    ax2.set_title('(b) Latency Comparison', fontweight='bold')
    ax2.set_xticks(range(len(model_latency)))
    ax2.set_xticklabels(model_latency.index, fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add speedup annotations
    baseline = model_latency.iloc[0]['mean']
    for i, (idx, row) in enumerate(model_latency.iterrows()):
        speedup = baseline / row['mean']
        if i > 0:
            ax2.text(i, row['mean'] + 100, f'{speedup:.1f}×',
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_model_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig2_model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f' Generated: {output_dir}/fig2_model_comparison.pdf')


def figure3_ablation_shapley(exp2, output_dir):
    """Figure 3: Ablation study via Shapley values (5 domains)"""

    # Pivot data
    pivot = exp2.pivot_table(
        index=['query_id', 'dataset_name', 'model'],
        columns='stage',
        values='shapley_value'
    ).reset_index()

    # Group by domain
    domain_stats = pivot.groupby('dataset_name')[['retrieve', 'synthesize', 'judge']].agg(['mean', 'std'])

    # Sort by retrieval importance
    domain_order = domain_stats[('retrieve', 'mean')].sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(domain_order))
    width = 0.25

    stages = ['retrieve', 'synthesize', 'judge']
    stage_labels = ['Retrieval', 'Synthesis', 'Judge']
    colors = ['#A8D8EA', '#79C7E3', '#4A9BAA']

    for i, stage in enumerate(stages):
        means = [domain_stats.loc[d, (stage, 'mean')] for d in domain_order]
        stds = [domain_stats.loc[d, (stage, 'std')] for d in domain_order]

        ax.bar(x + i*width, means, width, yerr=stds, label=stage_labels[i],
               color=colors[i], alpha=0.8, capsize=3)

    # Formatting
    ax.set_xlabel('Domain', fontweight='bold', fontsize=11)
    ax.set_ylabel('Shapley Value (Causal Contribution)', fontweight='bold', fontsize=11)
    ax.set_title('Stage Importance via Shapley Value Attribution (Ablation Study)',
                 fontweight='bold', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('Network Troubleshooting', 'Network')
                         .replace('Customer Support', 'Customer')
                         .replace('ITU Standards', 'ITU')
                         .replace('TBMP Legal', 'Legal')
                         .replace('Energy/Electricity', 'Energy')
                        for d in domain_order], rotation=0)
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.6)

    # Add interpretation
    ax.text(0.02, 0.98, 'Higher Shapley value = Greater causal contribution to quality',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_ablation_shapley.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig3_ablation_shapley.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f' Generated: {output_dir}/fig3_ablation_shapley.pdf')


def figure4_configuration_heatmap(exp1, output_dir):
    """Figure 4: Configuration heatmap across domains"""

    # Create pivot table: configs vs domains
    pivot = exp1.pivot_table(
        index='config',
        columns='dataset_name',
        values='relevance',
        aggfunc='mean'
    )

    # Sort columns by mean performance
    col_order = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[col_order]

    # Sort rows by mean performance
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # Rename columns
    pivot.columns = [c.replace('Network Troubleshooting', 'Network')
                      .replace('Customer Support', 'Customer')
                      .replace('ITU Standards', 'ITU')
                      .replace('TBMP Legal', 'Legal')
                      .replace('Energy/Electricity', 'Energy')
                     for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Grounding Score'}, ax=ax,
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Domain', fontweight='bold', fontsize=11)
    ax.set_ylabel('Configuration', fontweight='bold', fontsize=11)
    ax.set_title('Configuration Performance Heatmap Across Domains',
                 fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_config_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig4_config_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f' Generated: {output_dir}/fig4_config_heatmap.pdf')


def figure5_pareto_frontier(exp1, output_dir):
    """Figure 5: Quality-latency Pareto frontier"""

    # Aggregate by configuration
    config_stats = exp1.groupby('config').agg({
        'relevance': 'mean',
        'latency_ms': 'mean',
        'synth_model': 'first',
        'retriever': 'first',
        'judge_enabled': 'first'
    }).reset_index()

    # Clean model names
    config_stats['model_clean'] = config_stats['synth_model'].replace({
        'gpt2': 'GPT-2',
        'distilgpt2': 'DistilGPT-2',
        'Qwen/Qwen2.5-3B-Instruct': 'Qwen-3B'
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot by model
    for model in ['GPT-2', 'DistilGPT-2', 'Qwen-3B']:
        model_data = config_stats[config_stats['model_clean'] == model]
        ax.scatter(model_data['latency_ms'], model_data['relevance'],
                  label=model, s=100, alpha=0.7)

    # Identify Pareto frontier
    points = config_stats[['latency_ms', 'relevance']].values
    pareto_mask = np.zeros(len(points), dtype=bool)

    for i, (lat_i, qual_i) in enumerate(points):
        is_pareto = True
        for j, (lat_j, qual_j) in enumerate(points):
            if i != j:
                # Check if j dominates i (better quality AND lower latency)
                if qual_j >= qual_i and lat_j <= lat_i:
                    if qual_j > qual_i or lat_j < lat_i:
                        is_pareto = False
                        break
        pareto_mask[i] = is_pareto

    # Highlight Pareto frontier
    pareto_points = config_stats[pareto_mask].sort_values('latency_ms')
    ax.plot(pareto_points['latency_ms'], pareto_points['relevance'],
            'r--', linewidth=2, label='Pareto Frontier', zorder=5)
    ax.scatter(pareto_points['latency_ms'], pareto_points['relevance'],
              s=200, facecolors='none', edgecolors='red', linewidth=2, zorder=6)

    # Formatting
    ax.set_xlabel('Latency (ms)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Quality (Grounding Score)', fontweight='bold', fontsize=11)
    ax.set_title('Quality-Latency Trade-off and Pareto Frontier',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add annotations for key points
    best_quality = config_stats.loc[config_stats['relevance'].idxmax()]
    best_latency = config_stats.loc[config_stats['latency_ms'].idxmin()]

    ax.annotate(f'Best Quality\n{best_quality["config"]}',
                xy=(best_quality['latency_ms'], best_quality['relevance']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=8, ha='left')

    ax.annotate(f'Lowest Latency\n{best_latency["config"]}',
                xy=(best_latency['latency_ms'], best_latency['relevance']),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=8, ha='left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_pareto_frontier.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig5_pareto_frontier.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f' Generated: {output_dir}/fig5_pareto_frontier.pdf')


def figure6_judge_impact(exp1, output_dir):
    """Figure 6: Judge component impact analysis"""

    # Compare with/without judge
    judge_impact = exp1.groupby(['dataset_name', 'synth_model', 'judge_enabled']).agg({
        'relevance': 'mean',
        'latency_ms': 'mean'
    }).reset_index()

    # Clean names
    judge_impact['model_clean'] = judge_impact['synth_model'].replace({
        'gpt2': 'GPT-2',
        'distilgpt2': 'DistilGPT-2',
        'Qwen/Qwen2.5-3B-Instruct': 'Qwen-3B'
    })

    # Calculate improvement
    improvements = []
    for domain in judge_impact['dataset_name'].unique():
        for model in judge_impact['model_clean'].unique():
            without = judge_impact[(judge_impact['dataset_name'] == domain) &
                                  (judge_impact['model_clean'] == model) &
                                  (judge_impact['judge_enabled'] == False)]
            with_j = judge_impact[(judge_impact['dataset_name'] == domain) &
                                 (judge_impact['model_clean'] == model) &
                                 (judge_impact['judge_enabled'] == True)]

            if len(without) > 0 and len(with_j) > 0:
                qual_improvement = ((with_j['relevance'].values[0] - without['relevance'].values[0]) /
                                   without['relevance'].values[0] * 100)
                lat_overhead = with_j['latency_ms'].values[0] - without['latency_ms'].values[0]

                improvements.append({
                    'domain': domain,
                    'model': model,
                    'quality_improvement_%': qual_improvement,
                    'latency_overhead_ms': lat_overhead
                })

    improvements_df = pd.DataFrame(improvements)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Quality improvement
    pivot_qual = improvements_df.pivot(index='domain', columns='model', values='quality_improvement_%')
    pivot_qual.index = [d.replace('Network Troubleshooting', 'Network')
                         .replace('Customer Support', 'Customer')
                         .replace('ITU Standards', 'ITU')
                         .replace('TBMP Legal', 'Legal')
                         .replace('Energy/Electricity', 'Energy')
                        for d in pivot_qual.index]

    pivot_qual.plot(kind='bar', ax=ax1, width=0.8, color=['#66B2FF', '#FFA366', '#90EE90'])
    ax1.set_xlabel('Domain', fontweight='bold')
    ax1.set_ylabel('Quality Improvement (%)', fontweight='bold')
    ax1.set_title('(a) Judge Impact on Quality', fontweight='bold')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.legend(title='Model', frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Right: Latency overhead
    pivot_lat = improvements_df.pivot(index='domain', columns='model', values='latency_overhead_ms')
    pivot_lat.index = [d.replace('Network Troubleshooting', 'Network')
                        .replace('Customer Support', 'Customer')
                        .replace('ITU Standards', 'ITU')
                        .replace('TBMP Legal', 'Legal')
                        .replace('Energy/Electricity', 'Energy')
                       for d in pivot_lat.index]

    pivot_lat.plot(kind='bar', ax=ax2, width=0.8, color=['#66B2FF', '#FFA366', '#90EE90'])
    ax2.set_xlabel('Domain', fontweight='bold')
    ax2.set_ylabel('Latency Overhead (ms)', fontweight='bold')
    ax2.set_title('(b) Judge Impact on Latency', fontweight='bold')
    ax2.legend(title='Model', frameon=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_judge_impact.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig6_judge_impact.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f' Generated: {output_dir}/fig6_judge_impact.pdf')


def main():
    """Generate all publication figures."""

    print("="*80)
    print("GENERATING UPDATED PUBLICATION FIGURES")
    print("="*80)
    print()

    # Load data
    print("Loading experimental data...")
    exp1, exp2 = load_data()
    print(f"  Exp1: {len(exp1)} measurements")
    print(f"  Exp2: {len(exp2)} Shapley values")
    print()

    # Create output directory
    output_dir = 'paper/2026_ITU_conf/figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("Generating figures...")
    figure1_domain_performance(exp1, output_dir)
    figure2_model_comparison(exp1, output_dir)
    figure3_ablation_shapley(exp2, output_dir)
    figure4_configuration_heatmap(exp1, output_dir)
    figure5_pareto_frontier(exp1, output_dir)
    figure6_judge_impact(exp1, output_dir)

    print()
    print("="*80)
    print(" ALL FIGURES GENERATED")
    print("="*80)
    print(f"Location: {output_dir}/")
    print()
    print("Figures:")
    print("  - fig0_architecture.pdf (already created)")
    print("  - fig1_domain_performance.pdf (5 domains)")
    print("  - fig2_model_comparison.pdf (3 models with significance)")
    print("  - fig3_ablation_shapley.pdf (ablation study)")
    print("  - fig4_config_heatmap.pdf (configuration comparison)")
    print("  - fig5_pareto_frontier.pdf (quality-latency trade-off)")
    print("  - fig6_judge_impact.pdf (judge component analysis)")


if __name__ == '__main__':
    main()
