#!/usr/bin/env python3
"""
XPipe Publication-Quality Plotting
IEEE-standard figures for conference papers
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_quality_by_domain(df: pd.DataFrame,
                           quality_col: str = 'rougeL_f',
                           domain_col: str = 'dataset') -> plt.Figure:
    """
    Bar chart: Quality by domain with error bars.

    Args:
        df: Results DataFrame
        quality_col: Column name for quality metric
        domain_col: Column name for domain/dataset

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate data
    summary = df.groupby(domain_col)[quality_col].agg(['mean', 'std']).reset_index()

    # Bar plot with error bars
    x_pos = np.arange(len(summary))
    ax.bar(x_pos, summary['mean'], yerr=summary['std'],
           capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary[domain_col], rotation=45, ha='right')
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Quality (ROUGE-L F1)', fontsize=12)
    ax.set_title('Pipeline Quality Across Domains', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig


def plot_pareto_frontier(df: pd.DataFrame,
                        x_col: str = 'latency_ms',
                        y_col: str = 'quality',
                        pareto_col: Optional[str] = 'on_frontier',
                        label_col: Optional[str] = 'config') -> plt.Figure:
    """
    Scatter plot: Quality vs Latency with Pareto frontier.

    Args:
        df: Results DataFrame
        x_col: X-axis metric (e.g., latency)
        y_col: Y-axis metric (e.g., quality)
        pareto_col: Boolean column indicating Pareto-optimal points
        label_col: Column for point labels

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # All points
    ax.scatter(df[x_col], df[y_col],
               alpha=0.5, s=60, color='gray', label='All configurations')

    # Pareto frontier
    if pareto_col and pareto_col in df.columns:
        pareto = df[df[pareto_col] == True].sort_values(x_col)

        # Highlight Pareto points
        ax.scatter(pareto[x_col], pareto[y_col],
                  color='red', s=100, marker='*',
                  edgecolors='darkred', linewidth=1.5,
                  label='Pareto-optimal', zorder=5)

        # Draw frontier line
        ax.plot(pareto[x_col], pareto[y_col],
               'r--', linewidth=2, alpha=0.7, zorder=4)

        # Label top 3 Pareto points
        if label_col:
            top3 = pareto.nlargest(3, y_col)
            for _, row in top3.iterrows():
                ax.annotate(row[label_col],
                           (row[x_col], row[y_col]),
                           textcoords="offset points",
                           xytext=(10,5), fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow', alpha=0.7))

    ax.set_xlabel(f'{x_col.replace("_", " ").title()}', fontsize=12)
    ax.set_ylabel(f'{y_col.replace("_", " ").title()}', fontsize=12)
    ax.set_title('Quality-Latency Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_stage_importance(df: pd.DataFrame,
                          stage_col: str = 'stage',
                          value_col: str = 'shapley_value',
                          domain_col: str = 'dataset') -> plt.Figure:
    """
    Grouped bar chart: Shapley values by stage and domain.

    Args:
        df: Attribution results DataFrame
        stage_col: Column with stage names
        value_col: Column with importance values
        domain_col: Column with domain names

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot for grouped bars
    pivot = df.pivot_table(index=domain_col, columns=stage_col,
                          values=value_col, aggfunc='mean')

    # Plot
    pivot.plot(kind='bar', ax=ax, width=0.75, edgecolor='black', linewidth=1)

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Stage Importance (Shapley Value)', fontsize=12)
    ax.set_title('Causal Attribution: Stage Contributions', fontsize=14, fontweight='bold')
    ax.legend(title='Pipeline Stage', title_fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(df: pd.DataFrame,
                          true_col: str = 'failure_cause_human',
                          pred_col: str = 'failure_cause_auto',
                          stages: Optional[list] = None) -> plt.Figure:
    """
    Confusion matrix heatmap for error localization.

    Args:
        df: Results with human and automated labels
        true_col: Column with ground truth labels
        pred_col: Column with predicted labels
        stages: List of stage names (for ordering)

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    if stages is None:
        stages = sorted(df[true_col].unique())

    # Compute confusion matrix
    cm = confusion_matrix(df[true_col], df[pred_col], labels=stages)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=stages, yticklabels=stages,
           xlabel='Predicted Stage',
           ylabel='True Stage (Human)',
           title='Error Localization Accuracy')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_human_study_results(df: pd.DataFrame,
                             condition_col: str = 'condition',
                             metric_col: str = 'trust_score') -> plt.Figure:
    """
    Box plot: Human study results comparing conditions.

    Args:
        df: Human study responses
        condition_col: Column with condition labels
        metric_col: Column with measurement (trust, accuracy, time)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Box plot
    df.boxplot(column=metric_col, by=condition_col, ax=ax,
               patch_artist=True, showmeans=True)

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title('')  # Remove automatic title
    plt.suptitle('Human Evaluation: Effect of Explanations',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistical test result
    from scipy.stats import ttest_ind
    baseline = df[df[condition_col] == 'baseline'][metric_col]
    explained = df[df[condition_col] == 'explained'][metric_col]
    t_stat, p_value = ttest_ind(baseline, explained)

    ax.text(0.5, 0.95, f't-test: p = {p_value:.4f}{"*" if p_value < 0.05 else ""}',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_config_heatmap(df: pd.DataFrame,
                       row_var: str = 'retriever',
                       col_var: str = 'synthesizer',
                       value_var: str = 'quality') -> plt.Figure:
    """
    Heatmap: Configuration quality matrix.

    Args:
        df: Results DataFrame
        row_var: Variable for rows
        col_var: Variable for columns
        value_var: Metric to display

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot for heatmap
    pivot = df.pivot_table(index=row_var, columns=col_var,
                          values=value_var, aggfunc='mean')

    # Heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu',
               cbar_kws={'label': value_var.replace('_', ' ').title()},
               linewidths=0.5, ax=ax)

    ax.set_xlabel(col_var.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(row_var.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Pipeline Configuration Quality Matrix',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    print(" XPipe plotting module loaded successfully!")
    print("   Available functions:")
    print("   - plot_quality_by_domain()")
    print("   - plot_pareto_frontier()")
    print("   - plot_stage_importance()")
    print("   - plot_confusion_matrix()")
    print("   - plot_human_study_results()")
    print("   - plot_config_heatmap()")
