# xpipe/analysis/statistical_tests.py
# SPDX-License-Identifier: BSD-3-Clause
"""
Statistical Analysis for XPipeline Experiments

This module provides comprehensive statistical testing for comparing:
- Models (GPT-2, DistilGPT2, Qwen, Phi-3)
- Retrievers (simple_overlap, jaccard)
- Domains (Network, Customer Support, ITU, TBMP, Energy)
- Pipeline configurations

Features:
- Parametric tests: paired t-test, ANOVA
- Non-parametric tests: Wilcoxon signed-rank, Kruskal-Wallis
- Effect sizes: Cohen's d, eta-squared
- Confidence intervals (95%, 99%)
- Multiple comparison correction (Bonferroni, Holm)
- Automated significance reporting
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# Data Loading
# ============================================================================

def load_experiment_results(
    exp1_path: str = "output/exp1_multidomain/metrics/all_configs.csv",
    exp2_path: str = "output/exp2_attribution/metrics/shapley_values.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load experiment results from CSV files.

    Parameters
    ----------
    exp1_path : str
        Path to Experiment 1 results (multi-domain evaluation)
    exp2_path : str
        Path to Experiment 2 results (Shapley attribution)

    Returns
    -------
    tuple
        (exp1_df, exp2_df) DataFrames
    """
    exp1 = pd.read_csv(exp1_path, comment='#')
    exp2 = pd.read_csv(exp2_path, comment='#')
    return exp1, exp2


# ============================================================================
# Confidence Intervals
# ============================================================================

def calculate_confidence_intervals(
    data: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Calculate confidence intervals for a dataset.

    Parameters
    ----------
    data : array-like
        Numeric data
    confidence : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    dict
        {'mean', 'std', 'ci_lower', 'ci_upper', 'margin'}
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan,
                'ci_upper': np.nan, 'margin': np.nan}

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(len(data))

    # t-distribution critical value
    df = len(data) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_crit * se

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': mean - margin,
        'ci_upper': mean + margin,
        'margin': margin,
        'n': len(data)
    }


# ============================================================================
# Effect Sizes
# ============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Parameters
    ----------
    group1, group2 : array-like
        Two groups to compare

    Returns
    -------
    float
        Cohen's d (|d| < 0.2: small, 0.5: medium, 0.8: large)
    """
    g1 = np.asarray(group1)[~np.isnan(group1)]
    g2 = np.asarray(group2)[~np.isnan(group2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(g1) - np.mean(g2)) / pooled_std


# ============================================================================
# Pairwise Comparisons
# ============================================================================

def pairwise_comparison(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    method: str = 'ttest',
    correction: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons.

    Parameters
    ----------
    df : DataFrame
        Data with groups and metrics
    group_col : str
        Column name for grouping (e.g., 'synth_model')
    metric_col : str
        Column name for metric (e.g., 'relevance')
    method : str
        'ttest' (paired t-test) or 'wilcoxon' (non-parametric)
    correction : str
        'bonferroni', 'holm', or 'none'

    Returns
    -------
    DataFrame
        Pairwise comparison results with p-values and effect sizes
    """
    groups = df[group_col].unique()
    results = []

    for g1, g2 in combinations(groups, 2):
        data1 = df[df[group_col] == g1][metric_col].dropna().values
        data2 = df[df[group_col] == g2][metric_col].dropna().values

        if len(data1) < 2 or len(data2) < 2:
            continue

        # Statistical test
        if method == 'ttest':
            stat, p_value = stats.ttest_ind(data1, data2)
        elif method == 'wilcoxon':
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown method: {method}")

        # Effect size
        effect = cohens_d(data1, data2)

        results.append({
            'group1': g1,
            'group2': g2,
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'diff': np.mean(data1) - np.mean(data2),
            'statistic': stat,
            'p_value': p_value,
            'cohens_d': effect,
            'n1': len(data1),
            'n2': len(data2)
        })

    result_df = pd.DataFrame(results)

    # Multiple comparison correction
    if correction == 'bonferroni' and len(result_df) > 0:
        result_df['p_adjusted'] = result_df['p_value'] * len(result_df)
        result_df['p_adjusted'] = result_df['p_adjusted'].clip(upper=1.0)
    elif correction == 'holm' and len(result_df) > 0:
        sorted_idx = result_df['p_value'].argsort()
        n = len(result_df)
        p_adjusted = []
        for i, idx in enumerate(sorted_idx):
            p_adj = result_df.iloc[idx]['p_value'] * (n - i)
            p_adjusted.append(min(p_adj, 1.0))
        result_df['p_adjusted'] = pd.Series(p_adjusted, index=sorted_idx).sort_index()
    else:
        result_df['p_adjusted'] = result_df['p_value']

    # Significance markers
    result_df['significant'] = result_df['p_adjusted'] < 0.05
    result_df['sig_marker'] = result_df['p_adjusted'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

    return result_df.sort_values('p_adjusted')


# ============================================================================
# Group Comparisons
# ============================================================================

def compare_models(
    df: pd.DataFrame,
    metric: str = 'relevance',
    model_col: str = 'synth_model'
) -> Dict[str, Any]:
    """
    Compare all models statistically.

    Parameters
    ----------
    df : DataFrame
        Experiment 1 results
    metric : str
        Metric to compare (e.g., 'relevance', 'rougeL_f', 'latency_ms')
    model_col : str
        Column containing model names

    Returns
    -------
    dict
        {
            'anova': ANOVA results,
            'pairwise': DataFrame of pairwise comparisons,
            'summary': DataFrame of means and CIs per model
        }
    """
    # ANOVA
    groups = [df[df[model_col] == m][metric].dropna().values
              for m in df[model_col].unique()]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {'error': 'Not enough groups for comparison'}

    f_stat, p_anova = stats.f_oneway(*groups)

    # Pairwise comparisons
    pairwise = pairwise_comparison(df, model_col, metric, method='ttest', correction='bonferroni')

    # Summary statistics
    summary = []
    for model in df[model_col].unique():
        data = df[df[model_col] == model][metric].dropna().values
        ci = calculate_confidence_intervals(data)
        summary.append({
            'model': model,
            **ci
        })

    summary_df = pd.DataFrame(summary).sort_values('mean', ascending=False)

    return {
        'anova': {'f_statistic': f_stat, 'p_value': p_anova},
        'pairwise': pairwise,
        'summary': summary_df
    }


def compare_retrievers(
    df: pd.DataFrame,
    metric: str = 'relevance',
    retriever_col: str = 'retriever'
) -> Dict[str, Any]:
    """
    Compare retrievers (simple_overlap vs jaccard).

    Parameters
    ----------
    df : DataFrame
        Experiment 1 results
    metric : str
        Metric to compare
    retriever_col : str
        Column containing retriever names

    Returns
    -------
    dict
        {'ttest': results, 'effect_size': Cohen's d, 'summary': stats}
    """
    retrievers = df[retriever_col].unique()
    if len(retrievers) != 2:
        return {'error': f'Expected 2 retrievers, found {len(retrievers)}'}

    data1 = df[df[retriever_col] == retrievers[0]][metric].dropna().values
    data2 = df[df[retriever_col] == retrievers[1]][metric].dropna().values

    t_stat, p_value = stats.ttest_ind(data1, data2)
    effect = cohens_d(data1, data2)

    ci1 = calculate_confidence_intervals(data1)
    ci2 = calculate_confidence_intervals(data2)

    return {
        'ttest': {
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'effect_size': effect,
        'summary': {
            retrievers[0]: ci1,
            retrievers[1]: ci2
        }
    }


def compare_domains(
    df: pd.DataFrame,
    metric: str = 'relevance',
    domain_col: str = 'dataset_name'
) -> Dict[str, Any]:
    """
    Compare performance across domains.

    Parameters
    ----------
    df : DataFrame
        Experiment 1 results
    metric : str
        Metric to compare
    domain_col : str
        Column containing domain names

    Returns
    -------
    dict
        {'kruskal': results, 'pairwise': comparisons, 'summary': stats}
    """
    # Kruskal-Wallis (non-parametric ANOVA for domains with different sizes)
    groups = [df[df[domain_col] == d][metric].dropna().values
              for d in df[domain_col].unique()]
    groups = [g for g in groups if len(g) > 0]

    h_stat, p_kruskal = stats.kruskal(*groups)

    # Pairwise comparisons
    pairwise = pairwise_comparison(df, domain_col, metric, method='wilcoxon', correction='holm')

    # Summary statistics
    summary = []
    for domain in df[domain_col].unique():
        data = df[df[domain_col] == domain][metric].dropna().values
        ci = calculate_confidence_intervals(data)
        summary.append({
            'domain': domain,
            **ci
        })

    summary_df = pd.DataFrame(summary).sort_values('mean', ascending=False)

    return {
        'kruskal': {'h_statistic': h_stat, 'p_value': p_kruskal},
        'pairwise': pairwise,
        'summary': summary_df
    }


# ============================================================================
# Automated Reporting
# ============================================================================

def generate_significance_report(
    exp1_path: str = "output/exp1_multidomain/metrics/all_configs.csv",
    output_dir: str = "output/analysis"
) -> None:
    """
    Generate comprehensive significance report for Experiment 1.

    Creates:
    - model_comparison.csv
    - retriever_comparison.csv
    - domain_comparison.csv
    - significance_report.txt

    Parameters
    ----------
    exp1_path : str
        Path to Experiment 1 CSV
    output_dir : str
        Directory to save reports
    """
    # Load data
    df = pd.read_csv(exp1_path, comment='#')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate report
    with open(output_path / "significance_report.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE REPORT\n")
        f.write("XPipeline Experiment 1: Multi-Domain Evaluation\n")
        f.write("=" * 80 + "\n\n")

        # Model comparison
        f.write("1. MODEL COMPARISON (Relevance/Grounding)\n")
        f.write("-" * 80 + "\n")
        model_results = compare_models(df, metric='relevance')

        f.write(f"\nANOVA: F={model_results['anova']['f_statistic']:.3f}, "
                f"p={model_results['anova']['p_value']:.4f}\n\n")

        f.write("Mean ± 95% CI:\n")
        for _, row in model_results['summary'].iterrows():
            f.write(f"  {row['model']:40s} {row['mean']:.3f} "
                   f"± {row['margin']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]\n")

        f.write("\nPairwise Comparisons (Bonferroni corrected):\n")
        for _, row in model_results['pairwise'].head(10).iterrows():
            f.write(f"  {row['group1']} vs {row['group2']}: "
                   f"Δ={row['diff']:.3f}, p={row['p_adjusted']:.4f} {row['sig_marker']}, "
                   f"d={row['cohens_d']:.2f}\n")

        model_results['pairwise'].to_csv(output_path / "model_comparison.csv", index=False)

        # Retriever comparison
        f.write("\n\n2. RETRIEVER COMPARISON\n")
        f.write("-" * 80 + "\n")
        retr_results = compare_retrievers(df, metric='relevance')

        retrievers = list(retr_results['summary'].keys())
        r1_stats = retr_results['summary'][retrievers[0]]
        r2_stats = retr_results['summary'][retrievers[1]]

        f.write(f"\n{retrievers[0]}: {r1_stats['mean']:.3f} ± {r1_stats['margin']:.3f}\n")
        f.write(f"{retrievers[1]}: {r2_stats['mean']:.3f} ± {r2_stats['margin']:.3f}\n")
        f.write(f"\nt-test: t={retr_results['ttest']['statistic']:.3f}, "
               f"p={retr_results['ttest']['p_value']:.4f}\n")
        f.write(f"Effect size (Cohen's d): {retr_results['effect_size']:.3f}\n")
        f.write(f"Significant: {retr_results['ttest']['significant']}\n")

        # Domain comparison
        f.write("\n\n3. DOMAIN COMPARISON\n")
        f.write("-" * 80 + "\n")
        domain_results = compare_domains(df, metric='relevance')

        f.write(f"\nKruskal-Wallis: H={domain_results['kruskal']['h_statistic']:.3f}, "
                f"p={domain_results['kruskal']['p_value']:.4f}\n\n")

        f.write("Mean ± 95% CI per domain:\n")
        for _, row in domain_results['summary'].iterrows():
            f.write(f"  {row['domain']:30s} {row['mean']:.3f} "
                   f"± {row['margin']:.3f} (n={int(row['n'])})\n")

        domain_results['pairwise'].to_csv(output_path / "domain_comparison.csv", index=False)

        # Latency analysis
        f.write("\n\n4. LATENCY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        latency_results = compare_models(df, metric='latency_ms')

        f.write("\nMean latency (ms) per model:\n")
        for _, row in latency_results['summary'].iterrows():
            f.write(f"  {row['model']:40s} {row['mean']:.1f} ms "
                   f"± {row['margin']:.1f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Report saved to: {}\n".format(output_path))
        f.write("=" * 80 + "\n")

    print(f" Significance report generated: {output_path}/significance_report.txt")
    print(f" Model comparison: {output_path}/model_comparison.csv")
    print(f" Domain comparison: {output_path}/domain_comparison.csv")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Generate full significance report
    generate_significance_report()
