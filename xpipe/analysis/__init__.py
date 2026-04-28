# xpipe/analysis/__init__.py
"""
Statistical analysis module for XPipeline experiments.

Provides tools for:
- Statistical significance testing (t-tests, ANOVA, Wilcoxon)
- Confidence interval calculation
- Model/retriever/domain comparisons
- Automated significance reporting
"""

from .statistical_tests import (
    compare_models,
    compare_retrievers,
    compare_domains,
    pairwise_comparison,
    generate_significance_report,
    calculate_confidence_intervals,
    load_experiment_results
)

__all__ = [
    'compare_models',
    'compare_retrievers',
    'compare_domains',
    'pairwise_comparison',
    'generate_significance_report',
    'calculate_confidence_intervals',
    'load_experiment_results'
]
