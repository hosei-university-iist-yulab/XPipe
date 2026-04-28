#!/usr/bin/env python3
"""
XPipe Theoretical Framework
===========================
Causal attribution, Shapley values, and quality metrics for multi-stage LLM pipelines.

Journal Extension (5 Novel Equations):
1. Shapley-Latency Ratio (SLR): SLR_i = φ_i / latency_i
2. Shapley Interaction Index: I_ij = φ_{ij} - φ_i - φ_j
3. Variance/Convergence Bound: Var(φ̂_i) ≤ R² / (4N)
4. Failure Shapley: φ_i^F = E[Δ_i | v(M) < τ]
5. Blame Score: B_i = φ_i^F / φ_i
"""

from typing import List, Dict, Callable, Tuple, Optional
from collections import Counter
from itertools import permutations, combinations
import numpy as np
import pandas as pd
import re
import time


class AttributionAnalysis:
    """
    Causal attribution for multi-stage pipelines via Shapley values.

    Uses permutation sampling to efficiently estimate stage importance.
    Based on Shapley value theory (cooperative game theory).

    Journal Extension: Includes novel metrics (SLR, Interactions, Failure Shapley, Blame).
    """

    def compute_shapley_values(
        self,
        pipeline_fn: Callable,
        queries: List,
        stages: List[str] = None,
        n_samples: int = 100,
        verbose: bool = True,
        track_latency: bool = True
    ) -> pd.DataFrame:
        """
        Estimate Shapley values for each pipeline stage.

        Uses permutation sampling: For each random ordering of stages,
        compute marginal contribution when each stage is added.
        Average across permutations gives Shapley values.

        Args:
            pipeline_fn: Function (query, stages_list) -> {"quality": float, "latency_ms": float}
            queries: List of queries to evaluate
            stages: List of stage names (default: ["retrieve", "synthesize", "judge"])
            n_samples: Number of permutations to sample
            verbose: Print progress
            track_latency: Track per-stage latency for SLR computation

        Returns:
            DataFrame with columns:
            - query_id: str
            - stage: str
            - shapley_value: float (normalized contribution, sums to 1)
            - quality_impact: float (absolute contribution)
            - latency_ms: float (average stage latency)
            - slr: float (Shapley-Latency Ratio = φ_i / latency_i)
            - variance: float (variance of Shapley estimate)
            - ci_lower, ci_upper: float (95% confidence interval)
        """
        if stages is None:
            stages = ["retrieve", "synthesize", "judge"]

        results = []
        M = len(stages)

        for q_idx, query in enumerate(queries):
            if verbose and (q_idx + 1) % 5 == 0:
                print(f"  Processing query {q_idx + 1}/{len(queries)}...")

            # Full pipeline quality
            result_full = pipeline_fn(query, stages)
            q_full = result_full.get("quality", 0.0)

            # Empty pipeline (baseline)
            q_baseline = 0.0

            # Permutation sampling with latency tracking
            contributions = {s: [] for s in stages}
            latencies = {s: [] for s in stages}

            for _ in range(n_samples):
                # Random permutation of stages
                perm = np.random.permutation(stages).tolist()

                # Compute marginal contributions
                included = []
                prev_quality = q_baseline
                prev_latency = 0.0

                for stage in perm:
                    included.append(stage)

                    # Run pipeline with subset of stages
                    t0 = time.time()
                    result = pipeline_fn(query, included)
                    stage_time = (time.time() - t0) * 1000  # ms

                    curr_quality = result.get("quality", 0.0)
                    curr_latency = result.get("latency_ms", stage_time)

                    # Marginal contribution of this stage
                    marginal = curr_quality - prev_quality
                    contributions[stage].append(marginal)

                    # Latency of this stage (incremental)
                    if track_latency:
                        stage_latency = max(0.1, curr_latency - prev_latency)  # Avoid zero
                        latencies[stage].append(stage_latency)
                        prev_latency = curr_latency

                    prev_quality = curr_quality

            # Compute statistics for each stage
            total_impact = q_full - q_baseline

            for stage in stages:
                contrib_array = np.array(contributions[stage])
                avg_contribution = np.mean(contrib_array)
                variance = np.var(contrib_array, ddof=1) if len(contrib_array) > 1 else 0.0
                std_err = np.sqrt(variance / n_samples) if n_samples > 0 else 0.0

                # 95% CI using t-distribution approximation (z=1.96 for large n)
                ci_lower = avg_contribution - 1.96 * std_err
                ci_upper = avg_contribution + 1.96 * std_err

                # Normalized Shapley value (sums to 1 across stages)
                if total_impact > 0:
                    shapley_norm = avg_contribution / total_impact
                else:
                    shapley_norm = 1.0 / M  # Equal attribution if no quality

                # Average latency for this stage
                avg_latency = np.mean(latencies[stage]) if latencies[stage] else 1.0

                # Shapley-Latency Ratio (Novel Equation 1)
                # SLR = φ_i / latency_i (quality per millisecond)
                slr = shapley_norm / avg_latency if avg_latency > 0 else 0.0

                results.append({
                    'query_id': getattr(query, 'id', str(q_idx)),
                    'stage': stage,
                    'shapley_value': round(shapley_norm, 6),
                    'quality_impact': round(avg_contribution, 6),
                    'latency_ms': round(avg_latency, 2),
                    'slr': round(slr, 8),  # Shapley-Latency Ratio
                    'variance': round(variance, 8),
                    'ci_lower': round(ci_lower, 6),
                    'ci_upper': round(ci_upper, 6),
                    'n_samples': n_samples
                })

        return pd.DataFrame(results)

    def compute_shapley_interactions(
        self,
        pipeline_fn: Callable,
        queries: List,
        stages: List[str] = None,
        n_samples: int = 50,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compute Shapley Interaction Index between stage pairs.

        Novel Equation 2: I_ij = φ_{ij} - φ_i - φ_j

        Measures synergy (I > 0) or redundancy (I < 0) between stages.

        Args:
            pipeline_fn: Function (query, stages_list) -> {"quality": float}
            queries: List of queries to evaluate
            stages: List of stage names
            n_samples: Number of permutations to sample
            verbose: Print progress

        Returns:
            DataFrame with columns:
            - query_id: str
            - stage_i, stage_j: str (stage pair)
            - phi_i, phi_j: float (individual Shapley values)
            - phi_ij: float (joint contribution)
            - interaction_index: float (I_ij = φ_ij - φ_i - φ_j)
            - interaction_type: str ("synergy" | "redundancy" | "independent")
        """
        if stages is None:
            stages = ["retrieve", "synthesize", "judge"]

        results = []

        for q_idx, query in enumerate(queries):
            if verbose and (q_idx + 1) % 5 == 0:
                print(f"  Computing interactions for query {q_idx + 1}/{len(queries)}...")

            # First compute individual Shapley values
            individual_phi = {}
            for stage in stages:
                contribs = []
                for _ in range(n_samples):
                    perm = np.random.permutation(stages).tolist()
                    included = []
                    prev_q = 0.0
                    for s in perm:
                        included.append(s)
                        result = pipeline_fn(query, included)
                        curr_q = result.get("quality", 0.0)
                        if s == stage:
                            contribs.append(curr_q - prev_q)
                        prev_q = curr_q
                individual_phi[stage] = np.mean(contribs)

            # Compute pairwise interactions
            for i, j in combinations(stages, 2):
                # φ_ij: contribution when both i and j are added together
                joint_contribs = []
                for _ in range(n_samples):
                    # Sample permutations where i,j are adjacent
                    others = [s for s in stages if s not in (i, j)]

                    # Quality with neither i nor j
                    if others:
                        result_base = pipeline_fn(query, others)
                        q_base = result_base.get("quality", 0.0)
                    else:
                        q_base = 0.0

                    # Quality with both i and j added
                    result_both = pipeline_fn(query, others + [i, j])
                    q_both = result_both.get("quality", 0.0)

                    joint_contribs.append(q_both - q_base)

                phi_ij = np.mean(joint_contribs)
                phi_i = individual_phi[i]
                phi_j = individual_phi[j]

                # Interaction Index (Novel Equation 2)
                interaction = phi_ij - phi_i - phi_j

                # Classify interaction type
                if interaction > 0.01:
                    interaction_type = "synergy"
                elif interaction < -0.01:
                    interaction_type = "redundancy"
                else:
                    interaction_type = "independent"

                results.append({
                    'query_id': getattr(query, 'id', str(q_idx)),
                    'stage_i': i,
                    'stage_j': j,
                    'phi_i': round(phi_i, 6),
                    'phi_j': round(phi_j, 6),
                    'phi_ij': round(phi_ij, 6),
                    'interaction_index': round(interaction, 6),
                    'interaction_type': interaction_type
                })

        return pd.DataFrame(results)

    def compute_convergence_bound(
        self,
        shapley_df: pd.DataFrame,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Compute variance bounds and required sample size for convergence.

        Novel Equation 3: Var(φ̂_i) ≤ R² / (4N)

        Based on Hoeffding bound: P(|φ̂_i - φ_i| > ε) ≤ 2·exp(-2Nε²/R²)

        Args:
            shapley_df: DataFrame from compute_shapley_values()
            confidence: Desired confidence level (default 0.95)

        Returns:
            DataFrame with convergence statistics per stage:
            - stage: str
            - mean_shapley: float
            - empirical_variance: float
            - theoretical_var_bound: float (R²/4N)
            - current_ci_width: float
            - n_for_epsilon_01: int (samples needed for ε=0.01)
            - converged: bool (CI width < 0.05)
        """
        results = []

        # Group by stage
        for stage in shapley_df['stage'].unique():
            stage_data = shapley_df[shapley_df['stage'] == stage]

            shapley_values = stage_data['shapley_value'].values
            n_samples = stage_data['n_samples'].iloc[0] if 'n_samples' in stage_data else len(shapley_values)

            # Empirical statistics
            mean_phi = np.mean(shapley_values)
            empirical_var = np.var(shapley_values, ddof=1)

            # Quality range R (assume [0, 1] for normalized Shapley)
            R = 1.0

            # Theoretical variance bound: Var(φ̂) ≤ R² / (4N)
            theoretical_var_bound = (R ** 2) / (4 * n_samples)

            # Current CI width
            std_err = np.sqrt(empirical_var / len(shapley_values))
            z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
            ci_width = 2 * z * std_err

            # Samples needed for ε = 0.01 accuracy with given confidence
            # From Hoeffding: N ≥ R² * ln(2/α) / (2ε²)
            alpha = 1 - confidence
            epsilon = 0.01
            n_needed = int(np.ceil((R ** 2) * np.log(2 / alpha) / (2 * epsilon ** 2)))

            results.append({
                'stage': stage,
                'mean_shapley': round(mean_phi, 6),
                'empirical_variance': round(empirical_var, 8),
                'theoretical_var_bound': round(theoretical_var_bound, 8),
                'current_ci_width': round(ci_width, 6),
                'n_samples_used': n_samples,
                'n_for_epsilon_01': n_needed,
                'converged': ci_width < 0.05
            })

        return pd.DataFrame(results)

    def compute_failure_shapley(
        self,
        pipeline_fn: Callable,
        queries: List,
        stages: List[str] = None,
        failure_threshold: float = 0.1,
        n_samples: int = 50,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compute Failure Shapley - stage contributions conditioned on pipeline failure.

        Novel Equation 4: φ_i^F = E[Δ_i | v(M) < τ]

        Identifies which stages contribute most to failures.

        Args:
            pipeline_fn: Function (query, stages_list) -> {"quality": float}
            queries: List of queries to evaluate
            stages: List of stage names
            failure_threshold: Quality below this is considered failure
            n_samples: Number of permutations to sample
            verbose: Print progress

        Returns:
            DataFrame with columns:
            - query_id: str
            - stage: str
            - is_failure: bool (quality < threshold)
            - shapley_normal: float (φ_i from all queries)
            - shapley_failure: float (φ_i^F from failed queries only)
            - blame_score: float (B_i = φ_i^F / φ_i)
        """
        if stages is None:
            stages = ["retrieve", "synthesize", "judge"]

        # First pass: identify failed queries and compute all Shapley values
        all_results = []
        failure_contributions = {s: [] for s in stages}
        normal_contributions = {s: [] for s in stages}

        for q_idx, query in enumerate(queries):
            if verbose and (q_idx + 1) % 5 == 0:
                print(f"  Failure analysis for query {q_idx + 1}/{len(queries)}...")

            # Check if this query fails
            result_full = pipeline_fn(query, stages)
            quality = result_full.get("quality", 0.0)
            is_failure = quality < failure_threshold

            # Compute Shapley values for this query
            contributions = {s: [] for s in stages}

            for _ in range(n_samples):
                perm = np.random.permutation(stages).tolist()
                included = []
                prev_q = 0.0

                for stage in perm:
                    included.append(stage)
                    result = pipeline_fn(query, included)
                    curr_q = result.get("quality", 0.0)
                    marginal = curr_q - prev_q
                    contributions[stage].append(marginal)
                    prev_q = curr_q

            # Store contributions
            for stage in stages:
                avg_contrib = np.mean(contributions[stage])
                normal_contributions[stage].append(avg_contrib)

                if is_failure:
                    failure_contributions[stage].append(avg_contrib)

                all_results.append({
                    'query_id': getattr(query, 'id', str(q_idx)),
                    'stage': stage,
                    'is_failure': is_failure,
                    'quality': quality,
                    'marginal_contribution': round(avg_contrib, 6)
                })

        # Compute aggregated Failure Shapley and Blame Scores
        summary_results = []

        for stage in stages:
            # Normal Shapley (all queries)
            phi_normal = np.mean(normal_contributions[stage]) if normal_contributions[stage] else 0.0

            # Failure Shapley (failed queries only) - Novel Equation 4
            phi_failure = np.mean(failure_contributions[stage]) if failure_contributions[stage] else 0.0

            # Blame Score - Novel Equation 5
            # B_i = φ_i^F / φ_i
            if abs(phi_normal) > 1e-8:
                blame_score = phi_failure / phi_normal
            else:
                blame_score = 1.0 if phi_failure != 0 else 0.0

            summary_results.append({
                'stage': stage,
                'shapley_normal': round(phi_normal, 6),
                'shapley_failure': round(phi_failure, 6),
                'blame_score': round(blame_score, 4),
                'n_failures': len(failure_contributions[stage]),
                'n_total': len(normal_contributions[stage]),
                'failure_rate': round(len(failure_contributions[stage]) / max(1, len(normal_contributions[stage])), 4)
            })

        return pd.DataFrame(all_results), pd.DataFrame(summary_results)

    def localize_error(
        self,
        pipeline_fn: Callable,
        failed_queries: List,
        stages: List[str] = None,
        threshold: float = 0.5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Identify which stage caused pipeline failures.

        For each failed query, run leave-one-out ablations.
        Stage whose removal causes biggest quality drop is likely responsible.

        Args:
            pipeline_fn: Function (query, stages_list) -> {"quality": float}
            failed_queries: List of queries where pipeline failed
            stages: List of stage names
            threshold: Quality threshold for "failure" definition
            verbose: Print progress

        Returns:
            DataFrame with columns:
            - query_id: str
            - full_quality: float (original failed quality)
            - failure_cause: str (stage name most likely responsible)
            - confidence: float (0-1, strength of attribution)
            - impacts: dict (quality change when each stage removed)
        """
        if stages is None:
            stages = ["retrieve", "synthesize", "judge"]

        results = []

        for q_idx, query in enumerate(failed_queries):
            if verbose and (q_idx + 1) % 10 == 0:
                print(f"  Localizing error {q_idx + 1}/{len(failed_queries)}...")

            # Full pipeline (failed)
            result_full = pipeline_fn(query, stages)
            q_full = result_full.get("quality", 0.0)

            # Ablate each stage
            impacts = {}
            for stage in stages:
                # Run without this stage
                remaining = [s for s in stages if s != stage]

                if len(remaining) == 0:
                    q_ablated = 0.0
                else:
                    result = pipeline_fn(query, remaining)
                    q_ablated = result.get("quality", 0.0)

                impacts[stage] = q_ablated - q_full

            # Stage with most positive impact is likely cause
            failure_cause = max(impacts, key=lambda s: impacts[s])

            # Confidence = relative magnitude
            total_abs_impact = sum(abs(v) for v in impacts.values())
            if total_abs_impact > 0:
                confidence = abs(impacts[failure_cause]) / total_abs_impact
            else:
                confidence = 1.0 / len(stages)

            results.append({
                'query_id': getattr(query, 'id', f'q{q_idx}'),
                'full_quality': q_full,
                'failure_cause': failure_cause,
                'confidence': confidence,
                'stage_impacts': str(impacts)
            })

        return pd.DataFrame(results)


class JournalMetrics:
    """
    Novel metrics for XPipe Journal Extension.

    Contains implementations for:
    - Shapley-Latency Ratio (SLR)
    - Interaction Strength (IS)
    - Convergence diagnostics
    """

    @staticmethod
    def compute_slr_summary(shapley_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate Shapley-Latency Ratio statistics across queries.

        SLR_i = φ_i / latency_i (quality contribution per millisecond)

        Args:
            shapley_df: DataFrame from compute_shapley_values()

        Returns:
            DataFrame with SLR statistics per stage
        """
        results = []

        for stage in shapley_df['stage'].unique():
            stage_data = shapley_df[shapley_df['stage'] == stage]

            results.append({
                'stage': stage,
                'mean_shapley': round(stage_data['shapley_value'].mean(), 6),
                'mean_latency_ms': round(stage_data['latency_ms'].mean(), 2),
                'mean_slr': round(stage_data['slr'].mean(), 8),
                'std_slr': round(stage_data['slr'].std(), 8),
                'slr_rank': 0  # Will be filled
            })

        df = pd.DataFrame(results)
        df['slr_rank'] = df['mean_slr'].rank(ascending=False).astype(int)
        return df.sort_values('slr_rank')

    @staticmethod
    def compute_interaction_strength(interaction_df: pd.DataFrame) -> float:
        """
        Compute overall Interaction Strength (IS) for the pipeline.

        IS = Σ|I_ij| / Σφ_i

        High IS = stages are highly coupled, must optimize together
        Low IS = stages are independent, can optimize separately

        Args:
            interaction_df: DataFrame from compute_shapley_interactions()

        Returns:
            Float: Interaction strength value
        """
        total_interaction = interaction_df['interaction_index'].abs().sum()
        total_phi = interaction_df['phi_i'].sum() + interaction_df['phi_j'].sum()

        if total_phi > 0:
            return round(total_interaction / total_phi, 6)
        return 0.0

    @staticmethod
    def compute_blame_ranking(failure_summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stages by blame score for debugging prioritization.

        B_i > 1: Stage is disproportionately responsible for failures
        B_i < 1: Stage performs better on failures than average
        B_i = 1: Stage contributes equally to successes and failures

        Args:
            failure_summary_df: Summary DataFrame from compute_failure_shapley()

        Returns:
            DataFrame sorted by blame score (highest first)
        """
        df = failure_summary_df.copy()
        df['blame_rank'] = df['blame_score'].rank(ascending=False).astype(int)
        df['diagnosis'] = df['blame_score'].apply(
            lambda b: 'BOTTLENECK' if b > 1.2 else ('OK' if b < 0.8 else 'NEUTRAL')
        )
        return df.sort_values('blame_rank')


class QualityMetrics:
    """Standard quality metrics for text generation"""

    @staticmethod
    def grounding_score(answer: str, contexts: List[str]) -> float:
        """Fraction of answer tokens present in retrieved contexts."""
        answer_tokens = set(_tokenize(answer))
        if not answer_tokens:
            return 0.0
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(_tokenize(ctx))
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)

    @staticmethod
    def rouge_l_f1(prediction: str, reference: str) -> float:
        """ROUGE-L F1 score (longest common subsequence)."""
        from xpipe.metrics import rouge_l_fscore
        return rouge_l_fscore(prediction, reference)

    @staticmethod
    def f1_token(prediction: str, reference: str) -> float:
        """Token-level F1 score (bag-of-words)."""
        from xpipe.metrics import f1_token_score
        return f1_token_score(prediction, reference)

    @staticmethod
    def compute_all(prediction: str, reference: str, contexts: List[str]) -> Dict[str, float]:
        """Compute all quality metrics at once."""
        return {
            'grounding': QualityMetrics.grounding_score(prediction, contexts),
            'rouge_l_f1': QualityMetrics.rouge_l_f1(prediction, reference),
            'f1_token': QualityMetrics.f1_token(prediction, reference)
        }


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercasing."""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return text.split()


def compute_pareto_frontier(df: pd.DataFrame,
                           x_col: str = 'latency_ms',
                           y_col: str = 'quality',
                           maximize_y: bool = True) -> pd.DataFrame:
    """Identify Pareto-optimal points (quality vs latency trade-off)."""
    df = df.copy()
    df['on_frontier'] = False
    df = df.sort_values(x_col)
    best_y = -float('inf') if maximize_y else float('inf')

    for idx, row in df.iterrows():
        if maximize_y:
            if row[y_col] > best_y:
                df.at[idx, 'on_frontier'] = True
                best_y = row[y_col]
        else:
            if row[y_col] < best_y:
                df.at[idx, 'on_frontier'] = True
                best_y = row[y_col]

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("XPipe Theoretical Framework - Journal Extension")
    print("=" * 70)
    print("\n Module loaded successfully!")
    print("\nCore Classes:")
    print("  - AttributionAnalysis: Shapley values, interactions, failure analysis")
    print("  - JournalMetrics: SLR, interaction strength, blame ranking")
    print("  - QualityMetrics: grounding, ROUGE-L, F1")
    print("\nNovel Equations (Journal Extension):")
    print("  1. SLR (Shapley-Latency Ratio): φ_i / latency_i")
    print("  2. Interaction Index: I_ij = φ_ij - φ_i - φ_j")
    print("  3. Convergence Bound: Var(φ̂) ≤ R²/(4N)")
    print("  4. Failure Shapley: φ_i^F = E[Δ_i | v(M) < τ]")
    print("  5. Blame Score: B_i = φ_i^F / φ_i")
    print("=" * 70)
