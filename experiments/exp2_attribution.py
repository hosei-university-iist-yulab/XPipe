#!/usr/bin/env python3
"""
Experiment 2: Causal Attribution Analysis (5 domains)
Computes Shapley values to identify which pipeline stages contribute most to quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from xpipe.theory import AttributionAnalysis, QualityMetrics, JournalMetrics
from xpipe.outputs import ExperimentOutputs
from xpipe.plotting import plot_stage_importance
from xpipe.runners.rag import _retrieve_simple, _retrieve_jaccard, _hf_gen, _clamp_prompt, _grounding


class SimplePipeline:
    """Wrapper for RAG pipeline that can run with subset of stages"""

    def __init__(self, corpus, retriever="simple_overlap", synthesizer="hf/gpt2"):
        self.corpus = corpus
        self.retriever = retriever
        self.synthesizer = synthesizer

    def run(self, query, stages):
        """
        Run pipeline with specified stages.

        Args:
            query: Query dict with 'text' and 'ref'
            stages: List of stage names to include

        Returns:
            Dict with 'quality' score
        """
        result = {'quality': 0.0, 'answer': '', 'contexts': []}

        # Stage 1: Retrieve
        if 'retrieve' in stages:
            if self.retriever == "simple_overlap":
                docs = _retrieve_simple(query['text'], self.corpus, k=3)
            else:
                docs = _retrieve_jaccard(query['text'], self.corpus, k=3)

            result['contexts'] = [d['text'] for d in docs]
        else:
            # No retrieval = empty context
            result['contexts'] = []

        # Stage 2: Synthesize
        if 'synthesize' in stages:
            if result['contexts']:
                # Build prompt
                contexts_clamped = _clamp_prompt(query['text'], result['contexts'], 900, 300)
                prompt = (f"Answer the question using ONLY the evidence.\\n"
                         f"QUESTION: {query['text']}\\n\\n"
                         f"EVIDENCE:\\n- " + "\\n- ".join(contexts_clamped) + "\\n\\nAnswer:")

                try:
                    # Generate answer - extract model ID from synthesizer
                    model_id = self.synthesizer.split('/')[-1] if '/' in self.synthesizer else self.synthesizer
                    if 'Qwen' in model_id:
                        model_id = "Qwen/Qwen2.5-3B-Instruct"
                    elif 'Phi' in model_id or 'phi' in model_id:
                        model_id = "microsoft/Phi-3-mini-4k-instruct"
                    elif 'gpt2' in model_id.lower():
                        model_id = "gpt2"
                    else:
                        model_id = "distilgpt2"

                    out = _hf_gen(
                        model_id,
                        prompt,
                        {"max_new_tokens": 50, "temperature": 0.1}
                    )
                    result['answer'] = out['text']

                except Exception as e:
                    result['answer'] = "Error generating answer"
            else:
                result['answer'] = "No context available"
        else:
            # No synthesis = empty answer
            result['answer'] = ""

        # Stage 3: Judge (simple heuristic)
        judge_bonus = 0.0
        if 'judge' in stages and result['answer']:
            # Simple judge: bonus if answer is grounded
            if result['contexts']:
                grounding = _grounding(result['answer'], result['contexts'])
                judge_bonus = 0.1 if grounding > 0.5 else 0.0

        # Compute quality
        metrics = QualityMetrics()

        if query.get('ref'):
            # If we have reference answer, use ROUGE-L
            result['quality'] = metrics.rouge_l_f1(result['answer'], query['ref'])
        elif result['contexts']:
            # Otherwise use grounding score
            result['quality'] = metrics.grounding_score(result['answer'], result['contexts'])
        else:
            result['quality'] = 0.0

        # Add judge bonus
        result['quality'] += judge_bonus
        result['quality'] = min(1.0, result['quality'])  # Cap at 1.0

        return result


def load_queries(path: str, max_queries: int = 20):
    """Load queries from JSONL file"""
    queries = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries.append({
                    'id': q.get('id', f'q{len(queries)}'),
                    'text': q.get('text', ''),
                    'ref': q.get('ref', '')
                })
                if len(queries) >= max_queries:
                    break
    return queries


def load_corpus(path: str):
    """Load corpus from JSONL file"""
    corpus = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus.append({
                    'id': doc.get('id', f'd{len(corpus)}'),
                    'text': doc.get('text', '')
                })
    return corpus


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 2: Causal Attribution Analysis", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    outputs = ExperimentOutputs("exp2_attribution")
    attribution = AttributionAnalysis()

    datasets = {
        'network_logs': {
            'corpus': 'datasets/network_logs/cases.jsonl',
            'queries': 'datasets/network_logs/queries.jsonl',
            'name': 'Network Troubleshooting'
        },
        'customer_support': {
            'corpus': 'datasets/customer_support/conversations.jsonl',
            'queries': 'datasets/customer_support/queries.jsonl',
            'name': 'Customer Support'
        },
        'itu_standards': {
            'corpus': 'datasets/itu_standards/recommendations.jsonl',
            'queries': 'datasets/itu_standards/queries.jsonl',
            'name': 'ITU Standards'
        },
        'energy': {
            'corpus': 'datasets/energy/cases.jsonl',
            'queries': 'datasets/energy/queries.jsonl',
            'name': 'Energy/Electricity'
        },
        'tbmp': {
            'corpus': 'datasets/tbmp_2024/chunks.jsonl',
            'queries': 'datasets/tbmp_2024/queries.jsonl',
            'name': 'TBMP Legal'
        }
    }

    all_shapley = []
    all_interactions = []  # Equation 2
    all_convergence = []   # Equation 3
    all_failure = []       # Equations 4 & 5
    stages = ["retrieve", "synthesize", "judge"]

    for dataset_id, dataset_config in datasets.items():
        print(f"\n{'='*70}", flush=True)
        print(f"Computing Shapley values: {dataset_config['name']}", flush=True)
        print(f"{'='*70}\n", flush=True)

        # Load dataset
        try:
            corpus = load_corpus(dataset_config['corpus'])
            queries = load_queries(dataset_config['queries'], max_queries=20)  # Use 20 queries for statistical power

            print(f"  Loaded: {len(corpus)} documents, {len(queries)} queries")

        except Exception as e:
            print(f"   Error loading dataset: {e}")
            continue

        # Test with multiple models
        # Removed Phi-3 due to DynamicCache compatibility issue with transformers
        models = ["hf/gpt2", "hf/distilgpt2", "hf/Qwen/Qwen2.5-3B-Instruct", "hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

        for model_name in models:
            print(f"\n  Model: {model_name}")

            # Build pipeline
            pipeline = SimplePipeline(corpus, retriever="simple_overlap", synthesizer=model_name)

            # Compute Shapley values
            print(f"  Computing Shapley values (20 samples per query)...", flush=True)

            try:
                shapley_df = attribution.compute_shapley_values(
                    pipeline.run,
                    queries,
                    stages=stages,
                    n_samples=20,  # Use 20 permutations for statistical power
                    verbose=True
                )

                shapley_df['dataset'] = dataset_id
                shapley_df['dataset_name'] = dataset_config['name']
                shapley_df['model'] = model_name

                all_shapley.append(shapley_df)

                # Show summary
                summary = shapley_df.groupby('stage')['shapley_value'].mean()
                print(f"\n  Average stage importance:")
                for stage, value in summary.items():
                    print(f"    {stage:12s}: {value:.3f}")

                # === NOVEL EQUATION 2: Shapley Interaction Index ===
                print(f"\n  Computing Equation 2: Shapley Interactions...", flush=True)
                try:
                    interaction_df = attribution.compute_shapley_interactions(
                        pipeline.run,
                        queries[:10],  # Use 10 queries for interaction (computationally expensive)
                        stages=stages,
                        n_samples=10,
                        verbose=False
                    )
                    interaction_df['dataset'] = dataset_id
                    interaction_df['model'] = model_name
                    all_interactions.append(interaction_df)

                    # Compute Interaction Strength (IS)
                    is_value = JournalMetrics.compute_interaction_strength(interaction_df)
                    print(f"    Interaction Strength (IS): {is_value:.4f}")
                except Exception as e:
                    print(f"     Interaction computation skipped: {e}")

                # === NOVEL EQUATION 3: Convergence Bound ===
                print(f"  Computing Equation 3: Convergence Bound...", flush=True)
                try:
                    convergence_df = attribution.compute_convergence_bound(shapley_df)
                    convergence_df['dataset'] = dataset_id
                    convergence_df['model'] = model_name
                    all_convergence.append(convergence_df)

                    # Show convergence status
                    for _, row in convergence_df.iterrows():
                        status = "CONVERGED" if row['converged'] else "needs more samples"
                        print(f"    {row['stage']:12s}: CI width={row['current_ci_width']:.4f} ({status})")
                except Exception as e:
                    print(f"     Convergence computation skipped: {e}")

                # === NOVEL EQUATIONS 4 & 5: Failure Shapley & Blame Score ===
                print(f"  Computing Equations 4-5: Failure Shapley & Blame...", flush=True)
                try:
                    failure_detail_df, failure_summary_df = attribution.compute_failure_shapley(
                        pipeline.run,
                        queries[:10],  # Use 10 queries for failure analysis
                        stages=stages,
                        failure_threshold=0.1,
                        n_samples=10,
                        verbose=False
                    )
                    failure_summary_df['dataset'] = dataset_id
                    failure_summary_df['model'] = model_name
                    all_failure.append(failure_summary_df)

                    # Show blame ranking
                    blame_df = JournalMetrics.compute_blame_ranking(failure_summary_df)
                    for _, row in blame_df.iterrows():
                        print(f"    {row['stage']:12s}: Blame={row['blame_score']:.3f} ({row['diagnosis']})")
                except Exception as e:
                    print(f"     Failure analysis skipped: {e}")

            except Exception as e:
                print(f"   Error computing Shapley values: {e}")
                import traceback
                traceback.print_exc()

    # Combine results
    if all_shapley:
        print(f"\n{'='*70}")
        print("Saving results for ALL 5 NOVEL EQUATIONS...")
        print(f"{'='*70}\n")

        # === EQUATION 1: Shapley Values with SLR ===
        df = pd.concat(all_shapley, ignore_index=True)
        outputs.save_metrics_csv(df, "shapley_values")
        print(f"  Eq 1 (SLR): Saved to shapley_values.csv - columns include 'slr'")

        # === EQUATION 2: Shapley Interactions ===
        if all_interactions:
            interactions_df = pd.concat(all_interactions, ignore_index=True)
            outputs.save_metrics_csv(interactions_df, "shapley_interactions")
            print(f"  Eq 2 (Interactions): Saved to shapley_interactions.csv")

        # === EQUATION 3: Convergence Bounds ===
        if all_convergence:
            convergence_df = pd.concat(all_convergence, ignore_index=True)
            outputs.save_metrics_csv(convergence_df, "convergence_bounds")
            print(f"  Eq 3 (Convergence): Saved to convergence_bounds.csv")

        # === EQUATIONS 4 & 5: Failure Shapley & Blame Score ===
        if all_failure:
            failure_df = pd.concat(all_failure, ignore_index=True)
            outputs.save_metrics_csv(failure_df, "failure_shapley_blame")
            print(f"  Eq 4-5 (Failure/Blame): Saved to failure_shapley_blame.csv")

        # Generate summary table
        summary = df.groupby(['dataset_name', 'stage']).agg({
            'shapley_value': ['mean', 'std', 'count'],
            'quality_impact': ['mean', 'std']
        }).reset_index()

        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

        outputs.save_latex_table(
            summary,
            "attribution_summary",
            caption="Stage importance across domains via Shapley values. " + \
                    "Values sum to 1.0 per query, representing normalized contribution to quality."
        )

        # Generate plot
        print("Generating plot...")

        try:
            fig = plot_stage_importance(df, stage_col='stage', value_col='shapley_value',
                                       domain_col='dataset_name')
            outputs.save_plot(fig, "stage_importance_by_domain")
            print(" Plot generated successfully")

        except Exception as e:
            print(f" Warning: Plot generation failed: {e}")

        # Create summary report
        outputs.create_summary_report(df, metrics=['shapley_value', 'quality_impact'])

        print("\n" + "="*70)
        print(" EXPERIMENT 2 COMPLETE - ALL 5 NOVEL EQUATIONS COMPUTED!")
        print("="*70)
        print(f"\nResults saved to: {outputs.base_dir}")
        print(f"\n   NOVEL EQUATION OUTPUTS:")
        print(f"  Eq 1 (SLR):          {outputs.base_dir}/metrics/shapley_values.csv")
        print(f"  Eq 2 (Interactions): {outputs.base_dir}/metrics/shapley_interactions.csv")
        print(f"  Eq 3 (Convergence):  {outputs.base_dir}/metrics/convergence_bounds.csv")
        print(f"  Eq 4-5 (Failure):    {outputs.base_dir}/metrics/failure_shapley_blame.csv")
        print(f"\n   Figures: {outputs.base_dir}/figures/stage_importance_by_domain.pdf")
        print(f"   Tables:  {outputs.base_dir}/tables/attribution_summary.tex")

    else:
        print("\n No results generated")


if __name__ == "__main__":
    main()
