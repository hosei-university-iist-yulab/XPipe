#!/usr/bin/env python3
"""
Experiment 1: Multi-Domain Pipeline Evaluation
Evaluates RAG pipeline across 5 real-world domains with different configurations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from xpipe.runners.rag import run as run_rag
from xpipe.metrics import MetricLog
from xpipe.outputs import ExperimentOutputs
from xpipe.plotting import plot_quality_by_domain, plot_config_heatmap
import yaml


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


def load_queries(path: str):
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
    return queries


def main():
    import sys
    print("=" * 70, flush=True)
    print("EXPERIMENT 1: Multi-Domain Pipeline Evaluation", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Initialize outputs
    outputs = ExperimentOutputs("exp1_multidomain")

    # Dataset configurations (5 domains)
    datasets = {
        'network_logs': {
            'corpus_file': 'datasets/network_logs/cases.jsonl',
            'queries_file': 'datasets/network_logs/queries.jsonl',
            'name': 'Network Troubleshooting'
        },
        'customer_support': {
            'corpus_file': 'datasets/customer_support/conversations.jsonl',
            'queries_file': 'datasets/customer_support/queries.jsonl',
            'name': 'Customer Support'
        },
        'itu_standards': {
            'corpus_file': 'datasets/itu_standards/recommendations.jsonl',
            'queries_file': 'datasets/itu_standards/queries.jsonl',
            'name': 'ITU Standards'
        },
        'tbmp': {
            'corpus_file': 'datasets/tbmp_2024/chunks.jsonl',
            'queries_file': 'datasets/tbmp_2024/queries.jsonl',
            'name': 'TBMP Legal'
        },
        'energy': {
            'corpus_file': 'datasets/energy/cases.jsonl',
            'queries_file': 'datasets/energy/queries.jsonl',
            'name': 'Energy/Electricity'
        }
    }

    # Pipeline configurations to test
    retrievers = ["simple_overlap", "jaccard"]
    synthesizers = ["hf/gpt2", "hf/distilgpt2", "hf/Qwen/Qwen2.5-3B-Instruct", "hf/microsoft/Phi-3-mini-4k-instruct", "hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    judges = [None, "heuristic"]  # None = no judge

    all_results = []
    total_configs = len(datasets) * len(retrievers) * len(synthesizers) * len(judges)
    current = 0

    print(f"Testing {total_configs} configurations across {len(datasets)} domains\n")

    # Run experiments
    for dataset_id, dataset_config in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_config['name']}")
        print(f"{'='*70}\n")

        # Load dataset
        try:
            corpus = load_corpus(dataset_config['corpus_file'])
            queries = load_queries(dataset_config['queries_file'])

            # Use 20 queries per dataset for statistical power
            queries = queries[:20]  # n=20 enables t-tests with p<0.05

            print(f"  Loaded: {len(corpus)} documents, {len(queries)} queries", flush=True)

        except Exception as e:
            print(f"   Error loading dataset: {e}")
            continue

        for retriever in retrievers:
            for synthesizer in synthesizers:
                for judge in judges:
                    current += 1

                    judge_enabled = judge is not None
                    config_name = f"{retriever}+{synthesizer.split('/')[-1]}+{judge or 'none'}"

                    print(f"\n  [{current}/{total_configs}] Testing: {config_name}")

                    # Build configuration
                    cfg = {
                        "name": f"{dataset_id}_{config_name}",
                        "pipeline": "rag",
                        "logdir": "output/exp1_multidomain",
                        "retriever": {
                            "name": retriever,
                            "top_k": 3
                        },
                        "llms": {
                            "synthesize": {
                                "model": synthesizer,
                                "params": {
                                    "max_new_tokens": 100,
                                    "temperature": 0.1
                                }
                            },
                            "judge": {
                                "enabled": judge_enabled,
                                "model": "hf/distilgpt2" if judge_enabled else "",
                                "params": {
                                    "max_new_tokens": 8,
                                    "temperature": 0.0
                                }
                            }
                        },
                        "prompt": {
                            "budget_tokens": 512,
                            "ctx_per_doc_tokens": 150
                        },
                        "outputs": {
                            "run_jsonl": "",
                            "metrics_csv": ""
                        }
                    }

                    # Run pipeline
                    try:
                        metrics = MetricLog()
                        result = run_rag(cfg, {}, metrics, corpus, queries)

                        # Collect results
                        for row in metrics.rows:
                            row['dataset'] = dataset_id
                            row['dataset_name'] = dataset_config['name']
                            row['config'] = config_name
                            all_results.append(row)

                        # Show summary
                        df_run = pd.DataFrame(metrics.rows)
                        if 'rougeL_f' in df_run.columns and df_run['rougeL_f'].notna().any():
                            avg_quality = df_run['rougeL_f'].mean()
                            avg_latency = df_run['latency_ms'].mean()
                            print(f"      Quality: {avg_quality:.3f}, Latency: {avg_latency:.0f}ms")
                        else:
                            print(f"      Completed {len(metrics.rows)} queries")

                    except Exception as e:
                        print(f"       Error: {e}")
                        import traceback
                        traceback.print_exc()

    # Save results
    print(f"\n{'='*70}")
    print("Saving results...")
    print(f"{'='*70}\n")

    df = pd.DataFrame(all_results)

    # Save raw metrics
    outputs.save_metrics_csv(
        df,
        "all_configs",
        required_cols=['dataset', 'config', 'latency_ms']
    )

    # Generate summary table
    summary = df.groupby(['dataset_name', 'retriever', 'synthesizer_model']).agg({
        'rougeL_f': ['mean', 'std', 'count'],
        'f1_token': ['mean', 'std'],
        'relevance': ['mean', 'std'],
        'latency_ms': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    outputs.save_latex_table(
        summary.head(20),  # Top 20 rows
        "exp1_summary",
        caption="Pipeline performance across domains and configurations. " + \
                "Shows mean ± std for quality metrics (ROUGE-L, F1, relevance) and latency."
    )

    # Generate plots
    print("Generating plots...")

    if 'rougeL_f' in df.columns and df['rougeL_f'].notna().any():
        try:
            # Plot 1: Quality by domain
            fig1 = plot_quality_by_domain(df, quality_col='rougeL_f', domain_col='dataset_name')
            outputs.save_plot(fig1, "quality_by_domain")

            # Plot 2: Configuration heatmap
            fig2 = plot_config_heatmap(df, row_var='retriever', col_var='synthesizer_model',
                                      value_var='rougeL_f')
            outputs.save_plot(fig2, "config_quality_heatmap")

            print(" Plots generated successfully")

        except Exception as e:
            print(f" Warning: Plot generation failed: {e}")

    # Create summary report
    outputs.create_summary_report(df, metrics=['rougeL_f', 'f1_token', 'latency_ms'])

    print("\n" + "="*70)
    print(" EXPERIMENT 1 COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {outputs.base_dir}")
    print(f"  - Metrics: {outputs.base_dir}/metrics/all_configs.csv")
    print(f"  - Figures: {outputs.base_dir}/figures/")
    print(f"  - Tables: {outputs.base_dir}/tables/exp1_summary.tex")
    print(f"  - Summary: {outputs.base_dir}/summary_report.html")


if __name__ == "__main__":
    main()
