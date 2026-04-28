#!/usr/bin/env python3
"""
Experiment 3: Cost-Quality-Latency Pareto Analysis

Evaluates tradeoffs between cost, quality, and latency across different LLM models.
Tests 5 models (4 local FREE + 1 Claude API) on 4 domains with 18 queries each (360 total evaluations).

Models:
- FREE (local): GPT-2, DistilGPT2, Qwen2.5-3B, Phi-3-mini (GPU-accelerated when CUDA available)
- PAID (API): Claude-3-Haiku ($0.25/$1.25 per 1M tokens) - Budget tier only

Total cost: ~$0.63 for all API evaluations (72 queries × Claude-3-Haiku)

Goal: Generate 3D Pareto frontiers showing optimal model selection for different
      budget constraints and quality requirements.

Usage:
    # On GPU server with CUDA 5,6,7:
    CUDA_VISIBLE_DEVICES=5,6,7 python experiments/exp3_cost_quality_latency.py

    # Outputs saved to: output/exp3_cost_quality_latency/
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file for API keys
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
    print(f" Loaded API keys from {env_path}\n")

import json
import pandas as pd
import numpy as np
from xpipe.runners.rag import run as run_rag
from xpipe.metrics import MetricLog
from xpipe.outputs import ExperimentOutputs
import time


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
    print("=" * 80, flush=True)
    print("EXPERIMENT 3: Cost-Quality-Latency Pareto Analysis", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # Initialize outputs
    outputs = ExperimentOutputs("exp3_cost_quality_latency")

    # Dataset configurations (4 domains available)
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
        }
    }

    # Model configurations (cost-quality spectrum)
    # Format: (backend/model_id, display_name, tier)
    models = [
        # FREE Local Models (GPU-accelerated when CUDA available)
        ("hf/gpt2", "GPT-2 (124M)", "free"),
        ("hf/distilgpt2", "DistilGPT2 (82M)", "free"),
        ("hf/Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "free"),
        ("hf/microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini", "free"),

        # PAID API Models (Claude) - Budget tier only
        ("anthropic/claude-3-haiku-20240307", "Claude-3-Haiku", "budget"),
        # Note: Claude-3-7-Sonnet removed (too expensive: $13.02 for 72 queries)
        # Can add OpenAI/Google models when API keys available
    ]

    # Fixed configuration (to isolate model effects)
    retriever = "simple_overlap"
    judge_enabled = False  # No judge to isolate synthesis model performance

    all_results = []
    total_configs = len(datasets) * len(models)
    current = 0

    print(f"Testing {len(models)} models across {len(datasets)} domains")
    print(f"Total evaluations: {total_configs} configurations\n")
    print(f"Models under test:")
    for model_spec, display_name, tier in models:
        print(f"  - {display_name:25s} [{tier:7s}] ({model_spec})")
    print()

    # Run experiments
    experiment_start = time.time()

    for dataset_id, dataset_config in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_config['name']}")
        print(f"{'='*80}\n")

        # Load dataset
        try:
            corpus = load_corpus(dataset_config['corpus_file'])
            queries = load_queries(dataset_config['queries_file'])

            # Use 18 queries per dataset (72 total across 4 domains)
            # This gives us 72 × 5 models = 360 data points for analysis
            queries = queries[:18]

            print(f"  Loaded: {len(corpus)} documents, {len(queries)} queries", flush=True)

        except Exception as e:
            print(f"   Error loading dataset: {e}")
            continue

        for model_spec, display_name, tier in models:
            current += 1

            print(f"\n  [{current}/{total_configs}] Model: {display_name}")
            print(f"    Tier: {tier}, Backend: {model_spec}")

            # Build configuration
            cfg = {
                "name": f"{dataset_id}_{display_name.replace(' ', '_')}",
                "pipeline": "rag",
                "logdir": "output/exp3_cost_quality_latency",
                "retriever": {
                    "name": retriever,
                    "top_k": 3
                },
                "llms": {
                    "synthesize": {
                        "model": model_spec,
                        "params": {
                            "max_new_tokens": 150,  # Longer outputs for quality
                            "temperature": 0.7,     # Balanced creativity
                            "top_p": 0.95
                        }
                    },
                    "judge": {
                        "enabled": judge_enabled,
                        "model": "",
                        "params": {}
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
                    row['model_spec'] = model_spec
                    row['model_name'] = display_name
                    row['model_tier'] = tier
                    all_results.append(row)

                # Show summary
                df_run = pd.DataFrame(metrics.rows)
                if 'rougeL_f' in df_run.columns and df_run['rougeL_f'].notna().any():
                    avg_quality = df_run['rougeL_f'].mean()
                    avg_latency = df_run['latency_ms'].mean()
                    total_cost = df_run['total_cost_usd'].sum()
                    avg_cost = df_run['total_cost_usd'].mean()

                    print(f"     Quality: {avg_quality:.3f} ROUGE-L")
                    print(f"       Latency: {avg_latency:.0f}ms")
                    print(f"       Cost: ${total_cost:.6f} total (${avg_cost:.6f}/query)")
                    print(f"       Queries: {len(df_run)}")
                else:
                    print(f"     Completed {len(metrics.rows)} queries")

            except Exception as e:
                print(f"     Error: {e}")
                import traceback
                traceback.print_exc()

    experiment_duration = time.time() - experiment_start

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print(" No results to save!")
        return

    # Save raw metrics
    metrics_path = outputs.save_metrics_csv(df, "cost_quality_latency")
    print(f" Saved metrics: {metrics_path}")

    # Generate summary statistics
    summary = df.groupby(['model_name', 'model_tier']).agg({
        'rougeL_f': ['mean', 'std', 'count'],
        'latency_ms': ['mean', 'std'],
        'total_cost_usd': ['sum', 'mean'],
        'synth_prompt_tokens': 'mean',
        'synth_completion_tokens': 'mean'
    }).round(6)

    summary_path = outputs.dir / "model_summary.csv"
    summary.to_csv(summary_path)
    print(f" Saved summary: {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Model Performance Comparison")
    print(f"{'='*80}\n")
    print(summary.to_string())

    # Cost-effectiveness analysis
    print(f"\n{'='*80}")
    print("COST-EFFECTIVENESS ANALYSIS")
    print(f"{'='*80}\n")

    cost_effectiveness = df.groupby('model_name').agg({
        'rougeL_f': 'mean',
        'latency_ms': 'mean',
        'total_cost_usd': 'mean'
    }).round(6)

    # Calculate quality per dollar (handle free models)
    cost_effectiveness['quality_per_dollar'] = np.where(
        cost_effectiveness['total_cost_usd'] > 0,
        cost_effectiveness['rougeL_f'] / cost_effectiveness['total_cost_usd'],
        np.inf  # Free models have infinite ROI
    )

    # Calculate quality per second
    cost_effectiveness['quality_per_sec'] = cost_effectiveness['rougeL_f'] / (cost_effectiveness['latency_ms'] / 1000)

    cost_effectiveness = cost_effectiveness.sort_values('quality_per_dollar', ascending=False)
    print(cost_effectiveness.to_string())

    # Save cost-effectiveness
    ce_path = outputs.dir / "cost_effectiveness.csv"
    cost_effectiveness.to_csv(ce_path)
    print(f"\n Saved cost-effectiveness: {ce_path}")

    # Per-domain analysis
    print(f"\n{'='*80}")
    print("PER-DOMAIN ANALYSIS")
    print(f"{'='*80}\n")

    domain_summary = df.groupby(['dataset_name', 'model_tier']).agg({
        'rougeL_f': 'mean',
        'total_cost_usd': 'mean'
    }).round(6)

    print(domain_summary.to_string())

    domain_path = outputs.dir / "domain_summary.csv"
    domain_summary.to_csv(domain_path)
    print(f"\n Saved domain summary: {domain_path}")

    # Final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")
    print(f"Total configurations tested: {current}/{total_configs}")
    print(f"Total evaluations: {len(df)}")
    print(f"Total duration: {experiment_duration/60:.1f} minutes")
    print(f"Average time per evaluation: {experiment_duration/len(df):.1f}s")
    print(f"\nResults saved to: {outputs.dir}/")
    print(f"  - metrics.csv: Full per-query results")
    print(f"  - model_summary.csv: Aggregated model statistics")
    print(f"  - cost_effectiveness.csv: ROI analysis")
    print(f"  - domain_summary.csv: Per-domain performance")
    print()


if __name__ == "__main__":
    main()
