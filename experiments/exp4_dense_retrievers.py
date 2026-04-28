#!/usr/bin/env python3
"""
Experiment 4: Dense Retriever Comparison
=========================================

Journal Extension - Week 4-5

Research Question:
  Do dense neural retrievers (BERT, Contriever, E5) improve RAG quality
  compared to lexical baselines (overlap, Jaccard, BM25)?

Design:
  - 7 retrievers: simple_overlap, jaccard, bm25, bert, contriever, e5-large, splade
  - 4 LLM models: GPT-2, DistilGPT2, Qwen2.5-3B, Phi-3-mini (local, FREE)
  - 2 judge settings: enabled, disabled
  - 7 domains: TBMP, Network, Customer, ITU, Energy, EV/Battery, Solar PV
  - 14 queries per domain

  Total: 7 × 4 × 2 × 7 × 14 = 5,488 evaluations

Metrics:
  1. End-to-end quality: ROUGE-L, F1-token, relevance (grounding)
  2. Latency: Total time per query
  3. Cost: All FREE (local models + local retrievers)

Expected Results:
  - Dense retrievers improve end-to-end quality (+5-15% ROUGE-L)
  - E5-large best for technical domains, BM25 competitive
  - Trade-off: Dense retrievers slower but better quality

Output:
  - output/exp4_dense_retrievers/metrics/retriever_comparison.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import json
import pandas as pd
from typing import List, Dict, Any

from xpipe.runners.rag import run as run_rag
from xpipe.metrics import MetricLog
from xpipe.outputs import ExperimentOutputs


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

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value


# ============================================================================
# Experiment Configuration
# ============================================================================

# 7 Retrievers to compare
RETRIEVERS = [
    'simple_overlap',  # Baseline 1: Simple word overlap
    'jaccard',         # Baseline 2: Jaccard similarity
    'bm25',           # Sparse: Probabilistic retrieval
    'bert',           # Dense: BERT-base (110M params)
    'contriever',     # Dense: Facebook Contriever (110M)
    'e5-large',       # Dense: Microsoft E5 (335M, SOTA)
    'splade'          # Sparse Neural: SPLADE
]

# 4 LLM Models (all local, FREE)
MODELS = {
    'gpt2': {
        'id': 'gpt2',
        'backend': 'hf',
        'name': 'GPT-2',
        'params': {'max_new_tokens': 100, 'temperature': 0.1}
    },
    'distilgpt2': {
        'id': 'distilgpt2',
        'backend': 'hf',
        'name': 'DistilGPT2',
        'params': {'max_new_tokens': 100, 'temperature': 0.1}
    },
    'qwen': {
        'id': 'Qwen/Qwen2.5-3B-Instruct',
        'backend': 'hf',
        'name': 'Qwen2.5-3B',
        'params': {'max_new_tokens': 100, 'temperature': 0.1}
    },
    'phi3': {
        'id': 'microsoft/Phi-3-mini-4k-instruct',
        'backend': 'hf',
        'name': 'Phi-3-mini',
        'params': {'max_new_tokens': 100, 'temperature': 0.1}
    }
}

# 7 Domains (actual dataset names) - INCLUDING 3 SMART GRID/IoT DATASETS
DATASETS = {
    'tbmp': {
        'corpus_file': 'datasets/tbmp_2024/chunks.jsonl',
        'queries_file': 'datasets/tbmp_2024/queries.jsonl',
        'name': 'TBMP Legal'
    },
    'network': {
        'corpus_file': 'datasets/network_logs/cases.jsonl',
        'queries_file': 'datasets/network_logs/queries.jsonl',
        'name': 'Network Troubleshooting'
    },
    'customer': {
        'corpus_file': 'datasets/customer_support/conversations.jsonl',
        'queries_file': 'datasets/customer_support/queries.jsonl',
        'name': 'Customer Support'
    },
    'itu': {
        'corpus_file': 'datasets/itu_standards/recommendations.jsonl',
        'queries_file': 'datasets/itu_standards/queries.jsonl',
        'name': 'ITU Standards'
    },
    # SMART GRID / IoT DATASETS
    'energy': {
        'corpus_file': 'datasets/energy/cases.jsonl',
        'queries_file': 'datasets/energy/queries.jsonl',
        'name': 'Smart Grid/Energy'
    },
    'ev_battery': {
        'corpus_file': 'datasets/ev_battery/cases.jsonl',
        'queries_file': 'datasets/ev_battery/queries.jsonl',
        'name': 'EV/Battery Management'
    },
    'solar_pv': {
        'corpus_file': 'datasets/solar_pv/cases.jsonl',
        'queries_file': 'datasets/solar_pv/queries.jsonl',
        'name': 'Solar PV Systems'
    }
}

# Query limit per domain (to control runtime)
QUERIES_PER_DOMAIN = 14

# Retrieval parameters
RETRIEVAL_K = 5  # Retrieve top-5 documents


# ============================================================================
# Helper Functions
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 4: Dense Retriever Comparison")
    print("=" * 80)
    print(f"Retrievers: {len(RETRIEVERS)}")
    print(f"Models: {len(MODELS)}")
    print(f"Judge settings: 2 (enabled/disabled)")
    print(f"Domains: {len(DATASETS)}")
    print(f"Queries/domain: {QUERIES_PER_DOMAIN}")

    total_configs = len(RETRIEVERS) * len(MODELS) * 2  # 2 = judge on/off
    total_evals = total_configs * len(DATASETS) * QUERIES_PER_DOMAIN
    print(f"Total configs: {total_configs}")
    print(f"Total evaluations: {total_evals}")
    print("=" * 80)

    # Initialize outputs
    outputs = ExperimentOutputs("exp4_dense_retrievers")
    metrics = MetricLog()

    # Track timing
    start_time = time.time()
    config_num = 0

    # Iterate over all configurations
    for retriever_name in RETRIEVERS:
        for model_key, model_info in MODELS.items():
            for judge_enabled in [True, False]:
                config_num += 1

                model_name = model_info['name']
                judge_label = "judge-on" if judge_enabled else "judge-off"

                print(f"\n{'='*80}")
                print(f"[{config_num}/{total_configs}] {retriever_name} + {model_name} + {judge_label}")
                print(f"{'='*80}")

                # Build pipeline configuration
                cfg = {
                    "name": f"{retriever_name}_{model_key}_{judge_label}",
                    "pipeline": "rag",
                    "retriever": {
                        "name": retriever_name,
                        "top_k": RETRIEVAL_K
                    },
                    "llms": {
                        "synthesize": {
                            "model": f"hf/{model_info['id']}",
                            "params": model_info['params']
                        },
                        "judge": {
                            "enabled": judge_enabled,
                            "model": "hf/distilgpt2" if judge_enabled else "",
                            "params": {"max_new_tokens": 16, "temperature": 0.0}
                        }
                    },
                    "prompt": {
                        "budget_tokens": 900,
                        "ctx_per_doc_tokens": 300
                    }
                }

                # Iterate over domains
                for domain_key, domain_config in DATASETS.items():
                    print(f"\n   Domain: {domain_config['name']}")

                    try:
                        # Load dataset
                        corpus = load_corpus(domain_config['corpus_file'])
                        queries = load_queries(domain_config['queries_file'])

                        # Limit queries
                        queries = queries[:QUERIES_PER_DOMAIN]

                        print(f"    Loaded: {len(corpus)} docs, {len(queries)} queries")

                        # Run RAG pipeline
                        try:
                            result = run_rag(cfg, MODELS, metrics, corpus, queries)
                            print(f"     Completed {len(queries)} queries")

                        except Exception as e:
                            print(f"     Error running pipeline: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                    except Exception as e:
                        print(f"     Error loading domain: {e}")
                        continue

                # Print progress summary every 10 configs
                if config_num % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_per_config = elapsed / config_num
                    remaining_configs = total_configs - config_num
                    eta_seconds = avg_per_config * remaining_configs

                    print(f"\n Progress: {config_num}/{total_configs} configs")
                    print(f"   Elapsed: {format_time(elapsed)}")
                    print(f"   ETA: {format_time(eta_seconds)}")

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}\n")

    df = metrics.to_dataframe()

    if len(df) == 0:
        print(" No results to save!")
        return

    # Save raw metrics
    metrics_path = outputs.save_metrics_csv(df, "retriever_comparison")

    # Generate summary by retriever
    print(f"\n{'='*80}")
    print("SUMMARY: Retriever Performance")
    print(f"{'='*80}\n")

    retriever_summary = df.groupby(['retriever']).agg({
        'rougeL_f': ['mean', 'std', 'count'],
        'f1_token': ['mean', 'std'],
        'relevance': 'mean',
        'latency_ms': ['mean', 'std']
    }).round(4)

    print(retriever_summary.to_string())

    # Save retriever summary
    summary_path = outputs.base_dir / "metrics" / "retriever_summary.csv"
    retriever_summary.to_csv(summary_path)
    print(f"\n Saved retriever summary: {summary_path}")

    # Generate summary by domain
    print(f"\n{'='*80}")
    print("SUMMARY: Performance by Domain")
    print(f"{'='*80}\n")

    # Extract domain from item (assuming format: domain_qX)
    df['domain'] = df['item'].str.extract(r'^([a-z_]+)')[0]

    domain_summary = df.groupby(['domain', 'retriever']).agg({
        'rougeL_f': 'mean',
        'relevance': 'mean',
        'latency_ms': 'mean'
    }).round(4)

    print(domain_summary.to_string())

    # Save domain summary
    domain_path = outputs.base_dir / "metrics" / "domain_summary.csv"
    domain_summary.to_csv(domain_path)
    print(f"\n Saved domain summary: {domain_path}")

    # Model comparison
    print(f"\n{'='*80}")
    print("SUMMARY: Performance by Model")
    print(f"{'='*80}\n")

    # Extract model from synth_model
    model_summary = df.groupby(['synth_model', 'retriever']).agg({
        'rougeL_f': 'mean',
        'latency_ms': 'mean'
    }).round(4)

    print(model_summary.to_string())

    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("EXPERIMENT 4 COMPLETE")
    print(f"{'='*80}")
    print(f"Total evaluations: {len(df)}")
    print(f"Total time: {format_time(elapsed)}")
    print(f"Average time per eval: {elapsed/len(df):.2f}s")
    print(f"Results saved to: {outputs.base_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
