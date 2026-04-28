#!/bin/bash
# GPU Runner Script for XPipeline Experiments
# Supports CUDA GPUs 5, 6, 7 (or any specified)
#
# Usage:
#   ./scripts/run_experiments_gpu.sh 5        # Run on GPU 5
#   ./scripts/run_experiments_gpu.sh 6,7      # Run on GPUs 6 and 7
#   ./scripts/run_experiments_gpu.sh          # Auto-detect available GPU

set -e  # Exit on error

# Default to auto-detect if no GPU specified
GPU_ID=${1:-0}

echo "============================================================"
echo "XPipeline GPU Experiment Runner"
echo "============================================================"
echo ""
echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Check PyTorch CUDA availability
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "============================================================"
echo "EXPERIMENT 1: Multi-Domain Pipeline Evaluation"
echo "============================================================"
echo "Configuration:"
echo "  - Models: 4 (GPT-2, DistilGPT2, Qwen2.5-3B, Phi-3-mini)"
echo "  - Queries: 20 per domain"
echo "  - Domains: 4"
echo "  - Total runs: 1,280"
echo "  - Estimated time: 4-8 hours on GPU"
echo ""

# Run Experiment 1
python3 -u experiments/exp1_multidomain.py 2>&1 | tee output/exp1_gpu_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo "EXPERIMENT 2: Causal Attribution Analysis"
echo "============================================================"
echo "Configuration:"
echo "  - Models: 4"
echo "  - Queries: 20 per domain"
echo "  - Domains: 4"
echo "  - Permutations: 20 per query"
echo "  - Total computations: 6,400"
echo "  - Estimated time: 2-4 hours on GPU"
echo ""

# Run Experiment 2
python3 -u experiments/exp2_attribution.py 2>&1 | tee output/exp2_gpu_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - output/exp1_multidomain/metrics/all_configs.csv"
echo "  - output/exp2_attribution/metrics/shapley_values.csv"
echo ""

# Auto-generate figures with new data
echo "============================================================"
echo "GENERATING PUBLICATION FIGURES"
echo "============================================================"
echo ""
python3 scripts/generate_all_figures.py

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - 7 publication figures in paper/2026_ITU_conf/figures/"
echo "  - Experiment data in output/"
echo ""
echo "Next steps:"
echo "  1. Review figures: ls -lh paper/2026_ITU_conf/figures/"
echo "  2. Run statistical tests: python3 scripts/compute_statistics.py"
echo "  3. Revise paper with new results"
echo ""
