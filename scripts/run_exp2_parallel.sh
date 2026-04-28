#!/bin/bash
# Run Experiment 2 on GPU 6
set -e

GPU_ID=${1:-6}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "============================================================"
echo "EXPERIMENT 2: Causal Attribution Analysis (GPU ${GPU_ID})"
echo "============================================================"
echo "Started: $(date)"
echo "GPU: ${GPU_ID}"
echo ""

python3 -u experiments/exp2_attribution.py 2>&1 | tee output/exp2_parallel_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo " EXPERIMENT 2 COMPLETE (GPU ${GPU_ID})"
echo "============================================================"
echo "Completed: $(date)"
echo "Results: output/exp2_attribution/metrics/shapley_values.csv"
