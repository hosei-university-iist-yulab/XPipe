#!/bin/bash
# Run Experiment 1 on GPU 5
set -e

GPU_ID=${1:-5}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "============================================================"
echo "EXPERIMENT 1: Multi-Domain Pipeline Evaluation (GPU ${GPU_ID})"
echo "============================================================"
echo "Started: $(date)"
echo "GPU: ${GPU_ID}"
echo ""

python3 -u experiments/exp1_multidomain.py 2>&1 | tee output/exp1_parallel_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo " EXPERIMENT 1 COMPLETE (GPU ${GPU_ID})"
echo "============================================================"
echo "Completed: $(date)"
echo "Results: output/exp1_multidomain/metrics/all_configs.csv"
