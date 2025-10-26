#!/bin/bash
# Master Parallel Execution Script
# Runs Exp1 and Exp2 simultaneously on different GPUs

echo "============================================================"
echo "XPipeline Parallel Experiment Execution"
echo "============================================================"
echo "Started: $(date)"
echo ""
echo "Strategy:"
echo "  GPU 5: Experiment 1 (Multi-Domain, 1,280 runs, 4-8 hours)"
echo "  GPU 6: Experiment 2 (Attribution, 6,400 runs, 2-4 hours)"
echo ""
echo "Expected completion: 4-8 hours (longest experiment)"
echo "Time saved: 2-4 hours vs sequential execution"
echo "============================================================"
echo ""

# Make scripts executable
chmod +x scripts/run_exp1_parallel.sh
chmod +x scripts/run_exp2_parallel.sh

# Launch Experiment 1 on GPU 5 in background
echo "▶ Launching Experiment 1 on GPU 5..."
nohup ./scripts/run_exp1_parallel.sh 5 > output/exp1_parallel.log 2>&1 &
EXP1_PID=$!
echo "  PID: ${EXP1_PID}"
echo ${EXP1_PID} > output/exp1_parallel.pid

sleep 2

# Launch Experiment 2 on GPU 6 in background
echo "▶ Launching Experiment 2 on GPU 6..."
nohup ./scripts/run_exp2_parallel.sh 6 > output/exp2_parallel.log 2>&1 &
EXP2_PID=$!
echo "  PID: ${EXP2_PID}"
echo ${EXP2_PID} > output/exp2_parallel.pid

echo ""
echo "============================================================"
echo "Both experiments launched successfully!"
echo "============================================================"
echo ""
echo "Process IDs:"
echo "  Exp1 (GPU 5): ${EXP1_PID}"
echo "  Exp2 (GPU 6): ${EXP2_PID}"
echo ""
echo "Monitor with:"
echo "  watch -n 30 './monitor_experiments.sh'"
echo "  tail -f output/exp1_parallel.log"
echo "  tail -f output/exp2_parallel.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check status:"
echo "  ps -p ${EXP1_PID} ${EXP2_PID}"
echo ""
