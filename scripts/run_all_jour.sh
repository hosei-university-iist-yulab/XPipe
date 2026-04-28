#!/bin/bash
################################################################################
# XPipe Journal Extension - Run All Experiments Sequentially
# Runs Experiment 3, then Experiment 4
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "XPipe Journal Extension - Run All Experiments"
echo "================================================================================"
echo ""
echo "This will run:"
echo "  1. Experiment 3: Cost-Quality-Latency Analysis (360 evals, ~30 min, ~$0.63)"
echo "  2. Experiment 4: Dense Retriever Comparison (5,488 evals, ~6 hours, FREE)"
echo ""
echo "Total runtime: ~6.5 hours"
echo "Total cost: ~$0.63 (Experiment 3 API calls only)"
echo ""

read -p "Start all experiments? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "Step 1/2: Running Experiment 3"
echo "================================================================================"
echo ""

# Run Experiment 3
bash "$SCRIPT_DIR/run_exp3_jour.sh"

# Wait for Experiment 3 to complete
if [ -f output/exp3.pid ]; then
    EXP3_PID=$(cat output/exp3.pid)
    echo ""
    echo "Waiting for Experiment 3 to complete (PID: $EXP3_PID)..."
    echo "Monitor: tail -f output/exp3_run.log"
    echo ""

    # Wait for process to finish
    while ps -p $EXP3_PID > /dev/null 2>&1; do
        sleep 60
        echo "  [$(date +%H:%M:%S)] Experiment 3 still running..."
    done

    echo ""
    echo " Experiment 3 completed!"
    echo ""
fi

echo ""
echo "================================================================================"
echo "Step 2/2: Running Experiment 4"
echo "================================================================================"
echo ""

# Run Experiment 4
bash "$SCRIPT_DIR/run_exp4_jour.sh"

echo ""
echo "================================================================================"
echo "Both experiments launched!"
echo "================================================================================"
echo ""
echo "Monitor Experiment 4: tail -f output/exp4_run.log"
echo ""
