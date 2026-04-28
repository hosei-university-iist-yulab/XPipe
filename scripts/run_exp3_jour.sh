#!/bin/bash
################################################################################
# XPipe Journal Extension - Experiment 3 Runner
# Experiment: Cost-Quality-Latency Pareto Analysis
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================================================"
echo "XPipe Journal Extension - Experiment 3: Cost-Quality-Latency Analysis"
echo "================================================================================"
echo ""

# Configuration
EXPERIMENT="exp3_cost_quality_latency"
GPU_DEVICES="5,6,7"  # Change based on available GPUs
OUTPUT_LOG="output/exp3_run.log"

# Check if experiment script exists
if [ ! -f "$PROJECT_DIR/experiments/${EXPERIMENT}.py" ]; then
    echo " Error: Experiment script not found: experiments/${EXPERIMENT}.py"
    exit 1
fi

# Check if .env file exists (for API keys)
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "  Warning: .env file not found. API models (Claude) may fail."
    echo "   Create .env with ANTHROPIC_API_KEY if needed."
    echo ""
fi

# Display configuration
echo "Configuration:"
echo "  Experiment: ${EXPERIMENT}"
echo "  GPU Devices: ${GPU_DEVICES}"
echo "  Output Log: ${OUTPUT_LOG}"
echo "  Working Dir: ${PROJECT_DIR}"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    echo ""
else
    echo "  nvidia-smi not found. Running on CPU."
    GPU_DEVICES=""
    echo ""
fi

# Prompt for confirmation
read -p "Start Experiment 3? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 0
fi

# Create output directory
mkdir -p "$PROJECT_DIR/output"

# Change to project directory
cd "$PROJECT_DIR"

# Run experiment
echo ""
echo "================================================================================"
echo "Starting Experiment 3..."
echo "================================================================================"
echo ""
echo "Models to test (5):"
echo "  1. GPT-2 (124M)           - FREE (local)"
echo "  2. DistilGPT2 (82M)       - FREE (local)"
echo "  3. Qwen2.5-3B             - FREE (local)"
echo "  4. Phi-3-mini             - FREE (local)"
echo "  5. Claude-3-Haiku         - PAID (~$0.63 total)"
echo ""
echo "Domains (4): Network, Customer Support, ITU, TBMP"
echo "Total evaluations: 360 (5 models × 4 domains × 18 queries)"
echo ""
echo "Estimated runtime: 15-30 minutes"
echo "Estimated cost: ~$0.63 (Claude API only)"
echo ""

# Run with GPU selection
if [ -n "$GPU_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES nohup python3 experiments/${EXPERIMENT}.py > ${OUTPUT_LOG} 2>&1 &
else
    nohup python3 experiments/${EXPERIMENT}.py > ${OUTPUT_LOG} 2>&1 &
fi

PID=$!
echo " Experiment started in background"
echo "   PID: $PID"
echo "   Log: $OUTPUT_LOG"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_LOG"
echo ""
echo "Check if running:"
echo "  ps aux | grep ${EXPERIMENT}"
echo ""
echo "Kill if needed:"
echo "  kill $PID"
echo ""
echo "================================================================================"

# Save PID
echo $PID > output/exp3.pid
echo "PID saved to output/exp3.pid"
