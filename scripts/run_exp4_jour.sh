#!/bin/bash
################################################################################
# XPipe Journal Extension - Experiment 4 Runner
# Experiment: Dense Retriever Comparison
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================================================"
echo "XPipe Journal Extension - Experiment 4: Dense Retriever Comparison"
echo "================================================================================"
echo ""

# Configuration
EXPERIMENT="exp4_dense_retrievers"
GPU_DEVICES="5,6,7"  # Change based on available GPUs
OUTPUT_LOG="output/exp4_run.log"

# Check if experiment script exists
if [ ! -f "$PROJECT_DIR/experiments/${EXPERIMENT}.py" ]; then
    echo " Error: Experiment script not found: experiments/${EXPERIMENT}.py"
    exit 1
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
    echo "  nvidia-smi not found. Running on CPU (will be VERY slow for dense retrievers)."
    GPU_DEVICES=""
    echo ""
fi

# Check required dependencies
echo "Checking dependencies..."
python3 -c "import transformers; print('   transformers')" 2>/dev/null || echo "   transformers (REQUIRED)"
python3 -c "import torch; print('   torch')" 2>/dev/null || echo "   torch (REQUIRED)"
python3 -c "from rank_bm25 import BM25Okapi; print('   rank-bm25')" 2>/dev/null || echo "   rank-bm25 (install: pip install rank-bm25)"
echo ""

# Display experiment details
echo "================================================================================"
echo "Experiment Details"
echo "================================================================================"
echo ""
echo "Retrievers (7):"
echo "  Lexical:       simple_overlap, jaccard"
echo "  Sparse:        BM25, SPLADE"
echo "  Dense:         BERT-base, Contriever, E5-large"
echo ""
echo "Models (4 - all LOCAL, FREE):"
echo "  1. GPT-2 (124M)"
echo "  2. DistilGPT2 (82M)"
echo "  3. Qwen2.5-3B"
echo "  4. Phi-3-mini"
echo ""
echo "Domains (7):"
echo "  1. TBMP Legal"
echo "  2. Network Troubleshooting"
echo "  3. Customer Support"
echo "  4. ITU Standards"
echo "  5. Smart Grid/Energy           (IoT)"
echo "  6. EV/Battery Management       (IoT)"
echo "  7. Solar PV Systems            (IoT)"
echo ""
echo "Configuration:"
echo "  Judge settings: 2 (enabled/disabled)"
echo "  Queries per domain: 14"
echo ""
echo "Total evaluations: 5,488"
echo "  = 7 retrievers × 4 models × 2 judges × 7 domains × 14 queries"
echo ""
echo "Estimated runtime: 4-8 hours (depends on GPU)"
echo "Estimated cost: $0.00 (all local models)"
echo ""

# Prompt for confirmation
read -p "Start Experiment 4? This will take several hours. (y/n) " -n 1 -r
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
echo "Starting Experiment 4..."
echo "================================================================================"
echo ""

# Run with GPU selection
if [ -n "$GPU_DEVICES" ]; then
    echo "Using GPUs: $GPU_DEVICES"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES nohup python3 experiments/${EXPERIMENT}.py > ${OUTPUT_LOG} 2>&1 &
else
    echo "Using CPU (this will be very slow)"
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
echo "Monitor with live updates:"
echo "  watch -n 10 'tail -20 $OUTPUT_LOG'"
echo ""
echo "Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Check if running:"
echo "  ps aux | grep ${EXPERIMENT}"
echo ""
echo "Kill if needed:"
echo "  kill $PID"
echo ""
echo "================================================================================"
echo "Note: Dense retrievers (BERT, Contriever, E5) will download models on first run."
echo "      This may take 5-10 minutes. Monitor log for download progress."
echo "================================================================================"

# Save PID
echo $PID > output/exp4.pid
echo "PID saved to output/exp4.pid"
