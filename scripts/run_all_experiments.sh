#!/bin/bash
# XPipe - Master Experiment Runner
# Executes all 5 experiments for ITU Kaleidoscope 2026

set -e  # Exit on error

echo "======================================"
echo "XPipe Experiments - ITU Kaleidoscope 2026"
echo "======================================"
echo ""

# Check Python environment
if ! python -c "import pandas, matplotlib, seaborn" 2>/dev/null; then
    echo " Missing dependencies. Installing..."
    pip install pandas matplotlib seaborn scikit-learn scipy
fi

echo " Dependencies OK"
echo ""

# Experiment 1: Multi-Domain Evaluation
echo "▶ EXPERIMENT 1: Multi-Domain Pipeline Evaluation"
echo "  Datasets: TBMP, Network Logs, Customer Support, ITU Standards"
echo "  Configurations: 2 retrievers × 2 synthesizers × 2 judges = 8 per dataset"
python xpipe/experiments/exp1_multidomain.py
echo " Experiment 1 complete!"
echo ""

# Experiment 2: Causal Attribution
echo "▶ EXPERIMENT 2: Causal Attribution Analysis"
echo "  Computing Shapley values for all domains..."
python xpipe/experiments/exp2_attribution.py
echo " Experiment 2 complete!"
echo ""

# Experiment 3: Error Localization
echo "▶ EXPERIMENT 3: Error Localization"
echo "  Identifying failure causes..."
python xpipe/experiments/exp3_error_localization.py
echo " Experiment 3 complete!"
echo ""

# Experiment 4: Configuration Optimization
echo "▶ EXPERIMENT 4: Configuration Optimization"
echo "  Computing Pareto frontiers..."
python xpipe/experiments/exp4_optimization.py
echo " Experiment 4 complete!"
echo ""

# Experiment 5: Human Evaluation
echo "▶ EXPERIMENT 5: Human Evaluation Study"
echo "  Analyzing human study data..."
python xpipe/experiments/exp5_human_study.py
echo " Experiment 5 complete!"
echo ""

# Generate paper outputs
echo "▶ Generating publication-ready outputs..."
python xpipe/scripts/generate_paper_outputs.py
echo ""

echo "======================================"
echo " ALL EXPERIMENTS COMPLETE!"
echo "======================================"
echo ""
echo "Results location:"
echo "  - Metrics: xpipe/output/*/metrics/*.csv"
echo "  - Figures: xpipe/output/*/figures/*.pdf"
echo "  - Tables:  xpipe/output/*/tables/*.tex"
echo ""
echo "Next steps:"
echo "  1. Review results in output/ directories"
echo "  2. Check HTML summary reports"
echo "  3. Begin paper writing!"
echo ""
