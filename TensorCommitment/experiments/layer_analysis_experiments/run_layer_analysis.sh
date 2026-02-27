#!/bin/bash
# Run layer analysis pipeline to generate all plots
# 
# This script generates:
# - Individual model plots in results/
# - Combined all_models_comparison.png
#
# Usage: ./run_layer_analysis.sh
# Environment: Requires alphaprun conda environment (or testingICML with matplotlib/seaborn)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Layer Analysis Pipeline"
echo "============================================================"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Ensure results directory exists
mkdir -p results

# Generate individual model plots from JSON scores
echo "Generating individual model plots from pre-computed scores..."
echo ""

for scores_file in scores/*.json; do
    model_name=$(basename "$scores_file" _scores.json)
    output_file="results/${model_name}_plot.png"
    echo "  Processing: $model_name"
    python visualize_scores.py \
        --mode json_single \
        --json_file "$scores_file" \
        --output "$output_file"
done

echo ""
echo "Generating combined comparison plot from .npy data..."
python visualize_scores.py \
    --mode data_dir \
    --data_dir data/ \
    --metric alpha_peak \
    --output results/all_models_comparison.png

echo ""
echo "============================================================"
echo "Done! Plots saved to: $SCRIPT_DIR/results/"
echo "============================================================"
ls -la results/
