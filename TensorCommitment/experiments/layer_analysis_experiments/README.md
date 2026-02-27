# Layer Analysis

This folder contains scripts for analyzing layer significance scores across different LLM architectures using ESD (Empirical Spectral Distribution) metrics.

## Structure

```
layer_analysis/
├── lib/                    # ESD utilities from AlphaPruning
│   ├── esd_utils.py        # Core ESD metric computation
│   ├── prune.py            # Layer finding utilities
│   └── ...
├── data/                   # Pre-computed .npy metric files
│   ├── llama2-7b-hf/
│   │   └── alpha_peak.npy
│   ├── opt-125m/
│   │   └── alpha_peak.npy
│   └── ...
├── scores/                 # Pre-computed JSON significance scores
│   ├── llama2-7b_scores.json
│   ├── opt125m_scores.json
│   └── ...
├── results/                # Output plots
├── extract_layer_scores.py # Extract significance scores from metrics
├── visualize_scores.py     # Generate plots from scores/metrics
└── run_layer_analysis.sh   # Main runner script
```

## Quick Start

```bash
# Generate all plots
./run_layer_analysis.sh
```

## Output

The script generates:
- **Individual model plots**: `results/<model>_plot.png` - Bar charts of per-layer significance scores
- **Combined comparison**: `results/all_models_comparison.png` - Quarter-based boxplot comparison across all models

## Models Included

- OPT-125M
- LLaMA (7B, 13B, 30B, 65B)
- LLaMA-2 (7B, 13B, 70B)

## Dependencies

- numpy
- matplotlib
- seaborn
- (optional) transformers, torch - only needed if computing new metrics
