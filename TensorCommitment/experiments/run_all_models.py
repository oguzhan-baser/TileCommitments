#!/usr/bin/env python3
"""
Run benchmarks for all models and create combined plots.

This script runs the full benchmarking pipeline for multiple models
and creates both individual and combined visualizations.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import os

# Model configuration: (score_file, huggingface_name, short_name)
# Note: Model names must match the cached models in layer_selection_integration/llm_weights
# HuggingFace cache uses: models--org--model-name format
MODELS = [
    ("llama2-70b_scores.json", "meta-llama/Llama-2-70b-hf", "llama2_70b"),
    ("llama2-13b_scores.json", "meta-llama/Llama-2-13b-hf", "llama2_13b"),
    ("llama2-7b_scores.json", "meta-llama/Llama-2-7b-hf", "llama2_7b"),  # Fixed: added -hf suffix
    ("opt125m_scores.json", "facebook/opt-125m", "opt_125m"),
]


def run_model_benchmark(score_file, model_name, short_name, scores_dir, benchmark_dir, 
                        num_cases, num_verifiers, cache_dir, skip_norms=False, skip_cases=False):
    """Run benchmark for a single model."""
    print(f"\n{'='*70}")
    print(f"Processing model: {model_name}")
    print(f"Score file: {score_file}")
    print(f"{'='*70}\n")
    
    score_path = Path(scores_dir) / score_file
    if not score_path.exists():
        print(f"Warning: Score file not found: {score_path}")
        print(f"Skipping {model_name}")
        return False
    
    # Determine cache directory (use default if not specified)
    if cache_dir is None:
        cache_dir = str(Path(benchmark_dir).parent / "layerSelectionLib" / "layer_selection_integration" / "llm_weights")
    
    print(f"Using cache directory: {cache_dir}")
    
    # Run the full benchmark pipeline
    run_script = Path(benchmark_dir) / "run_full_benchmark.py"
    
    cmd = [
        sys.executable, str(run_script),
        "--model", model_name,
        "--scores_file", str(score_path),
        "--num_cases", str(num_cases),
        "--num_verifiers", str(num_verifiers),
        "--cache_dir", cache_dir,
    ]
    
    if skip_norms:
        cmd.append("--skip_norms")
    if skip_cases:
        cmd.append("--skip_cases")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: Benchmark failed for {model_name}")
        return False
    
    print(f"✓ Completed benchmark for {model_name}\n")
    return True


def create_combined_plot(benchmark_dir, results_dir, output_file):
    """Create combined plot from all model results."""
    print(f"\n{'='*70}")
    print("Creating combined plot")
    print(f"{'='*70}\n")
    
    # Find all results files
    results_path = Path(results_dir)
    results_files = list(results_path.glob("*_all_results.json"))
    
    if len(results_files) == 0:
        print(f"No results files found in {results_dir}")
        return False
    
    print(f"Found {len(results_files)} result files:")
    for f in results_files:
        print(f"  - {f.name}")
    
    # Run plotting script
    plot_script = Path(benchmark_dir) / "plot_results.py"
    
    cmd = [
        sys.executable, str(plot_script),
        "--results", str(results_dir),
        "--output", str(output_file),
        "--combined"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: Plotting failed")
        return False
    
    print(f"✓ Combined plot saved to {output_file}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks for all models and create combined plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--scores_dir', type=str, default='../layerSelectionLib/scores',
                       help='Directory containing score files')
    parser.add_argument('--benchmark_dir', type=str, default=None,
                       help='Benchmark directory (default: script location)')
    parser.add_argument('--num_cases', type=int, default=100,
                       help='Number of budget cases to generate')
    parser.add_argument('--num_verifiers', type=int, default=3,
                       help='Number of verifiers')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model weights (default: ../layer_selection_integration/llm_weights)')
    parser.add_argument('--skip_norms', action='store_true',
                       help='Skip weight norm extraction (use existing)')
    parser.add_argument('--skip_cases', action='store_true',
                       help='Skip budget case generation (use existing)')
    parser.add_argument('--skip_individual', action='store_true',
                       help='Skip individual model benchmarks (only create combined plot)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for combined plot (default: results/combined_comparison.png)')
    
    args = parser.parse_args()
    
    # Determine benchmark directory
    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir).resolve()
    else:
        benchmark_dir = Path(__file__).parent.absolute()
    
    results_dir = benchmark_dir / "results"
    scores_dir = Path(args.scores_dir).resolve()
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = results_dir / "combined_comparison.png"
    
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Scores directory: {scores_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    
    # Run benchmarks for each model
    if not args.skip_individual:
        successful_models = []
        for score_file, model_name, short_name in MODELS:
            success = run_model_benchmark(
                score_file, model_name, short_name,
                scores_dir, benchmark_dir,
                args.num_cases, args.num_verifiers, args.cache_dir,
                args.skip_norms, args.skip_cases
            )
            if success:
                successful_models.append(short_name)
        
        print(f"\n{'='*70}")
        print(f"Completed benchmarks for {len(successful_models)}/{len(MODELS)} models")
        print(f"{'='*70}\n")
    
    # Create combined plot
    create_combined_plot(benchmark_dir, results_dir, output_file)
    
    print(f"\n{'='*70}")
    print("ALL MODELS BENCHMARK COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Combined plot: {output_file}")
    print(f"\nIndividual model results:")
    for score_file, model_name, short_name in MODELS:
        result_file = results_dir / f"{short_name}_all_results.json"
        if result_file.exists():
            print(f"  ✓ {short_name}: {result_file}")


if __name__ == '__main__':
    main()

