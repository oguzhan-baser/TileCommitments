#!/usr/bin/env python3
"""
Master script to run the complete benchmarking pipeline.

This script orchestrates:
1. Generating budget cases
2. Extracting weight norms
3. Running benchmarks
4. Plotting results
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete benchmarking pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model name (e.g., facebook/opt-125m)')
    parser.add_argument('--scores_file', type=str, required=True,
                       help='Path to scores JSON file')
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
    parser.add_argument('--benchmark_dir', type=str, default=None,
                       help='Benchmark directory (default: script location)')
    
    args = parser.parse_args()
    
    # Determine benchmark directory: use script's location by default
    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir).resolve()
    else:
        # Use the directory where this script is located
        benchmark_dir = Path(__file__).parent.absolute()
    
    cases_dir = benchmark_dir / "cases"
    norms_dir = benchmark_dir / "weight_norms"
    results_dir = benchmark_dir / "results"
    
    # Add benchmark directory to path to import utilities
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))
    
    from model_utils import get_model_short_name
    
    # Extract model short name for file naming (consistent with existing files)
    model_short = get_model_short_name(args.model)
    
    # Step 1: Generate budget cases
    if not args.skip_cases:
        cases_script = benchmark_dir / "generate_budget_cases.py"
        run_command(
            [sys.executable, str(cases_script),
             "--num_cases", str(args.num_cases),
             "--num_verifiers", str(args.num_verifiers),
             "--output_dir", str(cases_dir)],
            "Generating budget cases"
        )
    else:
        print("Skipping budget case generation (using existing cases)")
    
    # Step 2: Extract weight norms
    norms_file = norms_dir / f"{model_short}_norms.json"
    if not args.skip_norms:
        norms_script = benchmark_dir / "extract_weight_norms.py"
        # Use default cache_dir if not specified
        cache_dir_arg = args.cache_dir if args.cache_dir else str(
            benchmark_dir.parent / "layerSelectionLib" / "layer_selection_integration" / "llm_weights"
        )
        
        # Build command - the script will automatically skip if file exists
        cmd = [
            sys.executable, str(norms_script),
            "--model", args.model,
            "--scores_file", args.scores_file,
            "--output", str(norms_file),
            "--cache_dir", cache_dir_arg
        ]
        
        # Note: extract_weight_norms.py will skip if file exists (unless --force is used)
        run_command(cmd, "Extracting weight norms")
    else:
        print(f"Skipping weight norm extraction (--skip_norms flag set)")
    
    # Step 3: Run benchmarks
    benchmark_script = benchmark_dir / "run_benchmarks.py"
    run_command(
        [sys.executable, str(benchmark_script),
         "--scores", args.scores_file,
         "--norms", str(norms_file),
         "--cases_dir", str(cases_dir),
         "--output_dir", str(results_dir),
         "--model_name", model_short],
        "Running optimization benchmarks"
    )
    
    # Step 4: Plot results
    plot_script = benchmark_dir / "plot_results.py"
    results_summary = results_dir / f"{model_short}_all_results.json"
    plot_output = results_dir / f"{model_short}_comparison.png"
    
    run_command(
        [sys.executable, str(plot_script),
         "--results", str(results_summary),
         "--output", str(plot_output)],
        "Plotting results"
    )
    
    print(f"\n{'='*70}")
    print("BENCHMARK PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Plot saved to: {plot_output}")
    print(f"\nTo view results:")
    print(f"  - Individual case results: {results_dir}/case_XXX_results.json")
    print(f"  - Summary: {results_summary}")
    print(f"  - Plot: {plot_output}")


if __name__ == '__main__':
    main()

