#!/usr/bin/env python3
"""
Master script to run all benchmark sweeps: Terkle, Verkle, and Merkle.

This script orchestrates running benchmarks for all three tree implementations
and consolidates results into a single JSONL file for plotting.

Usage:
    python run_all_sweeps.py                    # Run all with defaults
    python run_all_sweeps.py --terkle-only      # Run only Terkle
    python run_all_sweeps.py --verkle-only      # Run only Verkle
    python run_all_sweeps.py --merkle-only      # Run only Merkle
    python run_all_sweeps.py --plot             # Also generate plot after sweep

Prerequisites:
    Run ../install.sh to build and install all libraries first.
"""
import subprocess
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="results/dimension_benchmark.jsonl",
        help="Output file for all benchmark results (JSONL format)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing output file before running",
    )
    
    # Which implementations to run
    parser.add_argument("--terkle-only", action="store_true", help="Run only Terkle benchmarks")
    parser.add_argument("--verkle-only", action="store_true", help="Run only Verkle benchmarks")
    parser.add_argument("--merkle-only", action="store_true", help="Run only Merkle benchmarks")
    parser.add_argument("--no-terkle", action="store_true", help="Skip Terkle benchmarks")
    parser.add_argument("--no-verkle", action="store_true", help="Skip Verkle benchmarks")
    parser.add_argument("--no-merkle", action="store_true", help="Skip Merkle benchmarks")
    
    # Common sweep parameters
    parser.add_argument("--min-exp", type=int, default=1, help="Minimum exponent (base^min_exp)")
    parser.add_argument("--max-exp", type=int, default=4, help="Maximum exponent (base^max_exp)")
    parser.add_argument("--base-size", type=int, default=64, help="Base size for exponential scaling")
    parser.add_argument("--seeds", type=str, default="2,13,74", help="Comma-separated seeds")
    
    # Post-processing
    parser.add_argument("--plot", action="store_true", help="Generate plot after running sweeps")
    parser.add_argument("--plot-output", type=str, default="results/dimension_benchmark_plot.png", help="Plot output file")
    
    return parser.parse_args()


def run_terkle_sweep(args) -> bool:
    """Run Terkle (multivariate Verkle) benchmarks."""
    print("\n" + "="*70)
    print("Running TERKLE (Multivariate Verkle) Benchmarks")
    print("="*70)
    
    cmd = [
        sys.executable, "sweep_terkle.py",
        "--output", args.output,
        "--build",  # Build if needed
    ]
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Terkle sweep: {e}", file=sys.stderr)
        return False


def run_verkle_sweep(args) -> bool:
    """Run Verkle (univariate KZG) benchmarks."""
    print("\n" + "="*70)
    print("Running VERKLE (Univariate KZG) Benchmarks")
    print("="*70)
    
    cmd = [
        sys.executable, "sweep_verkle.py",
        "--output", args.output,
        "--min-exp", str(args.min_exp),
        "--max-exp", str(args.max_exp),
        "--base-size", str(args.base_size),
        "--seeds", args.seeds,
    ]
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Verkle sweep: {e}", file=sys.stderr)
        return False


def run_merkle_sweep(args) -> bool:
    """Run Merkle tree benchmarks."""
    print("\n" + "="*70)
    print("Running MERKLE Benchmarks")
    print("="*70)
    
    cmd = [
        sys.executable, "sweep_merkle.py",
        "--output", args.output,
        "--min-exp", str(args.min_exp),
        "--max-exp", str(args.max_exp),
        "--base-size", str(args.base_size),
        "--seeds", args.seeds,
    ]
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Merkle sweep: {e}", file=sys.stderr)
        return False


def generate_plot(args) -> bool:
    """Generate the benchmark comparison plot."""
    print("\n" + "="*70)
    print("Generating Benchmark Plot")
    print("="*70)
    
    cmd = [
        sys.executable, "plot_dimension_benchmark.py",
        "--input", args.output,
        "--output", args.plot_output,
        "--base-size", str(args.base_size),
    ]
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating plot: {e}", file=sys.stderr)
        return False


def main() -> None:
    args = parse_args()
    
    # Resolve output path
    output_path = (script_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output if requested
    if args.clear and output_path.exists():
        print(f"Clearing existing output file: {output_path}")
        output_path.unlink()
    
    # Determine which implementations to run
    run_terkle = True
    run_verkle = True
    run_merkle = True
    
    if args.terkle_only:
        run_verkle = run_merkle = False
    elif args.verkle_only:
        run_terkle = run_merkle = False
    elif args.merkle_only:
        run_terkle = run_verkle = False
    
    if args.no_terkle:
        run_terkle = False
    if args.no_verkle:
        run_verkle = False
    if args.no_merkle:
        run_merkle = False
    
    # Print configuration
    print("="*70)
    print("ICML Benchmark Sweep")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {output_path}")
    print(f"Parameters:")
    print(f"  - Size range: {args.base_size}^{args.min_exp} to {args.base_size}^{args.max_exp}")
    print(f"  - Seeds: {args.seeds}")
    print(f"Running:")
    print(f"  - Terkle: {'Yes' if run_terkle else 'No'}")
    print(f"  - Verkle: {'Yes' if run_verkle else 'No'}")
    print(f"  - Merkle: {'Yes' if run_merkle else 'No'}")
    print(f"  - Plot: {'Yes' if args.plot else 'No'}")
    print("="*70)
    
    # Track results
    results = {}
    
    # Run sweeps
    if run_merkle:
        results["merkle"] = run_merkle_sweep(args)
        
    if run_verkle:
        results["verkle"] = run_verkle_sweep(args)
    
    if run_terkle:
        results["terkle"] = run_terkle_sweep(args)

    # Generate plot if requested
    if args.plot:
        results["plot"] = generate_plot(args)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name.capitalize()}: {status}")
    
    print(f"\nResults saved to: {output_path}")
    if args.plot:
        plot_path = (script_dir / args.plot_output).resolve()
        print(f"Plot saved to: {plot_path}")
    
    print("="*70)
    
    # Exit with error if any sweep failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
