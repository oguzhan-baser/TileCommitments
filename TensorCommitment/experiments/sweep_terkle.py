#!/usr/bin/env python3
"""
Sweep script to run Terkle tree benchmarks.

This script runs the Rust benchmark binary for Terkle (multivariate Verkle tree).
The Rust binary handles the benchmark configurations internally and outputs
results in JSONL format.

Before running, ensure the terkleLib has been built:
    cd ../terkleLib && cargo build --release
"""
import subprocess
import argparse
import sys
import os
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="results/dimension_benchmark.jsonl",
        help="Output file to append results to (JSONL format)",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default="../terkleLib/target/release/benchmark_dimensions",
        help="Path to the Rust benchmark binary (relative to script dir)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the Rust binary before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    binary_path = (script_dir / args.binary).resolve()
    output_path = (script_dir / args.output).resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally build the binary
    if args.build:
        print("Building terkleLib...")
        terkle_dir = (script_dir / "../terkleLib").resolve()
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=terkle_dir,
            check=True,
        )
        print("✓ Build complete")

    # Check if binary exists
    if not binary_path.exists():
        print(f"Error: Benchmark binary not found: {binary_path}", file=sys.stderr)
        print("Run with --build to build it first, or run:", file=sys.stderr)
        print(f"  cd {script_dir / '../terkleLib'} && cargo build --release", file=sys.stderr)
        sys.exit(1)

    print(f"Running Terkle benchmark...")
    print(f"Binary: {binary_path}")
    print(f"Output: {output_path}")

    # Run the benchmark binary
    # The Rust binary expects positional args: [output_file] [num_seeds] [iterations_per_seed]
    cmd = [str(binary_path), str(output_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"✓ Terkle benchmark complete, results saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
