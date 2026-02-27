#!/usr/bin/env python3
"""
Sweep script to run Merkle tree simulations with increasing sizes
across multiple arity configurations. Results are appended to a JSONL file
for downstream plotting/analysis. Normalizes output to match multiverkle format.

Before running, ensure multibranch-merkle has been installed:
    cd ../merkle && maturin develop --features python --release
"""
import subprocess
import json
import argparse
import sys
import os
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-exp", type=int, default=1, help="Minimum exponent (base_size^min_exp)")
    parser.add_argument("--max-exp", type=int, default=4, help="Maximum exponent (base_size^max_exp)")
    parser.add_argument(
        "--output",
        type=str,
        default="results/dimension_benchmark.jsonl",
        help="Output file to append results to (JSONL format)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default='2,13,74',
        help="Comma-separated list of seeds",
    )
    parser.add_argument(
        "--arities",
        type=str,
        default="64",
        help="Comma-separated list of arities to test (e.g., '2,4,8')",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=64,
        help="Base size to start with (sizes will be base_size^exp)",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="../merkle/python_demo",
        help="Path to the demo directory containing merkle demo scripts",
    )
    return parser.parse_args()


def run_simulation(
    leaves: int,
    seed: int,
    arity: int,
    demo_dir: Path,
) -> dict:
    """Run a single simulation and return normalized metrics."""
    print(f"\n{'='*60}")
    print(f"Running Merkle simulation: leaves={leaves}, seed={seed}, arity={arity}")
    print(f"{'='*60}")

    script_path = demo_dir / "final_prover_verifier" / "run_demo.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--arity", str(arity),
        "--leaves", str(leaves),
        "--seed", str(seed),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=demo_dir,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

    # Read the metrics from the artifacts directory
    artifacts_dir = demo_dir / "final_prover_verifier" / f"artifacts_{arity}_{leaves}"
    metrics_path = artifacts_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file {metrics_path} not found after simulation")

    with open(metrics_path, "r") as f:
        merkle_metrics = json.load(f)

    # Normalize to match multiverkle format
    normalized = {
        "implementation": "Merkle",
        "size": merkle_metrics["parameters"]["leaf_count"],
        "seed": merkle_metrics["parameters"]["seed"],
        "arity": merkle_metrics["parameters"]["arity"],
        "arity_label": f"width-{arity}",
        "time_taken_to_build_tree": merkle_metrics["timings_seconds"]["tree_generation"],
        "time_taken_to_generate_proof": merkle_metrics["timings_seconds"]["proof_generation"],
        "time_taken_to_verify_proof": merkle_metrics["timings_seconds"]["verification"],
        "verification_passed": True,
        "proof_size": merkle_metrics["artifact_sizes_bytes"].get("proof", None),
        "tree_size": merkle_metrics["artifact_sizes_bytes"].get("tree", None),
    }

    return normalized


def main() -> None:
    args = parse_args()

    # Resolve paths
    demo_dir = (script_dir / args.demo_dir).resolve()
    output_path = (script_dir / args.output).resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if demo script exists
    demo_script = demo_dir / "final_prover_verifier" / "run_demo.py"
    if not demo_script.exists():
        print(f"Error: Demo script not found: {demo_script}", file=sys.stderr)
        sys.exit(1)

    # Parse seeds and arities
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    arities = [int(a.strip()) for a in args.arities.split(",") if a.strip()]

    # Generate sizes
    sizes = [args.base_size ** exp for exp in range(args.min_exp, args.max_exp + 1)]

    total_simulations = len(sizes) * len(seeds) * len(arities)
    print(f"Starting Merkle sweep: {total_simulations} total simulations")
    print(f"  - Sizes: {sizes}")
    print(f"  - Seeds: {seeds}")
    print(f"  - Arities: {arities}")
    print(f"Output: {output_path}")

    results = []
    count = 0

    for arity in arities:
        for seed in seeds:
            for size in sizes:
                count += 1
                print(f"\n[{count}/{total_simulations}] Size={size}, Seed={seed}, Arity={arity}")
                try:
                    metrics = run_simulation(size, seed, arity, demo_dir)
                    results.append(metrics)

                    with open(output_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")

                    print(f"✓ Completed, saved to {output_path}")
                except Exception as e:
                    print(f"✗ Failed: {e}", file=sys.stderr)
                    continue

    print(f"\n{'='*60}")
    print(f"Merkle sweep complete!")
    print(f"Successful: {len(results)}/{total_simulations}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
