#!/usr/bin/env python3
"""
Sweep script to run merkle tree simulations with increasing sizes
across multiple arity configurations. Results are appended to a JSONL file
for downstream plotting/analysis. Normalizes output to match multiverkle format.
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
        default="../../terkle/new/lagrange_multiverkle/dimension_benchmark.jsonl",
        help="Output file to append results to (JSONL format, relative to script dir)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default='2,13,74',
        help="Comma-separated list of seeds (e.g., '42,43,44') or use --num-runs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs with different seeds (seeds will be 1..num_runs)",
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
        help="Base size to start with (sizes will be base_size^exp, used as --leaves)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="final_prover_verifier/run_demo.py",
        help="Path to the simulation script (relative to script dir)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="mmpmerkle",
        help="Conda environment to activate (e.g., 'mmpmerkle')",
    )
    return parser.parse_args()


def run_simulation(
    leaves: int,
    seed: int,
    arity: int,
    script_path: str,
    conda_env: str = None,
) -> dict:
    """Run a single simulation and return normalized metrics."""
    print(f"\n{'='*60}")
    print(f"Running simulation with leaves: {leaves}, seed: {seed}, arity: {arity}")
    print(f"{'='*60}")

    # Build command
    # If conda environment is specified, use conda run with python
    if conda_env:
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "--no-capture-output",
            "python",
            str(script_path),
            "--arity",
            str(arity),
            "--leaves",
            str(leaves),
            "--seed",
            str(seed),
        ]
    else:
        cmd = [
            sys.executable,
            str(script_path),
            "--arity",
            str(arity),
            "--leaves",
            str(leaves),
            "--seed",
            str(seed),
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=script_dir,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(
            f"Error running simulation for leaves {leaves}, seed {seed}, arity {arity}:",
            file=sys.stderr,
        )
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

    # Read the metrics from the artifacts directory
    # The script saves to: artifacts_{arity}_{leaf_count}/metrics.json
    artifacts_dir = script_dir / "final_prover_verifier" / f"artifacts_{arity}_{leaves}"
    metrics_path = artifacts_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file {metrics_path} not found after simulation"
        )

    with open(metrics_path, "r") as f:
        merkle_metrics = json.load(f)

    # Normalize to match multiverkle format
    normalized = {
        "implementation": "merkle",
        "size": merkle_metrics["parameters"]["leaf_count"],
        "seed": merkle_metrics["parameters"]["seed"],
        "arity": merkle_metrics["parameters"]["arity"],
        "arity_label": f"width-{arity}",
        # Normalize timing fields
        "time_taken_to_build_tree": merkle_metrics["timings_seconds"]["tree_generation"],
        "time_taken_to_generate_proof": merkle_metrics["timings_seconds"][
            "proof_generation"
        ],
        "time_taken_to_verify_proof": merkle_metrics["timings_seconds"][
            "verification"
        ],
        "verification_passed": True,  # Script raises error if verification fails
        # Keep additional merkle-specific fields
        "commit_time": merkle_metrics["timings_seconds"].get("commit", None),
        "proof_size": merkle_metrics["artifact_sizes_bytes"].get("proof", None),
        "tree_size": merkle_metrics["artifact_sizes_bytes"].get("tree", None),
        "proof_index": merkle_metrics["proof_metadata"].get("index", None),
        "proof_value": merkle_metrics["proof_metadata"].get("value", None),
    }

    return normalized


def main() -> None:
    args = parse_args()

    # Resolve script path
    script_path = script_dir / args.script
    if not script_path.exists():
        print(
            f"Error: Simulation script not found: {script_path}", file=sys.stderr
        )
        sys.exit(1)

    # Resolve output path (can be absolute or relative)
    if os.path.isabs(args.output):
        output_path = Path(args.output)
    else:
        output_path = script_dir / args.output
    output_path = output_path.resolve()

    # Determine seeds to use
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            print(
                f"Error: Invalid seeds format: {args.seeds}. Use comma-separated integers.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        seeds = list(range(1, args.num_runs + 1))

    # Parse arities
    try:
        arities = [int(a.strip()) for a in args.arities.split(",") if a.strip()]
        if not arities:
            raise ValueError("No valid arities provided")
    except ValueError as e:
        print(f"Error: Invalid arities format: {args.arities}. {e}", file=sys.stderr)
        sys.exit(1)

    # Generate sizes: base_size^min_exp to base_size^max_exp
    sizes = [args.base_size ** exp for exp in range(args.min_exp, args.max_exp + 1)]

    total_simulations = len(sizes) * len(seeds) * len(arities)
    print(f"Starting sweep: {total_simulations} total simulations")
    print(f"  - {len(sizes)} different sizes: {sizes[0]} to {sizes[-1]}")
    print(f"  - {len(seeds)} different seeds: {seeds}")
    print(f"  - {len(arities)} different arities: {arities}")
    print(f"Output file: {output_path}")
    if args.conda_env:
        print(f"Conda environment: {args.conda_env}")

    results = []
    simulation_count = 0

    # Run all sizes for each seed and arity (allows incremental saving)
    for arity_idx, arity in enumerate(arities, 1):
        print(f"\n{'#'*60}")
        print(f"Running arity {arity} ({arity_idx}/{len(arities)})")
        print(f"{'#'*60}")

        for seed_idx, seed in enumerate(seeds, 1):
            print(
                f"\n{'-'*40}\nSeed {seed} ({seed_idx}/{len(seeds)}) for arity {arity}\n{'-'*40}"
            )
            for size_idx, size in enumerate(sizes, 1):
                simulation_count += 1
                print(
                    f"[{simulation_count}/{total_simulations}] "
                    f"Size: {size} (size {size_idx}/{len(sizes)}), "
                    f"Seed: {seed}, Arity: {arity}"
                )
                try:
                    metrics = run_simulation(
                        size,
                        seed,
                        arity,
                        str(script_path),
                        args.conda_env,
                    )
                    results.append(metrics)

                    # Append to output file immediately
                    with open(output_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")

                    print(
                        f"✓ Completed size {size}, seed {seed}, "
                        f"arity {arity}, saved to {output_path}"
                    )

                except Exception as e:
                    print(
                        f"✗ Failed for size {size}, seed {seed}, arity {arity}: {e}",
                        file=sys.stderr,
                    )
                    # Continue with next simulation
                    continue

    print(f"\n{'='*60}")
    print(f"Sweep completed!")
    print(f"Total successful simulations: {len(results)}/{total_simulations}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

