#!/usr/bin/env python3
"""
Sweep script to run pegasus verkle tree simulations with increasing sizes
across multiple width configurations. Results are appended to a JSONL file
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
        default="../../../../new/lagrange_multiverkle/dimension_benchmark.jsonl",
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
        "--widths",
        type=str,
        default="64",
        help="Comma-separated list of widths to test (e.g., '2,4,8')",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=64,
        help="Base size to start with (sizes will be base_size^exp)",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=1_000_000,
        help="Exclusive upper bound for sampled integers",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="new_verkle_demo_metrics.py",
        help="Path to the simulation script (relative to script dir)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="Conda environment to activate (e.g., 'pegasus')",
    )
    return parser.parse_args()


def run_simulation(
    size: int,
    seed: int,
    width: int,
    max_value: int,
    script_path: str,
    conda_env: str = None,
) -> dict:
    """Run a single simulation and return normalized metrics."""
    print(f"\n{'='*60}")
    print(f"Running simulation with size: {size}, seed: {seed}, width: {width}")
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
            "--size",
            str(size),
            "--seed",
            str(seed),
            "--width",
            str(width),
            "--max-value",
            str(max_value),
        ]
    else:
        cmd = [
            sys.executable,
            str(script_path),
            "--size",
            str(size),
            "--seed",
            str(seed),
            "--width",
            str(width),
            "--max-value",
            str(max_value),
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
            f"Error running simulation for size {size}, seed {seed}, width {width}:",
            file=sys.stderr,
        )
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

    # Read the metrics from the metrics directory
    metrics_dir = script_dir / "metrics"
    # The script saves as: metrics_width{width}_size{size}.json
    metrics_filename = f"metrics_width{width}_size{size}.json"
    metrics_path = metrics_dir / metrics_filename

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file {metrics_path} not found after simulation"
        )

    with open(metrics_path, "r") as f:
        pegasus_metrics = json.load(f)

    # Normalize to match multiverkle format
    normalized = {
        "implementation": "pegasus",
        "size": pegasus_metrics["parameters"]["size"],
        "seed": pegasus_metrics["parameters"]["seed"],
        "width": pegasus_metrics["parameters"]["width"],
        "arity_label": f"width-{width}",
        # Normalize timing fields
        "time_taken_to_build_tree": pegasus_metrics["timings_seconds"]["tree_build"],
        "time_taken_to_generate_proof": pegasus_metrics["timings_seconds"][
            "proof_generation"
        ],
        "time_taken_to_verify_proof": pegasus_metrics["timings_seconds"][
            "proof_verification"
        ],
        "verification_passed": pegasus_metrics["proof_verification_result"],
        # Keep additional pegasus-specific fields
        "proof_size": pegasus_metrics["sizes_bytes"].get("proof_estimated", None),
        "tree_size": pegasus_metrics["sizes_bytes"].get("tree_estimated", None),
        "proof_node_count": pegasus_metrics.get("proof_node_count", None),
        "challenge_index": pegasus_metrics.get("challenge_indices", None),
        "challenge_value": pegasus_metrics.get("challenge_values", None),
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

    # Parse widths
    try:
        widths = [int(w.strip()) for w in args.widths.split(",") if w.strip()]
        if not widths:
            raise ValueError("No valid widths provided")
    except ValueError as e:
        print(f"Error: Invalid widths format: {args.widths}. {e}", file=sys.stderr)
        sys.exit(1)

    # Generate sizes: base_size^min_exp to base_size^max_exp
    sizes = [args.base_size ** exp for exp in range(args.min_exp, args.max_exp + 1)]

    total_simulations = len(sizes) * len(seeds) * len(widths)
    print(f"Starting sweep: {total_simulations} total simulations")
    print(f"  - {len(sizes)} different sizes: {sizes[0]} to {sizes[-1]}")
    print(f"  - {len(seeds)} different seeds: {seeds}")
    print(f"  - {len(widths)} different widths: {widths}")
    print(f"Output file: {output_path}")
    if args.conda_env:
        print(f"Conda environment: {args.conda_env}")

    results = []
    simulation_count = 0

    # Run all sizes for each seed and width (allows incremental saving)
    for width_idx, width in enumerate(widths, 1):
        print(f"\n{'#'*60}")
        print(f"Running width {width} ({width_idx}/{len(widths)})")
        print(f"{'#'*60}")

        for seed_idx, seed in enumerate(seeds, 1):
            print(
                f"\n{'-'*40}\nSeed {seed} ({seed_idx}/{len(seeds)}) for width {width}\n{'-'*40}"
            )
            for size_idx, size in enumerate(sizes, 1):
                simulation_count += 1
                print(
                    f"[{simulation_count}/{total_simulations}] "
                    f"Size: {size} (size {size_idx}/{len(sizes)}), "
                    f"Seed: {seed}, Width: {width}"
                )
                try:
                    metrics = run_simulation(
                        size,
                        seed,
                        width,
                        args.max_value,
                        str(script_path),
                        args.conda_env,
                    )
                    results.append(metrics)

                    # Append to output file immediately
                    with open(output_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")

                    print(
                        f"✓ Completed size {size}, seed {seed}, "
                        f"width {width}, saved to {output_path}"
                    )

                except Exception as e:
                    print(
                        f"✗ Failed for size {size}, seed {seed}, width {width}: {e}",
                        file=sys.stderr,
                    )
                    # Continue with next simulation
                    continue
        print(results)                
    print(f"\n{'='*60}")
    print(f"Sweep completed!")
    print(f"Total successful simulations: {len(results)}/{total_simulations}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
