#!/usr/bin/env python3
"""
Unified runner for layer selection workflows.

Subcommands:
- extract:   run layer significance extraction
- visualize: render score/metric plots
- optimize:  solve MV-BCS layer assignment
- benchmark: run full objective benchmark pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INTEGRATION_DIR = ROOT / "layer_selection_integration"
OPTIMIZATION_DIR = ROOT / "OptimizationModule"
BENCHMARK_DIR = ROOT / "benchmark"


def run_command(cmd: list[str]) -> None:
    print(f"[layerSelectionLib] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Extract per-layer significance scores from AlphaPruning metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--output", required=True, help="Output JSON path for extracted scores.")
    parser.add_argument("--ww-metric", default="alpha_peak", help="ESD metric name.")
    parser.add_argument(
        "--cache-dir",
        default=str(INTEGRATION_DIR / "llm_weights"),
        help="Directory for model weight cache.",
    )
    parser.add_argument(
        "--ww-metric-cache",
        default=None,
        help="Directory for metric cache. If omitted, defaults to AlphaPruning/data/<model>/.",
    )
    parser.add_argument("--sparsity-ratio", type=float, default=0.85, help="Target sparsity ratio.")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Pruning ratio spread factor.")
    parser.add_argument(
        "--mapping-type",
        choices=["block_wise", "layer_wise"],
        default="block_wise",
        help="Metric-to-layer mapping strategy.",
    )
    parser.add_argument(
        "--significance-method",
        choices=["inverse_ratio"],
        default="inverse_ratio",
        help="Method for converting pruning ratios to significance scores.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write pretty-printed JSON output.")


def add_visualize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help="Visualize significance scores from JSON outputs or AlphaPruning metric files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["data_dir", "json", "json_single"],
        default="data_dir",
        help="Visualization mode.",
    )
    parser.add_argument("--output", required=True, help="Output plot path.")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "AlphaPruning" / "data"),
        help="Metric data directory for data_dir mode.",
    )
    parser.add_argument("--metric", default="alpha_peak", help="Metric filename stem for data_dir mode.")
    parser.add_argument(
        "--json-file",
        nargs="+",
        default=None,
        help="One or more JSON score files for json/json_single modes.",
    )
    parser.add_argument("--max-models", type=int, default=None, help="Max models to plot in data_dir mode.")
    parser.add_argument("--figsize", default="16,8", help='Figure size as "width,height".')
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Normalize plotted scores.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Use unnormalized plotted scores.",
    )
    parser.add_argument(
        "--use-raw-metrics",
        action="store_true",
        default=False,
        help="Visualize inverted raw ESD metrics directly.",
    )


def add_optimize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "optimize",
        help="Run MV-BCS optimizer on extracted layer scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", required=True, help="Path to layer score JSON file.")
    parser.add_argument(
        "--budgets",
        required=True,
        nargs="+",
        type=float,
        help="Space-separated verifier budgets.",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")


def add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Run full benchmark pipeline (cases + norms + benchmark + plotting).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--scores-file", required=True, help="Score JSON path for this model.")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of budget cases.")
    parser.add_argument("--num-verifiers", type=int, default=3, help="Number of verifiers.")
    parser.add_argument(
        "--cache-dir",
        default=str(INTEGRATION_DIR / "llm_weights"),
        help="Model cache directory for norm extraction.",
    )
    parser.add_argument("--skip-norms", action="store_true", help="Skip weight norm extraction.")
    parser.add_argument("--skip-cases", action="store_true", help="Skip budget case generation.")


def run_extract(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(INTEGRATION_DIR / "extract_layer_scores.py"),
        "--model",
        args.model,
        "--ww_metric",
        args.ww_metric,
        "--cache_dir",
        args.cache_dir,
        "--sparsity_ratio",
        str(args.sparsity_ratio),
        "--epsilon",
        str(args.epsilon),
        "--mapping_type",
        args.mapping_type,
        "--significance_method",
        args.significance_method,
        "--output",
        args.output,
    ]
    if args.ww_metric_cache:
        cmd.extend(["--ww_metric_cache", args.ww_metric_cache])
    if args.pretty:
        cmd.append("--pretty")
    run_command(cmd)


def run_visualize(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(INTEGRATION_DIR / "visualize_scores.py"),
        "--mode",
        args.mode,
        "--data_dir",
        args.data_dir,
        "--metric",
        args.metric,
        "--output",
        args.output,
        "--figsize",
        args.figsize,
    ]
    if args.max_models is not None:
        cmd.extend(["--max_models", str(args.max_models)])
    if args.json_file:
        cmd.extend(["--json_file", *args.json_file])
    if not args.normalize:
        cmd.append("--no_normalize")
    if args.use_raw_metrics:
        cmd.append("--use_raw_metrics")
    run_command(cmd)


def run_optimize(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "OptimizationModule.optimize_layers",
        "--scores",
        args.scores,
        "--budgets",
        *[str(budget) for budget in args.budgets],
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    if args.pretty:
        cmd.append("--pretty")
    run_command(cmd)


def run_benchmark(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(BENCHMARK_DIR / "run_full_benchmark.py"),
        "--model",
        args.model,
        "--scores_file",
        args.scores_file,
        "--num_cases",
        str(args.num_cases),
        "--num_verifiers",
        str(args.num_verifiers),
        "--cache_dir",
        args.cache_dir,
    ]
    if args.skip_norms:
        cmd.append("--skip_norms")
    if args.skip_cases:
        cmd.append("--skip_cases")
    run_command(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified runner for layerSelectionLib workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_extract_parser(subparsers)
    add_visualize_parser(subparsers)
    add_optimize_parser(subparsers)
    add_benchmark_parser(subparsers)

    args = parser.parse_args()

    if args.command == "extract":
        run_extract(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "optimize":
        run_optimize(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
