#!/usr/bin/env python3
"""
Plot existing TileCommitments verification artifacts without rerunning pipelines.

Inputs:
- commitment_results.json (from tensorCommitmentLib commit/prove/verify output)
- compute_crypto_verification_summary.json files (from layer verification runs)

Outputs:
- PNG plots into a target directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_commitment_results(commitment_results_path: Path, out_dir: Path) -> List[Path]:
    data = _load_json(commitment_results_path)
    point_results = data.get("point_results", [])
    if not point_results:
        raise RuntimeError(f"No point_results found in {commitment_results_path}")

    saved_paths: List[Path] = []

    query_labels = [f"q{i}" for i in range(len(point_results))]
    eval_times = [float(r.get("eval_time_s", 0.0)) for r in point_results]
    prove_times = [float(r.get("prove_time_s", 0.0)) for r in point_results]
    verify_times = [float(r.get("verify_time_s", 0.0)) for r in point_results]
    proof_bytes = [int(r.get("proof_bytes", 0)) for r in point_results]
    verified = [1 if bool(r.get("verified", False)) else 0 for r in point_results]
    gt_match = [1 if bool(r.get("ground_truth_match", False)) else 0 for r in point_results]

    x = list(range(len(point_results)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar([i - width for i in x], eval_times, width=width, label="eval_time_s", color="#4C72B0")
    ax.bar(x, prove_times, width=width, label="prove_time_s", color="#55A868")
    ax.bar([i + width for i in x], verify_times, width=width, label="verify_time_s", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.set_xlabel("Query")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Commitment Query Timings")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "commitment_query_timings.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    saved_paths.append(out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(query_labels, proof_bytes, color="#8172B2")
    axes[0].set_title("Proof Size by Query")
    axes[0].set_xlabel("Query")
    axes[0].set_ylabel("Proof bytes")

    axes[1].bar(query_labels, verified, label="verified", alpha=0.8, color="#55A868")
    axes[1].bar(query_labels, gt_match, label="ground_truth_match", alpha=0.6, color="#4C72B0")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_yticks([0, 1])
    axes[1].set_title("Verification Outcomes by Query")
    axes[1].set_xlabel("Query")
    axes[1].set_ylabel("Pass (0/1)")
    axes[1].legend()
    fig.tight_layout()
    out = out_dir / "commitment_proof_size_and_outcomes.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    saved_paths.append(out)

    return saved_paths


def _discover_compute_summaries(search_root: Path) -> List[Path]:
    return sorted(search_root.rglob("compute_crypto_verification_summary.json"))


def _extract_compute_row(summary_path: Path) -> Dict:
    data = _load_json(summary_path)
    run_name = summary_path.parent.name
    inputs = data.get("inputs", {})
    compute = data.get("compute_verification", {})
    proof_gen = data.get("proof_generation", {})
    crypto = data.get("crypto_verification", {})

    layer = str(inputs.get("layer", "unknown"))
    num_proofs = int(proof_gen.get("count", 0))
    approx_mismatch_count = int(proof_gen.get("approx_mismatch_count", 0))
    compute_matched = bool(compute.get("matched", False))
    all_verified = bool(crypto.get("all_verified", False))
    max_abs_diff = float(compute.get("max_abs_diff", 0.0))
    mean_abs_diff = float(compute.get("mean_abs_diff", 0.0))

    return {
        "run_name": run_name,
        "layer": layer,
        "num_proofs": num_proofs,
        "approx_mismatch_count": approx_mismatch_count,
        "compute_matched": 1 if compute_matched else 0,
        "all_verified": 1 if all_verified else 0,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def _plot_compute_summaries(summary_paths: List[Path], out_dir: Path) -> List[Path]:
    if not summary_paths:
        raise RuntimeError("No compute_crypto_verification_summary.json files found")

    rows = [_extract_compute_row(p) for p in summary_paths]
    rows.sort(key=lambda r: (r["layer"], r["run_name"]))

    labels = [f"{r['run_name']}\n(layer={r['layer']})" for r in rows]
    x = list(range(len(rows)))

    saved_paths: List[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 1.2), 10), sharex=True)
    axes[0].bar(x, [r["max_abs_diff"] for r in rows], color="#C44E52", alpha=0.9, label="max_abs_diff")
    axes[0].bar(x, [r["mean_abs_diff"] for r in rows], color="#4C72B0", alpha=0.7, label="mean_abs_diff")
    axes[0].set_title("Compute Verification Error Metrics Across Runs")
    axes[0].set_ylabel("Absolute difference")
    axes[0].legend()

    axes[1].bar(x, [r["num_proofs"] for r in rows], color="#55A868", alpha=0.9, label="num_proofs")
    axes[1].bar(x, [r["approx_mismatch_count"] for r in rows], color="#8172B2", alpha=0.85, label="approx_mismatch_count")
    axes[1].set_title("Proof Counts and Approximate Mismatches Across Runs")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    out = out_dir / "compute_summary_error_and_proof_counts.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    saved_paths.append(out)

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.2), 5))
    ax.bar([i - 0.18 for i in x], [r["compute_matched"] for r in rows], width=0.36, color="#4C72B0", label="compute_matched")
    ax.bar([i + 0.18 for i in x], [r["all_verified"] for r in rows], width=0.36, color="#55A868", label="all_verified")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1])
    ax.set_title("Binary Verification Outcomes Across Runs")
    ax.set_ylabel("Pass (0/1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "compute_summary_binary_outcomes.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    saved_paths.append(out)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot existing commitment + compute/crypto verification artifacts."
    )
    parser.add_argument(
        "--commitment-results",
        type=Path,
        default=Path(
            "TensorCommitment/activationCaptureLib/output/"
            "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_commitment/"
            "commitment_results.json"
        ),
        help="Path to commitment_results.json",
    )
    parser.add_argument(
        "--compute-root",
        type=Path,
        default=Path("TensorCommitment/activationCaptureLib/output"),
        help="Root directory to recursively discover compute_crypto_verification_summary.json files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("TensorCommitment/activationCaptureLib/output/existing_results_plots"),
        help="Directory for generated plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sns.set_theme(style="darkgrid")

    out_dir = args.outdir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    commitment_results_path = args.commitment_results.resolve()
    compute_root = args.compute_root.resolve()

    if not commitment_results_path.exists():
        raise FileNotFoundError(f"commitment_results.json not found: {commitment_results_path}")
    if not compute_root.exists():
        raise FileNotFoundError(f"compute root not found: {compute_root}")

    saved: List[Path] = []
    saved.extend(_plot_commitment_results(commitment_results_path, out_dir))

    summary_paths = _discover_compute_summaries(compute_root)
    saved.extend(_plot_compute_summaries(summary_paths, out_dir))

    print("\nGenerated plots:")
    for p in saved:
        print(f"- {p}")
    print(f"\nTotal plots: {len(saved)}")


if __name__ == "__main__":
    main()

