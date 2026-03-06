#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize commitment/proof/verification size and timing metrics for one pipeline run."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under activationCaptureLib/output/")
    parser.add_argument(
        "--layer-summary",
        type=Path,
        required=True,
        help="Path to compute_crypto_verification_summary.json for the selected layer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <run-dir>/run_metrics.json)",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_bytes_per_element(dtype_text: str) -> int:
    text = dtype_text.lower()
    if "float16" in text or "bfloat16" in text:
        return 2
    if "float32" in text:
        return 4
    if "float64" in text:
        return 8
    return 4


def product(values: List[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def find_commitment_dir(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*_int_activations_hypercube_commitment"))
    if not candidates:
        raise FileNotFoundError(f"No *_int_activations_hypercube_commitment directory found in: {run_dir}")
    return candidates[0]


def find_first_proof_bundle(layer_summary_path: Path) -> Path | None:
    proof_dir = layer_summary_path.parent / "proof_bundles"
    bundles = sorted(proof_dir.glob("proof_*.json"))
    if not bundles:
        return None
    return bundles[0]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    layer_summary_path = args.layer_summary.resolve()
    output_path = args.output.resolve() if args.output else (run_dir / "run_metrics.json").resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not layer_summary_path.is_file():
        raise FileNotFoundError(f"Layer summary file not found: {layer_summary_path}")

    commitment_dir = find_commitment_dir(run_dir)
    commitment_results_path = commitment_dir / "commitment_results.json"
    commitment_file_path = commitment_dir / "commitment.txt"
    if not commitment_results_path.is_file():
        raise FileNotFoundError(f"commitment_results.json not found: {commitment_results_path}")
    if not commitment_file_path.is_file():
        raise FileNotFoundError(f"commitment.txt not found: {commitment_file_path}")

    commitment_results = read_json(commitment_results_path)
    layer_summary = read_json(layer_summary_path)

    timing = commitment_results.get("timing", {})
    proof_stats = commitment_results.get("proof_stats", {})
    commitment_bytes = int(commitment_results.get("commitment_bytes", 0))
    commitment_file_bytes = int(commitment_file_path.stat().st_size)

    avg_opening_proof_bytes = float(proof_stats.get("avg_proof_bytes", 0.0))
    proof_elements_per_query = int(proof_stats.get("proof_elements_per_query", 0))

    layer_shape = [int(value) for value in layer_summary.get("compute_verification", {}).get("layer_shape", [])]
    dtype_text = str(layer_summary.get("inputs", {}).get("dtype", "torch.float32"))
    bytes_per_element = infer_bytes_per_element(dtype_text)
    layer_numel = product(layer_shape) if layer_shape else 0
    one_tensor_bytes = layer_numel * bytes_per_element
    input_plus_output_bytes = one_tensor_bytes * 2

    activations_pt = Path(str(layer_summary.get("inputs", {}).get("activations_pt", "")))
    activations_pt_bytes = int(activations_pt.stat().st_size) if activations_pt.is_file() else None

    first_proof_bundle = find_first_proof_bundle(layer_summary_path)
    sample_bundle_bytes = None
    sample_bundle_crypto_bytes = None
    sample_bundle_proof_elements = None
    proof_bundle_count = 0
    if first_proof_bundle is not None:
        proof_bundle_count = len(sorted((layer_summary_path.parent / "proof_bundles").glob("proof_*.json")))
        payload = read_json(first_proof_bundle)
        sample_bundle_proof_elements = len(payload.get("proof_hex", []))
        sample_bundle_crypto_bytes = sample_bundle_proof_elements * 32
        sample_bundle_bytes = int(first_proof_bundle.stat().st_size)

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "sizes": {
            "commitment": {
                "crypto_bytes": commitment_bytes,
                "file_bytes": commitment_file_bytes,
            },
            "opening_proof": {
                "avg_crypto_bytes": avg_opening_proof_bytes,
                "proof_elements_per_query": proof_elements_per_query,
                "sample_layer_bundle_crypto_bytes": sample_bundle_crypto_bytes,
                "sample_layer_bundle_file_bytes": sample_bundle_bytes,
                "sample_layer_bundle_proof_elements": sample_bundle_proof_elements,
                "sample_layer_bundle_count": proof_bundle_count,
            },
            "intermediate_activations_for_verification": {
                "layer_shape": layer_shape,
                "dtype": dtype_text,
                "bytes_per_element": bytes_per_element,
                "one_tensor_bytes": one_tensor_bytes,
                "input_plus_output_bytes": input_plus_output_bytes,
            },
            "activations_pt_file_bytes": activations_pt_bytes,
        },
        "timings_seconds": {
            "commit": float(timing.get("commit_time_s", 0.0)),
            "prove_tile_avg": float(timing.get("avg_prove_time_s", 0.0)),
            "verify_tile_avg": float(timing.get("avg_verify_time_s", 0.0)),
            "prove_total": float(timing.get("total_prove_time_s", 0.0)),
            "verify_total": float(timing.get("total_verify_time_s", 0.0)),
        },
        "source_files": {
            "commitment_results_json": str(commitment_results_path),
            "commitment_txt": str(commitment_file_path),
            "layer_summary_json": str(layer_summary_path),
            "sample_layer_proof_bundle_json": str(first_proof_bundle) if first_proof_bundle else None,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[metrics] wrote {output_path}")
    print(f"[metrics] commitment_bytes={commitment_bytes}")
    print(f"[metrics] opening_proof_avg_bytes={avg_opening_proof_bytes}")
    print(f"[metrics] commit_time_s={summary['timings_seconds']['commit']}")
    print(f"[metrics] prove_tile_avg_s={summary['timings_seconds']['prove_tile_avg']}")
    print(f"[metrics] verify_tile_avg_s={summary['timings_seconds']['verify_tile_avg']}")


if __name__ == "__main__":
    main()
