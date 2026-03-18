#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import requests

from splitcompute.protocol import verify_prover_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Modal prover endpoint and verify returned proof bundles locally."
    )
    parser.add_argument("--prover-url", type=str, required=True, help="Base URL of deployed Modal prover API")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-factor", type=int, default=16)
    parser.add_argument("--quantize", type=float, default=50.0)
    parser.add_argument("--min-dim", type=int, default=4)
    parser.add_argument("--max-dim", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modalpatch/output/verifier_runs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = args.prover_url.rstrip("/") + "/prove"
    payload: Dict[str, Any] = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new_tokens),
        "num_queries": int(args.num_queries),
        "seed": int(args.seed),
        "scale_factor": int(args.scale_factor),
        "quantize": float(args.quantize),
        "min_dim": int(args.min_dim),
        "max_dim": int(args.max_dim),
        "skip_interp_build": True,
    }

    response = requests.post(endpoint, json=payload, timeout=7200)
    response.raise_for_status()
    prover_response = response.json()

    verification = verify_prover_response(prover_response)
    run_id = str(prover_response.get("run_id", "unknown"))

    response_path = args.output_dir / f"{run_id}_prover_response.json"
    report_path = args.output_dir / f"{run_id}_verification_report.json"

    with response_path.open("w", encoding="utf-8") as handle:
        json.dump(prover_response, handle, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2)

    print("[INFO] Modal prover query completed")
    print(f"[INFO] Run ID:       {run_id}")
    print(f"[INFO] Verified:     {verification['verification']['all_verified']}")
    print(f"[INFO] Proof count:  {verification['verification']['num_proofs']}")
    print(f"[INFO] Response:     {response_path}")
    print(f"[INFO] Report:       {report_path}")

    if not verification["verification"]["all_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

