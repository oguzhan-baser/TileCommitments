#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from protocol import post_json, save_json, utc_tag, verify_prover_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verifier-side client. Requests proof material from remote prover server "
            "or verifies an already saved payload file."
        )
    )
    parser.add_argument("--prover-url", type=str, default="http://127.0.0.1:8081", help="Base URL for prover server")
    parser.add_argument("--model-name", type=str, default=None, help="Model in prover allowlist (required when querying prover)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (required when querying prover)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-factor", type=int, default=16)
    parser.add_argument("--quantize", type=float, default=50.0)
    parser.add_argument("--min-dim", type=int, default=4)
    parser.add_argument("--max-dim", type=int, default=10)
    parser.add_argument("--skip-interp-build", action="store_true", default=True)
    parser.add_argument(
        "--payload-file",
        type=Path,
        default=None,
        help="If provided, skip remote call and verify this saved prover payload JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "verifier_runs",
        help="Directory to save verifier response + report",
    )
    return parser.parse_args()


def load_payload_file(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Payload file not found: {resolved}")
    import json

    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_payload_from_prover(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.model_name:
        raise ValueError("--model-name is required when --payload-file is not provided")
    if not args.prompt:
        raise ValueError("--prompt is required when --payload-file is not provided")
    endpoint = args.prover_url.rstrip("/") + "/prove"
    request_payload = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new_tokens),
        "num_queries": int(args.num_queries),
        "seed": int(args.seed),
        "scale_factor": int(args.scale_factor),
        "quantize": float(args.quantize),
        "min_dim": int(args.min_dim),
        "max_dim": int(args.max_dim),
        "skip_interp_build": bool(args.skip_interp_build),
    }
    return post_json(endpoint, request_payload, timeout_s=7200)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.payload_file is not None:
        prover_response = load_payload_file(args.payload_file)
    else:
        prover_response = request_payload_from_prover(args)

    verification_report = verify_prover_response(prover_response)
    run_tag = utc_tag()
    response_path = output_dir / f"prover_response_{run_tag}.json"
    report_path = output_dir / f"verification_report_{run_tag}.json"
    save_json(response_path, prover_response)
    save_json(report_path, verification_report)

    all_verified = bool(verification_report["verification"]["all_verified"])
    num_proofs = int(verification_report["verification"]["num_proofs"])
    model_name = verification_report.get("model_name", "")
    generated_text = verification_report.get("inference_output", {}).get("generated_text", "")

    print("[INFO] Verifier client summary")
    print(f"[INFO] Model:            {model_name}")
    print(f"[INFO] Proof count:      {num_proofs}")
    print(f"[INFO] All verified:     {all_verified}")
    print(f"[INFO] Generated text:   {generated_text!r}")
    print(f"[INFO] Saved response:   {response_path}")
    print(f"[INFO] Saved report:     {report_path}")

    if not all_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

