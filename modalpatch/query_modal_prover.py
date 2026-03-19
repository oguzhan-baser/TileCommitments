#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def normalize_prover_base_url(url: str) -> str:
    base_url = url.strip().rstrip("/")
    if not base_url:
        raise ValueError("--prover-url cannot be empty")
    if "modal.com/apps/" in base_url and ".modal.run" not in base_url:
        raise ValueError(
            "You passed a Modal dashboard URL. Use the deployed web endpoint URL ending in '.modal.run'."
        )
    if base_url.endswith("/prove"):
        base_url = base_url[: -len("/prove")]
    return base_url


def post_prove_request(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_endpoints = [base_url + "/prove", base_url]
    attempts: list[tuple[str, int, str]] = []
    minimal_payload = {
        "model_name": payload["model_name"],
        "prompt": payload["prompt"],
    }

    for endpoint in candidate_endpoints:
        response = requests.post(endpoint, json=payload, timeout=7200, allow_redirects=True)
        status = int(response.status_code)
        body_head = response.text[:200].replace("\n", " ").strip()
        attempts.append((endpoint, status, body_head))
        if 200 <= status < 300:
            return response.json()
        if status == 422:
            fallback_response = requests.post(
                endpoint,
                json=minimal_payload,
                timeout=7200,
                allow_redirects=True,
            )
            fallback_status = int(fallback_response.status_code)
            fallback_body_head = fallback_response.text[:200].replace("\n", " ").strip()
            attempts.append((f"{endpoint} [minimal_payload]", fallback_status, fallback_body_head))
            if 200 <= fallback_status < 300:
                return fallback_response.json()
        if status not in (404, 405):
            details = response.text[:800].strip().replace("\n", " ")
            raise RuntimeError(
                f"POST {endpoint} failed with HTTP {status}. Response: {details}"
            )

    health_status = None
    models_status = None
    try:
        health_status = requests.get(base_url + "/health", timeout=20, allow_redirects=True).status_code
    except requests.RequestException:
        health_status = None
    try:
        models_status = requests.get(base_url + "/models", timeout=20, allow_redirects=True).status_code
    except requests.RequestException:
        models_status = None

    attempt_lines = "\n".join(
        f"  - POST {endpoint} -> {status} ({body_head})" for endpoint, status, body_head in attempts
    )
    raise RuntimeError(
        "Prover endpoint did not accept the request.\n"
        f"{attempt_lines}\n"
        f"  - GET {base_url}/health -> {health_status}\n"
        f"  - GET {base_url}/models -> {models_status}\n\n"
        "If you are using Modal, run `modal deploy modalpatch/modal_splitcompute_service.py` and copy the "
        "deployed `prover_api` URL (`...modal.run`). URLs from `modal run` / `-dev.modal.run` are temporary."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_url = normalize_prover_base_url(args.prover_url)
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

    prover_response = post_prove_request(base_url, payload)

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
