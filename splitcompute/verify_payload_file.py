#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import save_json, verify_prover_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a saved prover response payload JSON on verifier machine."
    )
    parser.add_argument("--payload", type=Path, required=True, help="Path to prover response JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for verifier report JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_path = args.payload.resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(f"Payload file not found: {payload_path}")

    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    report = verify_prover_response(payload)
    out_path = args.output.resolve() if args.output else payload_path.parent / "verification_report.json"
    save_json(out_path, report)

    all_verified = bool(report["verification"]["all_verified"])
    print("[INFO] Verified payload file")
    print(f"[INFO] Payload:      {payload_path}")
    print(f"[INFO] Proof count:  {report['verification']['num_proofs']}")
    print(f"[INFO] All verified: {all_verified}")
    print(f"[INFO] Report:       {out_path}")
    if not all_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

