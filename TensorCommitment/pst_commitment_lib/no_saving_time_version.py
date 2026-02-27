#!/usr/bin/env python3
"""
One-shot pipeline: run the BN254 interpolation demo, capture coefficients in-memory,
and immediately commit/prove/verify with tensorcommitments—no intermediate files.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import List, Sequence

import tensorcommitments


def run_poly_interp_capture(project_dir: Path, npy_path: Path) -> dict:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--",
        str(npy_path),
        "--coeffs",
        "-",
    ]
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def normalize_coeffs(payload: dict) -> tuple[List[int], int, List[int]]:
    dims = [int(d) for d in payload["dims"]]
    degree_bound = int(payload["degree_bound"])
    coeffs = [int(c) for c in payload["coefficients"]]

    if any(d != degree_bound for d in dims):
        raise ValueError(
            f"Non-uniform grid dims {dims}; TensorCommitmentWrapper currently needs all dims == degree_bound"
        )

    expected = degree_bound ** len(dims)
    if len(coeffs) < expected:
        coeffs = coeffs + [0] * (expected - len(coeffs))
    elif len(coeffs) > expected:
        raise ValueError(
            f"Coefficient count {len(coeffs)} exceeds expected {expected} "
            f"for degree_bound={degree_bound} and {len(dims)} variables"
        )

    return dims, degree_bound, coeffs


def sample_points(dims: Sequence[int], num_random: int) -> List[List[int]]:
    pts: List[List[int]] = []
    pts.append([0] * len(dims))
    if all(d > 1 for d in dims):
        pts.append([min(1, d - 1) for d in dims])
    rng = random.Random(0)
    for _ in range(num_random):
        pts.append([rng.randrange(d) for d in dims])
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "npy_path",
        type=Path,
        help="Path to the tensor .npy file (same input as poly_interp_demo)",
    )
    parser.add_argument(
        "--poly-project",
        type=Path,
        default=Path("poly_interp_demo"),
        help="Cargo project directory for the interpolation demo",
    )
    parser.add_argument(
        "--random-checks",
        type=int,
        default=2,
        help="Additional random grid points to test proofs at",
    )
    args = parser.parse_args()

    poly_project = args.poly_project.resolve()
    npy_path = args.npy_path.resolve()

    payload = run_poly_interp_capture(poly_project, npy_path)
    dims, degree_bound, coeffs = normalize_coeffs(payload)
    num_vars = len(dims)

    print(
        f"Captured {len(coeffs)} coefficients for {num_vars}-var tensor "
        f"(degree_bound={degree_bound})"
    )

    pst = tensorcommitments.TensorCommitmentWrapper(num_vars, degree_bound)
    commitment = pst.commit(coeffs)
    print(f"Commitment: {commitment}")

    for pt in sample_points(dims, args.random_checks):
        eval_val = pst.evaluate_polynomial(coeffs, pt)
        proof = pst.prove(coeffs, pt, eval_val)
        ok = pst.verify(commitment, pt, eval_val, proof)
        flag = "ok" if ok else "FAIL"
        print(f"  point {pt}: eval={eval_val} proof={flag}")


if __name__ == "__main__":
    main()

