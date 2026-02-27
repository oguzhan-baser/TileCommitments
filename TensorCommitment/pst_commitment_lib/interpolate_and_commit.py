#!/usr/bin/env python3
"""
Run the BN254 interpolation demo to extract coefficients, then commit/prove with tensorcommitments.
"""

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence

import tensorcommitments


def run_poly_interp(project_dir: Path, npy_path: Path, coeffs_path: Path) -> None:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--",
        str(npy_path),
        "--coeffs",
        str(coeffs_path),
    ]
    subprocess.run(cmd, cwd=project_dir, check=True)


def load_coefficients(coeffs_path: Path) -> tuple[List[int], int, List[int]]:
    with coeffs_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    dims = [int(d) for d in data["dims"]]
    degree_bound = int(data.get("degree_bound", max(dims)))
    coeffs = [int(c) for c in data["coefficients"]]

    if any(d != degree_bound for d in dims):
        raise ValueError(
            f"Non-uniform grid dimensions {dims}; PST wrapper currently "
            f"requires all dims == degree_bound={degree_bound}"
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
        help="Path to the tensor .npy file (same input used by poly_interp_demo)",
    )
    parser.add_argument(
        "--poly-project",
        type=Path,
        default=Path("poly_interp_demo"),
        help="Path to the interpolation demo Cargo project",
    )
    parser.add_argument(
        "--coeffs-json",
        type=Path,
        help="Where to store/read the coefficient dump (JSON). Defaults to a temp file.",
    )
    parser.add_argument(
        "--skip-interp",
        action="store_true",
        help="Do not rerun cargo; assume --coeffs-json already exists.",
    )
    parser.add_argument(
        "--random-checks",
        type=int,
        default=2,
        help="Number of additional random grid points to test proofs at.",
    )
    args = parser.parse_args()

    coeffs_path: Path
    cleanup = False
    if args.coeffs_json:
        coeffs_path = args.coeffs_json
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="coeffs_", suffix=".json", delete=False)
        coeffs_path = Path(tmp.name)
        tmp.close()
        cleanup = True

    try:
        if not args.skip_interp:
            run_poly_interp(args.poly_project, args.npy_path, coeffs_path)
        elif not coeffs_path.exists():
            raise FileNotFoundError(
                f"--skip-interp was set but coefficient file {coeffs_path} is missing"
            )

        dims, degree_bound, coeffs = load_coefficients(coeffs_path)
        num_vars = len(dims)

        print(
            f"Loaded {len(coeffs)} coefficients for {num_vars}-var tensor "
            f"with degree_bound={degree_bound}"
        )

        pst = tensorcommitments.TensorCommitmentWrapper(num_vars, degree_bound)
        commitment = pst.commit(coeffs)
        print(f"Commitment: {commitment}")

        points = sample_points(dims, args.random_checks)
        for pt in points:
            eval_val = pst.evaluate_polynomial(coeffs, pt)
            proof = pst.prove(coeffs, pt, eval_val)
            is_valid = pst.verify(commitment, pt, eval_val, proof)
            status = "ok" if is_valid else "FAIL"
            print(f"  point {pt}: eval={eval_val} proof={status}")

    finally:
        if cleanup and coeffs_path.exists():
            coeffs_path.unlink()


if __name__ == "__main__":
    main()

