#!/usr/bin/env python3
"""
Generate and verify a proof for one hypercube index.

This script reuses the existing TensorCommitment binding (`tensorcommitments`)
used by `TensorCommitment/tensorCommitmentLib/commit_prove_verify.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import tensorcommitments
except ImportError:
    sys.exit(
        "[ERROR] Cannot import tensorcommitments.\n"
        "Activate the right env and build bindings:\n"
        "  conda activate tilecommitments\n"
        "  cd TensorCommitment/pst_commitment_lib && maturin develop --features python --release"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given a saved polynomial + commitment, generate proof for a target index, "
            "save it, then verify with commitment+index+value."
        )
    )
    parser.add_argument("--poly-dir", type=Path, required=True, help="Directory containing coefficients.json.")
    parser.add_argument("--commitment-file", type=Path, required=True, help="Path to commitment.txt.")
    parser.add_argument("--index", type=str, default=None, help="Comma-separated multi-index, e.g. 3,1,0,5,2,4,1,0")
    parser.add_argument("--flat-index", type=int, default=None, help="Flat row-major index.")
    parser.add_argument("--value", type=int, default=None, help="Explicit claimed value at the selected index.")
    parser.add_argument(
        "--hypercube-dir",
        type=Path,
        default=None,
        help="Optional directory containing hypercube.npy; used to fetch value automatically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output proof bundle JSON path. Default: <poly-dir>/../proof_at_<index>.json",
    )
    return parser.parse_args()


def load_polynomial(poly_dir: Path) -> tuple[List[int], int, List[int], Path]:
    coeffs_path = poly_dir / "coefficients.json"
    if not coeffs_path.is_file():
        sys.exit(f"[ERROR] coefficients.json not found in {poly_dir}")

    with coeffs_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    dims = [int(v) for v in data["dims"]]
    degree_bound = int(data["degree_bound"])
    coefficients = [int(v) for v in data["coefficients"]]

    expected = 1
    for dim in dims:
        expected *= dim
    if len(coefficients) != expected:
        sys.exit(
            f"[ERROR] coefficient count mismatch: got {len(coefficients)}, expected {expected}"
        )

    return dims, degree_bound, coefficients, coeffs_path


def parse_multi_index(index_text: str, dims: Sequence[int]) -> List[int]:
    try:
        index = [int(part.strip()) for part in index_text.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid --index: {index_text}") from exc

    if len(index) != len(dims):
        raise ValueError(
            f"--index has {len(index)} coordinates, expected {len(dims)} for dims={list(dims)}"
        )
    for axis, (coord, dim) in enumerate(zip(index, dims)):
        if coord < 0 or coord >= dim:
            raise ValueError(f"index out of range at axis {axis}: {coord} not in [0, {dim - 1}]")
    return index


def flat_to_multi(flat_index: int, dims: Sequence[int]) -> List[int]:
    if flat_index < 0:
        raise ValueError("--flat-index must be non-negative")
    total = 1
    for dim in dims:
        total *= dim
    if flat_index >= total:
        raise ValueError(f"--flat-index={flat_index} out of range for total size {total}")

    coords_rev: List[int] = []
    remainder = flat_index
    for dim in reversed(dims):
        coords_rev.append(remainder % dim)
        remainder //= dim
    return list(reversed(coords_rev))


def multi_to_flat(index: Sequence[int], dims: Sequence[int]) -> int:
    flat = 0
    for coord, dim in zip(index, dims):
        flat = flat * dim + coord
    return flat


def resolve_index(args: argparse.Namespace, dims: Sequence[int]) -> List[int]:
    if args.index is None and args.flat_index is None:
        raise ValueError("provide either --index or --flat-index")
    if args.index is not None and args.flat_index is not None:
        raise ValueError("provide only one of --index or --flat-index")
    if args.index is not None:
        return parse_multi_index(args.index, dims)
    return flat_to_multi(args.flat_index, dims)


def resolve_value(
    args: argparse.Namespace,
    index: Sequence[int],
    dims: Sequence[int],
    degree_bound: int,
    coefficients: Sequence[int],
) -> tuple[int, str]:
    if args.value is not None:
        return int(args.value), "explicit"

    if args.hypercube_dir is not None:
        hypercube_path = args.hypercube_dir / "hypercube.npy"
        if not hypercube_path.is_file():
            sys.exit(f"[ERROR] hypercube.npy not found in {args.hypercube_dir}")
        hypercube = np.load(hypercube_path, allow_pickle=True)
        if tuple(hypercube.shape) != tuple(dims):
            sys.exit(
                f"[ERROR] hypercube shape mismatch: expected {tuple(dims)}, got {tuple(hypercube.shape)}"
            )
        return int(hypercube[tuple(index)]), f"hypercube:{hypercube_path}"

    wrapper = tensorcommitments.TensorCommitmentWrapper(len(dims), degree_bound)
    value = wrapper.evaluate_polynomial(list(coefficients), list(index))
    return int(value), "polynomial-eval"


def load_commitment(path: Path) -> str:
    if not path.is_file():
        sys.exit(f"[ERROR] commitment file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"[ERROR] commitment file is empty: {path}")
    return text


def default_output(poly_dir: Path, index: Sequence[int]) -> Path:
    suffix = "_".join(str(v) for v in index)
    return poly_dir.parent / f"proof_at_{suffix}.json"


def main() -> None:
    args = parse_args()
    poly_dir = args.poly_dir.resolve()
    commitment_file = args.commitment_file.resolve()

    dims, degree_bound, coefficients, coeffs_path = load_polynomial(poly_dir)
    index = resolve_index(args, dims)
    flat_index = multi_to_flat(index, dims)
    value, value_source = resolve_value(args, index, dims, degree_bound, coefficients)
    commitment_hex = load_commitment(commitment_file)

    wrapper = tensorcommitments.TensorCommitmentWrapper(len(dims), degree_bound)
    proof_hex = wrapper.prove(coefficients, index, value)

    output_path = args.output.resolve() if args.output else default_output(poly_dir, index)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "poly_dir": str(poly_dir),
        "coefficients_file": str(coeffs_path),
        "commitment_file": str(commitment_file),
        "commitment_hex": commitment_hex,
        "dims": dims,
        "num_variables": len(dims),
        "degree_bound": degree_bound,
        "index": list(index),
        "flat_index": flat_index,
        "value_int": str(value),
        "value_source": value_source,
        "proof_hex": proof_hex,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)

    with output_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    verified = wrapper.verify(
        saved["commitment_hex"],
        saved["index"],
        int(saved["value_int"]),
        saved["proof_hex"],
    )

    print("[INFO] Proof generation and verification")
    print(f"[INFO] Polynomial:   {coeffs_path}")
    print(f"[INFO] Commitment:   {commitment_file}")
    print(f"[INFO] Index:        {saved['index']} (flat={saved['flat_index']})")
    print(f"[INFO] Value:        {saved['value_int']}  source={saved['value_source']}")
    print(f"[INFO] Proof elems:  {len(saved['proof_hex'])}")
    print(f"[INFO] Proof file:   {output_path}")
    print(f"[INFO] Verified:     {verified}")

    if not verified:
        sys.exit(1)


if __name__ == "__main__":
    main()
