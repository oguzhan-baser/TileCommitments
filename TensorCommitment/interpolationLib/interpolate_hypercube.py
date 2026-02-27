#!/usr/bin/env python3
"""
Hypercube Polynomial Interpolation
====================================

Reads a ``hypercube.npy`` file (produced by activationCaptureLib's
``reshape_to_hypercube.py``) and performs N-D polynomial interpolation
over the BN254 scalar field using the Rust ``poly_interp_demo`` binary
from ``pst_commitment_lib``.

The resulting polynomial satisfies:

    p(i_0, i_1, ..., i_{d-1}) = hypercube[i_0, i_1, ..., i_{d-1}]

for every grid point, where indices map integers into the BN254 field
(negative values use additive inverses: -x -> p - x).

Outputs
-------
1. ``coefficients.json``  -- polynomial coefficients as BN254 field
   element strings, together with dimension info and degree bound.
2. ``interpolation_metadata.json`` -- provenance: source paths, timing,
   data statistics, and the full hypercube metadata.

Quick start::

    python interpolate_hypercube.py \\
        --input-dir ../activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube

Or point directly at a ``.npy`` file::

    python interpolate_hypercube.py \\
        --npy ../activationCaptureLib/output/.../hypercube.npy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Relative path from this script to the poly_interp_demo Cargo project
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_POLY_PROJECT = (
    _SCRIPT_DIR.parent / "pst_commitment_lib" / "poly_interp_demo"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate a hypercube into a multivariate polynomial over BN254.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Hypercube directory (must contain hypercube.npy and "
             "hypercube_metadata.json).",
    )
    src.add_argument(
        "--npy",
        type=str,
        default=None,
        help="Direct path to a .npy file to interpolate.",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write output files. Default: "
             "<input-dir>_polynomial/ or next to the .npy file.",
    )
    p.add_argument(
        "--poly-project",
        type=str,
        default=str(_DEFAULT_POLY_PROJECT),
        help="Path to the poly_interp_demo Cargo project.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not rebuild the Rust binary (assume it is up-to-date).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hypercube(
    input_dir: Optional[Path],
    npy_path: Optional[Path],
) -> tuple[np.ndarray, Dict[str, Any], Path]:
    """Load hypercube data and optional metadata.

    Returns (array, metadata_dict, resolved_npy_path).
    """
    if input_dir is not None:
        input_dir = Path(input_dir).resolve()
        npy_file = input_dir / "hypercube.npy"
        meta_file = input_dir / "hypercube_metadata.json"
        if not npy_file.is_file():
            sys.exit(f"[ERROR] hypercube.npy not found in {input_dir}")
        arr = np.load(npy_file, allow_pickle=True)
        meta: Dict[str, Any] = {}
        if meta_file.is_file():
            with meta_file.open("r") as f:
                meta = json.load(f)
        return arr, meta, npy_file

    assert npy_path is not None
    npy_file = Path(npy_path).resolve()
    if not npy_file.is_file():
        sys.exit(f"[ERROR] File not found: {npy_file}")
    arr = np.load(npy_file, allow_pickle=True)
    # Try to find metadata beside the npy
    meta_file = npy_file.parent / "hypercube_metadata.json"
    meta = {}
    if meta_file.is_file():
        with meta_file.open("r") as f:
            meta = json.load(f)
    return arr, meta, npy_file


def convert_to_int64_npy(arr: np.ndarray, out_path: Path) -> None:
    """Convert an object-dtype (or any int-like) array to int64 and save as .npy.

    The Rust ``ndarray-npy`` crate requires a standard numeric dtype.
    """
    if arr.dtype == object:
        # Check that all values fit in int64
        flat = arr.flat
        sample_min = min(flat[i] for i in range(min(len(flat), arr.size)))
        sample_max = max(flat[i] for i in range(min(len(flat), arr.size)))
        i64_min, i64_max = -(2**63), 2**63 - 1
        if sample_min < i64_min or sample_max > i64_max:
            sys.exit(
                f"[ERROR] Values exceed int64 range: min={sample_min}, max={sample_max}. "
                f"The Rust interpolation binary requires values in [{i64_min}, {i64_max}]."
            )
        arr_i64 = np.array(arr, dtype=np.int64)
    elif arr.dtype == np.int64:
        arr_i64 = arr
    else:
        arr_i64 = arr.astype(np.int64)

    np.save(out_path, arr_i64)


def build_rust_binary(project_dir: Path) -> None:
    """Run ``cargo build --release`` in the poly_interp_demo project."""
    print(f"[INFO] Building Rust interpolation binary in {project_dir} ...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[ERROR] cargo build failed:")
        print(result.stderr)
        sys.exit(1)
    print("[INFO] Build succeeded.")


def run_interpolation(project_dir: Path, npy_path: Path) -> Dict[str, Any]:
    """Run the Rust interpolation binary and capture coefficient JSON from stdout.

    Uses ``--coeffs -`` to stream JSON to stdout.
    """
    cmd = [
        "cargo", "run", "--release", "--",
        str(npy_path),
        "--coeffs", "-",
    ]
    print(f"[INFO] Running interpolation: {' '.join(cmd[:4])} ...")
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[ERROR] Interpolation failed:")
        print(result.stderr)
        sys.exit(1)

    # Diagnostic logs go to stderr (via the log_println! macro in quiet mode)
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            print(f"  [rust] {line}")

    # JSON comes from stdout
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse interpolation output as JSON: {exc}")
        print("stdout was:", result.stdout[:500])
        sys.exit(1)

    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve paths ---
    poly_project = Path(args.poly_project).resolve()
    if not (poly_project / "Cargo.toml").is_file():
        sys.exit(f"[ERROR] No Cargo.toml found in poly project: {poly_project}")

    # --- Load hypercube ---
    print("[INFO] Loading hypercube ...")
    t0 = time.time()
    arr, hc_meta, source_npy = load_hypercube(
        args.input_dir, args.npy
    )
    load_time = time.time() - t0

    dims = list(arr.shape)
    total_elements = int(arr.size)
    print(f"[INFO] Shape: {tuple(dims)}  ({len(dims)}-D, {total_elements} elements)")
    print(f"[INFO] dtype: {arr.dtype}   load time: {load_time:.2f}s")

    # --- Compute data stats ---
    flat = arr.flatten()
    if arr.dtype == object:
        int_flat = [int(v) for v in flat]
        data_min = min(int_flat)
        data_max = max(int_flat)
        has_negative = data_min < 0
    else:
        data_min = int(flat.min())
        data_max = int(flat.max())
        has_negative = data_min < 0

    print(f"[INFO] Value range: [{data_min}, {data_max}]  "
          f"has_negative={has_negative}")

    # --- Convert to int64 .npy (temp file) ---
    tmp_npy = None
    npy_for_rust = source_npy

    if arr.dtype != np.int64:
        print("[INFO] Converting object-dtype to int64 for Rust compatibility ...")
        t0 = time.time()
        tmp = tempfile.NamedTemporaryFile(
            prefix="hypercube_int64_", suffix=".npy", delete=False
        )
        tmp.close()
        tmp_npy = Path(tmp.name)
        convert_to_int64_npy(arr, tmp_npy)
        npy_for_rust = tmp_npy
        convert_time = time.time() - t0
        print(f"[INFO] Saved int64 copy to {tmp_npy}  ({convert_time:.2f}s)")
    else:
        convert_time = 0.0

    try:
        # --- Build Rust binary ---
        if not args.skip_build:
            build_rust_binary(poly_project)

        # --- Run interpolation ---
        t0 = time.time()
        payload = run_interpolation(poly_project, npy_for_rust)
        interp_time = time.time() - t0
        print(f"[INFO] Interpolation completed in {interp_time:.2f}s")

        num_coeffs = len(payload.get("coefficients", []))
        degree_bound = payload.get("degree_bound", 0)
        print(f"[INFO] Obtained {num_coeffs} coefficients  "
              f"(degree_bound={degree_bound})")

        # --- Determine output directory ---
        if args.output_dir:
            out_dir = Path(args.output_dir).resolve()
        elif args.input_dir:
            out_dir = Path(args.input_dir).resolve().parent / (
                Path(args.input_dir).resolve().name + "_polynomial"
            )
        else:
            out_dir = source_npy.parent / (source_npy.stem + "_polynomial")
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Save coefficients ---
        coeffs_path = out_dir / "coefficients.json"
        with coeffs_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        coeffs_size = coeffs_path.stat().st_size
        print(f"[INFO] Saved coefficients -> {coeffs_path}  "
              f"({coeffs_size / 1024 / 1024:.2f} MB)")

        # --- Save interpolation metadata ---
        interp_meta = {
            "source_npy": str(source_npy),
            "source_dir": str(source_npy.parent),
            "hypercube_metadata": hc_meta if hc_meta else None,
            "polynomial": {
                "dims": payload.get("dims", dims),
                "degree_bound": degree_bound,
                "num_coefficients": num_coeffs,
                "field": "BN254 scalar field (Fr)",
                "field_modulus": "21888242871839275222246405745257275088548364400416034343698204186575808495617",
                "coefficient_order": "row-major (C-order) lexicographic over "
                                     "(a_0, a_1, ..., a_{d-1})",
                "monomial_meaning": "coeffs[flat(a_0,...,a_{d-1})] is the "
                                    "coefficient of x_0^{a_0} * ... * x_{d-1}^{a_{d-1}}",
            },
            "data_stats": {
                "value_min": data_min,
                "value_max": data_max,
                "has_negative_values": has_negative,
                "negative_encoding": "Negative integer -x is mapped to "
                                     "Fr::zero() - Fr::from(x), i.e. the "
                                     "additive inverse in the BN254 field.",
            },
            "timing": {
                "load_time_s": round(load_time, 3),
                "dtype_convert_time_s": round(convert_time, 3),
                "interpolation_time_s": round(interp_time, 3),
                "total_time_s": round(load_time + convert_time + interp_time, 3),
            },
            "artifacts": {
                "coefficients_json": str(coeffs_path),
                "coefficients_size_bytes": coeffs_size,
            },
        }

        meta_path = out_dir / "interpolation_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(interp_meta, f, indent=2)
        print(f"[INFO] Saved metadata    -> {meta_path}")

        print("\n=== Summary ===")
        print(f"  Input:          {source_npy}")
        print(f"  Shape:          {tuple(dims)}")
        print(f"  Coefficients:   {num_coeffs}")
        print(f"  Degree bound:   {degree_bound}")
        print(f"  Output dir:     {out_dir}")
        print(f"  Total time:     {load_time + convert_time + interp_time:.2f}s")
        print("[INFO] Done.")

    finally:
        # Clean up temp file
        if tmp_npy is not None and tmp_npy.exists():
            tmp_npy.unlink()
            print(f"[INFO] Cleaned up temp file: {tmp_npy}")


if __name__ == "__main__":
    main()
