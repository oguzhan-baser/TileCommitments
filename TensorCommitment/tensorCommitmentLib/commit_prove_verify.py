#!/usr/bin/env python3
"""
Tensor Commitment: Commit, Prove, Verify
==========================================

Loads polynomial coefficients (produced by ``interpolationLib``) and runs the
full PST tensor commitment pipeline:

1. **Setup**  -- generate commitment and verification keys.
2. **Commit** -- produce a single binding commitment to the polynomial.
3. **Prove**  -- for selected evaluation points, generate evaluation proofs.
4. **Verify** -- check each proof against the commitment.

Optionally cross-checks polynomial evaluations against the original
``hypercube.npy`` ground truth to confirm end-to-end correctness.

Requires the ``tensorcommitments`` Python module (built from
``pst_commitment_lib`` via ``maturin develop --features python``).

Quick start::

    python commit_prove_verify.py \\
        --poly-dir ../activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_polynomial
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorcommitments
except ImportError:
    sys.exit(
        "[ERROR] Cannot import tensorcommitments. "
        "Build it first:\n"
        "  cd finalcode/pst_commitment_lib && maturin develop --features python --release"
    )


# ---------------------------------------------------------------------------
# BN254 field modulus (for sign recovery when cross-checking ground truth)
# ---------------------------------------------------------------------------
BN254_P = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PointResult:
    """Result of commit-prove-verify for one evaluation point."""
    point: List[int]
    evaluation: int
    proof_hex: List[str]
    proof_bytes: int
    verified: bool
    ground_truth_int: Optional[int] = None
    ground_truth_match: Optional[bool] = None
    eval_time_s: float = 0.0
    prove_time_s: float = 0.0
    verify_time_s: float = 0.0


@dataclass
class PipelineResult:
    """Aggregated results of the full pipeline run."""
    commitment_hex: str
    commitment_bytes: int
    num_variables: int
    degree_bound: int
    num_coefficients: int
    num_queries: int
    all_verified: bool
    all_ground_truth_match: Optional[bool]
    setup_time_s: float
    commit_time_s: float
    total_eval_time_s: float
    total_prove_time_s: float
    total_verify_time_s: float
    avg_prove_time_s: float
    avg_verify_time_s: float
    avg_proof_bytes: float
    point_results: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--poly-dir",
        type=str,
        required=True,
        help="Directory containing coefficients.json "
             "(output of interpolationLib).",
    )
    p.add_argument(
        "--hypercube-dir",
        type=str,
        default=None,
        help="Directory containing hypercube.npy for ground-truth "
             "cross-checking. Auto-detected if --poly-dir follows the "
             "naming convention (<name>_polynomial -> <name>).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write results. Default: <poly-dir>/../<model>_commitment/",
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Total number of evaluation points to test.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting evaluation points.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_polynomial(poly_dir: Path) -> Tuple[List[int], List[int], int]:
    """Load coefficients.json.

    Returns (dims, coefficients_as_ints, degree_bound).
    """
    coeffs_path = poly_dir / "coefficients.json"
    if not coeffs_path.is_file():
        sys.exit(f"[ERROR] coefficients.json not found in {poly_dir}")

    print(f"[INFO] Loading polynomial from {coeffs_path} ...")
    t0 = time.time()
    with coeffs_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    load_time = time.time() - t0

    dims = [int(d) for d in data["dims"]]
    degree_bound = int(data["degree_bound"])
    coeffs = [int(c) for c in data["coefficients"]]

    expected = 1
    for d in dims:
        expected *= d
    if len(coeffs) != expected:
        sys.exit(
            f"[ERROR] Coefficient count {len(coeffs)} != expected "
            f"{expected} for dims {dims}"
        )

    print(f"[INFO] Loaded {len(coeffs)} coefficients in {load_time:.2f}s")
    print(f"[INFO] dims={dims}, degree_bound={degree_bound}, "
          f"num_vars={len(dims)}")
    return dims, coeffs, degree_bound


def auto_detect_hypercube_dir(poly_dir: Path) -> Optional[Path]:
    """Try to find the hypercube directory from the polynomial directory name.

    Convention: polynomial dir is ``<name>_polynomial``, hypercube dir is
    ``<name>`` (sibling).
    """
    name = poly_dir.name
    if name.endswith("_polynomial"):
        candidate = poly_dir.parent / name[: -len("_polynomial")]
        if (candidate / "hypercube.npy").is_file():
            return candidate
    return None


def load_hypercube(hc_dir: Path) -> np.ndarray:
    """Load hypercube.npy."""
    npy_path = hc_dir / "hypercube.npy"
    if not npy_path.is_file():
        sys.exit(f"[ERROR] hypercube.npy not found in {hc_dir}")
    return np.load(npy_path, allow_pickle=True)


def to_field(v: int) -> int:
    """Map a (possibly negative) integer to BN254 field element."""
    return v % BN254_P


def select_points(
    dims: Sequence[int], num_queries: int, seed: int
) -> List[List[int]]:
    """Select a mix of deterministic and random grid points."""
    d = len(dims)
    pts: List[List[int]] = []

    # Deterministic points
    pts.append([0] * d)                           # origin
    pts.append([n - 1 for n in dims])              # max corner
    pts.append([n // 2 for n in dims])             # center
    if d >= 2:
        pts.append([1] * d)                        # all ones

    # Random points
    rng = random.Random(seed)
    while len(pts) < num_queries:
        pts.append([rng.randrange(n) for n in dims])

    # Deduplicate preserving order
    seen = set()
    unique: List[List[int]] = []
    for pt in pts:
        key = tuple(pt)
        if key not in seen:
            seen.add(key)
            unique.append(pt)

    return unique[:num_queries]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dims: List[int],
    coeffs: List[int],
    degree_bound: int,
    points: List[List[int]],
    hypercube: Optional[np.ndarray],
) -> PipelineResult:
    """Run the full commit-prove-verify pipeline."""

    num_vars = len(dims)
    num_coeffs = len(coeffs)

    # --- Setup ---
    print(f"\n[SETUP] Generating keys for {num_vars} variables, "
          f"degree_bound={degree_bound} ({num_coeffs} CRS elements) ...")
    t0 = time.time()
    pst = tensorcommitments.TensorCommitmentWrapper(num_vars, degree_bound)
    setup_time = time.time() - t0
    print(f"[SETUP] Done in {setup_time:.2f}s")

    # --- Commit ---
    print(f"\n[COMMIT] Committing to {num_coeffs} coefficients ...")
    t0 = time.time()
    commitment = pst.commit(coeffs)
    commit_time = time.time() - t0
    commitment_bytes = len(bytes.fromhex(commitment))
    print(f"[COMMIT] Done in {commit_time:.2f}s")
    print(f"[COMMIT] Commitment: {commitment[:64]}...")
    print(f"[COMMIT] Size: {commitment_bytes} bytes")

    # --- Prove & Verify ---
    print(f"\n[PROVE/VERIFY] Processing {len(points)} evaluation points ...\n")

    point_results: List[PointResult] = []
    all_verified = True
    all_gt_match: Optional[bool] = True if hypercube is not None else None

    total_eval = 0.0
    total_prove = 0.0
    total_verify = 0.0

    for i, pt in enumerate(points):
        # Evaluate
        t0 = time.time()
        eval_val = pst.evaluate_polynomial(coeffs, pt)
        eval_time = time.time() - t0
        total_eval += eval_time

        # Prove
        t0 = time.time()
        proof = pst.prove(coeffs, pt, eval_val)
        prove_time = time.time() - t0
        total_prove += prove_time

        # Verify
        t0 = time.time()
        verified = pst.verify(commitment, pt, eval_val, proof)
        verify_time = time.time() - t0
        total_verify += verify_time

        proof_bytes = sum(len(bytes.fromhex(h)) for h in proof)

        if not verified:
            all_verified = False

        # Ground-truth cross-check
        gt_int: Optional[int] = None
        gt_match: Optional[bool] = None
        gt_note = ""
        if hypercube is not None:
            gt_int = int(hypercube[tuple(pt)])
            gt_field = to_field(gt_int)
            gt_match = (eval_val == gt_field)
            if not gt_match:
                all_gt_match = False
            sign = " (neg)" if gt_int < 0 else ""
            gt_note = (
                f"  gt={gt_int}{sign}  "
                f"gt_match={'OK' if gt_match else 'FAIL'}"
            )

        status = "OK" if verified else "FAIL"
        print(
            f"  [{i+1:2d}/{len(points)}] point={pt}  "
            f"verify={status}  "
            f"proof={proof_bytes}B  "
            f"prove={prove_time:.3f}s  "
            f"verify={verify_time:.4f}s"
            f"{gt_note}"
        )

        point_results.append(PointResult(
            point=pt,
            evaluation=eval_val,
            proof_hex=proof,
            proof_bytes=proof_bytes,
            verified=verified,
            ground_truth_int=gt_int,
            ground_truth_match=gt_match,
            eval_time_s=round(eval_time, 4),
            prove_time_s=round(prove_time, 4),
            verify_time_s=round(verify_time, 4),
        ))

    n = len(points)
    avg_proof_bytes = (
        sum(r.proof_bytes for r in point_results) / n if n else 0.0
    )

    return PipelineResult(
        commitment_hex=commitment,
        commitment_bytes=commitment_bytes,
        num_variables=num_vars,
        degree_bound=degree_bound,
        num_coefficients=num_coeffs,
        num_queries=n,
        all_verified=all_verified,
        all_ground_truth_match=all_gt_match,
        setup_time_s=round(setup_time, 3),
        commit_time_s=round(commit_time, 3),
        total_eval_time_s=round(total_eval, 3),
        total_prove_time_s=round(total_prove, 3),
        total_verify_time_s=round(total_verify, 3),
        avg_prove_time_s=round(total_prove / n, 3) if n else 0.0,
        avg_verify_time_s=round(total_verify / n, 4) if n else 0.0,
        avg_proof_bytes=round(avg_proof_bytes, 1),
        point_results=[asdict(r) for r in point_results],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    poly_dir = Path(args.poly_dir).resolve()
    if not poly_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {poly_dir}")

    # --- Load polynomial ---
    dims, coeffs, degree_bound = load_polynomial(poly_dir)

    # --- Detect / load hypercube for ground-truth checks ---
    hc_dir: Optional[Path] = None
    hypercube: Optional[np.ndarray] = None

    if args.hypercube_dir:
        hc_dir = Path(args.hypercube_dir).resolve()
    else:
        hc_dir = auto_detect_hypercube_dir(poly_dir)

    if hc_dir is not None:
        print(f"[INFO] Loading ground-truth hypercube from {hc_dir}")
        hypercube = load_hypercube(hc_dir)
        if list(hypercube.shape) != dims:
            print(
                f"[WARN] Hypercube shape {hypercube.shape} != polynomial dims "
                f"{dims}. Disabling ground-truth checks."
            )
            hypercube = None
        else:
            print(f"[INFO] Ground-truth cross-checking enabled")
    else:
        print("[INFO] No hypercube directory found; skipping ground-truth checks")

    # --- Select evaluation points ---
    points = select_points(dims, args.num_queries, args.seed)
    print(f"[INFO] Selected {len(points)} evaluation points "
          f"(seed={args.seed})")

    # --- Run pipeline ---
    result = run_pipeline(dims, coeffs, degree_bound, points, hypercube)

    # --- Print summary ---
    print(f"\n{'=' * 64}")
    print(f"COMMITMENT PIPELINE SUMMARY")
    print(f"{'=' * 64}")
    print(f"  Polynomial:       {result.num_variables} variables, "
          f"degree_bound={result.degree_bound}, "
          f"{result.num_coefficients} coefficients")
    print(f"  Commitment:       {result.commitment_bytes} bytes")
    print(f"  Queries:          {result.num_queries}")
    print(f"  All verified:     {result.all_verified}")
    if result.all_ground_truth_match is not None:
        print(f"  Ground-truth OK:  {result.all_ground_truth_match}")
    print(f"  Setup time:       {result.setup_time_s:.3f}s")
    print(f"  Commit time:      {result.commit_time_s:.3f}s")
    print(f"  Avg prove time:   {result.avg_prove_time_s:.3f}s")
    print(f"  Avg verify time:  {result.avg_verify_time_s:.4f}s")
    print(f"  Avg proof size:   {result.avg_proof_bytes:.0f} bytes")
    print(f"{'=' * 64}")

    if not result.all_verified:
        print("\n[FAIL] Some proofs did not verify!")
        sys.exit(1)
    if result.all_ground_truth_match is False:
        print("\n[FAIL] Some evaluations did not match ground-truth!")
        sys.exit(1)

    # --- Determine output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        # Convention: sibling of poly_dir named *_commitment
        base = poly_dir.name
        if base.endswith("_polynomial"):
            out_name = base[: -len("_polynomial")] + "_commitment"
        else:
            out_name = base + "_commitment"
        out_dir = poly_dir.parent / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save commitment ---
    commit_path = out_dir / "commitment.txt"
    commit_path.write_text(result.commitment_hex + "\n", encoding="utf-8")
    print(f"\n[INFO] Saved commitment       -> {commit_path}")

    # --- Save full results ---
    results_dict: Dict[str, Any] = {
        "commitment_hex": result.commitment_hex,
        "commitment_bytes": result.commitment_bytes,
        "polynomial": {
            "num_variables": result.num_variables,
            "degree_bound": result.degree_bound,
            "num_coefficients": result.num_coefficients,
            "source_dir": str(poly_dir),
        },
        "hypercube_source": str(hc_dir) if hc_dir else None,
        "verification_summary": {
            "num_queries": result.num_queries,
            "all_proofs_verified": result.all_verified,
            "all_ground_truth_matched": result.all_ground_truth_match,
        },
        "timing": {
            "setup_time_s": result.setup_time_s,
            "commit_time_s": result.commit_time_s,
            "total_eval_time_s": result.total_eval_time_s,
            "total_prove_time_s": result.total_prove_time_s,
            "total_verify_time_s": result.total_verify_time_s,
            "avg_prove_time_s": result.avg_prove_time_s,
            "avg_verify_time_s": result.avg_verify_time_s,
        },
        "proof_stats": {
            "avg_proof_bytes": result.avg_proof_bytes,
            "proof_elements_per_query": result.num_variables,
        },
        "point_results": result.point_results,
    }

    results_path = out_dir / "commitment_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[INFO] Saved results          -> {results_path}")

    # --- Save proofs separately for easy loading ---
    proofs_dict = {
        "commitment_hex": result.commitment_hex,
        "num_variables": result.num_variables,
        "degree_bound": result.degree_bound,
        "proofs": [
            {
                "point": r.point,
                "evaluation": r.evaluation,
                "proof_hex": r.proof_hex,
            }
            for r in [PointResult(**d) for d in result.point_results]
        ],
    }
    proofs_path = out_dir / "proofs.json"
    with proofs_path.open("w", encoding="utf-8") as f:
        json.dump(proofs_dict, f, indent=2)
    print(f"[INFO] Saved proofs           -> {proofs_path}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
