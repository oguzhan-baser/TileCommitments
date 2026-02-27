#!/usr/bin/env python3
"""
Two-party style demo (with intermediate files) for the multivariate tensor commitment:

- Prover:
    * runs the BN254 interpolation demo on the provided .npy tensor
    * saves the coefficient JSON (same format as `poly_interp_demo`) and the commitment
    * answers verifier queries with evaluations + proofs
- Verifier:
    * loads the same tensor
    * samples random grid points and looks up the ground-truth values
    * asks the prover for evaluations/proofs on those points
    * checks equality vs. its own tensor lookups and verifies each proof against the commitment
"""

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

import tensorcommitments


def run_poly_interp(project_dir: Path, npy_path: Path, coeffs_path: Path) -> float:
    start = perf_counter()
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
    return perf_counter() - start


def load_coefficients(coeffs_path: Path) -> Tuple[List[int], int, List[int]]:
    with coeffs_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    dims = [int(d) for d in data["dims"]]
    degree_bound = int(data.get("degree_bound", max(dims)))
    coeffs = [int(c) for c in data["coefficients"]]

    if any(d != degree_bound for d in dims):
        raise ValueError(
            f"Non-uniform grid dimensions {dims}; TensorCommitmentWrapper currently "
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


def ensure_coefficients(
    project_dir: Path,
    npy_path: Path,
    coeffs_path: Path,
    skip_interp: bool,
) -> Tuple[List[int], int, List[int], float]:
    interp_time = 0.0
    if not skip_interp or not coeffs_path.exists():
        interp_time = run_poly_interp(project_dir, npy_path, coeffs_path)
    dims, degree_bound, coeffs = load_coefficients(coeffs_path)
    return dims, degree_bound, coeffs, interp_time


def sample_points(dims: Sequence[int], num_points: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    pts: List[List[int]] = []
    pts.append([0] * len(dims))
    if all(d > 1 for d in dims):
        pts.append([min(1, d - 1) for d in dims])
    while len(pts) < num_points:
        pts.append([rng.randrange(d) for d in dims])
    return pts[:num_points]


def read_commitment(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


@dataclass
class QueryResponse:
    point: List[int]
    evaluation: int
    proof: List[str]


class Prover:
    def __init__(
        self,
        coeffs: List[int],
        dims: List[int],
        degree_bound: int,
        pst: tensorcommitments.TensorCommitmentWrapper,
        commitment_path: Path,
    ) -> None:
        self.coeffs = coeffs
        self.dims = dims
        self.degree_bound = degree_bound
        self.pst = pst
        self.commitment_path = commitment_path
        self.commitment: str | None = None

    def save_commitment(self) -> Tuple[str, float]:
        t0 = perf_counter()
        self.commitment = self.pst.commit(self.coeffs)
        commit_time = perf_counter() - t0
        self.commitment_path.write_text(self.commitment + "\n", encoding="utf-8")
        return self.commitment, commit_time

    def answer_queries(
        self, points: Sequence[Sequence[int]]
    ) -> Tuple[List[QueryResponse], Dict[str, float]]:
        responses: List[QueryResponse] = []
        eval_time = 0.0
        prove_time = 0.0
        total_proof_bytes = 0
        t_total = perf_counter()
        for pt in points:
            pt_list = list(pt)
            t0 = perf_counter()
            evaluation = self.pst.evaluate_polynomial(self.coeffs, pt_list)
            eval_time += perf_counter() - t0

            t0 = perf_counter()
            proof = self.pst.prove(self.coeffs, pt_list, evaluation)
            prove_time += perf_counter() - t0
            total_proof_bytes += sum(len(bytes.fromhex(h)) for h in proof)

            responses.append(QueryResponse(point=pt_list, evaluation=evaluation, proof=proof))
        answer_time = perf_counter() - t_total
        prove_time = prove_time / len(points)
        eval_time = eval_time / len(points)
        stats = {
            "answer_time": answer_time,
            "eval_time": eval_time,
            "prove_time": prove_time,
            "proof_bytes": float(total_proof_bytes),
        }
        return responses, stats


class Verifier:
    def __init__(
        self,
        tensor: np.ndarray,
        dims: List[int],
        pst: tensorcommitments.TensorCommitmentWrapper,
        commitment_path: Path,
    ) -> None:
        if tensor.shape != tuple(dims):
            raise ValueError(f"Tensor shape {tensor.shape} does not match dims {dims}")
        self.tensor = tensor
        self.dims = dims
        self.pst = pst
        self.commitment = read_commitment(commitment_path)

    def evaluate_tensor(self, point: Sequence[int]) -> int:
        value = int(self.tensor[tuple(point)])
        if value < 0:
            raise ValueError("Tensor contains negative values; unsupported in current pipeline.")
        return value

    def verify_batch(
        self,
        expected_values: Dict[Tuple[int, ...], int],
        responses: Sequence[QueryResponse],
    ) -> Dict[str, float]:
        verify_time = 0.0
        for resp in responses:
            key = tuple(resp.point)
            expected = expected_values[key]
            if resp.evaluation != expected:
                raise ValueError(
                    f"Mismatch at {resp.point}: prover sent {resp.evaluation}, "
                    f"verifier expected {expected}"
                )
            t0 = perf_counter()
            ok = self.pst.verify(self.commitment, resp.point, resp.evaluation, resp.proof)
            verify_time += perf_counter() - t0
            if not ok:
                raise ValueError(f"Proof verification failed at point {resp.point}")
        return {"verify_time": verify_time}


def human_bytes(num: float) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(num) < step:
            return f"{num:0.2f} {unit}"
        num /= step
    return f"{num:0.2f} PB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("npy_path", type=Path, help="Input tensor stored as .npy")
    parser.add_argument(
        "--poly-project",
        type=Path,
        default=Path("poly_interp_demo"),
        help="Cargo project directory for the interpolation demo",
    )
    parser.add_argument(
        "--coeffs-json",
        type=Path,
        default=Path("coefficients.json"),
        help="Where the prover stores the coefficient dump",
    )
    parser.add_argument(
        "--commitment-file",
        type=Path,
        default=Path("commitment.txt"),
        help="Where the prover stores the commitment string",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=4,
        help="How many verifier queries to issue (including the deterministic seeds)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for verifier sampling",
    )
    parser.add_argument(
        "--skip-interp",
        action="store_true",
        help="Reuse existing coefficient JSON instead of rerunning cargo",
    )
    args = parser.parse_args()

    npy_path = args.npy_path.resolve()
    project_dir = args.poly_project.resolve()
    coeffs_path = args.coeffs_json.resolve()
    commitment_path = args.commitment_file.resolve()

    dims, degree_bound, coeffs, interp_time = ensure_coefficients(
        project_dir, npy_path, coeffs_path, args.skip_interp
    )
    pst = tensorcommitments.TensorCommitmentWrapper(len(dims), degree_bound)

    prover = Prover(coeffs, dims, degree_bound, pst, commitment_path)
    commitment, commit_time = prover.save_commitment()
    print(f"[Prover] Commitment stored in {commitment_path}: {commitment}")

    t0 = perf_counter()
    tensor = np.load(npy_path)
    tensor_load_time = perf_counter() - t0
    verifier = Verifier(tensor, dims, pst, commitment_path)

    points = sample_points(dims, args.num_queries, args.seed)
    expected = {tuple(pt): verifier.evaluate_tensor(pt) for pt in points}
    print(f"[Verifier] Selected {len(points)} query points.")

    responses, prover_stats = prover.answer_queries(points)
    print("[Prover] Returned evaluations and proofs for requested points.")

    verifier_stats = verifier.verify_batch(expected, responses)
    print("[Verifier] All responses matched tensor values and verified against the commitment.")

    # Size metrics
    input_size = npy_path.stat().st_size if npy_path.exists() else os.path.getsize(npy_path)
    coeff_size = coeffs_path.stat().st_size if coeffs_path.exists() else 0
    commitment_bytes = len(bytes.fromhex(commitment))
    avg_proof_bytes = (
        prover_stats["proof_bytes"] / len(responses) if responses else 0.0
    )

    print("\n=== Metrics ===")
    print(f"Input tensor size:      {human_bytes(input_size)}")
    print(f"Coefficient JSON size:  {human_bytes(coeff_size)}")
    print(f"Commitment size:        {commitment_bytes} bytes")
    print(
        f"Proof size (avg):       {avg_proof_bytes:0.2f} bytes "
        f"(total {prover_stats['proof_bytes']:0.0f})"
    )
    print(f"Interpolation time:     {interp_time:0.3f} s" if interp_time else "Interpolation time:     skipped (reused JSON)")
    print(f"Commit time:            {commit_time:0.3f} s")
    print(f"Tensor load time:       {tensor_load_time:0.3f} s")
    print(f"Prover avg eval time:   {prover_stats['eval_time']:0.3f} s")
    print(f"Prover avg prove time:  {prover_stats['prove_time']:0.3f} s")
    print(f"Prover answer time:     {prover_stats['answer_time']:0.3f} s")
    print(f"Verifier verify time:   {verifier_stats['verify_time']:0.3f} s")


if __name__ == "__main__":
    main()

