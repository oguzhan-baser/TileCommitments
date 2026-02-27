
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np
from pegasus_verkle import BatchProof, KzgVerkleTree, SingleProof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=8, help="branching factor (>=2)")
    parser.add_argument("--size", type=int, default=8**3, help="size of the tree (>=1)")
    # parser.add_argument("--queries", type=int, default=2, help="indices per round")
    parser.add_argument("--seed", type=int, default=42, help="rng seed for reproducibility")
    parser.add_argument(
        "--max-value",
        type=int,
        default=1_000_000,
        help="exclusive upper bound for sampled integers",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="metrics",
        help="directory (relative to this script) to store metrics JSON output",
    )
    return parser.parse_args()


def estimate_tree_size_bytes(width: int, depth: int) -> int:
    if width <= 1:
        return 0
    total_nodes = (width ** (depth + 1) - 1) // (width - 1)
    node_bytes = 48 + width * 32  # commitment + coefficients estimate
    return total_nodes * node_bytes


def estimate_proof_size_bytes(proof: BatchProof) -> int:
    return getattr(proof, "node_count", 0) * 200  # heuristic used elsewhere


def commitment_size_bytes(root_hex: str) -> int:
    clean = root_hex[2:] if root_hex.startswith(("0x", "0X")) else root_hex
    return len(clean) // 2


def main() -> None:
    args = parse_args()

    if args.width < 2:
        raise ValueError("--width must be at least 2")

    depth = int(np.floor(np.log(args.size) / np.log(args.width)))
    if depth < 1:
        raise ValueError("computed depth must be at least 1; adjust --size or --width")

    actual_size = args.width ** depth
    if actual_size != args.size:
        print(
            f"Warning: size {args.size} is not a perfect power of width {args.width}. "
            f"Using size {actual_size} instead (width^{depth})."
        )
        args.size = actual_size

    rng = np.random.default_rng(args.seed)
    values = rng.integers(1, args.max_value, size=args.size, dtype=np.int64).tolist()

    print("Tree generation started...")
    tree_start = time.time()
    tree = KzgVerkleTree(values, width=args.width)
    root = tree.root_hex()
    tree_end = time.time()
    tree_time = tree_end - tree_start
    print(f"Tree generation time: {tree_time:.4f}s")
    print(f"Root commitment: {root}")
    print("Tree generation completed...")

    print("Challenge generation started...")
    challenge_start = time.time()
    # challenge = sorted(random.sample(range(args.size), min(args.queries, args.size)))
    challenge = random.randint(0, args.size-1)
    challenge_end = time.time()
    challenge_time = challenge_end - challenge_start
    print(f"Challenge generation time: {challenge_time:.4f}s")
    print(f"Challenge: {challenge}")
    print("Challenge generation completed...")

    print("Proof generation started...")
    proof_start = time.time()
    proof = tree.prove_single(challenge)
    proof_end = time.time()
    proof_time = proof_end - proof_start
    print(f"Proof generation time: {proof_time:.4f}s")
    print(f"Proof type: {type(proof).__name__}, node_count: {proof.node_count}")
    print("Proof generation completed...")

    print("Proof verification started...")
    proof_ver_start = time.time()
    proof_ver_result = proof.verify(root, challenge, values[challenge])
    proof_ver_end = time.time()
    proof_ver_time = proof_ver_end - proof_ver_start
    print(f"Proof verification time: {proof_ver_time:.4f}s")
    print("Proof verification completed...")
    print(f"Proof verification result: {proof_ver_result}")

    if proof_ver_result:
        print("\n✅ Proof verification successful!")
    else:
        print("\n❌ Proof verification failed!")

    tree_size = estimate_tree_size_bytes(args.width, depth)
    proof_size = estimate_proof_size_bytes(proof)
    commitment_bytes = commitment_size_bytes(root)
    dataset_bytes = len(values) * 8  # values are 64-bit integers

    metrics = {
        "parameters": {
            "width": args.width,
            "size": args.size,
            "depth": depth,
            "max_value": args.max_value,
            "seed": args.seed,
        },
        "timings_seconds": {
            "tree_build": tree_time,
            "challenge_generation": challenge_time,
            "proof_generation": proof_time,
            "proof_verification": proof_ver_time,
        },
        "sizes_bytes": {
            "tree_estimated": tree_size,
            "proof_estimated": proof_size,
            "commitment": commitment_bytes,
            "dataset": dataset_bytes,
        },
        "proof_node_count": getattr(proof, "node_count", None),
        "challenge_indices": challenge,
        "challenge_values": values[challenge],
        "root_commitment_hex": root,
        "proof_verification_result": proof_ver_result,
    }

    metrics_dir = Path(__file__).parent / args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_filename = (
        f"metrics_width{args.width}_size{args.size}.json"
    )
    metrics_path = metrics_dir / metrics_filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
