#!/usr/bin/env python3
"""
Combined compute + crypto verification for one layer.

Stages:
1) Compute verification:
   Recompute a chosen layer output from the saved token sequence and compare
   against the saved activation tensor.
2) Proof generation:
   Map selected elements of that layer into hypercube indices, then generate
   opening proofs using saved polynomial + commitment.
3) Crypto verification:
   Verify saved proof bundles using commitment + index + value.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVATION_LIB_DIR = SCRIPT_DIR / "TensorCommitment" / "activationCaptureLib"

if str(ACTIVATION_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_LIB_DIR))

import capture_activations as capture_lib  # type: ignore
import convert_to_npy as convert_lib  # type: ignore
import prove_verify_index as prove_index_lib  # type: ignore

try:
    import tensorcommitments
except ImportError:
    sys.exit(
        "[ERROR] Cannot import tensorcommitments.\n"
        "Activate env and build bindings first:\n"
        "  conda activate tilecommitments\n"
        "  cd TensorCommitment/pst_commitment_lib && maturin develop --features python --release"
    )

BN254_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def resolve_local_hf_snapshot(model_name: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    if not cache_root.is_dir():
        return None

    ref_main = cache_root / "refs" / "main"
    if ref_main.is_file():
        revision = ref_main.read_text(encoding="utf-8").strip()
        candidate = cache_root / "snapshots" / revision
        if candidate.is_dir():
            return candidate

    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshots:
            return snapshots[-1]
    return None


def load_model_for_compute_verification(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    try:
        model, _ = capture_lib.load_model_and_tokenizer(model_name, device, dtype)
        return model
    except Exception as exc:
        local_snapshot = resolve_local_hf_snapshot(model_name)
        if local_snapshot is None:
            raise RuntimeError(
                "Failed to load model/tokenizer via capture_activations utility, and no local Hugging Face snapshot was found."
            ) from exc

        print(
            "[WARN] Falling back to local snapshot model load without tokenizer "
            f"(reason: {type(exc).__name__}: {exc})"
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(local_snapshot),
            torch_dtype=dtype,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compute verification + proof generation + crypto verification for a chosen layer."
    )
    parser.add_argument("--activations-pt", type=Path, required=True, help="Path to <model>_activations.pt")
    parser.add_argument("--hypercube-dir", type=Path, required=True, help="Directory with hypercube.npy and hypercube_metadata.json")
    parser.add_argument("--poly-dir", type=Path, required=True, help="Directory with coefficients.json")
    parser.add_argument("--commitment-file", type=Path, required=True, help="Path to commitment.txt")
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Layer to verify: 'embedding' or integer block id (e.g. 0, 1, ...)",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available, else cpu)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Model dtype (default: float16 on CUDA, float32 on CPU).",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="allclose rtol for compute verification")
    parser.add_argument("--atol", type=float, default=1e-3, help="allclose atol for compute verification")
    parser.add_argument(
        "--proof-mode",
        type=str,
        choices=["sample", "all", "positions"],
        default="sample",
        help="How many layer elements to prove.",
    )
    parser.add_argument("--num-proofs", type=int, default=8, help="Number of proofs when --proof-mode=sample")
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Comma-separated local flat positions for --proof-mode=positions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for sample selection")
    parser.add_argument(
        "--allow-approx-commitment",
        action="store_true",
        help=(
            "Allow computed scaled value to differ from committed hypercube value. "
            "When enabled, proofs are generated for committed values and mismatches are reported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for proof bundles + summary. Default: <activations parent>/layer_verify_<layer>",
    )
    return parser.parse_args()


def load_pt_artifacts(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        sys.exit(f"[ERROR] Activations file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    for key in ("model_name", "prompt", "token_sequence", "hidden_states"):
        if key not in payload:
            sys.exit(f"[ERROR] Missing key '{key}' in activations file: {path}")
    return payload


def resolve_layer(layer_arg: str) -> Tuple[str, int]:
    if layer_arg == "embedding":
        return "embedding", 0
    try:
        layer_id = int(layer_arg)
    except ValueError as exc:
        raise ValueError("layer must be 'embedding' or an integer like 0,1,2") from exc
    if layer_id < 0:
        raise ValueError("layer integer must be >= 0")
    return f"layer_{layer_id}", layer_id + 1


def load_hypercube_metadata(hypercube_dir: Path) -> Dict[str, Any]:
    meta_path = hypercube_dir / "hypercube_metadata.json"
    if not meta_path.is_file():
        sys.exit(f"[ERROR] hypercube_metadata.json not found in {hypercube_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_positions(mode: str, count: int, requested: str | None, numel: int, seed: int) -> List[int]:
    if numel <= 0:
        return []
    if mode == "all":
        return list(range(numel))
    if mode == "positions":
        if not requested:
            raise ValueError("--positions required when --proof-mode=positions")
        try:
            positions = [int(part.strip()) for part in requested.split(",") if part.strip() != ""]
        except ValueError as exc:
            raise ValueError(f"invalid --positions: {requested}") from exc
        if not positions:
            raise ValueError("--positions produced no indices")
        for pos in positions:
            if pos < 0 or pos >= numel:
                raise ValueError(f"position {pos} out of range [0, {numel - 1}]")
        dedup = []
        seen = set()
        for pos in positions:
            if pos not in seen:
                seen.add(pos)
                dedup.append(pos)
        return dedup

    # sample mode: evenly spaced deterministic positions
    if count <= 0:
        raise ValueError("--num-proofs must be > 0 for sample mode")
    if count >= numel:
        return list(range(numel))
    rng = np.random.default_rng(seed)
    # blend deterministic spacing + random offset for better coverage
    base = np.linspace(0, numel - 1, count, dtype=int).tolist()
    jitter = rng.integers(0, max(1, numel // max(1, count)), size=count).tolist()
    sampled = []
    seen = set()
    for b, j in zip(base, jitter):
        pos = min(numel - 1, b + j)
        if pos not in seen:
            seen.add(pos)
            sampled.append(pos)
    while len(sampled) < count:
        candidate = int(rng.integers(0, numel))
        if candidate not in seen:
            seen.add(candidate)
            sampled.append(candidate)
    return sorted(sampled)


def bundle_output_dir(args: argparse.Namespace, activations_path: Path, layer_label: str) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    return (activations_path.parent / f"layer_verify_{layer_label}").resolve()


def main() -> None:
    args = parse_args()

    activations_path = args.activations_pt.resolve()
    hypercube_dir = args.hypercube_dir.resolve()
    poly_dir = args.poly_dir.resolve()
    commitment_file = args.commitment_file.resolve()
    layer_label, hidden_index = resolve_layer(args.layer)

    artifacts = load_pt_artifacts(activations_path)
    saved_hidden_states: Sequence[torch.Tensor] = artifacts["hidden_states"]
    if hidden_index >= len(saved_hidden_states):
        sys.exit(
            f"[ERROR] Requested {layer_label} -> hidden index {hidden_index}, "
            f"but only {len(saved_hidden_states)} hidden-state tensors available."
        )
    saved_layer = saved_hidden_states[hidden_index].cpu()
    token_sequence = artifacts["token_sequence"].cpu()
    model_name = artifacts["model_name"]

    device = capture_lib.resolve_device(args.device)
    dtype = capture_lib.resolve_dtype(args.dtype, device)

    print(f"[INFO] Stage 1 - compute verification for {layer_label}")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Device: {device}, dtype: {dtype}")

    model = load_model_for_compute_verification(model_name, device, dtype)
    computed_hidden_states = capture_lib.capture_hidden_states(model, token_sequence)
    computed_layer = computed_hidden_states[hidden_index].cpu()
    del model

    abs_diff = (saved_layer.float() - computed_layer.float()).abs()
    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())
    compute_match = bool(
        torch.allclose(
            saved_layer.float(),
            computed_layer.float(),
            rtol=args.rtol,
            atol=args.atol,
        )
    )

    print(f"[INFO] Compute compare: match={compute_match}, max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}")

    if not compute_match:
        print(
            "[STOP] Compute verification failed. "
            "Please confirm expected tolerance/model runtime settings before proceeding."
        )
        sys.exit(2)

    print("[INFO] Stage 1 passed. Proceeding to proof generation.")

    print(f"[INFO] Stage 2 - generate proofs for {layer_label}")
    meta = load_hypercube_metadata(hypercube_dir)
    hypercube_path = hypercube_dir / "hypercube.npy"
    if not hypercube_path.is_file():
        sys.exit(f"[ERROR] hypercube.npy not found in {hypercube_dir}")
    hypercube = np.load(hypercube_path, allow_pickle=True)

    layer_map = meta.get("layer_map", [])
    layer_entry = next((entry for entry in layer_map if entry.get("layer") == layer_label), None)
    if layer_entry is None:
        sys.exit(f"[ERROR] Layer '{layer_label}' not found in hypercube_metadata.layer_map")

    flat_offset = int(layer_entry["flat_offset"])
    layer_numel = int(layer_entry["num_elements"])
    hyper_dims = [int(v) for v in meta["hypercube"]["dimensions"]]
    total_real_elements = int(meta["hypercube"]["total_real_elements"])

    if layer_numel != int(saved_layer.numel()):
        sys.exit(
            f"[ERROR] layer_numel mismatch: metadata={layer_numel}, saved={saved_layer.numel()}"
        )

    conversion_params = meta.get("conversion_params", {})
    if "effective_scale" not in conversion_params:
        sys.exit("[ERROR] conversion_params.effective_scale missing from hypercube metadata")
    effective_scale = int(conversion_params["effective_scale"])
    saved_scaled = convert_lib.convert_tensor_to_scaled_int(saved_layer, effective_scale)
    computed_scaled = convert_lib.convert_tensor_to_scaled_int(computed_layer, effective_scale)
    saved_scaled_flat = saved_scaled.flatten()
    computed_scaled_flat = computed_scaled.flatten()

    dims, degree_bound, coefficients, coeffs_path = prove_index_lib.load_polynomial(poly_dir)
    if list(dims) != list(hyper_dims):
        sys.exit(
            f"[ERROR] Polynomial dims {dims} do not match hypercube dims {hyper_dims}"
        )
    commitment_hex = prove_index_lib.load_commitment(commitment_file)
    wrapper = tensorcommitments.TensorCommitmentWrapper(len(dims), degree_bound)

    positions = pick_positions(args.proof_mode, args.num_proofs, args.positions, layer_numel, args.seed)
    output_dir = bundle_output_dir(args, activations_path, layer_label)
    proofs_dir = output_dir / "proof_bundles"
    proofs_dir.mkdir(parents=True, exist_ok=True)

    proof_records: List[Dict[str, Any]] = []
    approx_mismatch_count = 0
    for order, layer_flat_index in enumerate(positions):
        global_flat = flat_offset + int(layer_flat_index)
        if global_flat >= total_real_elements:
            sys.exit(
                f"[ERROR] Global flat index {global_flat} maps into padding region; "
                "cannot generate commitment proof for padded entries."
            )

        index = [int(v) for v in np.unravel_index(global_flat, tuple(hyper_dims))]
        computed_value_int = int(computed_scaled_flat[layer_flat_index])
        saved_value_int = int(saved_scaled_flat[layer_flat_index])
        hypercube_value = int(hypercube[tuple(index)])

        if saved_value_int != hypercube_value:
            sys.exit(
                f"[ERROR] Value mismatch at index {index}: "
                f"saved_scaled={saved_value_int}, hypercube={hypercube_value}"
            )

        if computed_value_int != hypercube_value:
            if not args.allow_approx_commitment:
                sys.exit(
                    f"[ERROR] Value mismatch at index {index}: "
                    f"computed_scaled={computed_value_int}, committed={hypercube_value}. "
                    "Re-run with --allow-approx-commitment to prove committed values while recording this mismatch."
                )
            approx_mismatch_count += 1
            print(
                f"[WARN] Approx commitment at index {index}: "
                f"computed_scaled={computed_value_int}, committed={hypercube_value}"
            )
        value_int = hypercube_value

        poly_eval = int(wrapper.evaluate_polynomial(coefficients, index))
        value_field = value_int % BN254_P
        if poly_eval != value_field:
            sys.exit(
                f"[ERROR] Polynomial eval mismatch at index {index}: "
                f"poly_eval={poly_eval}, expected_field={value_field}, raw_value={value_int}"
            )

        proof_hex = wrapper.prove(coefficients, index, value_int)
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
            "index": index,
            "flat_index": global_flat,
            "value_int": str(value_int),
            "value_source": f"committed_hypercube:{layer_label}",
            "proof_hex": proof_hex,
            "layer_label": layer_label,
            "layer_flat_index": int(layer_flat_index),
            "hypercube_value_int": str(hypercube_value),
            "saved_scaled_value_int": str(saved_value_int),
            "computed_scaled_value_int": str(computed_value_int),
            "polynomial_eval_int": str(poly_eval),
        }

        bundle_path = proofs_dir / f"proof_{order:04d}_flat_{global_flat}.json"
        with bundle_path.open("w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2)

        proof_records.append(
            {
                "bundle_path": str(bundle_path),
                "index": index,
                "flat_index": global_flat,
                "layer_flat_index": int(layer_flat_index),
                "value_int": str(value_int),
                "saved_scaled_value_int": str(saved_value_int),
                "computed_scaled_value_int": str(computed_value_int),
                "proof_elements": len(proof_hex),
            }
        )

    print(
        f"[INFO] Generated {len(proof_records)} proof bundle(s) in {proofs_dir} "
        f"(approx_mismatches={approx_mismatch_count})"
    )

    print("[INFO] Stage 3 - crypto verification of saved bundles")
    verification_records: List[Dict[str, Any]] = []
    all_verified = True
    for record in proof_records:
        bundle_path = Path(record["bundle_path"])
        with bundle_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        verified = bool(
            wrapper.verify(
                payload["commitment_hex"],
                payload["index"],
                int(payload["value_int"]),
                payload["proof_hex"],
            )
        )
        verification_records.append(
            {
                "bundle_path": str(bundle_path),
                "index": payload["index"],
                "verified": verified,
            }
        )
        if not verified:
            all_verified = False

    print(f"[INFO] Crypto verification result: all_verified={all_verified}")
    if not all_verified:
        print("[STOP] At least one crypto verification failed. Please review bundle outputs.")
        sys.exit(3)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "activations_pt": str(activations_path),
            "hypercube_dir": str(hypercube_dir),
            "poly_dir": str(poly_dir),
            "commitment_file": str(commitment_file),
            "layer": layer_label,
            "proof_mode": args.proof_mode,
            "num_proofs": args.num_proofs,
            "positions": args.positions,
            "rtol": args.rtol,
            "atol": args.atol,
            "allow_approx_commitment": args.allow_approx_commitment,
        },
        "compute_verification": {
            "matched": compute_match,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "layer_shape": list(saved_layer.shape),
            "effective_scale": effective_scale,
        },
        "proof_generation": {
            "count": len(proof_records),
            "approx_mismatch_count": approx_mismatch_count,
            "proof_records": proof_records,
        },
        "crypto_verification": {
            "all_verified": all_verified,
            "records": verification_records,
        },
    }

    summary_path = output_dir / "compute_crypto_verification_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[INFO] Summary written to {summary_path}")
    print("[INFO] Completed all stages successfully.")


if __name__ == "__main__":
    main()
