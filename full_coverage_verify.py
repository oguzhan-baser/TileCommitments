#!/usr/bin/env python3
"""
Full-coverage verification (compute + crypto) across all model layers.

For each transformer layer (layer_0 ... layer_{N-1}):
1) Pick one random activation entry and save index metadata.
2) Generate one opening proof for the corresponding hypercube index/value.
3) Run crypto verification for that proof.
4) Recompute that layer output from the saved layer input using only that layer's
   weights, and compare the selected entry to the saved activation entry.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

import compute_crypto_verify_layer as layer_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-coverage verification: one random entry per layer, with proof + crypto + per-layer compute check."
    )
    parser.add_argument("--activations-pt", type=Path, required=True, help="Path to <model>_activations.pt")
    parser.add_argument("--hypercube-dir", type=Path, required=True, help="Directory with hypercube.npy and hypercube_metadata.json")
    parser.add_argument("--poly-dir", type=Path, required=True, help="Directory containing coefficients.json")
    parser.add_argument("--commitment-file", type=Path, required=True, help="Path to commitment.txt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-layer entry selection")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda). Use --device cpu to force CPU.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Model dtype (default: float16 on CUDA, float32 on CPU).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="none",
        choices=["none", "auto", "balanced", "balanced_low_0", "sequential"],
        help="Transformers device_map for model sharding (default: none).",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        type=str,
        default=None,
        help="Per-GPU memory budget (e.g. 20GiB) used when device-map is enabled.",
    )
    parser.add_argument(
        "--max-memory-cpu",
        type=str,
        default=None,
        help="CPU RAM budget (e.g. 128GiB) used with device-map max_memory.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="Compute verification rtol (entry-level)")
    parser.add_argument("--atol", type=float, default=5e-2, help="Compute verification atol (entry-level)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <activations parent>/full_coverage_verification",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, activations_path: Path) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    return (activations_path.parent / "full_coverage_verification").resolve()


def require_layer_entry(layer_map: Sequence[Dict[str, Any]], layer_label: str) -> Dict[str, Any]:
    entry = next((item for item in layer_map if item.get("layer") == layer_label), None)
    if entry is None:
        raise ValueError(f"Layer '{layer_label}' not found in hypercube_metadata.layer_map")
    return entry


def choose_layer_entries(
    saved_hidden_states: Sequence[torch.Tensor],
    layer_map: Sequence[Dict[str, Any]],
    hyper_dims: Sequence[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    num_layers = len(saved_hidden_states) - 1
    selections: List[Dict[str, Any]] = []

    for layer_idx in range(num_layers):
        layer_label = f"layer_{layer_idx}"
        layer_out = saved_hidden_states[layer_idx + 1]
        layer_numel = int(layer_out.numel())
        selected_flat = int(rng.integers(0, layer_numel))
        tensor_index = [int(v) for v in np.unravel_index(selected_flat, tuple(layer_out.shape))]

        layer_entry = require_layer_entry(layer_map, layer_label)
        flat_offset = int(layer_entry["flat_offset"])
        global_flat = flat_offset + selected_flat
        hypercube_index = [int(v) for v in np.unravel_index(global_flat, tuple(hyper_dims))]

        selections.append(
            {
                "layer_index": layer_idx,
                "layer_label": layer_label,
                "layer_shape": [int(v) for v in layer_out.shape],
                "layer_flat_index": selected_flat,
                "tensor_index": tensor_index,
                "global_flat_index": global_flat,
                "hypercube_index": hypercube_index,
                "saved_value_float": float(layer_out.flatten()[selected_flat].item()),
            }
        )

    return selections


def compute_single_layer_output(
    model: torch.nn.Module,
    layer_idx: int,
    layer_input: torch.Tensor,
    dtype: torch.dtype,
    include_final_norm: bool = False,
) -> torch.Tensor:
    fallback_device = layer_utils.get_model_input_device(model, torch.device("cpu"))
    decoder_layer = model.model.layers[layer_idx]
    layer_device = layer_utils.infer_module_device(decoder_layer, fallback_device)

    layer_input = layer_input.to(device=layer_device, dtype=dtype)
    sequence_len = layer_input.shape[1]
    cache_position = torch.arange(sequence_len, device=layer_device)
    position_ids = cache_position.unsqueeze(0)

    mask_kwargs = {
        "config": model.model.config,
        "input_embeds": layer_input,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(model.model, "has_sliding_layers", False):
        causal_masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = model.model.rotary_emb(layer_input, position_ids)
    attention_type = getattr(decoder_layer, "attention_type", "full_attention")
    mask = causal_masks.get(attention_type, causal_masks["full_attention"])

    with torch.no_grad():
        output = decoder_layer(
            layer_input,
            attention_mask=mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
        )

    if isinstance(output, tuple):
        output = output[0]

    if include_final_norm:
        if not hasattr(model.model, "norm") or model.model.norm is None:
            raise ValueError("Model does not expose model.norm required for final hidden-state verification")
        norm_device = layer_utils.infer_module_device(model.model.norm, layer_device)
        output = output.to(device=norm_device, dtype=dtype)
        with torch.no_grad():
            output = model.model.norm(output)

    return output.detach().cpu()


def main() -> None:
    args = parse_args()
    activations_path = args.activations_pt.resolve()
    hypercube_dir = args.hypercube_dir.resolve()
    poly_dir = args.poly_dir.resolve()
    commitment_file = args.commitment_file.resolve()

    artifacts = layer_utils.load_pt_artifacts(activations_path)
    saved_hidden_states: Sequence[torch.Tensor] = artifacts["hidden_states"]
    model_name = artifacts["model_name"]
    num_layers = len(saved_hidden_states) - 1

    if num_layers <= 0:
        raise ValueError("No transformer layers found in saved hidden_states")

    metadata = layer_utils.load_hypercube_metadata(hypercube_dir)
    hypercube = np.load(hypercube_dir / "hypercube.npy", allow_pickle=True)
    layer_map = metadata.get("layer_map", [])
    hyper_dims = [int(v) for v in metadata["hypercube"]["dimensions"]]
    total_real_elements = int(metadata["hypercube"]["total_real_elements"])

    conversion_params = metadata.get("conversion_params", {})
    if "effective_scale" not in conversion_params:
        raise ValueError("conversion_params.effective_scale missing from hypercube metadata")
    effective_scale = int(conversion_params["effective_scale"])

    output_dir = resolve_output_dir(args, activations_path)
    proof_dir = output_dir / "proof_bundles"
    output_dir.mkdir(parents=True, exist_ok=True)
    proof_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Stage 1 - selecting one random entry per layer (total layers={num_layers})")
    selections = choose_layer_entries(saved_hidden_states, layer_map, hyper_dims, args.seed)
    selections_path = output_dir / "selected_entries.json"
    with selections_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "seed": args.seed,
                "num_layers": num_layers,
                "entries": selections,
            },
            handle,
            indent=2,
        )
    print(f"[INFO] Saved selected indices: {selections_path}")

    print("[INFO] Stage 2 - generating one proof per selected layer entry")
    dims, degree_bound, coefficients, coefficients_path = layer_utils.prove_index_lib.load_polynomial(poly_dir)
    if list(dims) != list(hyper_dims):
        raise ValueError(f"Polynomial dims {dims} do not match hypercube dims {hyper_dims}")
    commitment_hex = layer_utils.prove_index_lib.load_commitment(commitment_file)
    wrapper = layer_utils.tensorcommitments.TensorCommitmentWrapper(len(dims), degree_bound)

    proof_records: List[Dict[str, Any]] = []
    for item in selections:
        layer_idx = int(item["layer_index"])
        layer_label = item["layer_label"]
        layer_flat_index = int(item["layer_flat_index"])
        global_flat = int(item["global_flat_index"])
        hypercube_index = [int(v) for v in item["hypercube_index"]]

        if global_flat >= total_real_elements:
            raise ValueError(
                f"Selected index for {layer_label} maps into hypercube padding region: global_flat={global_flat}"
            )

        saved_layer = saved_hidden_states[layer_idx + 1]
        saved_scaled_flat = layer_utils.convert_lib.convert_tensor_to_scaled_int(saved_layer, effective_scale).flatten()
        saved_scaled_value = int(saved_scaled_flat[layer_flat_index])
        committed_value = int(hypercube[tuple(hypercube_index)])
        if saved_scaled_value != committed_value:
            raise ValueError(
                f"Saved scaled value mismatch for {layer_label}: "
                f"saved_scaled={saved_scaled_value}, committed={committed_value}"
            )

        poly_eval = int(wrapper.evaluate_polynomial(coefficients, hypercube_index))
        if poly_eval != (committed_value % layer_utils.BN254_P):
            raise ValueError(
                f"Polynomial eval mismatch for {layer_label}: poly_eval={poly_eval}, committed={committed_value}"
            )

        proof_hex = wrapper.prove(coefficients, hypercube_index, committed_value)
        bundle = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "layer_index": layer_idx,
            "layer_label": layer_label,
            "layer_flat_index": layer_flat_index,
            "tensor_index": item["tensor_index"],
            "global_flat_index": global_flat,
            "hypercube_index": hypercube_index,
            "value_int": str(committed_value),
            "saved_scaled_value_int": str(saved_scaled_value),
            "proof_hex": proof_hex,
            "commitment_hex": commitment_hex,
            "dims": dims,
            "degree_bound": degree_bound,
            "poly_dir": str(poly_dir),
            "coefficients_file": str(coefficients_path),
            "commitment_file": str(commitment_file),
        }
        bundle_path = proof_dir / f"proof_{layer_label}.json"
        with bundle_path.open("w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2)

        proof_records.append(
            {
                "layer_label": layer_label,
                "bundle_path": str(bundle_path),
                "hypercube_index": hypercube_index,
                "value_int": str(committed_value),
            }
        )

    if len(proof_records) != num_layers:
        raise RuntimeError(f"Expected {num_layers} proofs, generated {len(proof_records)}")
    print(f"[INFO] Generated {len(proof_records)} proof bundles in {proof_dir}")

    print("[INFO] Stage 3 - crypto verification for every generated proof")
    crypto_records: List[Dict[str, Any]] = []
    all_crypto_verified = True
    for record in proof_records:
        with Path(record["bundle_path"]).open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        verified = bool(
            wrapper.verify(
                bundle["commitment_hex"],
                bundle["hypercube_index"],
                int(bundle["value_int"]),
                bundle["proof_hex"],
            )
        )
        crypto_records.append(
            {
                "layer_label": bundle["layer_label"],
                "bundle_path": record["bundle_path"],
                "verified": verified,
            }
        )
        if not verified:
            all_crypto_verified = False

    print(f"[INFO] Crypto verification all_verified={all_crypto_verified}")
    if not all_crypto_verified:
        raise RuntimeError("At least one crypto verification failed")

    print("[INFO] Stage 4/5 - per-layer compute verification for selected entry in each layer")
    device = layer_utils.capture_lib.resolve_device(args.device)
    dtype = layer_utils.capture_lib.resolve_dtype(args.dtype, device)
    device_map = layer_utils.normalize_device_map(args.device_map)
    if device.type != "cuda":
        if device_map is not None:
            print("[WARN] --device-map ignored because --device is not CUDA")
        device_map = None
    max_memory = layer_utils.build_max_memory_map(device, args.max_memory_per_gpu, args.max_memory_cpu)
    model = layer_utils.load_model_for_compute_verification(
        model_name,
        device,
        dtype,
        device_map=device_map,
        max_memory=max_memory,
    )
    placement = layer_utils.describe_model_placement(model)
    print(f"[INFO] Model placement: {placement['module_count_by_device']}")

    compute_records: List[Dict[str, Any]] = []
    all_compute_match = True
    for item in selections:
        layer_idx = int(item["layer_index"])
        layer_label = item["layer_label"]
        layer_flat_index = int(item["layer_flat_index"])

        layer_input = saved_hidden_states[layer_idx]
        saved_output = saved_hidden_states[layer_idx + 1]
        include_final_norm = layer_idx == (num_layers - 1)
        computed_output = compute_single_layer_output(
            model,
            layer_idx,
            layer_input,
            dtype,
            include_final_norm=include_final_norm,
        )

        saved_value = float(saved_output.flatten()[layer_flat_index].item())
        computed_value = float(computed_output.flatten()[layer_flat_index].item())
        abs_diff = float(abs(saved_value - computed_value))
        matches = bool(np.isclose(saved_value, computed_value, rtol=args.rtol, atol=args.atol))

        compute_records.append(
            {
                "layer_index": layer_idx,
                "layer_label": layer_label,
                "layer_flat_index": layer_flat_index,
                "tensor_index": item["tensor_index"],
                "saved_value_float": saved_value,
                "computed_value_float": computed_value,
                "abs_diff": abs_diff,
                "matches": matches,
            }
        )
        if not matches:
            all_compute_match = False

    del model
    print(f"[INFO] Compute verification all_matched={all_compute_match}")

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "activations_pt": str(activations_path),
            "hypercube_dir": str(hypercube_dir),
            "poly_dir": str(poly_dir),
            "commitment_file": str(commitment_file),
            "model_name": model_name,
            "seed": args.seed,
            "num_layers": num_layers,
            "rtol": args.rtol,
            "atol": args.atol,
            "device": str(device),
            "dtype": str(dtype),
            "device_map": device_map,
            "max_memory": max_memory,
        },
        "model_placement": placement,
        "selected_entries_file": str(selections_path),
        "proof_generation": {
            "expected_count": num_layers,
            "generated_count": len(proof_records),
            "proof_records": proof_records,
        },
        "crypto_verification": {
            "all_verified": all_crypto_verified,
            "records": crypto_records,
        },
        "compute_verification": {
            "all_matched": all_compute_match,
            "records": compute_records,
        },
    }
    summary_path = output_dir / "full_coverage_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[INFO] Summary saved to {summary_path}")
    if not all_compute_match:
        raise SystemExit(4)

    print("[INFO] Full-coverage verification completed successfully.")


if __name__ == "__main__":
    main()
