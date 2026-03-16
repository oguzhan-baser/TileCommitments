#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_LIB_DIR = REPO_ROOT / "TensorCommitment" / "activationCaptureLib"
if str(ACTIVATION_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_LIB_DIR))

import capture_activations as capture_lib  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-prover sequential inference + per-prover commitments + sampled proof verification "
            "with one-prover baseline comparison."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for a single forward pass")
    parser.add_argument("--num-provers", type=int, default=3, help="Number of sequential provers")
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids to use (default: first N GPUs).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Computation dtype for prover stages.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for commitment sampling stages")
    parser.add_argument("--scale-factor", type=int, default=16, help="Scale factor for float->int conversion")
    parser.add_argument("--quantize", type=float, default=50.0, help="Quantization percentage for conversion")
    parser.add_argument("--min-dim", type=int, default=4, help="Hypercube minimum dimension")
    parser.add_argument("--max-dim", type=int, default=10, help="Hypercube maximum dimension")
    parser.add_argument("--num-queries", type=int, default=10, help="Sampled commitment prove/verify queries per part")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Allclose rtol for one-vs-multiprover comparison")
    parser.add_argument("--atol", type=float, default=5e-2, help="Allclose atol for one-vs-multiprover comparison")
    parser.add_argument("--skip-interp-build", action="store_true", help="Pass --skip-build to interpolation stage")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: multiproversetting/output/run_YYYYmmdd_HHMMSS_utc).",
    )
    return parser.parse_args()


def resolve_output_dir(path_arg: Path | None) -> Path:
    if path_arg is not None:
        return path_arg.resolve()
    run_tag = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_utc"
    return (REPO_ROOT / "multiproversetting" / "output" / run_tag).resolve()


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_arg]


def resolve_prover_devices(num_provers: int, device_ids_arg: str | None) -> List[torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multiprover mode but no GPU was detected.")

    total = torch.cuda.device_count()
    if device_ids_arg:
        parsed = [int(part.strip()) for part in device_ids_arg.split(",") if part.strip()]
    else:
        parsed = list(range(total))

    if len(parsed) < num_provers:
        raise ValueError(
            f"Requested num_provers={num_provers}, but only {len(parsed)} device ids were provided/available: {parsed}"
        )
    selected = parsed[:num_provers]
    if len(set(selected)) != len(selected):
        raise ValueError(f"Each prover must use a distinct GPU. Got duplicates in: {selected}")
    if any(idx < 0 or idx >= total for idx in selected):
        raise ValueError(f"Invalid CUDA index in {selected}. Available GPU indices: 0..{total - 1}")
    return [torch.device(f"cuda:{idx}") for idx in selected]


def split_layer_ranges(total_layers: int, num_provers: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for prover_idx in range(num_provers):
        start = (prover_idx * total_layers) // num_provers
        end = ((prover_idx + 1) * total_layers) // num_provers
        ranges.append((start, end))
    return ranges


def ensure_model_layout(model: torch.nn.Module) -> None:
    if not hasattr(model, "model"):
        raise ValueError("Expected model.model for decoder stack access; unsupported architecture for this script.")
    if not hasattr(model.model, "layers"):
        raise ValueError("Expected model.model.layers; unsupported architecture for this script.")
    if not hasattr(model.model, "rotary_emb"):
        raise ValueError("Expected model.model.rotary_emb; unsupported architecture for this script.")


def forward_decoder_layer(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    position_ids: torch.Tensor,
    position_embeddings: Any,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    attempts: List[Dict[str, Any]] = [
        {
            "attention_mask": mask,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "cache_position": cache_position,
        },
        {
            "attention_mask": mask,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "past_key_value": None,
            "use_cache": False,
            "cache_position": cache_position,
        },
        {
            "attention_mask": mask,
            "position_ids": position_ids,
            "past_key_value": None,
            "use_cache": False,
        },
        {
            "attention_mask": mask,
            "position_ids": position_ids,
            "use_cache": False,
        },
        {"attention_mask": mask},
    ]

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            out = layer(hidden, **kwargs)
            if isinstance(out, tuple):
                return out[0]
            return out
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Decoder-layer forward failed unexpectedly.")


def run_layer_range(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    start: int,
    end: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    outputs_cpu: List[torch.Tensor] = []
    t_start = time.perf_counter()
    rotary_emb = model.model.rotary_emb.to(device=device)

    for layer_idx in range(start, end):
        layer = model.model.layers[layer_idx].to(device=device, dtype=dtype)
        hidden = hidden.to(device=device, dtype=dtype)
        seq_len = hidden.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": hidden,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(model.model, "has_sliding_layers", False):
            causal_masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = rotary_emb(hidden, position_ids)
        attention_type = getattr(layer, "attention_type", "full_attention")
        mask = causal_masks.get(attention_type, causal_masks["full_attention"])

        with torch.no_grad():
            hidden = forward_decoder_layer(
                layer,
                hidden,
                mask,
                position_ids,
                position_embeddings,
                cache_position,
            )

        outputs_cpu.append(hidden.detach().cpu())

        layer.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rotary_emb.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wall_seconds = time.perf_counter() - t_start
    return hidden, outputs_cpu, wall_seconds


def run_partitioned_inference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    ranges: Sequence[Tuple[int, int]],
    devices: Sequence[torch.device],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    embedding_module = model.get_input_embeddings()
    if embedding_module is None:
        raise ValueError("Model does not expose input embeddings.")

    first_device = devices[0]
    embedding_module = embedding_module.to(device=first_device, dtype=dtype)
    with torch.no_grad():
        hidden = embedding_module(input_ids.to(first_device))
    embedding_output_cpu = hidden.detach().cpu()
    embedding_module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_prover: List[Dict[str, Any]] = []
    all_layer_outputs_cpu: List[torch.Tensor] = []
    exchange_records: List[Dict[str, Any]] = []

    for prover_idx, ((start, end), device) in enumerate(zip(ranges, devices)):
        prover_input_cpu = hidden.detach().cpu()
        hidden, layer_outputs, range_wall_seconds = run_layer_range(
            model=model,
            hidden=hidden,
            start=start,
            end=end,
            device=device,
            dtype=dtype,
        )
        all_layer_outputs_cpu.extend(layer_outputs)
        per_prover.append(
            {
                "prover_index": prover_idx,
                "device": str(device),
                "layer_start": start,
                "layer_end_exclusive": end,
                "input_tensor_cpu": prover_input_cpu,
                "layer_outputs_cpu": layer_outputs,
                "compute_seconds": float(range_wall_seconds),
            }
        )
        if prover_idx + 1 < len(devices):
            exchange_bytes = int(hidden.numel() * hidden.element_size())
            exchange_records.append(
                {
                    "from_prover_index": prover_idx,
                    "to_prover_index": prover_idx + 1,
                    "from_device": str(device),
                    "to_device": str(devices[prover_idx + 1]),
                    "tensor_shape": list(hidden.shape),
                    "tensor_dtype": str(hidden.dtype),
                    "payload_bytes": exchange_bytes,
                    "payload_mib": float(exchange_bytes / (1024.0 * 1024.0)),
                }
            )
            hidden = hidden.to(devices[prover_idx + 1])

    last_device = devices[-1]
    hidden = hidden.to(last_device, dtype=dtype)

    final_hidden = hidden
    if hasattr(model.model, "norm") and model.model.norm is not None:
        norm_module = model.model.norm.to(device=last_device, dtype=dtype)
        with torch.no_grad():
            final_hidden = norm_module(final_hidden)
        norm_module.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_head = model.get_output_embeddings()
    if output_head is None and hasattr(model, "lm_head"):
        output_head = model.lm_head
    if output_head is None:
        raise ValueError("Model does not expose output projection head for logits.")

    output_head = output_head.to(device=last_device, dtype=dtype)
    with torch.no_grad():
        logits = output_head(final_hidden)
    output_head.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())

    return {
        "embedding_output_cpu": embedding_output_cpu,
        "all_layer_outputs_cpu": all_layer_outputs_cpu,
        "per_prover": per_prover,
        "exchange_records": exchange_records,
        "final_hidden_cpu": final_hidden.detach().cpu(),
        "logits_cpu": logits.detach().cpu(),
        "next_token_id": next_token_id,
    }


def save_activation_artifact(
    artifact_path: Path,
    model_name: str,
    prompt: str,
    token_sequence: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "prompt": prompt,
            "generated_text": "",
            "token_sequence": token_sequence.cpu(),
            "hidden_states": tuple(hidden_states),
        },
        artifact_path,
    )


def run_cmd(args: List[str], cwd: Path) -> None:
    print(f"[cmd] {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd), check=True)


def run_commitment_pipeline_for_part(
    activations_pt: Path,
    part_dir: Path,
    scale_factor: int,
    quantize: float,
    min_dim: int,
    max_dim: int,
    num_queries: int,
    seed: int,
    skip_interp_build: bool,
) -> Dict[str, Any]:
    part_dir.mkdir(parents=True, exist_ok=True)
    int_dir = part_dir / "int_activations"
    hypercube_dir = part_dir / "hypercube"
    poly_dir = part_dir / "polynomial"
    commitment_dir = part_dir / "commitment"

    t0 = time.perf_counter()
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "TensorCommitment" / "activationCaptureLib" / "convert_to_npy.py"),
            "--input",
            str(activations_pt),
            "--output-dir",
            str(int_dir),
            "--scale-factor",
            str(scale_factor),
            "--quantize",
            str(quantize),
        ],
        cwd=REPO_ROOT,
    )
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "TensorCommitment" / "activationCaptureLib" / "reshape_to_hypercube.py"),
            "--input-dir",
            str(int_dir),
            "--output-dir",
            str(hypercube_dir),
            "--min-dim",
            str(min_dim),
            "--max-dim",
            str(max_dim),
        ],
        cwd=REPO_ROOT,
    )
    interp_cmd = [
        sys.executable,
        str(REPO_ROOT / "TensorCommitment" / "interpolationLib" / "interpolate_hypercube.py"),
        "--input-dir",
        str(hypercube_dir),
        "--output-dir",
        str(poly_dir),
    ]
    if skip_interp_build:
        interp_cmd.append("--skip-build")
    run_cmd(interp_cmd, cwd=REPO_ROOT)

    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "TensorCommitment" / "tensorCommitmentLib" / "commit_prove_verify.py"),
            "--poly-dir",
            str(poly_dir),
            "--hypercube-dir",
            str(hypercube_dir),
            "--output-dir",
            str(commitment_dir),
            "--num-queries",
            str(num_queries),
            "--seed",
            str(seed),
        ],
        cwd=REPO_ROOT,
    )
    total_wall_s = time.perf_counter() - t0

    commitment_results_path = commitment_dir / "commitment_results.json"
    if not commitment_results_path.is_file():
        raise FileNotFoundError(f"Missing commitment_results.json at {commitment_results_path}")
    with commitment_results_path.open("r", encoding="utf-8") as handle:
        commitment_results = json.load(handle)

    verification_summary = commitment_results.get("verification_summary", {})
    all_verified = bool(
        commitment_results.get("all_verified", verification_summary.get("all_proofs_verified", False))
    )
    all_ground_truth = commitment_results.get(
        "all_ground_truth_match",
        verification_summary.get("all_ground_truth_matched", None),
    )
    if not all_verified:
        raise RuntimeError(f"Random-sample crypto verification failed for {part_dir.name}")

    return {
        "part_dir": str(part_dir),
        "int_dir": str(int_dir),
        "hypercube_dir": str(hypercube_dir),
        "poly_dir": str(poly_dir),
        "commitment_dir": str(commitment_dir),
        "commitment_results_json": str(commitment_results_path),
        "pipeline_wall_seconds": total_wall_s,
        "all_verified": all_verified,
        "all_ground_truth_match": all_ground_truth,
        "commitment_results": commitment_results,
    }


def to_jsonable_prover_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        out.append(
            {
                "prover_index": int(rec["prover_index"]),
                "device": str(rec["device"]),
                "layer_start": int(rec["layer_start"]),
                "layer_end_exclusive": int(rec["layer_end_exclusive"]),
                "input_shape": list(rec["input_tensor_cpu"].shape),
                "num_layer_outputs": len(rec["layer_outputs_cpu"]),
                "layer_output_shapes": [list(t.shape) for t in rec["layer_outputs_cpu"]],
                "compute_seconds": float(rec.get("compute_seconds", 0.0)),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)
    devices = resolve_prover_devices(args.num_provers, args.device_ids)
    print(f"[stage1] Loading model={args.model}")
    print(f"[stage1] Provers={args.num_provers}, devices={[str(d) for d in devices]}, dtype={dtype}")

    model, tokenizer = capture_lib.load_model_and_tokenizer(
        args.model,
        torch.device("cpu"),
        dtype,
        device_map=None,
        max_memory=None,
        mxfp4_mode="auto",
    )
    ensure_model_layout(model)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    token_sequence = inputs["input_ids"].cpu()
    total_layers = len(model.model.layers)
    if total_layers <= 0:
        raise ValueError("Model exposes zero decoder layers.")
    ranges = split_layer_ranges(total_layers, args.num_provers)

    assignment = {
        "model_name": args.model,
        "prompt": args.prompt,
        "num_layers": total_layers,
        "num_provers": args.num_provers,
        "dtype": args.dtype,
        "devices": [str(device) for device in devices],
        "layer_ranges": [{"prover_index": i, "start": s, "end_exclusive": e} for i, (s, e) in enumerate(ranges)],
    }
    with (output_dir / "prover_assignment.json").open("w", encoding="utf-8") as handle:
        json.dump(assignment, handle, indent=2)
    print(f"[stage1] Saved prover assignment: {output_dir / 'prover_assignment.json'}")

    print("[stage1] Running multiprover sequential inference")
    t_multi_compute = time.perf_counter()
    multi_result = run_partitioned_inference(
        model=model,
        input_ids=token_sequence,
        ranges=ranges,
        devices=devices,
        dtype=dtype,
    )
    multi_compute_seconds = time.perf_counter() - t_multi_compute
    print(f"[stage1] Multiprover compute completed in {multi_compute_seconds:.3f}s")

    print("[stage2] Running one-prover baseline inference")
    t_one_compute = time.perf_counter()
    one_result = run_partitioned_inference(
        model=model,
        input_ids=token_sequence,
        ranges=[(0, total_layers)],
        devices=[devices[0]],
        dtype=dtype,
    )
    one_compute_seconds = time.perf_counter() - t_one_compute
    print(f"[stage2] One-prover compute completed in {one_compute_seconds:.3f}s")

    hidden_match = bool(
        torch.allclose(
            one_result["final_hidden_cpu"].float(),
            multi_result["final_hidden_cpu"].float(),
            rtol=args.rtol,
            atol=args.atol,
        )
    )
    logits_match = bool(
        torch.allclose(
            one_result["logits_cpu"].float(),
            multi_result["logits_cpu"].float(),
            rtol=args.rtol,
            atol=args.atol,
        )
    )
    hidden_abs_diff = torch.abs(one_result["final_hidden_cpu"].float() - multi_result["final_hidden_cpu"].float())
    logits_abs_diff = torch.abs(one_result["logits_cpu"].float() - multi_result["logits_cpu"].float())
    comparison = {
        "hidden_allclose": hidden_match,
        "logits_allclose": logits_match,
        "one_prover_next_token_id": int(one_result["next_token_id"]),
        "multi_prover_next_token_id": int(multi_result["next_token_id"]),
        "next_token_match": int(one_result["next_token_id"]) == int(multi_result["next_token_id"]),
        "hidden_max_abs_diff": float(hidden_abs_diff.max().item()),
        "hidden_mean_abs_diff": float(hidden_abs_diff.mean().item()),
        "logits_max_abs_diff": float(logits_abs_diff.max().item()),
        "logits_mean_abs_diff": float(logits_abs_diff.mean().item()),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
    }
    with (output_dir / "one_vs_multi_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    print(f"[stage2] Saved comparison: {output_dir / 'one_vs_multi_comparison.json'}")
    if not comparison["next_token_match"]:
        raise RuntimeError("Stage 2 failed: next-token mismatch between one-prover and multi-prover outputs.")
    if not (comparison["hidden_allclose"] and comparison["logits_allclose"]):
        raise RuntimeError("Stage 2 failed: one-prover and multi-prover tensors are not allclose under configured tolerances.")
    print("[stage2] Inference outputs match one-prover baseline.")

    print("[stage3] Preparing per-prover activation artifacts")
    one_dir = output_dir / "one_prover"
    one_artifact = one_dir / "activations.pt"
    one_hidden_states = [one_result["embedding_output_cpu"], *one_result["all_layer_outputs_cpu"]]
    save_activation_artifact(one_artifact, args.model, args.prompt, token_sequence, one_hidden_states)

    multi_dir = output_dir / "multi_prover"
    prover_artifacts: List[Path] = []
    for rec in multi_result["per_prover"]:
        prover_idx = int(rec["prover_index"])
        part_dir = multi_dir / f"prover_{prover_idx:02d}"
        artifact_path = part_dir / "activations.pt"
        part_hidden_states = [rec["input_tensor_cpu"], *rec["layer_outputs_cpu"]]
        save_activation_artifact(artifact_path, args.model, args.prompt, token_sequence, part_hidden_states)
        prover_artifacts.append(artifact_path)
    with (output_dir / "multiprover_compute_layout.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "per_prover": to_jsonable_prover_records(multi_result["per_prover"]),
                "embedding_exchanges": multi_result.get("exchange_records", []),
            },
            handle,
            indent=2,
        )
    print(f"[stage3] Saved per-prover compute layout: {output_dir / 'multiprover_compute_layout.json'}")

    print("[stage3] Running one-prover commitment pipeline")
    one_commit = run_commitment_pipeline_for_part(
        activations_pt=one_artifact,
        part_dir=one_dir,
        scale_factor=args.scale_factor,
        quantize=args.quantize,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        num_queries=args.num_queries,
        seed=args.seed,
        skip_interp_build=args.skip_interp_build,
    )

    print("[stage3] Running per-prover commitment pipelines")
    per_prover_commit: List[Dict[str, Any]] = []
    for prover_idx, artifact_path in enumerate(prover_artifacts):
        commit_info = run_commitment_pipeline_for_part(
            activations_pt=artifact_path,
            part_dir=multi_dir / f"prover_{prover_idx:02d}",
            scale_factor=args.scale_factor,
            quantize=args.quantize,
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            num_queries=args.num_queries,
            seed=args.seed,
            skip_interp_build=args.skip_interp_build,
        )
        per_prover_commit.append(commit_info)

    print("[stage4] Verifying sampled commitment proofs for all provers")
    if not bool(one_commit.get("all_verified", False)):
        raise RuntimeError("One-prover sampled proof verification failed.")
    for idx, info in enumerate(per_prover_commit):
        if not bool(info.get("all_verified", False)):
            raise RuntimeError(f"Prover {idx} sampled proof verification failed.")
    print("[stage4] Sampled proof verification succeeded for one-prover and all multi-prover parts.")

    print("[stage5] Measuring one-prover vs multi-prover overhead")
    multi_commit_total = float(sum(info["pipeline_wall_seconds"] for info in per_prover_commit))
    one_commit_total = float(one_commit["pipeline_wall_seconds"])
    one_total = one_compute_seconds + one_commit_total
    multi_total = multi_compute_seconds + multi_commit_total
    per_prover_compute_seconds = [float(rec.get("compute_seconds", 0.0)) for rec in multi_result["per_prover"]]
    exchange_records = multi_result.get("exchange_records", [])
    total_exchange_bytes = int(sum(int(item.get("payload_bytes", 0)) for item in exchange_records))
    num_exchanges = int(len(exchange_records))
    avg_exchange_bytes = float(total_exchange_bytes / num_exchanges) if num_exchanges > 0 else 0.0

    overhead = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "prompt": args.prompt,
            "num_provers": args.num_provers,
            "devices": [str(device) for device in devices],
            "dtype": args.dtype,
            "scale_factor": args.scale_factor,
            "quantize": args.quantize,
            "num_queries": args.num_queries,
        },
        "stage_timings_seconds": {
            "one_prover_compute": one_compute_seconds,
            "multi_prover_compute": multi_compute_seconds,
            "one_prover_commit_pipeline_total": one_commit_total,
            "multi_prover_commit_pipeline_total": multi_commit_total,
            "one_prover_total_compute_plus_commit": one_total,
            "multi_prover_total_compute_plus_commit": multi_total,
        },
        "multi_prover_compute_summary": {
            "per_prover_compute_seconds": per_prover_compute_seconds,
            "avg_inference_time_across_agents_seconds": (
                float(sum(per_prover_compute_seconds) / len(per_prover_compute_seconds))
                if per_prover_compute_seconds
                else None
            ),
            "max_inference_time_across_agents_seconds": (
                float(max(per_prover_compute_seconds)) if per_prover_compute_seconds else None
            ),
        },
        "embedding_exchange_summary": {
            "num_exchanges": num_exchanges,
            "total_exchange_bytes": total_exchange_bytes,
            "total_exchange_mib": float(total_exchange_bytes / (1024.0 * 1024.0)),
            "average_exchange_bytes": avg_exchange_bytes,
            "average_exchange_mib": float(avg_exchange_bytes / (1024.0 * 1024.0)),
            "exchanges": exchange_records,
        },
        "overhead": {
            "compute_delta_seconds": multi_compute_seconds - one_compute_seconds,
            "compute_ratio_multi_over_one": (multi_compute_seconds / one_compute_seconds) if one_compute_seconds > 0 else None,
            "commit_pipeline_delta_seconds": multi_commit_total - one_commit_total,
            "commit_pipeline_ratio_multi_over_one": (multi_commit_total / one_commit_total) if one_commit_total > 0 else None,
            "total_delta_seconds": multi_total - one_total,
            "total_ratio_multi_over_one": (multi_total / one_total) if one_total > 0 else None,
        },
        "one_vs_multi_match": comparison,
        "one_prover_commit_summary": {
            "all_verified": bool(one_commit.get("all_verified", False)),
            "all_ground_truth_match": one_commit.get("all_ground_truth_match", None),
            "timing": one_commit["commitment_results"].get("timing", {}),
            "proof_stats": one_commit["commitment_results"].get("proof_stats", {}),
            "pipeline_wall_seconds": one_commit_total,
        },
        "multi_prover_commit_summary": [
            {
                "prover_index": idx,
                "all_verified": bool(info.get("all_verified", False)),
                "all_ground_truth_match": info.get("all_ground_truth_match", None),
                "timing": info["commitment_results"].get("timing", {}),
                "proof_stats": info["commitment_results"].get("proof_stats", {}),
                "pipeline_wall_seconds": float(info["pipeline_wall_seconds"]),
                "part_dir": info["part_dir"],
            }
            for idx, info in enumerate(per_prover_commit)
        ],
        "artifacts": {
            "output_dir": str(output_dir),
            "prover_assignment_json": str(output_dir / "prover_assignment.json"),
            "comparison_json": str(output_dir / "one_vs_multi_comparison.json"),
            "layout_json": str(output_dir / "multiprover_compute_layout.json"),
        },
    }

    with (output_dir / "overhead_report.json").open("w", encoding="utf-8") as handle:
        json.dump(overhead, handle, indent=2)
    print(f"[stage5] Saved overhead report: {output_dir / 'overhead_report.json'}")

    print("\n[done] Multi-prover pipeline completed successfully.")
    print(f"[done] Output directory: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
