#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

import compute_crypto_verify_layer as layer_utils
import full_coverage_verify as fullcov
import prove_verify_index as prove_index

try:
    import tensorcommitments
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] Cannot import tensorcommitments.\n"
        "Activate env and build bindings first:\n"
        "  conda activate tilecommitments\n"
        "  cd TensorCommitment/pst_commitment_lib && maturin develop --features python --release"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Empirically measure transfer-size and compute/proving overhead for one completed "
            "TileCommitments run directory (stages 5–8)."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under activationCaptureLib/output/")
    parser.add_argument("--model-id", type=str, default=None, help="HF model id. If omitted, read from activations artifact.")
    parser.add_argument("--full-coverage-dir", type=Path, default=None, help="Directory with selected_entries.json and proof_bundles/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for measurement artifacts.")
    parser.add_argument(
        "--root-report",
        type=Path,
        default=Path("final_empirical_report.md"),
        help="Repo-level markdown summary path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Request payload field (for prover input size).")
    parser.add_argument("--scale-factor", type=int, default=16, help="Request payload field (for prover input size).")
    parser.add_argument("--quantize", type=float, default=50.0, help="Request payload field (for prover input size).")
    parser.add_argument("--min-dim", type=int, default=4, help="Request payload field (for prover input size).")
    parser.add_argument("--max-dim", type=int, default=10, help="Request payload field (for prover input size).")
    parser.add_argument("--num-queries", type=int, default=10, help="Request payload field (for prover input size).")
    parser.add_argument("--seed", type=int, default=42, help="Request payload field (for prover input size).")
    parser.add_argument("--skip-gpu-overhead", action="store_true", help="Skip verifier GPU compute-overhead measurement.")
    parser.add_argument("--skip-cpu-overhead", action="store_true", help="Skip verifier CPU compute-overhead measurement.")
    parser.add_argument("--skip-model-transfer", action="store_true", help="Skip model-parameter transfer measurements.")
    parser.add_argument(
        "--force-recompute-model-transfer",
        action="store_true",
        help="Recompute model transfer artifacts even if model_param_transfer_sizes.json already exists.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def file_size(path: Path) -> int:
    return int(path.stat().st_size)


def dir_size(path: Path) -> int:
    total = 0
    for fp in path.rglob("*"):
        if fp.is_file():
            total += fp.stat().st_size
    return int(total)


def safe_clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for fp in sorted(path.rglob("*"), reverse=True):
        if fp.is_file():
            fp.unlink()
        elif fp.is_dir():
            try:
                fp.rmdir()
            except OSError:
                pass


def tar_gz_dir(src_dir: Path, out_targz: Path, base_dir: Path) -> int:
    with tarfile.open(out_targz, "w:gz") as tar:
        for fp in sorted(src_dir.rglob("*")):
            if fp.is_file():
                arcname = str(fp.relative_to(base_dir))
                tar.add(fp, arcname=arcname)
    return file_size(out_targz)


def gzip_file(src: Path, dst: Path) -> int:
    with src.open("rb") as in_f, gzip.open(dst, "wb", compresslevel=6) as out_f:
        shutil.copyfileobj(in_f, out_f, length=1024 * 1024)
    return file_size(dst)


def detect_artifacts(run_dir: Path, model_id_hint: str | None) -> Dict[str, Path]:
    if model_id_hint:
        safe = sanitize_model_name(model_id_hint)
        act_pt = run_dir / f"{safe}_activations.pt"
    else:
        candidates = sorted(run_dir.glob("*_activations.pt"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *_activations.pt in {run_dir}, found {len(candidates)}. "
                "Pass --model-id explicitly."
            )
        act_pt = candidates[0]
        safe = act_pt.stem.removesuffix("_activations")

    if not act_pt.is_file():
        raise FileNotFoundError(f"activations file not found: {act_pt}")

    hc_dir = run_dir / f"{safe}_int_activations_hypercube"
    poly_dir = run_dir / f"{safe}_int_activations_hypercube_polynomial"
    commit_dir = run_dir / f"{safe}_int_activations_hypercube_commitment"
    for req in (hc_dir / "hypercube.npy", poly_dir / "coefficients.json", commit_dir / "commitment.txt"):
        if not req.is_file():
            raise FileNotFoundError(f"Required artifact missing: {req}")

    return {
        "activations_pt": act_pt,
        "hypercube_dir": hc_dir,
        "poly_dir": poly_dir,
        "commitment_dir": commit_dir,
    }


def build_non_model_transfer_sizes(
    *,
    output_dir: Path,
    activations_payload: Dict[str, Any],
    model_id: str,
    prompt: str,
    full_coverage_dir: Path,
    commitment_file: Path,
    request_config: Dict[str, Any],
) -> Dict[str, Any]:
    verifier_pkg = output_dir / "verifier_min_payload"
    prover_pkg = output_dir / "prover_min_request"
    safe_clear_dir(verifier_pkg)
    safe_clear_dir(prover_pkg)
    verifier_pkg.mkdir(parents=True, exist_ok=True)
    prover_pkg.mkdir(parents=True, exist_ok=True)

    selected_entries = full_coverage_dir / "selected_entries.json"
    proof_dir = full_coverage_dir / "proof_bundles"
    if not selected_entries.is_file():
        raise FileNotFoundError(f"selected_entries.json not found: {selected_entries}")
    if not proof_dir.is_dir():
        raise FileNotFoundError(f"proof_bundles directory not found: {proof_dir}")

    # verifier-side minimal package for current full_coverage_verify implementation
    (verifier_pkg / "commitment.txt").write_bytes(commitment_file.read_bytes())
    (verifier_pkg / "selected_entries.json").write_bytes(selected_entries.read_bytes())
    verifier_proof_dir = verifier_pkg / "proof_bundles"
    verifier_proof_dir.mkdir(exist_ok=True)
    for proof_file in sorted(proof_dir.glob("proof_*.json")):
        (verifier_proof_dir / proof_file.name).write_bytes(proof_file.read_bytes())

    hidden_states: Sequence[torch.Tensor] = activations_payload["hidden_states"]
    layer_inputs = tuple(t.cpu() for t in hidden_states[:-1])
    torch.save(
        {
            "model_name": activations_payload["model_name"],
            "prompt": activations_payload["prompt"],
            "layer_inputs": layer_inputs,
        },
        verifier_pkg / "layer_inputs.pt",
    )
    manifest = {
        "generated_at_utc": utc_now(),
        "model_name": activations_payload["model_name"],
        "note": "Minimal verifier payload for current full_coverage_verify implementation (crypto+compute).",
        "files": [str(p.relative_to(verifier_pkg)) for p in sorted(verifier_pkg.rglob("*")) if p.is_file()],
    }
    (verifier_pkg / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # prover-side minimal request package
    request_payload = {
        "model_name": model_id,
        "prompt": prompt,
        **request_config,
    }
    (prover_pkg / "request_payload.json").write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

    verifier_raw = dir_size(verifier_pkg)
    verifier_tgz = tar_gz_dir(verifier_pkg, output_dir / "verifier_min_payload.tar.gz", output_dir)
    prover_raw = dir_size(prover_pkg)
    prover_tgz = tar_gz_dir(prover_pkg, output_dir / "prover_min_request.tar.gz", output_dir)

    return {
        "verifier_min_payload": {
            "raw_bytes": verifier_raw,
            "compressed_bytes": verifier_tgz,
            "component_bytes": {
                "commitment_txt": file_size(verifier_pkg / "commitment.txt"),
                "selected_entries_json": file_size(verifier_pkg / "selected_entries.json"),
                "proof_bundles_total": dir_size(verifier_proof_dir),
                "layer_inputs_pt": file_size(verifier_pkg / "layer_inputs.pt"),
                "manifest_json": file_size(verifier_pkg / "manifest.json"),
            },
            "package_dir": str(verifier_pkg),
            "package_tar_gz": str(output_dir / "verifier_min_payload.tar.gz"),
        },
        "prover_min_request": {
            "raw_bytes": prover_raw,
            "compressed_bytes": prover_tgz,
            "component_bytes": {
                "request_payload_json": file_size(prover_pkg / "request_payload.json"),
            },
            "package_dir": str(prover_pkg),
            "package_tar_gz": str(output_dir / "prover_min_request.tar.gz"),
        },
    }


def measure_model_transfer_sizes(model_id: str, output_dir: Path) -> Dict[str, Any]:
    snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    safetensors_files = sorted(snapshot.glob("*.safetensors"))
    if not safetensors_files:
        safetensors_files = sorted(snapshot.rglob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in snapshot: {snapshot}")

    # full prover model transfer size
    prover_raw = int(sum(fp.stat().st_size for fp in safetensors_files))
    if len(safetensors_files) == 1:
        raw_file = safetensors_files[0]
        prover_gz = output_dir / "prover_model_params_full.safetensors.gz"
        prover_comp = gzip_file(raw_file, prover_gz)
        prover_payload = {
            "raw_bytes": prover_raw,
            "compressed_bytes": prover_comp,
            "raw_file": str(raw_file),
            "compressed_file": str(prover_gz),
            "num_safetensors_files": 1,
            "compression_note": "gzip-compressed full model.safetensors",
        }
    else:
        prover_targz = output_dir / "prover_model_params_all_safetensors.tar.gz"
        with tarfile.open(prover_targz, "w:gz") as tar:
            for fp in safetensors_files:
                tar.add(fp, arcname=fp.name)
        prover_payload = {
            "raw_bytes": prover_raw,
            "compressed_bytes": file_size(prover_targz),
            "raw_file": str(snapshot),
            "compressed_file": str(prover_targz),
            "num_safetensors_files": len(safetensors_files),
            "compression_note": "tar.gz over multiple safetensors shards",
        }

    # verifier model subset transfer size (current compute path only needs decoder stack + norm + rotary)
    subset_prefixes = ("model.layers.", "model.norm.", "model.rotary_emb.")
    subset_tensors: Dict[str, torch.Tensor] = {}
    for sf in safetensors_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(subset_prefixes):
                    subset_tensors[key] = f.get_tensor(key)

    subset_sf = output_dir / "verifier_model_params_subset.safetensors"
    save_file(subset_tensors, str(subset_sf))
    subset_gz = output_dir / "verifier_model_params_subset.safetensors.gz"
    subset_comp = gzip_file(subset_sf, subset_gz)

    return {
        "model_id": model_id,
        "snapshot_path": str(snapshot),
        "verifier_model_params_subset": {
            "raw_bytes": file_size(subset_sf),
            "compressed_bytes": subset_comp,
            "file": str(subset_sf),
            "compressed_file": str(subset_gz),
            "key_prefixes": list(subset_prefixes),
            "num_tensors": len(subset_tensors),
        },
        "prover_model_params_full_inference": prover_payload,
    }


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def measure_compute_overhead(
    *,
    model_name: str,
    saved_hidden_states: Sequence[torch.Tensor],
    selections: Sequence[Dict[str, Any]],
    device_str: str,
    dtype_str: str,
) -> Dict[str, Any]:
    device = layer_utils.capture_lib.resolve_device(device_str)
    dtype = layer_utils.capture_lib.resolve_dtype(dtype_str, device)
    model = layer_utils.load_model_for_compute_verification(
        model_name,
        device,
        dtype,
        device_map=None,
        max_memory=None,
    )

    first = selections[0]
    layer_idx = int(first["layer_index"])
    layer_flat = int(first["layer_flat_index"])

    sync_if_cuda(device)
    node_t0 = time.perf_counter()
    node_out = fullcov.compute_single_layer_output(
        model,
        layer_idx,
        saved_hidden_states[layer_idx],
        dtype,
        include_final_norm=(layer_idx == (len(saved_hidden_states) - 2)),
    )
    sync_if_cuda(device)
    node_time = time.perf_counter() - node_t0
    node_saved = float(saved_hidden_states[layer_idx + 1].flatten()[layer_flat].item())
    node_comp = float(node_out.flatten()[layer_flat].item())

    sync_if_cuda(device)
    tile_t0 = time.perf_counter()
    abs_diffs: List[float] = []
    for item in selections:
        layer_index = int(item["layer_index"])
        layer_flat_index = int(item["layer_flat_index"])
        include_final_norm = layer_index == (len(saved_hidden_states) - 2)
        out = fullcov.compute_single_layer_output(
            model,
            layer_index,
            saved_hidden_states[layer_index],
            dtype,
            include_final_norm=include_final_norm,
        )
        saved_value = float(saved_hidden_states[layer_index + 1].flatten()[layer_flat_index].item())
        comp_value = float(out.flatten()[layer_flat_index].item())
        abs_diffs.append(abs(saved_value - comp_value))
    sync_if_cuda(device)
    tile_time = time.perf_counter() - tile_t0

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": str(device),
        "dtype": str(dtype),
        "node": {
            "time_seconds": node_time,
            "abs_diff": abs(node_saved - node_comp),
        },
        "tile": {
            "time_seconds_total": tile_time,
            "time_seconds_avg_per_layer": tile_time / max(1, len(selections)),
            "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
            "mean_abs_diff": float(sum(abs_diffs) / len(abs_diffs)) if abs_diffs else 0.0,
            "num_layers": len(selections),
        },
    }


def measure_crypto_overhead(proof_paths: Sequence[Path]) -> Dict[str, Any]:
    with proof_paths[0].open("r", encoding="utf-8") as handle:
        first_bundle = json.load(handle)
    wrapper = tensorcommitments.TensorCommitmentWrapper(
        int(first_bundle.get("num_variables", len(first_bundle["dims"]))),
        int(first_bundle["degree_bound"]),
    )

    node_start = time.perf_counter()
    node_ok = bool(
        wrapper.verify(
            first_bundle["commitment_hex"],
            first_bundle["hypercube_index"],
            int(first_bundle["value_int"]),
            first_bundle["proof_hex"],
        )
    )
    node_time = time.perf_counter() - node_start

    tile_start = time.perf_counter()
    all_ok = True
    for proof_path in proof_paths:
        payload = load_json(proof_path)
        ok = bool(
            wrapper.verify(
                payload["commitment_hex"],
                payload["hypercube_index"],
                int(payload["value_int"]),
                payload["proof_hex"],
            )
        )
        if not ok:
            all_ok = False
    tile_time = time.perf_counter() - tile_start

    return {
        "node": {
            "verified": node_ok,
            "time_seconds": node_time,
            "proof_file": str(proof_paths[0]),
        },
        "tile": {
            "verified": all_ok,
            "time_seconds_total": tile_time,
            "time_seconds_avg_per_layer": tile_time / max(1, len(proof_paths)),
            "num_layer_proofs": len(proof_paths),
        },
    }


def measure_prover_overhead(
    *,
    poly_dir: Path,
    commitment_file: Path,
    hypercube: np.ndarray,
    selections: Sequence[Dict[str, Any]],
    commitment_results_path: Path,
) -> Dict[str, Any]:
    dims, degree_bound, coefficients, _ = prove_index.load_polynomial(poly_dir)
    _ = prove_index.load_commitment(commitment_file)
    wrapper = tensorcommitments.TensorCommitmentWrapper(len(dims), degree_bound)

    first = selections[0]
    node_index = [int(v) for v in first["hypercube_index"]]
    node_value = int(hypercube[tuple(node_index)])
    t0 = time.perf_counter()
    node_proof = wrapper.prove(coefficients, node_index, node_value)
    node_prove_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    proof_sizes = []
    for item in selections:
        index = [int(v) for v in item["hypercube_index"]]
        value = int(hypercube[tuple(index)])
        proof = wrapper.prove(coefficients, index, value)
        proof_sizes.append(sum(len(bytes.fromhex(x)) for x in proof))
    tile_prove_s = time.perf_counter() - t1

    commitment_results = load_json(commitment_results_path)
    return {
        "node": {
            "prove_time_seconds": node_prove_s,
            "proof_elements": len(node_proof),
            "proof_bytes": sum(len(bytes.fromhex(x)) for x in node_proof),
        },
        "tile": {
            "prove_time_seconds_total": tile_prove_s,
            "prove_time_seconds_avg_per_layer": tile_prove_s / max(1, len(selections)),
            "num_layer_proofs": len(selections),
            "proof_bytes_avg": float(sum(proof_sizes) / len(proof_sizes)) if proof_sizes else 0.0,
        },
        "commit_prove_verify_pipeline": commitment_results.get("timing", {}),
    }


def render_markdown(
    *,
    run_dir: Path,
    model_id: str,
    size_report: Dict[str, Any],
    compute_report: Dict[str, Any],
) -> str:
    verifier_payload = size_report["verifier_min_payload"]
    prover_payload = size_report["prover_min_request"]
    model_sizes = size_report.get("model_parameters_transfer_files", {})
    verifier_model = model_sizes.get("verifier", {})
    prover_model = model_sizes.get("prover", {})

    crypto = compute_report["verifier_overhead"]["crypto"]
    compute = compute_report["verifier_overhead"]["compute"]
    prover = compute_report["prover_overhead"]

    lines = [
        "# Final Empirical Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Model: `{model_id}`",
        f"- Generated at (UTC): `{utc_now()}`",
        "",
        "## Data Transfer Sizes",
        "",
        "### Verifier (non-model payload)",
        f"- Raw: **{verifier_payload['raw_bytes']} B** ({human_bytes(verifier_payload['raw_bytes'])})",
        f"- Compressed: **{verifier_payload['compressed_bytes']} B** ({human_bytes(verifier_payload['compressed_bytes'])})",
        "",
        "### Prover (non-model request payload)",
        f"- Raw: **{prover_payload['raw_bytes']} B** ({human_bytes(prover_payload['raw_bytes'])})",
        f"- Compressed: **{prover_payload['compressed_bytes']} B** ({human_bytes(prover_payload['compressed_bytes'])})",
        "",
        "### Verifier minimal model-parameter subset (compute path)",
        f"- Raw: **{verifier_model.get('raw_bytes', 0)} B** ({human_bytes(verifier_model.get('raw_bytes', 0))})",
        f"- Compressed: **{verifier_model.get('compressed_bytes', 0)} B** ({human_bytes(verifier_model.get('compressed_bytes', 0))})",
        "",
        "### Prover full model parameters (inference path)",
        f"- Raw: **{prover_model.get('raw_bytes', 0)} B** ({human_bytes(prover_model.get('raw_bytes', 0))})",
        f"- Compressed: **{prover_model.get('compressed_bytes', 0)} B** ({human_bytes(prover_model.get('compressed_bytes', 0))})",
        "",
        "## Verifier Compute Overhead",
        "",
        "### Crypto verification",
        f"- Node: `{crypto['node']['time_seconds']:.6f} s`",
        f"- Tile total: `{crypto['tile']['time_seconds_total']:.6f} s`",
        f"- Tile avg/layer: `{crypto['tile']['time_seconds_avg_per_layer']:.6f} s`",
        "",
        "### Compute verification (CPU)",
    ]

    cpu = compute.get("cpu")
    if cpu and "node" in cpu:
        lines.extend(
            [
                f"- Node: `{cpu['node']['time_seconds']:.6f} s`, abs diff `{cpu['node']['abs_diff']:.6g}`",
                (
                    f"- Tile total: `{cpu['tile']['time_seconds_total']:.6f} s`, "
                    f"avg/layer `{cpu['tile']['time_seconds_avg_per_layer']:.6f} s`, "
                    f"max abs diff `{cpu['tile']['max_abs_diff']:.6g}`"
                ),
            ]
        )
    else:
        lines.append("- Skipped")

    lines.append("")
    lines.append("### Compute verification (GPU)")
    gpu = compute.get("gpu")
    if gpu and "node" in gpu:
        lines.extend(
            [
                f"- Node: `{gpu['node']['time_seconds']:.6f} s`, abs diff `{gpu['node']['abs_diff']:.6g}`",
                (
                    f"- Tile total: `{gpu['tile']['time_seconds_total']:.6f} s`, "
                    f"avg/layer `{gpu['tile']['time_seconds_avg_per_layer']:.6f} s`, "
                    f"max abs diff `{gpu['tile']['max_abs_diff']:.6g}`"
                ),
            ]
        )
    else:
        lines.append("- Unavailable / skipped")

    lines.extend(
        [
            "",
            "## Prover Compute Overhead",
            "",
            f"- Node proof generation: `{prover['node']['prove_time_seconds']:.6f} s`",
            (
                f"- Tile proof generation total: `{prover['tile']['prove_time_seconds_total']:.6f} s` "
                f"(avg/layer `{prover['tile']['prove_time_seconds_avg_per_layer']:.6f} s`)"
            ),
            f"- Commit stage time: `{prover['commit_prove_verify_pipeline'].get('commit_time_s', 0.0)} s`",
            f"- Commit script total prove time: `{prover['commit_prove_verify_pipeline'].get('total_prove_time_s', 0.0)} s`",
            f"- Commit script total verify time: `{prover['commit_prove_verify_pipeline'].get('total_verify_time_s', 0.0)} s`",
            "",
            "## Source Reports",
            "",
            f"- `{run_dir / 'empirical_measurements' / 'stage5_stage6_size_report.json'}`",
            f"- `{run_dir / 'empirical_measurements' / 'stage7_stage8_compute_report.json'}`",
            f"- `{run_dir / 'empirical_measurements' / 'model_param_transfer_sizes.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    artifact_paths = detect_artifacts(run_dir, args.model_id)
    activations_payload = layer_utils.load_pt_artifacts(artifact_paths["activations_pt"])
    model_id = args.model_id or str(activations_payload["model_name"])
    prompt = str(activations_payload["prompt"])
    safe_model = sanitize_model_name(model_id)

    full_coverage_dir = (args.full_coverage_dir.resolve() if args.full_coverage_dir else run_dir / "full_coverage_verification")
    if not full_coverage_dir.is_dir():
        raise FileNotFoundError(f"full-coverage directory not found: {full_coverage_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "empirical_measurements")
    output_dir.mkdir(parents=True, exist_ok=True)

    # stage 5/6: transfer sizes
    request_config = {
        "max_new_tokens": args.max_new_tokens,
        "scale_factor": args.scale_factor,
        "quantize": args.quantize,
        "min_dim": args.min_dim,
        "max_dim": args.max_dim,
        "num_queries": args.num_queries,
        "seed": args.seed,
    }
    size_report = {
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "model_id": model_id,
    }
    size_report.update(
        build_non_model_transfer_sizes(
            output_dir=output_dir,
            activations_payload=activations_payload,
            model_id=model_id,
            prompt=prompt,
            full_coverage_dir=full_coverage_dir,
            commitment_file=artifact_paths["commitment_dir"] / "commitment.txt",
            request_config=request_config,
        )
    )

    model_transfer_report = None
    if not args.skip_model_transfer:
        model_transfer_path = output_dir / "model_param_transfer_sizes.json"
        if model_transfer_path.is_file() and not args.force_recompute_model_transfer:
            model_transfer_report = load_json(model_transfer_path)
            if model_transfer_report.get("model_id") != model_id:
                raise ValueError(
                    "Existing model_param_transfer_sizes.json belongs to a different model. "
                    "Re-run with --force-recompute-model-transfer."
                )
        else:
            model_transfer_report = measure_model_transfer_sizes(model_id, output_dir)
            save_json(model_transfer_path, model_transfer_report)

        size_report["model_parameters_transfer_files"] = {
            "verifier": model_transfer_report["verifier_model_params_subset"],
            "prover": model_transfer_report["prover_model_params_full_inference"],
        }

    size_report_path = output_dir / "stage5_stage6_size_report.json"
    save_json(size_report_path, size_report)

    # stage 7/8: compute overhead
    selections_payload = load_json(full_coverage_dir / "selected_entries.json")
    selections = selections_payload.get("entries", [])
    if not selections:
        raise RuntimeError(f"No entries found in {full_coverage_dir / 'selected_entries.json'}")
    proof_paths = sorted((full_coverage_dir / "proof_bundles").glob("proof_layer_*.json"))
    if not proof_paths:
        raise RuntimeError(f"No proof_layer_*.json files found in {full_coverage_dir / 'proof_bundles'}")

    hypercube = np.load(artifact_paths["hypercube_dir"] / "hypercube.npy", allow_pickle=True)
    compute_report: Dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "model_name": str(activations_payload["model_name"]),
        "tile_definition": "one random selected entry per layer across full network",
        "verifier_overhead": {
            "crypto": measure_crypto_overhead(proof_paths),
            "compute": {},
        },
        "prover_overhead": {},
    }

    if not args.skip_cpu_overhead:
        compute_report["verifier_overhead"]["compute"]["cpu"] = measure_compute_overhead(
            model_name=str(activations_payload["model_name"]),
            saved_hidden_states=activations_payload["hidden_states"],
            selections=selections,
            device_str="cpu",
            dtype_str="float32",
        )
    else:
        compute_report["verifier_overhead"]["compute"]["cpu"] = {"skipped": True}

    if args.skip_gpu_overhead:
        compute_report["verifier_overhead"]["compute"]["gpu"] = {"skipped": True}
    else:
        if torch.cuda.is_available():
            compute_report["verifier_overhead"]["compute"]["gpu"] = measure_compute_overhead(
                model_name=str(activations_payload["model_name"]),
                saved_hidden_states=activations_payload["hidden_states"],
                selections=selections,
                device_str="cuda",
                dtype_str="float16",
            )
        else:
            compute_report["verifier_overhead"]["compute"]["gpu"] = {"available": False}

    compute_report["prover_overhead"] = measure_prover_overhead(
        poly_dir=artifact_paths["poly_dir"],
        commitment_file=artifact_paths["commitment_dir"] / "commitment.txt",
        hypercube=hypercube,
        selections=selections,
        commitment_results_path=artifact_paths["commitment_dir"] / "commitment_results.json",
    )

    compute_report_path = output_dir / "stage7_stage8_compute_report.json"
    save_json(compute_report_path, compute_report)

    # markdown summary (run-level + repo-level)
    markdown = render_markdown(
        run_dir=run_dir,
        model_id=model_id,
        size_report=size_report,
        compute_report=compute_report,
    )
    run_md_path = output_dir / "final_empirical_report.md"
    run_md_path.write_text(markdown, encoding="utf-8")

    root_report_path = args.root_report.resolve()
    root_report_path.write_text(markdown, encoding="utf-8")

    print(f"[INFO] wrote {size_report_path}")
    if model_transfer_report is not None:
        print(f"[INFO] wrote {output_dir / 'model_param_transfer_sizes.json'}")
    print(f"[INFO] wrote {compute_report_path}")
    print(f"[INFO] wrote {run_md_path}")
    print(f"[INFO] wrote {root_report_path}")


if __name__ == "__main__":
    main()
