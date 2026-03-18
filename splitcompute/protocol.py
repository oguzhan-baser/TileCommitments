#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TENSOR_ROOT = REPO_ROOT / "TensorCommitment"

CAPTURE_SCRIPT = TENSOR_ROOT / "activationCaptureLib" / "capture_activations.py"
CONVERT_SCRIPT = TENSOR_ROOT / "activationCaptureLib" / "convert_to_npy.py"
RESHAPE_SCRIPT = TENSOR_ROOT / "activationCaptureLib" / "reshape_to_hypercube.py"
INTERPOLATE_SCRIPT = TENSOR_ROOT / "interpolationLib" / "interpolate_hypercube.py"
COMMIT_SCRIPT = TENSOR_ROOT / "tensorCommitmentLib" / "commit_prove_verify.py"

DEFAULT_MODELS_FILE = Path(__file__).resolve().parent / "fixed_models.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def run_cmd(args: List[str], cwd: Path = REPO_ROOT) -> None:
    print(f"[cmd] {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd), check=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_fixed_models(models_file: Path | None = None) -> List[str]:
    file_path = (models_file or DEFAULT_MODELS_FILE).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Fixed models file not found: {file_path}")
    payload = load_json(file_path)
    models = payload.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError(f"Invalid or empty models list in {file_path}")
    cleaned = [str(item).strip() for item in models if str(item).strip()]
    if not cleaned:
        raise ValueError(f"No valid model names in {file_path}")
    return cleaned


def ensure_model_allowed(model_name: str, allowed_models: List[str]) -> None:
    if model_name not in allowed_models:
        raise ValueError(
            f"Model '{model_name}' is not in fixed allowlist: {allowed_models}"
        )


def extract_inference_output(activations_pt: Path) -> Dict[str, Any]:
    payload = torch.load(activations_pt, map_location="cpu")
    token_sequence = payload.get("token_sequence")
    token_ids = token_sequence.tolist() if hasattr(token_sequence, "tolist") else []
    return {
        "model_name": str(payload.get("model_name", "")),
        "prompt": str(payload.get("prompt", "")),
        "generated_text": str(payload.get("generated_text", "")),
        "token_ids": token_ids,
    }


def make_proof_bundles_from_proofs_json(proofs_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    commitment_hex = str(proofs_payload["commitment_hex"])
    num_variables = int(proofs_payload["num_variables"])
    degree_bound = int(proofs_payload["degree_bound"])
    proofs = proofs_payload.get("proofs", [])

    bundles: List[Dict[str, Any]] = []
    for idx, proof_row in enumerate(proofs):
        point = [int(v) for v in proof_row["point"]]
        value_int = int(proof_row["evaluation"])
        proof_hex = [str(v) for v in proof_row["proof_hex"]]
        bundles.append(
            {
                "bundle_id": idx,
                "num_variables": num_variables,
                "degree_bound": degree_bound,
                "commitment_hex": commitment_hex,
                "index": point,
                "value_int": str(value_int),
                "proof_hex": proof_hex,
            }
        )
    return bundles


def run_prover_pipeline(
    *,
    model_name: str,
    prompt: str,
    output_root: Path,
    allowed_models: List[str],
    max_new_tokens: int = 16,
    num_queries: int = 10,
    seed: int = 42,
    scale_factor: int = 16,
    quantize: float = 50.0,
    min_dim: int = 4,
    max_dim: int = 10,
    skip_interp_build: bool = True,
) -> Dict[str, Any]:
    ensure_model_allowed(model_name, allowed_models)
    output_root = output_root.resolve()
    run_id = f"run_{utc_tag()}_{sanitize_model_name(model_name)}"
    run_dir = output_root / run_id
    capture_dir = run_dir / "capture"
    int_dir = run_dir / "int_activations"
    hypercube_dir = run_dir / "hypercube"
    poly_dir = run_dir / "polynomial"
    commitment_dir = run_dir / "commitment"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(CAPTURE_SCRIPT),
            "--models",
            model_name,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--output-dir",
            str(capture_dir),
            "--seed",
            str(seed),
        ]
    )

    safe = sanitize_model_name(model_name)
    activations_pt = capture_dir / f"{safe}_activations.pt"
    if not activations_pt.is_file():
        raise FileNotFoundError(f"Expected activation artifact not found: {activations_pt}")
    inference_output = extract_inference_output(activations_pt)

    run_cmd(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            "--input",
            str(activations_pt),
            "--output-dir",
            str(int_dir),
            "--scale-factor",
            str(scale_factor),
            "--quantize",
            str(quantize),
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(RESHAPE_SCRIPT),
            "--input-dir",
            str(int_dir),
            "--output-dir",
            str(hypercube_dir),
            "--min-dim",
            str(min_dim),
            "--max-dim",
            str(max_dim),
        ]
    )
    interp_cmd = [
        sys.executable,
        str(INTERPOLATE_SCRIPT),
        "--input-dir",
        str(hypercube_dir),
        "--output-dir",
        str(poly_dir),
    ]
    if skip_interp_build:
        interp_cmd.append("--skip-build")
    run_cmd(interp_cmd)

    run_cmd(
        [
            sys.executable,
            str(COMMIT_SCRIPT),
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
        ]
    )

    commitment_results_path = commitment_dir / "commitment_results.json"
    proofs_path = commitment_dir / "proofs.json"
    if not commitment_results_path.is_file():
        raise FileNotFoundError(f"Missing expected file: {commitment_results_path}")
    if not proofs_path.is_file():
        raise FileNotFoundError(f"Missing expected file: {proofs_path}")

    commitment_results = load_json(commitment_results_path)
    proofs_payload = load_json(proofs_path)
    bundles = make_proof_bundles_from_proofs_json(proofs_payload)

    response = {
        "format_version": 1,
        "generated_at_utc": utc_now_iso(),
        "run_id": run_id,
        "request": {
            "model_name": model_name,
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "num_queries": int(num_queries),
            "seed": int(seed),
            "scale_factor": int(scale_factor),
            "quantize": float(quantize),
            "min_dim": int(min_dim),
            "max_dim": int(max_dim),
            "skip_interp_build": bool(skip_interp_build),
        },
        "inference_output": inference_output,
        "commitment": {
            "commitment_hex": str(proofs_payload["commitment_hex"]),
            "num_variables": int(proofs_payload["num_variables"]),
            "degree_bound": int(proofs_payload["degree_bound"]),
        },
        "proof_bundles": bundles,
        "commitment_results_summary": {
            "verification_summary": commitment_results.get("verification_summary", {}),
            "timing": commitment_results.get("timing", {}),
            "proof_stats": commitment_results.get("proof_stats", {}),
        },
        "artifact_paths_on_prover": {
            "run_dir": str(run_dir),
            "capture_dir": str(capture_dir),
            "int_dir": str(int_dir),
            "hypercube_dir": str(hypercube_dir),
            "poly_dir": str(poly_dir),
            "commitment_dir": str(commitment_dir),
            "activations_pt": str(activations_pt),
            "commitment_results_json": str(commitment_results_path),
            "proofs_json": str(proofs_path),
        },
    }

    save_json(run_dir / "prover_response.json", response)
    return response


def post_json(url: str, payload: Dict[str, Any], timeout_s: int = 3600) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def get_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
    with request.urlopen(url, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def verify_proof_bundles(proof_bundles: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import tensorcommitments  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Cannot import tensorcommitments. Activate tilecommitments env and build bindings."
        ) from exc

    wrappers: Dict[Tuple[int, int], Any] = {}
    proof_results: List[Dict[str, Any]] = []
    all_verified = True

    for bundle in proof_bundles:
        num_variables = int(bundle["num_variables"])
        degree_bound = int(bundle["degree_bound"])
        key = (num_variables, degree_bound)
        if key not in wrappers:
            wrappers[key] = tensorcommitments.TensorCommitmentWrapper(num_variables, degree_bound)
        wrapper = wrappers[key]

        index = [int(v) for v in bundle["index"]]
        value_int = int(bundle["value_int"])
        proof_hex = [str(v) for v in bundle["proof_hex"]]
        commitment_hex = str(bundle["commitment_hex"])
        verified = bool(wrapper.verify(commitment_hex, index, value_int, proof_hex))
        all_verified = all_verified and verified
        proof_results.append(
            {
                "bundle_id": int(bundle.get("bundle_id", -1)),
                "index": index,
                "value_int": str(value_int),
                "proof_elements": len(proof_hex),
                "verified": verified,
            }
        )

    return {
        "all_verified": all_verified,
        "num_proofs": len(proof_results),
        "proof_results": proof_results,
    }


def verify_prover_response(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    bundles = response_payload.get("proof_bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("Response payload missing non-empty proof_bundles list")
    verification = verify_proof_bundles(bundles)
    return {
        "verified_at_utc": utc_now_iso(),
        "run_id": response_payload.get("run_id", ""),
        "model_name": response_payload.get("request", {}).get("model_name", ""),
        "inference_output": response_payload.get("inference_output", {}),
        "verification": verification,
    }

