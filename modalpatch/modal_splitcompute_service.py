#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import modal

APP_NAME = "tilecommitments-splitcompute-prover"
REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_CATALOG_PATH = REPO_ROOT / "modalpatch" / "model_catalog.json"

REMOTE_REPO_ROOT = Path("/root/TileCommitments")
REMOTE_MODALPATCH_DIR = REMOTE_REPO_ROOT / "modalpatch"
REMOTE_SPLITCOMPUTE_DIR = REMOTE_REPO_ROOT / "splitcompute"
REMOTE_TENSOR_ROOT = REMOTE_REPO_ROOT / "TensorCommitment"

REMOTE_CAPTURE_SCRIPT = REMOTE_TENSOR_ROOT / "activationCaptureLib" / "capture_activations.py"
REMOTE_CONVERT_SCRIPT = REMOTE_TENSOR_ROOT / "activationCaptureLib" / "convert_to_npy.py"
REMOTE_RESHAPE_SCRIPT = REMOTE_TENSOR_ROOT / "activationCaptureLib" / "reshape_to_hypercube.py"
REMOTE_INTERPOLATE_SCRIPT = REMOTE_TENSOR_ROOT / "interpolationLib" / "interpolate_hypercube.py"
REMOTE_COMMIT_SCRIPT = REMOTE_TENSOR_ROOT / "tensorCommitmentLib" / "commit_prove_verify.py"

HF_CACHE_DIR = Path("/cache/hf")
ARTIFACT_DIR = Path("/data/splitcompute")

HF_CACHE_VOL = modal.Volume.from_name("tilecommitments-hf-cache", create_if_missing=True)
ARTIFACT_VOL = modal.Volume.from_name("tilecommitments-splitcompute-artifacts", create_if_missing=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def _load_model_catalog(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"Invalid catalog format: {path}")
    table: Dict[str, float] = {}
    for row in models:
        model_id = str(row.get("id", "")).strip()
        params_b = float(row.get("params_b", 0.0))
        if model_id:
            table[model_id] = params_b
    if not table:
        raise ValueError(f"No models found in catalog: {path}")
    return table


def _resolve_model_catalog_path() -> Path:
    candidates = [
        MODEL_CATALOG_PATH,
        Path(__file__).resolve().with_name("model_catalog.json"),
        Path("/root/model_catalog.json"),
        Path("/root/modalpatch/model_catalog.json"),
        REMOTE_MODALPATCH_DIR / "model_catalog.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find model_catalog.json. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


MODEL_PARAM_TABLE = _load_model_catalog(_resolve_model_catalog_path())


def _extract_params_b_from_name(model_name: str) -> float | None:
    match_b = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if match_b:
        return float(match_b.group(1))
    match_m = re.search(r"(\d+(?:\.\d+)?)\s*[mM]\b", model_name)
    if match_m:
        return float(match_m.group(1)) / 1000.0
    return None


GPU_MEMORY_GB = {
    "T4": 16.0,
    "L4": 24.0,
    "A10G": 24.0,
    "L40S": 48.0,
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "H100": 80.0,
}

GPU_PRIORITY = [
    "T4",
    "L4",
    "A10G",
    "L40S",
    "A100-40GB",
    "A100-80GB",
    "H100",
]


def _estimate_required_vram_gb(
    params_b: float,
    *,
    dtype_bytes: float = 2.0,
    runtime_overhead_factor: float = 1.4,
) -> float:
    # Approximation:
    # 1B params * 2 bytes (fp16/bf16) ~= 2GB raw weights.
    # Multiply by overhead factor to account for runtime buffers and framework overhead.
    return params_b * dtype_bytes * runtime_overhead_factor


def select_gpu_for_model(model_name: str) -> Dict[str, Any]:
    params_b = MODEL_PARAM_TABLE.get(model_name)
    if params_b is None:
        inferred = _extract_params_b_from_name(model_name)
        if inferred is None:
            raise ValueError(
                f"Model '{model_name}' is not in modal catalog and parameter count could not be inferred."
            )
        params_b = inferred

    required_vram = _estimate_required_vram_gb(params_b)
    model_name_lower = model_name.lower()
    # GPT-OSS-120B compatibility override:
    # prefer A100-80GB over H100 because H100 path can hit kernel/dtype mismatch issues.
    if model_name_lower == "openai/gpt-oss-120b":
        selected = "A100-80GB"
    else:
        selected = GPU_PRIORITY[-1]
        for gpu_name in GPU_PRIORITY:
            if GPU_MEMORY_GB[gpu_name] >= required_vram:
                selected = gpu_name
                break
    # Rule-of-thumb sizing for multi-GPU placement:
    # model params in billions * 2 bytes ~= total model memory in GiB.
    # Example: 120B -> 240GiB -> 3x 80GiB GPUs.
    rule_of_thumb_memory_gb = params_b * 2.0
    selected_gpu_count = max(1, math.ceil(rule_of_thumb_memory_gb / GPU_MEMORY_GB[selected]))
    return {
        "model_name": model_name,
        "params_b": params_b,
        "estimated_required_vram_gb": required_vram,
        "rule_of_thumb_memory_gb": rule_of_thumb_memory_gb,
        "selected_gpu": selected,
        "selected_gpu_memory_gb": GPU_MEMORY_GB[selected],
        "selected_gpu_count": selected_gpu_count,
    }


def _run_cmd(
    args: List[str],
    cwd: Path = REMOTE_REPO_ROOT,
    env_overrides: Dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(args, cwd=str(cwd), check=True, env=env)


def _commit_volume_if_supported(volume: modal.Volume, label: str) -> None:
    try:
        volume.commit()
        print(f"[INFO] Volume commit: {label}")
    except AttributeError:
        return


def _reload_volume_if_supported(volume: modal.Volume, label: str) -> None:
    try:
        volume.reload()
        print(f"[INFO] Volume reload: {label}")
    except AttributeError:
        return


def _extract_inference_output(activations_pt: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(activations_pt, map_location="cpu")
    token_sequence = payload.get("token_sequence")
    token_ids = token_sequence.tolist() if hasattr(token_sequence, "tolist") else []
    return {
        "model_name": str(payload.get("model_name", "")),
        "prompt": str(payload.get("prompt", "")),
        "generated_text": str(payload.get("generated_text", "")),
        "token_ids": token_ids,
    }


def _count_non_finite_layers(stats_json_path: Path) -> int:
    if not stats_json_path.is_file():
        return 0
    with stats_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    layer_stats = payload.get("layer_stats", [])
    non_finite = 0
    for row in layer_stats:
        values = [row.get("min"), row.get("max"), row.get("mean")]
        try:
            finite = all(math.isfinite(float(v)) for v in values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            non_finite += 1
    return non_finite


def _proof_bundles_from_proofs_json(proofs_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    commitment_hex = str(proofs_payload["commitment_hex"])
    num_variables = int(proofs_payload["num_variables"])
    degree_bound = int(proofs_payload["degree_bound"])
    proofs = proofs_payload.get("proofs", [])

    bundles: List[Dict[str, Any]] = []
    for idx, row in enumerate(proofs):
        bundles.append(
            {
                "bundle_id": idx,
                "num_variables": num_variables,
                "degree_bound": degree_bound,
                "commitment_hex": commitment_hex,
                "index": [int(v) for v in row["point"]],
                "value_int": str(int(row["evaluation"])),
                "proof_hex": [str(v) for v in row["proof_hex"]],
            }
        )
    return bundles


def _ensure_model_allowed(model_name: str) -> None:
    if model_name not in MODEL_PARAM_TABLE:
        raise ValueError(
            f"Model '{model_name}' is not in modalpatch/model_catalog.json. Allowed={list(MODEL_PARAM_TABLE.keys())}"
        )


def _build_capture_attempts(model_name: str, params_b: float) -> List[List[str]]:
    model_name_lower = model_name.lower()

    if model_name_lower == "openai/gpt-oss-120b":
        return [
            ["--dtype", "bfloat16", "--mxfp4-mode", "dequantize"],
            ["--dtype", "float16", "--mxfp4-mode", "dequantize"],
        ]

    # Large FP8 checkpoints and very large models are numerically fragile in fp16.
    # Prefer bf16 first, then fall back to fp16 for compatibility.
    if "fp8" in model_name_lower or params_b >= 100.0:
        return [
            ["--dtype", "bfloat16"],
            ["--dtype", "float16"],
        ]

    return [[]]


def _gpu_capture_impl(payload: Dict[str, Any], selected_gpu: str) -> Dict[str, Any]:
    _ensure_model_allowed(str(payload["model_name"]))
    run_id = str(payload["run_id"])
    model_name = str(payload["model_name"])
    prompt = str(payload["prompt"])
    max_new_tokens = int(payload["max_new_tokens"])
    seed = int(payload["seed"])
    selected_gpu_count = int(payload.get("selected_gpu_count", 1))
    params_b = float(payload.get("params_b", MODEL_PARAM_TABLE[model_name]))

    run_dir = ARTIFACT_DIR / run_id
    capture_dir = run_dir / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_model_name(model_name)
    activations_pt = capture_dir / f"{safe}_activations.pt"
    stats_json = capture_dir / f"{safe}_stats.json"

    capture_cmd = [
        sys.executable,
        str(REMOTE_CAPTURE_SCRIPT),
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
        "--fail-on-error",
    ]
    sharding_args: List[str] = []
    if selected_gpu_count > 1:
        per_gpu_mib = int(GPU_MEMORY_GB[selected_gpu] * 1024 * 85 // 100)
        offload_dir = run_dir / "offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        sharding_args.extend(
            [
                "--device-map",
                "auto",
                "--max-memory-per-gpu",
                f"{per_gpu_mib}MiB",
                "--offload-folder",
                str(offload_dir),
            ]
        )

    env_overrides = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    capture_attempts = _build_capture_attempts(model_name, params_b)
    if model_name.lower() == "openai/gpt-oss-120b":
        env_overrides["CUDA_LAUNCH_BLOCKING"] = "1"
    last_capture_error: Exception | None = None
    non_finite_layers = 0
    for attempt_idx, attempt_args in enumerate(capture_attempts, start=1):
        cmd = [*capture_cmd, *attempt_args, *sharding_args]
        try:
            _run_cmd(cmd, env_overrides=env_overrides)
            non_finite_layers = _count_non_finite_layers(stats_json)
            if non_finite_layers > 0:
                raise RuntimeError(
                    "Activation capture produced non-finite values "
                    f"for {model_name}: non_finite_layers={non_finite_layers}."
                )
            last_capture_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_capture_error = exc
            if len(capture_attempts) > 1 or non_finite_layers > 0:
                print(f"[WARN] Capture attempt {attempt_idx}/{len(capture_attempts)} failed for {model_name}.")
                print(f"[WARN] attempt_args={attempt_args}  reason={type(exc).__name__}: {exc}")

    if last_capture_error is not None:
        raise last_capture_error

    if not activations_pt.is_file():
        raise FileNotFoundError(f"Expected activation artifact missing: {activations_pt}")
    _commit_volume_if_supported(ARTIFACT_VOL, "artifacts")
    _commit_volume_if_supported(HF_CACHE_VOL, "hf_cache")

    inference_output = _extract_inference_output(activations_pt)
    return {
        "run_id": run_id,
        "model_name": model_name,
        "prompt": prompt,
        "selected_gpu": selected_gpu,
        "selected_gpu_count": selected_gpu_count,
        "capture_dir": str(capture_dir),
        "activations_pt": str(activations_pt),
        "non_finite_layers": non_finite_layers,
        "inference_output": inference_output,
    }


def _cpu_commit_prove_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    _reload_volume_if_supported(ARTIFACT_VOL, "artifacts")
    _reload_volume_if_supported(HF_CACHE_VOL, "hf_cache")

    run_id = str(payload["run_id"])
    model_name = str(payload["model_name"])
    prompt = str(payload["prompt"])
    selected_gpu = str(payload["selected_gpu"])
    params_b = float(payload["params_b"])
    estimated_required_vram_gb = float(payload["estimated_required_vram_gb"])
    activations_pt = Path(str(payload["activations_pt"]))
    stats_json = activations_pt.parent / f"{_sanitize_model_name(model_name)}_stats.json"

    num_queries = int(payload["num_queries"])
    seed = int(payload["seed"])
    scale_factor = int(payload["scale_factor"])
    quantize = float(payload["quantize"])
    min_dim = int(payload["min_dim"])
    max_dim = int(payload["max_dim"])
    skip_interp_build = bool(payload["skip_interp_build"])

    run_dir = ARTIFACT_DIR / run_id
    int_dir = run_dir / "int_activations"
    hypercube_dir = run_dir / "hypercube"
    poly_dir = run_dir / "polynomial"
    commitment_dir = run_dir / "commitment"
    run_dir.mkdir(parents=True, exist_ok=True)

    non_finite_layers = _count_non_finite_layers(stats_json)
    if non_finite_layers > 0:
        raise ValueError(
            "Captured activations contain non-finite values and cannot be committed. "
            f"model={model_name} non_finite_layers={non_finite_layers} stats={stats_json}. "
            "Please re-run capture with a more stable dtype (prefer bfloat16 for large/FP8 models)."
        )

    _run_cmd(
        [
            sys.executable,
            str(REMOTE_CONVERT_SCRIPT),
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
    _run_cmd(
        [
            sys.executable,
            str(REMOTE_RESHAPE_SCRIPT),
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
        str(REMOTE_INTERPOLATE_SCRIPT),
        "--input-dir",
        str(hypercube_dir),
        "--output-dir",
        str(poly_dir),
    ]
    if skip_interp_build:
        interp_cmd.append("--skip-build")
    _run_cmd(interp_cmd)

    _run_cmd(
        [
            sys.executable,
            str(REMOTE_COMMIT_SCRIPT),
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
    with commitment_results_path.open("r", encoding="utf-8") as handle:
        commitment_results = json.load(handle)
    with proofs_path.open("r", encoding="utf-8") as handle:
        proofs_payload = json.load(handle)

    bundles = _proof_bundles_from_proofs_json(proofs_payload)
    verification_summary = commitment_results.get("verification_summary", {})
    all_verified = bool(verification_summary.get("all_proofs_verified", False))

    response_payload = {
        "format_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "request": {
            "model_name": model_name,
            "prompt": prompt,
            "max_new_tokens": int(payload["max_new_tokens"]),
            "num_queries": num_queries,
            "seed": seed,
            "scale_factor": scale_factor,
            "quantize": quantize,
            "min_dim": min_dim,
            "max_dim": max_dim,
            "skip_interp_build": skip_interp_build,
        },
        "inference_output": payload["inference_output"],
        "commitment": {
            "commitment_hex": str(proofs_payload["commitment_hex"]),
            "num_variables": int(proofs_payload["num_variables"]),
            "degree_bound": int(proofs_payload["degree_bound"]),
        },
        "proof_bundles": bundles,
        "commitment_results_summary": {
            "verification_summary": verification_summary,
            "timing": commitment_results.get("timing", {}),
            "proof_stats": commitment_results.get("proof_stats", {}),
            "all_verified_on_prover": all_verified,
        },
        "modal_dispatch": {
            "selected_gpu": selected_gpu,
            "selected_gpu_count": int(payload.get("selected_gpu_count", 1)),
            "params_b": params_b,
            "estimated_required_vram_gb": estimated_required_vram_gb,
            "rule_of_thumb_memory_gb": float(payload.get("rule_of_thumb_memory_gb", 0.0)),
        },
        "artifact_paths_on_prover": {
            "run_dir": str(run_dir),
            "capture_dir": str(Path(payload["capture_dir"])),
            "activations_pt": str(activations_pt),
            "int_dir": str(int_dir),
            "hypercube_dir": str(hypercube_dir),
            "poly_dir": str(poly_dir),
            "commitment_dir": str(commitment_dir),
            "commitment_results_json": str(commitment_results_path),
            "proofs_json": str(proofs_path),
        },
    }

    out_path = run_dir / "prover_response.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(response_payload, handle, indent=2)
    return response_payload


IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands("curl https://sh.rustup.rs -sSf | sh -s -- -y")
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": str(HF_CACHE_DIR),
            "HF_HUB_CACHE": str(HF_CACHE_DIR / "hub"),
            "TRANSFORMERS_CACHE": str(HF_CACHE_DIR / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .pip_install(
        "fastapi[standard]==0.115.4",
        "pydantic==2.10.6",
        "numpy==1.26.4",
        "torch>=2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "datasets>=2.19.0",
        "sentencepiece>=0.2.0",
        "huggingface_hub>=0.24.0",
        "maturin>=1.7.0",
    )
    .add_local_dir(
        str(REPO_ROOT / "TensorCommitment"),
        remote_path=str(REMOTE_TENSOR_ROOT),
        copy=True,
        ignore=["**/__pycache__", "**/.pytest_cache", "activationCaptureLib/output", "experiments/output"],
    )
    .add_local_dir(
        str(REPO_ROOT / "splitcompute"),
        remote_path=str(REMOTE_SPLITCOMPUTE_DIR),
        copy=True,
        ignore=["**/__pycache__", "output"],
    )
    .add_local_dir(
        str(REPO_ROOT / "modalpatch"),
        remote_path=str(REMOTE_MODALPATCH_DIR),
        copy=True,
        ignore=["**/__pycache__", "output"],
    )
    .add_local_file(
        str(REPO_ROOT / "modalpatch" / "model_catalog.json"),
        remote_path="/root/model_catalog.json",
        copy=True,
    )
    .run_commands(
        "cd /root/TileCommitments/TensorCommitment/pst_commitment_lib && maturin build --features python --release",
        "pip install /root/TileCommitments/TensorCommitment/pst_commitment_lib/target/wheels/*.whl",
        "cd /root/TileCommitments/TensorCommitment/pst_commitment_lib/poly_interp_demo && cargo build --release",
    )
)


APP = modal.App(APP_NAME, image=IMAGE)
# Modal CLI expects a module-level `app` by default for `modal deploy <file.py>`.
# Keep uppercase alias for internal readability.
app = APP

SHARED_MOUNTS = {
    str(HF_CACHE_DIR): HF_CACHE_VOL,
    str(ARTIFACT_DIR): ARTIFACT_VOL,
}


@APP.function(timeout=60 * 60, cpu=4, memory=16384, volumes=SHARED_MOUNTS)
def run_cpu_commit_stage(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _cpu_commit_prove_impl(payload)


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="T4",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_t4(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "T4")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="L4",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_l4(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "L4")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="A10G",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_a10g(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "A10G")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="L40S",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_l40s(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "L40S")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="A100-40GB",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_a100_40(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "A100-40GB")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="A100-80GB",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_a100_80(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "A100-80GB")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="A100-80GB:2",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_a100_80_2(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "A100-80GB")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="A100-80GB:3",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_a100_80_3(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "A100-80GB")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:2",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_2(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:3",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_3(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:4",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_4(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:5",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_5(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:6",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_6(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:7",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_7(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


@APP.function(
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    gpu="H100:8",
    volumes=SHARED_MOUNTS,
    scaledown_window=30,
    min_containers=0,
)
def run_gpu_capture_h100_8(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _gpu_capture_impl(payload, "H100")


GPU_TO_CAPTURE_FN = {
    "T4": run_gpu_capture_t4,
    "L4": run_gpu_capture_l4,
    "A10G": run_gpu_capture_a10g,
    "L40S": run_gpu_capture_l40s,
    "A100-40GB": run_gpu_capture_a100_40,
    "A100-80GB": run_gpu_capture_a100_80,
    "H100": run_gpu_capture_h100,
}

GPU_TO_CAPTURE_FN_WITH_COUNT = {
    ("A100-80GB", 1): run_gpu_capture_a100_80,
    ("A100-80GB", 2): run_gpu_capture_a100_80_2,
    ("A100-80GB", 3): run_gpu_capture_a100_80_3,
    ("H100", 1): run_gpu_capture_h100,
    ("H100", 2): run_gpu_capture_h100_2,
    ("H100", 3): run_gpu_capture_h100_3,
    ("H100", 4): run_gpu_capture_h100_4,
    ("H100", 5): run_gpu_capture_h100_5,
    ("H100", 6): run_gpu_capture_h100_6,
    ("H100", 7): run_gpu_capture_h100_7,
    ("H100", 8): run_gpu_capture_h100_8,
}


@APP.function(timeout=60 * 60, cpu=2, memory=4096, volumes=SHARED_MOUNTS)
def prefetch_models(model_ids: List[str] | None = None) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    models = model_ids if model_ids else list(MODEL_PARAM_TABLE.keys())
    downloaded: List[str] = []
    for model_id in models:
        _ensure_model_allowed(model_id)
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(HF_CACHE_DIR / "hub"),
            local_files_only=False,
            resume_download=True,
        )
        downloaded.append(model_id)
    return {
        "prefetched_models": downloaded,
        "hf_cache_dir": str(HF_CACHE_DIR),
        "generated_at_utc": _utc_now_iso(),
    }


@APP.local_entrypoint()
def warm_models(models: str = "") -> None:
    selected = [m.strip() for m in models.split(",") if m.strip()] if models else None
    result = prefetch_models.remote(selected)
    print(json.dumps(result, indent=2))


@APP.function(timeout=60 * 60, cpu=2, memory=4096, volumes=SHARED_MOUNTS)
@modal.asgi_app()
def prover_api():
    from fastapi import FastAPI, HTTPException

    web_app = FastAPI(title="TileCommitments SplitCompute Prover (Modal)")

    @web_app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "app": APP_NAME,
            "generated_at_utc": _utc_now_iso(),
        }

    @web_app.get("/models")
    def models() -> Dict[str, Any]:
        rows = []
        for model_id in MODEL_PARAM_TABLE:
            policy = select_gpu_for_model(model_id)
            rows.append(
                {
                    "id": model_id,
                    "params_b": policy["params_b"],
                    "estimated_required_vram_gb": round(policy["estimated_required_vram_gb"], 3),
                    "rule_of_thumb_memory_gb": round(policy.get("rule_of_thumb_memory_gb", 0.0), 3),
                    "selected_gpu": policy["selected_gpu"],
                    "selected_gpu_count": int(policy.get("selected_gpu_count", 1)),
                }
            )
        return {"models": [row["id"] for row in rows], "catalog": rows}

    @web_app.post("/prove")
    def prove(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model_name = str(req.get("model_name", "")).strip()
            prompt = str(req.get("prompt", "")).strip()
            if not model_name:
                raise HTTPException(status_code=422, detail="model_name is required")
            if not prompt:
                raise HTTPException(status_code=422, detail="prompt is required")

            _ensure_model_allowed(model_name)
            policy = select_gpu_for_model(model_name)
            selected_gpu = policy["selected_gpu"]
            selected_gpu_count = int(policy.get("selected_gpu_count", 1))
            if selected_gpu_count == 1:
                capture_fn = GPU_TO_CAPTURE_FN[selected_gpu]
            else:
                capture_fn = GPU_TO_CAPTURE_FN_WITH_COUNT.get((selected_gpu, selected_gpu_count))
                if capture_fn is None:
                    raise ValueError(
                        f"No capture function configured for {selected_gpu} with count={selected_gpu_count}"
                    )
            run_id = f"run_{_utc_tag()}_{_sanitize_model_name(model_name)}"
            shared_payload = {
                "run_id": run_id,
                "model_name": model_name,
                "prompt": prompt,
                "max_new_tokens": int(req.get("max_new_tokens", 16)),
                "num_queries": int(req.get("num_queries", 10)),
                "seed": int(req.get("seed", 42)),
                "scale_factor": int(req.get("scale_factor", 16)),
                "quantize": float(req.get("quantize", 50.0)),
                "min_dim": int(req.get("min_dim", 4)),
                "max_dim": int(req.get("max_dim", 10)),
                "skip_interp_build": bool(req.get("skip_interp_build", True)),
                "params_b": float(policy["params_b"]),
                "estimated_required_vram_gb": float(policy["estimated_required_vram_gb"]),
                "rule_of_thumb_memory_gb": float(policy.get("rule_of_thumb_memory_gb", 0.0)),
                "selected_gpu_count": selected_gpu_count,
            }
            capture_out = capture_fn.remote(shared_payload)
            stage2_in = dict(shared_payload)
            stage2_in.update(capture_out)
            final_response = run_cpu_commit_stage.remote(stage2_in)
            return final_response
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))

    return web_app
