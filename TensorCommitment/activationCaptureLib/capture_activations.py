#!/usr/bin/env python3
"""
Activation Capture Library
==========================

Loads one or more Hugging Face causal language models, generates text for a
given prompt, then performs a full forward pass over the complete token
sequence (prompt + generated continuation) to extract every intermediate
hidden-state tensor.

For each model two artifacts are written:
  1.  <model>_activations.pt  – PyTorch file with all hidden-state tensors,
      the token sequence, prompt, and generated text.
  2.  <model>_stats.json      – Per-layer statistics (shape, min, max, mean,
      dtype, precision bits) and an aggregate size summary.

A consolidated JSON report is also printed to stdout after all models are
processed.

Quick start:
    python capture_activations.py \
        --models sshleifer/tiny-gpt2 \
        --prompt "Hello world" \
        --max-new-tokens 8 \
        --output-dir ./output
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AwqConfig
from transformers.utils.quantization_config import AwqBackend

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Capture intermediate activations from Hugging Face causal LMs.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more Hugging Face model identifiers (e.g. gpt2, meta-llama/Llama-3.2-1B-Instruct).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt fed into each model.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate (default: 64).",
    )
    p.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (only used when --do-sample is set, default: 0.8).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Model precision (default: float16 on CUDA, float32 on CPU).",
    )
    p.add_argument(
        "--device-map",
        type=str,
        default="none",
        choices=["none", "auto", "balanced", "balanced_low_0", "sequential"],
        help="Transformers device_map for model sharding (default: none).",
    )
    p.add_argument(
        "--max-memory-per-gpu",
        type=str,
        default=None,
        help="Per-GPU memory budget (e.g. 20GiB) used when device-map is enabled.",
    )
    p.add_argument(
        "--max-memory-cpu",
        type=str,
        default=None,
        help="CPU RAM budget (e.g. 128GiB) used with device-map max_memory.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for .pt and .json artifacts (default: ./output).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable fully deterministic execution. Forces greedy decoding, "
             "deterministic CUDA/cuDNN ops, and sets CUBLAS_WORKSPACE_CONFIG. "
             "Same input + same environment = bit-identical activations.",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str | None) -> torch.device:
    """Return an explicit device or auto-detect CUDA availability."""
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str | None, device: torch.device) -> torch.dtype:
    """Map a CLI string to a torch.dtype, with sensible defaults."""
    if dtype_arg is not None:
        return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_arg]
    return torch.float16 if device.type == "cuda" else torch.float32


def normalize_device_map(device_map_arg: str | None) -> str | None:
    if device_map_arg is None:
        return None
    value = device_map_arg.strip().lower()
    if value in {"none", "", "null"}:
        return None
    return value


def build_max_memory_map(
    device: torch.device,
    per_gpu: str | None,
    cpu_memory: str | None,
) -> Dict[int | str, str] | None:
    if per_gpu is None and cpu_memory is None:
        return None

    max_memory: Dict[int | str, str] = {}
    if per_gpu is not None:
        if device.type != "cuda":
            raise ValueError("--max-memory-per-gpu can only be used with CUDA devices")
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 0:
            raise ValueError("--max-memory-per-gpu set but no CUDA devices were detected")
        for gpu_idx in range(gpu_count):
            max_memory[gpu_idx] = per_gpu
    if cpu_memory is not None:
        max_memory["cpu"] = cpu_memory
    return max_memory


def get_model_input_device(model: AutoModelForCausalLM, fallback: torch.device) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        embedding_device = embeddings.weight.device
        if embedding_device.type != "meta":
            return embedding_device

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    for buffer in model.buffers():
        if buffer.device.type != "meta":
            return buffer.device
    return fallback


def sanitize_model_name(model_name: str) -> str:
    """Convert 'org/model' into a filesystem-safe string."""
    return model_name.replace("/", "_").replace(":", "_")


def seed_everything(seed: int) -> None:
    """Set random seeds for torch (CPU + CUDA)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode() -> None:
    """Lock down every known source of non-determinism in PyTorch.

    After this call, identical inputs on the same hardware / software stack
    will produce bit-identical activations.  Requires:
      - greedy decoding (no sampling)
      - same device, dtype, PyTorch version, and transformers version
    """
    # Force deterministic algorithm selection in all PyTorch ops.
    torch.use_deterministic_algorithms(True)

    # cuDNN: disable benchmark auto-tuner (non-deterministic kernel selection)
    # and force the deterministic convolution algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # cuBLAS: required for deterministic matrix-multiply reductions on CUDA.
    # Without this, torch.use_deterministic_algorithms(True) raises an error
    # on CUDA when cuBLAS GEMMs are invoked.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_precision_bits(tensor: torch.Tensor) -> int:
    """Return the number of bits used by the tensor's dtype."""
    if tensor.dtype.is_floating_point:
        return torch.finfo(tensor.dtype).bits
    return torch.iinfo(tensor.dtype).bits

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifacts:
    """Bundles every artifact produced for a single model run."""
    model_name: str
    prompt: str
    generated_text: str
    token_sequence: torch.Tensor        # (1, seq_len)
    hidden_states: Tuple[torch.Tensor, ...]  # one tensor per layer

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    device_map: str | None = None,
    max_memory: Dict[int | str, str] | None = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download / load a causal LM and its tokenizer from Hugging Face."""
    load_kwargs: Dict[str, object] = {"torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    def _is_awq_kernel_error(exc: Exception) -> bool:
        err = str(exc)
        return (
            "External ExLlama kernels are not properly installed" in err
            or "External ExLlamaV2 kernels are not properly installed" in err
            or "gptqmodel_exllamav2_awq_kernels" in err
            or "gptqmodel_awq_kernels" in err
        )

    def _awq_fallback_quant_config(
        trust_remote_code: bool,
        backend: str,
    ) -> AwqConfig | None:
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            quant_cfg = getattr(config, "quantization_config", None)
            if not isinstance(quant_cfg, dict):
                return None
            quant_method = str(quant_cfg.get("quant_method", "")).lower()
            if "awq" not in quant_method and "awq" not in str(quant_cfg.get("version", "")).lower():
                return None

            override = dict(quant_cfg)
            override["backend"] = backend
            override["version"] = "gemm"
            override["format"] = "gemm"
            return AwqConfig.from_dict(override)
        except Exception as cfg_exc:  # noqa: BLE001
            print(
                "[WARN] Failed to build AWQ fallback quantization config "
                f"(reason: {type(cfg_exc).__name__}: {cfg_exc})"
            )
            return None

    def _load(
        trust_remote_code: bool,
        quantization_config: AwqConfig | None = None,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        tokenizer.padding_side = "left"

        model_kwargs = dict(load_kwargs)
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            model.to(device)
        model.eval()
        return model, tokenizer

    def _load_with_awq_fallback(trust_remote_code: bool) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        try:
            return _load(trust_remote_code=trust_remote_code)
        except Exception as exc:
            if not _is_awq_kernel_error(exc):
                raise

            fallback_backends = [
                AwqBackend.GEMM.value,
                AwqBackend.TORCH_AWQ.value,
            ]
            last_error: Exception = exc
            for backend in fallback_backends:
                quantization_config = _awq_fallback_quant_config(
                    trust_remote_code=trust_remote_code,
                    backend=backend,
                )
                if quantization_config is None:
                    continue
                print(
                    "[WARN] AWQ kernel backend unavailable; retrying with "
                    f"AWQ backend='{backend}' (reason: {type(last_error).__name__}: {last_error})"
                )
                try:
                    return _load(
                        trust_remote_code=trust_remote_code,
                        quantization_config=quantization_config,
                    )
                except Exception as inner_exc:
                    if not _is_awq_kernel_error(inner_exc):
                        raise
                    last_error = inner_exc
                    continue
            raise last_error

    try:
        return _load_with_awq_fallback(trust_remote_code=False)
    except Exception as exc:
        err = str(exc)
        should_retry_remote_code = (
            "trust_remote_code" in err
            or "Could not import module" in err
            or "requires you to execute the configuration file" in err
        )
        if not should_retry_remote_code:
            raise
        print(
            "[WARN] Standard model load failed; retrying with trust_remote_code=True "
            f"(reason: {type(exc).__name__}: {exc})"
        )
        return _load_with_awq_fallback(trust_remote_code=True)


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Tuple[torch.Tensor, str]:
    """Generate a continuation and return the full sequence + decoded text."""
    model_input_device = get_model_input_device(model, torch.device("cpu"))
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model_input_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
    }

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    sequences = out.sequences
    prompt_len = inputs["input_ids"].shape[-1]
    generated_text = tokenizer.decode(
        sequences[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return sequences, generated_text


def capture_hidden_states(
    model: AutoModelForCausalLM,
    token_sequences: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Run a forward pass with output_hidden_states=True and return all layers.

    Returns a tuple of tensors, one per layer:
      index 0  = embedding output
      index i  = output of transformer block i-1
    Each tensor has shape (batch, seq_len, hidden_dim).
    """
    model_input_device = get_model_input_device(model, torch.device("cpu"))
    token_sequences = token_sequences.to(model_input_device)
    attention_mask = torch.ones_like(token_sequences, device=model_input_device)
    with torch.no_grad():
        outputs = model(
            input_ids=token_sequences,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    return tuple(t.detach().cpu() for t in outputs.hidden_states)


def summarize_hidden_states(
    model_name: str,
    hidden_states: Sequence[torch.Tensor],
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Compute per-layer statistics and an aggregate size summary."""
    layer_stats: List[Dict[str, object]] = []
    sizes: List[int] = []

    for idx, tensor in enumerate(hidden_states):
        label = "embedding" if idx == 0 else f"layer_{idx - 1}"
        numel = tensor.numel()
        sizes.append(numel)

        tf = tensor.float()
        layer_stats.append({
            "model": model_name,
            "layer": label,
            "shape": list(tensor.shape),
            "num_elements": numel,
            "min": tf.min().item(),
            "max": tf.max().item(),
            "mean": tf.mean().item(),
            "dtype": str(tensor.dtype),
            "precision_bits": get_precision_bits(tensor),
        })

    size_summary = {
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "avg_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
    }
    return layer_stats, size_summary

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    output_dir: Path,
    artifacts: ModelArtifacts,
    layer_stats: List[Dict[str, object]],
    size_summary: Dict[str, float],
) -> None:
    """Write .pt (tensors) and .json (stats) files for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = sanitize_model_name(artifacts.model_name)

    pt_path = output_dir / f"{safe}_activations.pt"
    json_path = output_dir / f"{safe}_stats.json"

    torch.save(
        {
            "model_name": artifacts.model_name,
            "prompt": artifacts.prompt,
            "generated_text": artifacts.generated_text,
            "token_sequence": artifacts.token_sequence.cpu(),
            "hidden_states": artifacts.hidden_states,
        },
        pt_path,
    )

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name": artifacts.model_name,
                "prompt": artifacts.prompt,
                "generated_text": artifacts.generated_text,
                "size_summary": size_summary,
                "layer_stats": layer_stats,
            },
            fh,
            indent=2,
        )

    print(f"[INFO] Saved activations -> {pt_path}")
    print(f"[INFO] Saved statistics  -> {json_path}")

# ---------------------------------------------------------------------------
# Per-model orchestrator
# ---------------------------------------------------------------------------

def process_model(
    model_name: str,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    device_map: str | None = None,
    max_memory: Dict[int | str, str] | None = None,
) -> Dict[str, object] | None:
    """End-to-end pipeline for a single model. Returns a report dict or None on failure."""
    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None
    try:
        print(f"\n{'='*60}")
        print(f"[INFO] Loading model: {model_name}")
        print(f"{'='*60}")
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device,
            dtype,
            device_map=device_map,
            max_memory=max_memory,
        )

        # --- Generate ---
        sequences, generated_text = run_generation(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        print(f"[INFO] Generated text: {generated_text!r}")

        # --- Capture activations ---
        hidden_states = capture_hidden_states(model, sequences)

        for idx, tensor in enumerate(hidden_states):
            label = "embedding" if idx == 0 else f"layer_{idx - 1}"
            print(
                f"  [SHAPE] {label:>12s}  shape={str(tuple(tensor.shape)):>24s}  "
                f"dtype={tensor.dtype}  elements={tensor.numel()}"
            )

        # --- Summarize & save ---
        layer_stats, size_summary = summarize_hidden_states(model_name, hidden_states)

        save_artifacts(
            output_dir,
            ModelArtifacts(
                model_name=model_name,
                prompt=prompt,
                generated_text=generated_text,
                token_sequence=sequences.cpu(),
                hidden_states=hidden_states,
            ),
            layer_stats,
            size_summary,
        )

        return {
            "model_name": model_name,
            "generated_text": generated_text,
            "size_summary": size_summary,
            "layer_stats": layer_stats,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {model_name}: {exc}")
        return None
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Deterministic mode (must run before any CUDA op) ---
    if args.deterministic:
        enable_deterministic_mode()
        if args.do_sample:
            print("[WARN] --deterministic overrides --do-sample: forcing greedy decoding.")
            args.do_sample = False
        print("[INFO] Deterministic mode ENABLED")

    seed_everything(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    device_map = normalize_device_map(args.device_map)
    if device.type != "cuda":
        if device_map is not None:
            print("[WARN] --device-map ignored because --device is not CUDA")
        device_map = None
    max_memory = build_max_memory_map(device, args.max_memory_per_gpu, args.max_memory_cpu)
    output_dir = Path(args.output_dir)

    print(
        f"[INFO] device={device}  dtype={dtype}  deterministic={args.deterministic}  "
        f"device_map={device_map or 'none'}"
    )
    if max_memory is not None:
        print(f"[INFO] max_memory={max_memory}")
    print(f"[INFO] output -> {output_dir.resolve()}")

    report: List[Dict[str, object]] = []
    for name in args.models:
        entry = process_model(
            name,
            args.prompt,
            args,
            device,
            dtype,
            output_dir,
            device_map=device_map,
            max_memory=max_memory,
        )
        if entry is not None:
            report.append(entry)

    # Print consolidated report
    print(f"\n{'='*60}")
    print("[JSON REPORT]")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
