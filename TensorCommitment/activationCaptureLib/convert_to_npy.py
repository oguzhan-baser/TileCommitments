#!/usr/bin/env python3
"""
Integer-Scaled Activation Converter
====================================

Reads a .pt activation file (produced by capture_activations.py) and converts
every hidden-state tensor from floating point to scaled arbitrary-precision
integers, saved as individual .npy files per layer.

Conversion formula
------------------
    integer_value = round( float_value * 10^effective_scale )

Where:
    effective_scale = scale_factor - floor(scale_factor * quantize / 100)

Defaults:
    scale_factor = 32 for fp32, 16 for fp16/bfloat16
    quantize     = 0  (no digit removal)

Quick start:
    python convert_to_npy.py --input output/model_activations.pt
    python convert_to_npy.py --input output/model_activations.pt --scale-factor 24 --quantize 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Defaults — scale factor mirrors the dtype bit-width
# ---------------------------------------------------------------------------

DEFAULT_SCALE_FACTORS: Dict[str, int] = {
    "torch.float32": 32,
    "torch.float16": 16,
    "torch.bfloat16": 16,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert .pt floating-point activations to scaled-integer .npy files.",
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a .pt activation file produced by capture_activations.py.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <input_dir>/<model>_int_activations/).",
    )
    p.add_argument(
        "--scale-factor",
        type=int,
        default=None,
        help="Total number of decimal digits in the scaled integer "
             "(default: 32 for fp32, 16 for fp16/bfloat16).",
    )
    p.add_argument(
        "--quantize",
        type=float,
        default=0.0,
        help="Percentage (0-100) of least-significant scale digits to discard "
             "before integer conversion. Reduces the effective scale, yielding "
             "smaller integers at the cost of precision (default: 0).",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def compute_effective_scale(
    scale_factor: int,
    quantize_pct: float,
) -> Tuple[int, int]:
    """Return (effective_scale, truncated_digits).

    effective_scale = scale_factor - floor(scale_factor * quantize_pct / 100)
    """
    truncated = int(scale_factor * quantize_pct / 100.0)
    effective = scale_factor - truncated
    if effective < 0:
        effective = 0
    return effective, truncated


def convert_tensor_to_scaled_int(
    tensor: torch.Tensor,
    effective_scale: int,
) -> np.ndarray:
    """Convert a float tensor to a numpy object-array of Python ints.

    Uses Python's ``decimal.Decimal`` for exact decimal arithmetic so that
    **no precision is lost** relative to the original floating-point value.
    The exact binary-float value is multiplied by 10^effective_scale and
    rounded (half-even) to produce an arbitrary-precision Python ``int``.

    Parameters
    ----------
    tensor : torch.Tensor
        Floating-point activation tensor (any shape).
    effective_scale : int
        The exponent: each value is multiplied by 10^effective_scale.

    Returns
    -------
    np.ndarray
        Same shape as *tensor*, dtype=object, containing Python ``int`` values.
    """
    # Decimal context needs enough precision for the multiplication.
    getcontext().prec = effective_scale + 40
    multiplier = Decimal(10) ** effective_scale

    original_shape = tensor.shape
    # .double() promotes fp16/bfloat16 to float64 without precision loss,
    # then .tolist() yields native Python floats.
    flat_vals: list = tensor.double().flatten().tolist()
    n = len(flat_vals)

    result = np.empty(n, dtype=object)
    for i, val in enumerate(flat_vals):
        # Decimal(float) captures the EXACT decimal expansion of the
        # binary float — no rounding from str() conversion.
        result[i] = int(
            (Decimal(val) * multiplier).to_integral_value(rounding=ROUND_HALF_EVEN)
        )

    return result.reshape(original_shape)


def layer_label(idx: int) -> str:
    """Consistent naming: index 0 -> 'embedding', else 'layer_<i-1>'."""
    return "embedding" if idx == 0 else f"layer_{idx - 1}"


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load .pt file ---
    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"[ERROR] File not found: {input_path}")

    print(f"[INFO] Loading {input_path}")
    try:
        data = torch.load(input_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only parameter
        data = torch.load(input_path, map_location="cpu")

    model_name: str = data["model_name"]
    prompt: str = data["prompt"]
    generated_text: str = data["generated_text"]
    hidden_states: Tuple[torch.Tensor, ...] = data["hidden_states"]

    # --- Resolve scale factor ---
    sample_dtype = str(hidden_states[0].dtype)
    scale_factor = args.scale_factor if args.scale_factor is not None else DEFAULT_SCALE_FACTORS.get(sample_dtype, 32)

    if args.quantize < 0 or args.quantize > 100:
        sys.exit("[ERROR] --quantize must be between 0 and 100.")

    effective_scale, truncated_digits = compute_effective_scale(scale_factor, args.quantize)

    print(f"[INFO] Model           : {model_name}")
    print(f"[INFO] Original dtype  : {sample_dtype}")
    print(f"[INFO] Scale factor    : {scale_factor}")
    print(f"[INFO] Quantize        : {args.quantize}%  ({truncated_digits} digits removed)")
    print(f"[INFO] Effective scale : {effective_scale}")
    print(f"[INFO] Layers          : {len(hidden_states)}")

    # --- Output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        safe_name = sanitize_model_name(model_name)
        out_dir = input_path.parent / f"{safe_name}_int_activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert each layer ---
    layer_info: List[Dict[str, Any]] = []
    total_start = time.time()

    for idx, tensor in enumerate(hidden_states):
        label = layer_label(idx)
        shape = list(tensor.shape)
        numel = tensor.numel()

        print(
            f"  [{idx + 1}/{len(hidden_states)}] {label:>12s}  "
            f"shape={str(tuple(shape)):>24s}  elements={numel:>8d}  ... ",
            end="",
            flush=True,
        )

        t0 = time.time()
        int_array = convert_tensor_to_scaled_int(tensor, effective_scale)
        elapsed = time.time() - t0

        npy_path = out_dir / f"{label}.npy"
        np.save(npy_path, int_array, allow_pickle=True)
        print(f"done  ({elapsed:.1f}s)")

        # Collect a sample value for the metadata (first element)
        sample_original = float(tensor.flatten()[0].item())
        sample_converted = int(int_array.flat[0])

        layer_info.append({
            "layer": label,
            "original_shape": shape,
            "num_elements": numel,
            "original_dtype": str(tensor.dtype),
            "npy_file": npy_path.name,
            "sample_original": sample_original,
            "sample_converted": sample_converted,
        })

    total_elapsed = time.time() - total_start

    # --- Write conversion metadata ---
    metadata = {
        "model_name": model_name,
        "prompt": prompt,
        "generated_text": generated_text,
        "conversion": {
            "scale_factor": scale_factor,
            "quantize_pct": args.quantize,
            "truncated_digits": truncated_digits,
            "effective_scale": effective_scale,
            "formula": "integer = round(float_value * 10^effective_scale)",
            "reverse": "float_approx = integer / 10^effective_scale",
        },
        "layers": layer_info,
    }

    meta_path = out_dir / "conversion_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\n[INFO] Conversion complete in {total_elapsed:.1f}s")
    print(f"[INFO] Output  : {out_dir.resolve()}")
    print(f"[INFO] Metadata: {meta_path.name}")
    print(f"[INFO] To load a layer:")
    print(f"         arr = np.load('{out_dir / 'layer_0.npy'}', allow_pickle=True)")


if __name__ == "__main__":
    main()
