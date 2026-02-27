#!/usr/bin/env python3
"""
Hypercube Reshaping of Integer Activations
===========================================

Reads integer-scaled ``.npy`` files produced by ``convert_to_npy.py``,
concatenates all layers (flattened, in layer order) into a single 1-D
vector, then reshapes it into the **least-sparse hypercube** whose
dimension sizes each fall within a configurable range (default 4–10).

Outputs
-------
1. ``hypercube.npy``         – the reshaped activation hypercube.
2. ``hypercube_metadata.json`` – full recovery mapping between hypercube
   coordinates, flat indices, and original (layer, position) tuples.

Quick start:
    python reshape_to_hypercube.py \\
        --input-dir output/model_int_activations/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reshape integer activations into a least-sparse hypercube.",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with .npy layer files (and optional conversion_metadata.json) "
             "produced by convert_to_npy.py.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <input-dir>_hypercube/).",
    )
    p.add_argument(
        "--min-dim",
        type=int,
        default=4,
        help="Minimum size for each hypercube dimension (default: 4).",
    )
    p.add_argument(
        "--max-dim",
        type=int,
        default=10,
        help="Maximum size for each hypercube dimension (default: 10).",
    )
    p.add_argument(
        "--num-dims",
        type=int,
        default=None,
        help="Force a specific number of dimensions (default: auto-select for least sparsity).",
    )
    p.add_argument(
        "--pad-seed",
        type=int,
        default=42,
        help="Random seed for generating padding fill values (default: 42).",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Hypercube dimension optimizer (uniform dimensions)
# ---------------------------------------------------------------------------

def find_optimal_dims(
    n: int,
    min_d: int = 4,
    max_d: int = 10,
    num_dims: Optional[int] = None,
) -> Tuple[int, ...]:
    """Find a uniform hypercube ``(d, d, ..., d)`` with ``d`` in
    [min_d, max_d] whose product ``d^k`` is the smallest value >= *n*.

    All dimensions have **equal size**.  The algorithm tries every
    candidate dimension size ``d`` in the allowed range and, for each,
    computes the smallest ``k`` such that ``d^k >= n``.  If *num_dims*
    is given, only that ``k`` is considered.  The (d, k) pair with the
    least padding wins.

    Parameters
    ----------
    n : int
        Total number of elements to fit.
    min_d, max_d : int
        Allowed range for the (single) dimension size.
    num_dims : int or None
        If set, force exactly this many dimensions.

    Returns
    -------
    tuple[int, ...]
        Dimension sizes — all entries are the same value.
    """
    if n <= 0:
        raise ValueError(f"Element count must be positive, got {n}")
    if min_d < 2:
        raise ValueError(f"min_dim must be >= 2, got {min_d}")
    if max_d < min_d:
        raise ValueError(f"max_dim ({max_d}) must be >= min_dim ({min_d})")

    best_dims: Optional[Tuple[int, ...]] = None
    best_product: int = sys.maxsize

    for d in range(min_d, max_d + 1):
        if num_dims is not None:
            # Fixed number of dimensions — just check this one k.
            k = num_dims
            product = d ** k
            if product < n:
                continue
        else:
            # Find smallest k where d^k >= n.
            k = max(1, math.ceil(math.log(n) / math.log(d)))
            product = d ** k
            # Guard against floating-point rounding in the log.
            while product < n:
                k += 1
                product = d ** k

        if product < best_product:
            best_product = product
            best_dims = (d,) * k

    if best_dims is None:
        if num_dims is not None:
            raise ValueError(
                f"Cannot fit {n} elements into {num_dims} uniform dimensions "
                f"with dimension sizes in [{min_d}, {max_d}] "
                f"(max product = {max_d ** num_dims})."
            )
        raise ValueError(
            f"No uniform hypercube configuration found for {n} elements "
            f"with dimension range [{min_d}, {max_d}]."
        )

    return best_dims

# ---------------------------------------------------------------------------
# Layer loading
# ---------------------------------------------------------------------------

def _layer_sort_key(path: Path) -> Tuple[int, int]:
    """Sort: embedding first (index -1), then layer_0, layer_1, ... by number."""
    name = path.stem
    if name == "embedding":
        return (0, -1)
    if name.startswith("layer_"):
        try:
            return (1, int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return (2, 0)


def load_layers(
    input_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], Dict[str, Any]]:
    """Load per-layer .npy files in the correct order.

    Returns
    -------
    layers_meta : list of dicts
        Per-layer metadata (name, shape, element count, file name).
    arrays : list of np.ndarray
        The loaded arrays (object-dtype with Python ints).
    source_meta : dict
        Full contents of conversion_metadata.json (empty dict if absent).
    """
    meta_path = input_dir / "conversion_metadata.json"
    source_meta: Dict[str, Any] = {}

    if meta_path.is_file():
        with meta_path.open("r") as f:
            source_meta = json.load(f)

        layers_info = source_meta.get("layers", [])
        layers_meta: List[Dict[str, Any]] = []
        arrays: List[np.ndarray] = []

        for entry in layers_info:
            npy_path = input_dir / entry["npy_file"]
            arr = np.load(npy_path, allow_pickle=True)
            layers_meta.append({
                "layer": entry["layer"],
                "original_shape": entry.get("original_shape", list(arr.shape)),
                "num_elements": int(arr.size),
                "npy_file": entry["npy_file"],
            })
            arrays.append(arr)

        return layers_meta, arrays, source_meta

    # Fallback: discover .npy files by name
    npy_files = sorted(input_dir.glob("*.npy"), key=_layer_sort_key)
    if not npy_files:
        sys.exit(f"[ERROR] No .npy files found in {input_dir}")

    layers_meta = []
    arrays = []
    for npy_path in npy_files:
        arr = np.load(npy_path, allow_pickle=True)
        layers_meta.append({
            "layer": npy_path.stem,
            "original_shape": list(arr.shape),
            "num_elements": int(arr.size),
            "npy_file": npy_path.name,
        })
        arrays.append(arr)

    return layers_meta, arrays, source_meta

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {input_dir}")

    # --- Load layers ---
    print(f"[INFO] Loading layers from {input_dir}")
    layers_meta, arrays, source_meta = load_layers(input_dir)

    # --- Flatten & concatenate ---
    flat_parts: List[np.ndarray] = []
    layer_map: List[Dict[str, Any]] = []
    offset = 0

    for meta_entry, arr in zip(layers_meta, arrays):
        flat = arr.flatten()
        flat_parts.append(flat)
        size = int(flat.size)
        layer_map.append({
            "layer": meta_entry["layer"],
            "original_shape": meta_entry["original_shape"],
            "num_elements": size,
            "flat_offset": offset,
            "flat_end": offset + size,
        })
        print(f"  {meta_entry['layer']:>12s}  "
              f"shape={str(tuple(meta_entry['original_shape'])):>24s}  "
              f"elements={size:>8d}  offset={offset}")
        offset += size

    concatenated = np.concatenate(flat_parts)
    total_elements = int(concatenated.size)
    print(f"[INFO] Total elements (all layers): {total_elements}")

    # --- Find optimal hypercube dimensions ---
    print(
        f"[INFO] Searching for optimal hypercube  "
        f"(dim range [{args.min_dim}, {args.max_dim}]"
        + (f", forced k={args.num_dims}" if args.num_dims else "")
        + ")"
    )

    t0 = time.time()
    dims = find_optimal_dims(
        total_elements,
        min_d=args.min_dim,
        max_d=args.max_dim,
        num_dims=args.num_dims,
    )
    search_ms = (time.time() - t0) * 1000

    product = math.prod(dims)
    padding = product - total_elements
    sparsity_pct = 100.0 * padding / product if product > 0 else 0.0

    print(f"[INFO] Dimensions : {dims}  ({len(dims)}-D)")
    print(f"[INFO] Product    : {product}")
    print(f"[INFO] Padding    : {padding} elements  ({sparsity_pct:.2f}% sparsity)")
    print(f"[INFO] Search time: {search_ms:.1f}ms")

    # --- Pad with random values sampled from real activations ---
    if padding > 0:
        rng = np.random.default_rng(args.pad_seed)
        # Sample random indices into the real data and use those values as fill.
        random_indices = rng.integers(0, total_elements, size=padding)
        pad_arr = concatenated[random_indices]
        padded = np.concatenate([concatenated, pad_arr])
        print(f"[INFO] Padding filled with {padding} values randomly sampled from real data (seed={args.pad_seed})")
    else:
        padded = concatenated

    hypercube = padded.reshape(dims)

    # --- Save outputs ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = input_dir.parent / (input_dir.name + "_hypercube")
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "hypercube.npy"
    np.save(npy_path, hypercube, allow_pickle=True)
    print(f"[INFO] Saved hypercube  -> {npy_path}")

    # --- Save recovery metadata ---
    dims_list = list(dims)
    metadata: Dict[str, Any] = {
        "source_dir": str(input_dir.resolve()),
        "model_name": source_meta.get("model_name", "unknown"),
        "conversion_params": source_meta.get("conversion", {}),
        "hypercube": {
            "dimensions": dims_list,
            "num_dimensions": len(dims_list),
            "product": product,
            "total_real_elements": total_elements,
            "padding_elements": padding,
            "sparsity_pct": round(sparsity_pct, 4),
            "pad_fill": "random (sampled from real activations)",
            "pad_seed": args.pad_seed,
            "dim_range": [args.min_dim, args.max_dim],
            "flatten_order": "C (row-major)",
        },
        "layer_map": layer_map,
        "recovery": {
            "description": (
                "All indices use C-order (row-major) flattening. "
                "Layers are concatenated in the order listed in layer_map. "
                "Positions with flat_index >= total_real_elements are padding."
            ),
            "flat_to_hypercube": (
                "coords = numpy.unravel_index(flat_index, dimensions)"
            ),
            "hypercube_to_flat": (
                "flat_index = numpy.ravel_multi_index(coords, dimensions)"
            ),
            "flat_to_layer": (
                "Find the entry in layer_map where "
                "flat_offset <= flat_index < flat_end. "
                "position_in_layer = flat_index - flat_offset. "
                "Use numpy.unravel_index(position_in_layer, original_shape) "
                "to recover the multi-dimensional index within that layer."
            ),
            "is_padding": (
                f"flat_index >= {total_elements}"
            ),
        },
    }

    meta_path = out_dir / "hypercube_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved metadata   -> {meta_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
