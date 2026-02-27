# TensorCommitments: A Lightweight Verifiable Inference for Language Models

## Podcast

https://github.com/user-attachments/assets/e68a3551-a9b5-4732-88b9-4db2cd6dea74


## Overview

![assets/intro_overview_fig.png](assets/intro_overview_fig.png)

This directory contains everything needed to:

1. **Capture** hidden-state activations from a Hugging Face language model
2. **Convert** floating-point activations to integer-scaled values
3. **Reshape** into an N-dimensional hypercube
4. **Interpolate** the hypercube into a multivariate polynomial over the BN254 field
5. **Commit** to the polynomial cryptographically (32-byte digest)
6. **Prove/Verify** any activation value with a succinct proof (256 bytes, 7ms verification)

## Directory Structure

```
finalcode/
├── README.md                    # This file
├── activationCaptureLib/        # Steps 1-3: capture, convert, reshape
│   ├── capture_activations.py   #   Step 1: extract hidden states from HF models
│   ├── convert_to_npy.py        #   Step 2: float -> integer scaling
│   ├── reshape_to_hypercube.py  #   Step 3: flatten + reshape into hypercube
│   └── output/                  #   All intermediate and final data files
├── layerSelectionLib/           # Optional: layer significance extraction + budgeted selection
│   └── run_layer_selection.py   #   Unified runner (extract / visualize / optimize / benchmark)
├── interpolationLib/            # Step 4: polynomial interpolation
│   └── interpolate_hypercube.py #   Hypercube -> BN254 polynomial coefficients
├── tensorCommitmentLib/         # Steps 5-6: commit, prove, verify
│   └── commit_prove_verify.py   #   Tensor commitment + proof generation
└── pst_commitment_lib/          # Rust cryptographic library (self-contained)
│     ├── src/                     #   TensorCommitment implementations
│     ├── poly_interp_demo/        #   N-D interpolation binary (Rust)
│     ├── Cargo.toml               #   Rust crate config (theseus)
│     └── build_python.sh          #   Builds tensorcommitments Python module
├── terkleLib/              # Terkle tree Rust library (depends on tensorCommitmentLib)
│   ├── Cargo.toml          # Package: terkle
│   ├── pyproject.toml      # Python module: terkle
│   └── src/
│       ├── lib.rs          # Core MultiverkleTree implementation
│       ├── python.rs       # Python bindings
│       └── bin/
│           └── benchmark_dimensions.rs  # Benchmark binary
│
├── CleanPegasus/           # Verkle tree implementation (univariate KZG)
│   ├── verkle-tree/        # Rust library
│   │   └── pointproofs/    # KZG commitment implementation
│   └── bindings/python/    # Python bindings and demos
│       └── demo/
│           ├── sweep_simulations.py
│           └── new_verkle_demo_metrics.py
│
├── merkle/                 # Multi-branch Merkle tree (SHA-256)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # MultiMerkleTree implementation
│   │   └── python.rs       # Python bindings
│   └── python_demo/
│       └── final_prover_verifier/
│           └── run_demo.py
│
└── experiments/            # Experiment scripts and results
    ├── run_all_sweeps.py   # Master orchestration script
    ├── sweep_terkle.py     # Terkle benchmark sweep
    ├── sweep_verkle.py     # Verkle benchmark sweep
    ├── sweep_merkle.py     # Merkle benchmark sweep
    ├── plot_dimension_benchmark.py  # Plotting script
    └── results/            # Output directory
    │    ├── dimension_benchmark.jsonl
    │    └── dimension_benchmark_plot.png
    │    ├── weight_norms/                # Per-model weight norms (generated)
    │    └── {model_short}_norms.json
    │    ├── case_*_results.json
    │    ├── {model_short}_all_results.json
    │    ├── {model_short}_comparison.png
    │    └── combined_comparison.png  # When running all models
    ├── scalability/                 # Scalability experiments
    │    ├── run_scalability_benchmarks.py
    │    ├── plot_scalability.py
    │    └── results/
    |── layer_analysis_experiments
    |── scalability_experiments
    ├── generate_budget_cases.py     # Step 1: sample budget cases
    ├── extract_weight_norms.py      # Step 2: extract L1/L2/L∞ norms
    ├── run_experiment_benchmarks.py # Step 3: run all methods on cases
    ├── plot_results.py              # Step 4: boxplots and summary
    ├── run_full_benchmark.py        # Single model: run steps 1–4
    ├── run_all_models.py            # All models + combined plot
    ├── test_benchmark.py            # Quick sanity check
    └── model_utils.py               # Shared helpers
```

`layerSelectionLib` is independent from the tensor commitment pipeline and can be used as a standalone module for layer scoring and selection.

## Quick Start

### Prerequisites

- **Rust toolchain** (`cargo`, `rustc`) -- install via [rustup](https://rustup.rs/)
- **Python 3.10+** with a conda or virtual environment
- **pip packages**: `numpy`, `maturin`, `torch`, `transformers`

### Setup

```bash
# Create environment
conda create -n tc python=3.10
conda activate tc

# Install Python dependencies
pip install numpy maturin torch transformers

# Build the Rust cryptographic library (Python bindings)
cd finalcode/pst_commitment_lib
maturin develop --features python --release
cd ../..
```

### Run the full pipeline

```bash
cd finalcode

# Step 1: Capture activations (requires GPU or CPU)
python activationCaptureLib/capture_activations.py \
    --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt "Explain verifiable inference." \
    --max-new-tokens 30 \
    --output-dir activationCaptureLib/output

# Step 2: Convert to integers
python activationCaptureLib/convert_to_npy.py \
    --input activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_activations.pt \
    --scale-factor 16 --quantize 50

# Step 3: Reshape into hypercube
python activationCaptureLib/reshape_to_hypercube.py \
    --input-dir activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations

# Step 4: Interpolate into polynomial
python interpolationLib/interpolate_hypercube.py \
    --input-dir activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube

# Steps 5-6: Commit, prove, verify
python tensorCommitmentLib/commit_prove_verify.py \
    --poly-dir activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_polynomial
```

## Pipeline Data Flow

```
Model inference
       |
       v
  activations.pt  (hidden states, ~50 MB)
       |  capture_activations.py
       v
  int_activations/  (per-layer .npy, integer-scaled)
       |  convert_to_npy.py + reshape_to_hypercube.py
       v
  hypercube.npy  (N-D tensor, ~8 MB)
       |  interpolate_hypercube.py
       v
  coefficients.json  (BN254 polynomial, ~135 MB)
       |  commit_prove_verify.py
       v
  commitment.txt  (32 bytes)  +  proofs.json  (~8 KB)
```




# activationCaptureLib

Capture and convert intermediate hidden-state activations from Hugging Face causal language models.

## Scripts

| Script | Purpose |
|--------|---------|
| `capture_activations.py` | Run inference, extract all hidden-state tensors, save as `.pt` + `.json` |
| `convert_to_npy.py` | Convert saved `.pt` activations to scaled arbitrary-precision integers in `.npy` format |
| `reshape_to_hypercube.py` | Concatenate all integer layers and reshape into a least-sparse hypercube |

## What It Does

**Step 1 -- Capture** (`capture_activations.py`):

1. Loads one or more HF causal LMs (`AutoModelForCausalLM`).
2. Generates a continuation for a given prompt.
3. Runs a second forward pass over the full token sequence (prompt + generation) with `output_hidden_states=True` to capture every layer's hidden-state tensor.
4. Saves two files per model:
   - **`<model>_activations.pt`** -- all hidden-state tensors, token sequence, prompt, and generated text.
   - **`<model>_stats.json`** -- per-layer statistics (shape, min, max, mean, dtype, precision bits) and aggregate size summary.
5. Prints a consolidated JSON report to stdout.

**Step 2 -- Convert** (`convert_to_npy.py`):

1. Loads a `.pt` activation file from Step 1.
2. Scales every floating-point value to an arbitrary-precision integer using exact decimal arithmetic: `integer = round(float_value * 10^effective_scale)`.
3. Saves one `.npy` file per layer (object-dtype arrays of Python `int`).
4. Writes `conversion_metadata.json` with all parameters needed to reverse the conversion.

**Step 3 -- Reshape** (`reshape_to_hypercube.py`):

1. Loads all per-layer `.npy` files from Step 2.
2. Flattens each layer and concatenates them into a single 1-D vector.
3. Finds the least-sparse hypercube shape whose dimension sizes are each in [4, 10] (configurable).
4. Pads (if needed) and reshapes into the hypercube.
5. Saves `hypercube.npy` and `hypercube_metadata.json` with full recovery mappings between hypercube coordinates, flat indices, and original (layer, position) tuples.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. CUDA is used automatically if available; otherwise falls back to CPU.

## Usage

```bash
python capture_activations.py \
    --models sshleifer/tiny-gpt2 \
    --prompt "Hello world" \
    --max-new-tokens 8 \
    --output-dir ./output
```

Multiple models in a single run:

```bash
python capture_activations.py \
    --models meta-llama/Llama-3.2-1B-Instruct facebook/opt-125m \
    --prompt "Explain verifiable inference in one paragraph." \
    --max-new-tokens 32 \
    --output-dir ./output
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | *(required)* | HF model identifiers (space-separated) |
| `--prompt` | *(required)* | Input text prompt |
| `--max-new-tokens` | 64 | Max tokens to generate |
| `--do-sample` | off | Enable sampling (otherwise greedy) |
| `--temperature` | 0.8 | Sampling temperature (ignored without `--do-sample`) |
| `--device` | auto | `cuda` / `cpu` / `mps` |
| `--dtype` | auto | `float16` / `float32` / `bfloat16` |
| `--output-dir` | `./output` | Where to write artifacts |
| `--seed` | 42 | Random seed |
| `--deterministic` | off | Bit-identical activations across runs (see below) |

## Deterministic Mode

Pass `--deterministic` to guarantee that the same input produces **bit-identical activations** across runs:

```bash
python capture_activations.py \
    --models sshleifer/tiny-gpt2 \
    --prompt "Hello world" \
    --max-new-tokens 8 \
    --deterministic \
    --output-dir ./output
```

This flag automatically:

1. Forces **greedy decoding** (overrides `--do-sample` if set).
2. Enables `torch.use_deterministic_algorithms(True)` -- all PyTorch ops use deterministic kernels.
3. Sets `cudnn.deterministic = True` and `cudnn.benchmark = False` -- prevents non-deterministic cuDNN kernel selection.
4. Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` -- required for deterministic cuBLAS GEMM reductions on CUDA.
5. Seeds all RNGs via `--seed` (default 42).

**Requirements for reproducibility across runs:**

- Same `--device` and `--dtype` (CPU vs CUDA and fp16 vs fp32 give different numerics)
- Same PyTorch and transformers versions
- Same hardware (different GPU architectures may use different floating-point reduction orders)

## Output Structure

After running all three scripts:

```
output/
├── sshleifer_tiny-gpt2_activations.pt              # Step 1
├── sshleifer_tiny-gpt2_stats.json                  # Step 1
├── sshleifer_tiny-gpt2_int_activations/            # Step 2
│   ├── conversion_metadata.json
│   ├── embedding.npy
│   ├── layer_0.npy
│   └── layer_1.npy
└── sshleifer_tiny-gpt2_int_activations_hypercube/  # Step 3
    ├── hypercube.npy
    └── hypercube_metadata.json
```

### `_activations.pt` contents

```python
data = torch.load("output/sshleifer_tiny-gpt2_activations.pt")
data["model_name"]      # "sshleifer/tiny-gpt2"
data["prompt"]          # "Hello world"
data["generated_text"]  # " stairs stairs ..."
data["token_sequence"]  # tensor of shape (1, seq_len)
data["hidden_states"]   # tuple of tensors, one per layer
                        #   index 0 = embedding output
                        #   index i = transformer block i-1 output
                        #   each shape: (1, seq_len, hidden_dim)
```

### `_stats.json` structure

```json
{
  "model_name": "sshleifer/tiny-gpt2",
  "prompt": "Hello world",
  "generated_text": "...",
  "size_summary": { "min_size": 20, "max_size": 20, "avg_size": 20.0 },
  "layer_stats": [
    {
      "layer": "embedding",
      "shape": [1, 10, 2],
      "num_elements": 20,
      "min": -0.05,
      "max": 0.11,
      "mean": 0.004,
      "dtype": "torch.float32",
      "precision_bits": 32
    }
  ]
}
```

## Integer Conversion (`convert_to_npy.py`)

### Usage

Basic conversion (uses default scale factor based on dtype):

```bash
python convert_to_npy.py --input output/sshleifer_tiny-gpt2_activations.pt
```

Custom scale factor and quantization:

```bash
python convert_to_npy.py \
    --input output/sshleifer_tiny-gpt2_activations.pt \
    --scale-factor 16 \
    --quantize 50 \
    --output-dir ./converted/sshleifer_tiny-gpt2_activations
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to a `.pt` activation file |
| `--output-dir` | auto | Output directory (default: `<model>_int_activations/` next to input) |
| `--scale-factor` | auto | Total decimal digits in scaled integer (default: 32 for fp32, 16 for fp16/bfloat16) |
| `--quantize` | 0 | Percentage (0-100) of least-significant digits to discard |

### How scaling works

Each float is multiplied by `10^effective_scale` and rounded to an integer:

```
effective_scale = scale_factor - floor(scale_factor * quantize / 100)
integer_value   = round(float_value * 10^effective_scale)
```

**Examples with a float value of `0.04248`:**

| scale_factor | quantize | effective_scale | integer result |
|-------------|----------|-----------------|----------------|
| 32 | 0% | 32 | `42480000000000000000000000000000` (preserves exact float) |
| 16 | 0% | 16 | `424800000000000` |
| 32 | 25% | 24 | `42480000000000000000000000` |
| 16 | 50% | 8 | `4248000` |

### Precision guarantee

- Uses Python's `decimal.Decimal` for exact decimal arithmetic (not float64 math).
- `Decimal(float_value)` captures the **exact** binary-float representation.
- No precision is lost relative to the original dtype. The saved integer is a lossless encoding of the original float at the given scale.

### Reversing the conversion

```python
import numpy as np
import json

meta = json.load(open("output/sshleifer_tiny-gpt2_int_activations/conversion_metadata.json"))
scale = meta["conversion"]["effective_scale"]

arr = np.load("output/sshleifer_tiny-gpt2_int_activations/layer_0.npy", allow_pickle=True)
floats = arr.astype(float) / (10 ** scale)  # approximate reverse
```

### `conversion_metadata.json` structure

```json
{
  "model_name": "sshleifer/tiny-gpt2",
  "prompt": "Hello world",
  "generated_text": "...",
  "conversion": {
    "scale_factor": 32,
    "quantize_pct": 0.0,
    "truncated_digits": 0,
    "effective_scale": 32,
    "formula": "integer = round(float_value * 10^effective_scale)",
    "reverse": "float_approx = integer / 10^effective_scale"
  },
  "layers": [
    {
      "layer": "embedding",
      "original_shape": [1, 10, 2],
      "num_elements": 20,
      "original_dtype": "torch.float32",
      "npy_file": "embedding.npy",
      "sample_original": 0.04248,
      "sample_converted": 42480000000000000000000000000000
    }
  ]
}
```

## Hypercube Reshaping (`reshape_to_hypercube.py`)

### Usage

```bash
# Default: auto-selects dimensions in [4, 10] for minimum padding
python reshape_to_hypercube.py \
    --input-dir output/sshleifer_tiny-gpt2_int_activations/

# Force 4 dimensions
python reshape_to_hypercube.py \
    --input-dir output/sshleifer_tiny-gpt2_int_activations/ \
    --num-dims 4

# Custom dimension range
python reshape_to_hypercube.py \
    --input-dir output/sshleifer_tiny-gpt2_int_activations/ \
    --min-dim 5 --max-dim 8
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Directory with `.npy` layer files from `convert_to_npy.py` |
| `--output-dir` | auto | Output directory (default: `<input-dir>_hypercube/`) |
| `--min-dim` | 4 | Minimum size per hypercube dimension |
| `--max-dim` | 10 | Maximum size per hypercube dimension |
| `--num-dims` | auto | Force a specific number of dimensions |
| `--pad-seed` | 42 | Random seed for reproducible padding fill |

### How it works

1. All per-layer `.npy` files are flattened (C-order) and concatenated in layer order (embedding, layer_0, layer_1, ...) into a single 1-D vector of length N.
2. The search finds a **uniform** dimension size `d` (in [min_dim, max_dim]) and the smallest `k` such that `d^k >= N`. All dimensions are equal. The `(d, k)` pair with the least padding wins.
3. If `d^k > N`, the extra positions are filled with values **randomly sampled from the real activations** (seeded for reproducibility via `--pad-seed`). The hypercube is then reshaped into `k` dimensions of size `d`.

**Example:** 3 layers with 20 elements each = 60 total.

```
Optimal shape: (4, 4, 4)   product=64   padding=4   sparsity=6.25%
```

**Example:** 28 layers of 86,016 elements each = 2,408,448 total.

```
Optimal shape: (9, 9, 9, 9, 9, 9, 9)   product=4,782,969   ...
  vs.         (8, 8, 8, 8, 8, 8, 8, 8)  product=16,777,216  ...
  vs.         (10, 10, 10, 10, 10, 10, 10) product=10,000,000 ...
Winner: (9, 9, 9, 9, 9, 9, 9) — smallest d^k >= 2,408,448
```

### Recovery: mapping between representations

The metadata file stores everything needed to convert between the three representations:

```python
import numpy as np
import json

# Load
meta = json.load(open("output/.../hypercube_metadata.json"))
dims = tuple(meta["hypercube"]["dimensions"])
layer_map = meta["layer_map"]
total_real = meta["hypercube"]["total_real_elements"]

cube = np.load("output/.../hypercube.npy", allow_pickle=True)

# --- Hypercube coords -> flat index ---
coords = (2, 1, 3)  # example position in the hypercube
flat_idx = np.ravel_multi_index(coords, dims)

# --- Flat index -> hypercube coords ---
coords = np.unravel_index(flat_idx, dims)

# --- Flat index -> (layer, position within layer) ---
if flat_idx >= total_real:
    print("This is a padding position")
else:
    for entry in layer_map:
        if entry["flat_offset"] <= flat_idx < entry["flat_end"]:
            layer_name = entry["layer"]
            pos_in_layer = flat_idx - entry["flat_offset"]
            # Multi-dim index within the original layer tensor:
            layer_coords = np.unravel_index(pos_in_layer, entry["original_shape"])
            print(f"Layer: {layer_name}, position: {layer_coords}")
            break

# --- (layer, position) -> hypercube coords ---
# e.g., layer_0, flat position 42
target_offset = layer_map[1]["flat_offset"]  # layer_0 is index 1
flat_idx = target_offset + 42
coords = np.unravel_index(flat_idx, dims)
```


# interpolationLib

Polynomial interpolation of hypercube activation tensors over the BN254 scalar field.

Takes the `hypercube.npy` files produced by `activationCaptureLib` and converts them into multivariate polynomials whose coefficients live in the BN254 finite field — ready for cryptographic commitment via `pst_commitment_lib`.

## What It Does

Given an N-dimensional hypercube of integer-scaled activations on the grid `{0, ..., n_0-1} x ... x {0, ..., n_{d-1}-1}`, the script finds the unique multivariate polynomial:

```
p(x_0, x_1, ..., x_{d-1}) = sum  c_{a_0,...,a_{d-1}} * x_0^{a_0} * ... * x_{d-1}^{a_{d-1}}
```

such that `p(i_0, ..., i_{d-1}) = hypercube[i_0, ..., i_{d-1}]` for every grid point.

The interpolation uses a **tensor-product Vandermonde inversion** implemented in Rust (from `pst_commitment_lib/poly_interp_demo`), which decomposes the N-D problem into a sequence of 1-D inverse Vandermonde transforms along each axis. All arithmetic is performed in the BN254 scalar field (prime order ~2^254).

### Negative value handling

Neural network activations can be negative. In the BN254 field F_p, a negative integer `-x` is encoded as its additive inverse `p - x`. This is the standard representation — all field operations (addition, multiplication, polynomial evaluation) remain correct because the encoding is a ring homomorphism.

## Prerequisites

- **Rust toolchain** (`cargo` in `$PATH`)
- **Python 3.10+** with `numpy`
- The `pst_commitment_lib/poly_interp_demo` Cargo project (auto-detected via relative path within `finalcode/`)

## Usage

### From a hypercube directory

```bash
cd finalcode/interpolationLib

python interpolate_hypercube.py \
    --input-dir ../activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube
```

### From a specific .npy file

```bash
python interpolate_hypercube.py \
    --npy ../activationCaptureLib/output/.../hypercube.npy
```

### Custom output directory

```bash
python interpolate_hypercube.py \
    --input-dir ../activationCaptureLib/output/..._hypercube \
    --output-dir ./my_output
```

### Skip Rust rebuild (if binary is already compiled)

```bash
python interpolate_hypercube.py \
    --input-dir ../activationCaptureLib/output/..._hypercube \
    --skip-build
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required\*)* | Hypercube directory containing `hypercube.npy` and `hypercube_metadata.json` |
| `--npy` | *(required\*)* | Direct path to a `.npy` file (alternative to `--input-dir`) |
| `--output-dir` | auto | Output directory (default: `<input-dir>_polynomial/`) |
| `--poly-project` | auto | Path to `poly_interp_demo` Cargo project |
| `--skip-build` | off | Skip `cargo build` (assume binary is up-to-date) |

\* Exactly one of `--input-dir` or `--npy` is required.

## Output

```
<input-dir>_polynomial/
├── coefficients.json           # The polynomial
└── interpolation_metadata.json # Provenance and timing
```

### `coefficients.json`

```json
{
  "dims": [6, 6, 6, 6, 6, 6, 6, 6],
  "degree_bound": 6,
  "coefficients": ["5908203", "9484905244463685...", ...]
}
```

- **`dims`**: Shape of the hypercube / polynomial index space.
- **`degree_bound`**: Maximum degree per variable (= max dimension size).
- **`coefficients`**: Flat array of BN254 field elements as decimal strings, in **row-major (C-order) lexicographic** order over the exponent tuple `(a_0, a_1, ..., a_{d-1})`.

The coefficient at flat index `k` corresponds to the monomial `x_0^{a_0} * ... * x_{d-1}^{a_{d-1}}` where:

```
k = a_0 * (n_1 * n_2 * ... * n_{d-1}) + a_1 * (n_2 * ... * n_{d-1}) + ... + a_{d-1}
```

This ordering matches the commitment key structure in `pst_commitment_lib`, so the coefficients can be passed directly to `TensorCommitmentWrapper.commit()`.

### `interpolation_metadata.json`

Contains:
- Source file paths and the full `hypercube_metadata.json` (for traceability)
- Polynomial description (field, modulus, coefficient order)
- Data statistics (value range, whether negatives are present)
- Timing breakdown (load, dtype conversion, interpolation)

## How It Works

1. **Load** the hypercube `.npy` (may be object-dtype with arbitrary-precision Python ints).
2. **Convert** to `int64` dtype (required by the Rust `ndarray-npy` reader). Checks that all values fit; exits with an error if they overflow.
3. **Build** the Rust `poly_interp_demo` binary (`cargo build --release`).
4. **Run** the binary with `--coeffs -` so coefficient JSON streams to stdout. Diagnostic logs go to stderr.
5. **Save** `coefficients.json` and `interpolation_metadata.json` to the output directory.
6. **Clean up** the temporary int64 `.npy` file.

### Why Python, not a shell script?

The dtype conversion (object-dtype → int64) requires `numpy` — a shell script would need to call Python for that step anyway. The Python overhead is ~0.5s out of ~6s total; the Rust binary does all the heavy arithmetic.

## Precision and int64 limits

The hypercube values are integer-scaled floats: `integer = round(float * 10^effective_scale)`.

fp32 has ~7 significant decimal digits. int64 holds up to ~18.9 decimal digits. The constraint is:

```
max |activation| * 10^effective_scale  <  2^63  ≈  9.22 * 10^18
```

For typical neural network activations (magnitude < 10,000) with the recommended `effective_scale = 8`:

```
10,000 * 10^8 = 10^12  <<  9.22 * 10^18   (safe by 6 orders of magnitude)
```

This captures all ~7 meaningful fp32 digits. Going beyond `effective_scale ≈ 15` risks int64 overflow for large activations, with no precision benefit (extra digits are fp32 rounding noise).

| Max \|activation\| | Max safe `effective_scale` | Recommended |
|:---:|:---:|:---:|
| ~10 | 17 | 8 |
| ~100 | 16 | 8 |
| ~1,000 | 15 | 8 |
| ~10,000 | 14 | 8 |

## Example run

```
$ python interpolate_hypercube.py --input-dir .../DeepSeek_hypercube --skip-build

[INFO] Loading hypercube ...
[INFO] Shape: (6, 6, 6, 6, 6, 6, 6, 6)  (8-D, 1679616 elements)
[INFO] dtype: object   load time: 0.09s
[INFO] Value range: [-260000000000, 245600000000]  has_negative=True
[INFO] Converting object-dtype to int64 for Rust compatibility ...
[INFO] Running interpolation ...
[INFO] Interpolation completed in 5.37s
[INFO] Obtained 1679616 coefficients  (degree_bound=6)
[INFO] Saved coefficients -> .../coefficients.json  (135.34 MB)

=== Summary ===
  Shape:          (6, 6, 6, 6, 6, 6, 6, 6)
  Coefficients:   1679616
  Degree bound:   6
  Total time:     5.86s
```

## Integration with tensor commitment

The output `coefficients.json` is directly compatible with `pst_commitment_lib`:

```python
import json
import tensorcommitments

# Load polynomial
with open("coefficients.json") as f:
    poly = json.load(f)

coeffs = [int(c) for c in poly["coefficients"]]
num_vars = len(poly["dims"])
degree_bound = poly["degree_bound"]

# Commit
pst = tensorcommitments.TensorCommitmentWrapper(num_vars, degree_bound)
commitment = pst.commit(coeffs)

# Prove & verify at a point
point = [0, 0, 0, 0, 0, 0, 0, 0]
eval_val = pst.evaluate_polynomial(coeffs, point)
proof = pst.prove(coeffs, point, eval_val)
assert pst.verify(commitment, point, eval_val, proof)
```


# tensorCommitmentLib

Cryptographic tensor commitment for multivariate polynomials over the BN254 curve.

Takes the polynomial coefficients produced by `interpolationLib` and runs the full **TensorCommitment** pipeline: setup, commit, prove, and verify. Optionally cross-checks evaluations against the original hypercube ground truth.

## What It Does

1. **Setup** -- Generates a structured reference string (commitment key + verification key) for the given polynomial dimensions.
2. **Commit** -- Produces a single 32-byte binding commitment to the 1.68M-coefficient polynomial.
3. **Prove** -- For each queried evaluation point, generates a succinct proof (256 bytes) that the polynomial evaluates to the claimed value.
4. **Verify** -- Checks each proof against the commitment using pairings (7ms per verification).
5. **Cross-check** -- Compares polynomial evaluations against the original `hypercube.npy` to confirm the full pipeline is correct end-to-end.

## Prerequisites

- Python 3.10+ with `numpy`
- The `tensorcommitments` Python module (built from `pst_commitment_lib`):

```bash
cd finalcode/pst_commitment_lib
pip install maturin
maturin develop --features python --release
```

## Usage

### Basic (auto-detects hypercube for ground-truth checks)

```bash
cd finalcode/tensorCommitmentLib

python commit_prove_verify.py \
    --poly-dir ../activationCaptureLib/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_polynomial
```

The script auto-detects the corresponding `_hypercube` sibling directory for ground-truth cross-checking.

### With explicit paths

```bash
python commit_prove_verify.py \
    --poly-dir ../activationCaptureLib/output/..._polynomial \
    --hypercube-dir ../activationCaptureLib/output/..._hypercube \
    --output-dir ./my_output \
    --num-queries 20
```

### Skip ground-truth (polynomial only)

If no hypercube is available, the script still commits, proves, and verifies -- it just skips the ground-truth cross-check.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--poly-dir` | *(required)* | Directory containing `coefficients.json` from `interpolationLib` |
| `--hypercube-dir` | auto | Directory with `hypercube.npy` for ground-truth cross-checking |
| `--output-dir` | auto | Output directory (default: `<name>_commitment/` sibling) |
| `--num-queries` | 10 | Number of evaluation points to test |
| `--seed` | 42 | Random seed for point selection |

## Output

```
<name>_commitment/
├── commitment.txt            # The 32-byte commitment (hex)
├── commitment_results.json   # Full results with timing and verification
└── proofs.json               # Portable proofs for each query point
```

### `commitment.txt`

The commitment as a hex string (64 characters = 32 bytes). This is a compressed BN254 G1 point.

### `commitment_results.json`

```json
{
  "commitment_hex": "e875b7a0...",
  "commitment_bytes": 32,
  "polynomial": {
    "num_variables": 8,
    "degree_bound": 6,
    "num_coefficients": 1679616,
    "source_dir": "..."
  },
  "verification_summary": {
    "num_queries": 10,
    "all_proofs_verified": true,
    "all_ground_truth_matched": true
  },
  "timing": {
    "setup_time_s": 1.498,
    "commit_time_s": 1.918,
    "avg_prove_time_s": 4.419,
    "avg_verify_time_s": 0.007,
    "..."
  },
  "point_results": [ ... ]
}
```

### `proofs.json`

Portable proof bundle -- contains the commitment, evaluation points, claimed values, and proof hex strings. A verifier with only this file and the verification key can check all proofs:

```json
{
  "commitment_hex": "e875b7a0...",
  "num_variables": 8,
  "degree_bound": 6,
  "proofs": [
    {
      "point": [0, 0, 0, 0, 0, 0, 0, 0],
      "evaluation": 5908203,
      "proof_hex": ["abcd...", "ef01...", ...]
    }
  ]
}
```

## Performance (DeepSeek-R1-Distill-Qwen-1.5B activations)

| Metric | Value |
|--------|-------|
| Polynomial | 8 variables, degree 6, 1,679,616 coefficients |
| Setup | 1.5s |
| Commit | 1.9s |
| Commitment size | **32 bytes** |
| Avg prove time | 4.4s |
| Avg verify time | **7ms** |
| Proof size | **256 bytes** (8 compressed G1 points, one per variable) |

Key takeaway: a 1.68M-element tensor (8 MB as `.npy`) is committed to a **32-byte** digest. Each evaluation proof is **256 bytes** and verifies in **7 milliseconds**.

## How the scheme works

The commitment uses:

- **Commitment key**: `g1^{t_0^{a_0} * t_1^{a_1} * ... * t_{m-1}^{a_{m-1}}}` for all degree tuples, where `t_i` are secret trapdoor values.
- **Commit**: `C = MSM(CRS, coefficients)` -- a single multi-scalar multiplication producing one G1 point.
- **Prove** at point `(v_0, ..., v_{m-1})`: For each variable `i`, divide the polynomial by `(x_i - v_i)` to get quotient `q_i`. The proof is `pi_i = MSM(CRS, q_i)` -- one G1 point per variable.
- **Verify**: Check the pairing equation `sum e(pi_i, [t_i - v_i]_2) == e(eval * g1 - C, g2)`.

The scheme is **binding** (can't open to a different polynomial) and proofs are **succinct** (constant size per variable, independent of polynomial size).

## Integration in the full pipeline

```
activationCaptureLib          interpolationLib          tensorCommitmentLib
  capture_activations.py         interpolate_hypercube.py    commit_prove_verify.py
  convert_to_npy.py                |                           |
  reshape_to_hypercube.py          |                           |
        |                          |                           |
   hypercube.npy  ------>  coefficients.json  ------>  commitment.txt
   (8 MB)                  (135 MB)                    proofs.json (8 KB)
                                                       commitment_results.json
```

## CommitmentLib

This folder is a self-contained implementation of multivariate tensor commitments with
polynomial interpolation over the BN254 curve. It provides Python bindings for committing
to multivariate polynomials, generating evaluation proofs, and verifying them. Copying or zipping `theseustensorcommitment/` is enough to rebuild everything and
reproduce the examples without depending on files outside this directory. See **[SETUP.md](SETUP.md)** for detailed step-by-step instructions.

**Quick version:**
```bash
conda create -n theseus python=3.10
conda activate theseus
cd theseustensorcommitment
pip install -r requirements.txt
./build_python.sh
python no_saving_time_version.py poly_interp_demo/tensorofshape10by10by10by10by10by10.npy
```

### Layout

```
theseustensorcommitment/
├── Cargo.toml / Cargo.lock         # Rust crate (theseus)
├── src/                            # Rust implementation (TensorCommitment + Python bindings)
├── benches/                        # Bench targets referenced by Cargo.toml
├── poly_interp_demo/               # N-D interpolator over BN254 + sample tensors
├── build_python.sh                 # Helper to build+install bindings via maturin
├── requirements.txt                # Python deps for build scripts
├── interpolate_and_commit.py       # JSON-export + commitment verification pipeline
├── no_saving_time_version.py       # Streamed (no-intermediate) interpolation+commit
├── prover_verifier.py              # Two-party style prover/verifier demo
└── README.md                       # This file
```

### Prerequisites

- Rust toolchain (`rustup`), with `cargo` in your `$PATH`
- Python 3.10+ with `pip`
- `numpy` (installed automatically if you use `requirements.txt`)
- Optional: `maturin` (automatically installed when running `build_python.sh`)

#### Quick start with conda

```bash
conda create -n theseus python=3.10
conda activate theseus
cd theseustensorcommitment
pip install -r requirements.txt
```

The requirements file installs `numpy`, `maturin`, and `setuptools-rust`, which are
the only Python-side dependencies needed for the scripts and build helpers.

### Build the Tensor Commitment Library

```bash
cd theseustensorcommitment

# Option A: install the Python module into your active environment
./build_python.sh        # internally runs: maturin develop --features python

# Option B: just build the shared object locally
cargo build --features python
ln -sf "$(pwd)/target/debug/libtensorcommitments.so" "$(pwd)/target/debug/tensorcommitments.so"
```

All Python scripts below assume you run them from `theseustensorcommitment/`, and that
`PYTHONPATH=target/debug` (if you chose option B). When using `build_python.sh`
the module is installed into the environment and `PYTHONPATH` is not needed.

### Example inputs

`poly_interp_demo/` ships with multiple tensors:

- `tensor10by10by10by10by10by10.npy` – 6-D grid with 10 points per axis (used in examples)
- Other `.npy` files for larger stress tests

### Workflows

#### 1. Streamed interpolation + commitment proof (no intermediate JSON)

```bash
cd theseustensorcommitment
PYTHONPATH=target/debug \
python no_saving_time_version.py poly_interp_demo/tensorofshape10by10by10by10by10by10.npy
```

This command:
1. Runs the N-D interpolator (Rust) and captures coefficients via stdout.
2. Commits to the polynomial using `tensorcommitments`.
3. Evaluates/proves on a few grid points and verifies the proofs.

#### 2. File-based export + commit (`interpolate_and_commit.py`)

```bash
cd theseustensorcommitment

# Produce coefficients.json (can be reused)
cargo run --manifest-path poly_interp_demo/Cargo.toml --release -- \
  poly_interp_demo/tensor10by10by10by10by10by10.npy --coeffs coeffs.json

# Or let the script run cargo automatically:
PYTHONPATH=target/debug \
python interpolate_and_commit.py \
  poly_interp_demo/tensor10by10by10by10by10by10.npy \
  --coeffs-json coeffs.json
```

#### 3. Prover/Verifier protocol with metrics

```bash
cd theseustensorcommitment
PYTHONPATH=target/debug \
python prover_verifier.py \
  poly_interp_demo/tensor10by10by10by10by10by10.npy \
  --num-queries 5
```

The script simulates:
- Prover: interpolates, saves `coefficients.json`, commits, and answers queries.
- Verifier: loads the tensor, samples random points, and checks proofs.
It also reports file sizes (tensor, polynomial, proofs, commitment) and timing
breakdowns for each phase.

Use `--skip-interp` to reuse an existing JSON, or `--coeffs-json/--commitment-file`
to control where artifacts are stored.

### Usage in Python

```python
import tensorcommitments

# Create a wrapper for a 6-variable polynomial with degree bound 10
wrapper = tensorcommitments.TensorCommitmentWrapper(6, 10)

# Commit to polynomial coefficients (list of integers)
commitment = wrapper.commit(coefficients)

# Evaluate polynomial at a point
eval_value = wrapper.evaluate_polynomial(coefficients, [2, 3, 1, 0, 5, 4])

# Generate proof
proof = wrapper.prove(coefficients, [2, 3, 1, 0, 5, 4], eval_value)

# Verify proof
is_valid = wrapper.verify(commitment, [2, 3, 1, 0, 5, 4], eval_value, proof)
```

### Notes

- All commands assume you run them inside `theseustensorcommitment/`.
- Feel free to remove large intermediates (e.g. `coefficients.json`,
  `commitment.txt`) after use—they can always be regenerated from the `.npy`.
- The Rust crate is self-contained; updates here don't affect other projects.


# layerSelectionLib

Layer significance extraction, contiguous budgeted assignment, and benchmarking for verifier-aware model partitioning.

This is an optional module in `finalcode`, independent from the activation -> interpolation -> commitment pipeline.

This README consolidates the library documentation that was previously spread across integration, optimization, benchmark, and AlphaPruning helper docs.

`AlphaPruning/README.md` is intentionally kept as separate upstream project documentation.

## What This Library Does

1. Extracts per-layer significance from AlphaPruning ESD metrics.
2. Solves Multi-Verifier Budgeted Contiguous Selection (MV-BCS) optimally.
3. Benchmarks multiple objective strategies against AlphaPrun scoring.
4. Provides visualization and model-comparison tooling.

## Module Layout

```text
layerSelectionLib/
├── README.md
├── run_layer_selection.py
├── AlphaPruning/
├── layer_selection_integration/
│   ├── extract_layer_scores.py
│   └── visualize_scores.py
├── OptimizationModule/
│   ├── layer_selection_optimizer.py
│   ├── optimize_layers.py
│   └── scores_loader.py
├── benchmark/
│   ├── run_full_benchmark.py
│   ├── run_all_models.py
│   ├── run_benchmarks.py
│   ├── generate_budget_cases.py
│   ├── extract_weight_norms.py
│   └── plot_results.py
├── scores/
├── plots/
└── extract_values_from_LLMs.sh
```

## Installation

### Minimal dependencies (main workflows)

```bash
pip install numpy torch transformers matplotlib seaborn accelerate weightwatcher datasets sentencepiece
```

### AlphaPruning reference environment (from original install guide)

```bash
conda create -n prune_llm python=3.9
conda activate prune_llm
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.47.1 datasets==2.19.0 numpy==1.24.3 wandb sentencepiece
pip install accelerate==0.26.0
pip install weightwatcher
```

### Extra dependency for image classifiers

```bash
pip install timm==0.4.12
```

### Hugging Face access

For gated/private models and benchmark scripts:

```bash
export HF_TOKEN=your_token_here
```

## Primary Interface

Use the unified runner:

```bash
cd finalcode/layerSelectionLib
python run_layer_selection.py --help
```

### Extract scores

```bash
python run_layer_selection.py extract \
  --model facebook/opt-125m \
  --output scores/opt125m_scores.json \
  --pretty
```

### Visualize scores

```bash
python run_layer_selection.py visualize \
  --mode json_single \
  --json-file scores/opt125m_scores.json \
  --output plots/opt125m_plot.png
```

### Optimize contiguous assignments

```bash
python run_layer_selection.py optimize \
  --scores scores/opt125m_scores.json \
  --budgets 10000000 20000000 15000000 \
  --output scores/opt125m_optimized.json \
  --pretty
```

### Run full benchmark pipeline

```bash
python run_layer_selection.py benchmark \
  --model facebook/opt-125m \
  --scores-file scores/opt125m_scores.json \
  --num-cases 100 \
  --num-verifiers 3
```

## Layer Score Pipeline

### Conceptual flow

```text
weights -> ESD metrics (alpha_peak/...) -> pruning ratios -> significance scores
```

### Relationship used by this library

- Lower pruning ratio means more critical layer.
- Default significance mapping is:
  - `significance_score = 1.0 - pruning_ratio`

### Why values can look compressed or near 0-1

1. Visualization in `data_dir` mode normalizes to [0, 1].
2. Raw scores from JSON are `1 - ratio`, often naturally clustered.
3. Default `block_wise` mapping assigns same metric across layers in a block.
4. `epsilon` and global sparsity scaling constrain ratio spread.

To increase differentiation:
- use `--mapping_type layer_wise`
- increase `--epsilon`
- inspect raw ESD metrics directly (`--use_raw_metrics` in visualization)

## ESD `.npy` Metric Files

Precomputed files in `AlphaPruning/data/<model>/<metric>.npy` are 1D arrays of per-prunable-layer metrics. If metric files are missing, extraction scripts can compute them on the fly (can take minutes to an hour depending on model size).

## Block vs Layer Definitions

### OPT-125M

- 12 transformer blocks
- 6 prunable layers per block:
  - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.out_proj`, `fc1`, `fc2`
- total 72 prunable layers

### LLaMA-style (e.g., 7B)

- 32 transformer blocks
- 7 prunable layers per block:
  - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
- total 224 prunable layers for 7B

`*_scores.json` includes both per-layer entries and per-block aggregates.

## Score Extraction Details

Direct script (without wrapper):

```bash
cd layer_selection_integration
python extract_layer_scores.py \
  --model facebook/opt-125m \
  --ww_metric alpha_peak \
  --sparsity_ratio 0.7 \
  --epsilon 0.3 \
  --output scores.json \
  --pretty
```

### Key extraction arguments

| Argument | Meaning | Default |
|---|---|---|
| `--model` | HF model id or local path | required |
| `--ww_metric` | ESD metric | `alpha_peak` |
| `--cache_dir` | model cache | `llm_weights` |
| `--ww_metric_cache` | metric cache dir | auto |
| `--sparsity_ratio` | target sparsity | script default |
| `--epsilon` | ratio spread control | script default |
| `--mapping_type` | `block_wise` or `layer_wise` | `block_wise` |
| `--output` | JSON output path | required |

### Output schema (`*_scores.json`)

- `model_name`, `total_parameters`, `num_prunable_layers`, `num_blocks`
- `layers[]` with:
  - `index`
  - `name`
  - `num_parameters` (cost)
  - `significance_score` (benefit)
  - `pruning_ratio`
  - `weight_shape`
- `blocks[]` with per-block summary
- `metadata` with min/max/mean statistics

## Optimization Module (MV-BCS)

Problem:
- `L` layers with value `v_i >= 0` and cost `c_i > 0`
- `M` verifiers with budgets `B_k`
- choose one contiguous interval per verifier (possibly empty), pairwise disjoint, maximize total value under per-verifier budgets

Algorithm:
- Dynamic Programming with sliding-window deque optimization
- Time complexity: `O(M * L)`
- Deterministic and optimal for defined objective

### CLI usage

```bash
python -m OptimizationModule.optimize_layers \
  --scores scores/opt125m_scores.json \
  --budgets 10000000 20000000 15000000 \
  --output results.json \
  --pretty
```

### Python API usage

```python
from OptimizationModule import LayerSelectionOptimizer, load_scores_from_json

values, costs, metadata = load_scores_from_json("scores/opt125m_scores.json")
optimizer = LayerSelectionOptimizer(values, costs, [10000000, 20000000, 15000000])
optimal_benefit, assignments = optimizer.solve()
```

### Validation/testing

```bash
python OptimizationModule/test_optimizer.py
```

## Benchmark Suite

Benchmarks compare multiple objective choices while evaluating outcomes on AlphaPrun benefit.

### Full pipeline in one command

```bash
cd benchmark
python run_full_benchmark.py \
  --model facebook/opt-125m \
  --scores_file ../scores/opt125m_scores.json \
  --num_cases 100 \
  --num_verifiers 3
```

### Step-by-step

```bash
python generate_budget_cases.py --num_cases 100 --num_verifiers 3 --output_dir cases

python extract_weight_norms.py \
  --model facebook/opt-125m \
  --scores_file ../scores/opt125m_scores.json \
  --output weight_norms/opt125m_norms.json \
  --cache_dir ../layer_selection_integration/llm_weights

python run_benchmarks.py \
  --scores ../scores/opt125m_scores.json \
  --norms weight_norms/opt125m_norms.json \
  --cases_dir cases \
  --output_dir results \
  --model_name opt125m

python plot_results.py \
  --results results/opt125m_all_results.json \
  --output results/opt125m_comparison.png
```

### Run all configured models

```bash
python run_all_models.py
```

Useful flags:
- `--skip_norms`
- `--skip_cases`
- `--skip_individual`
- `--output`
- `--num_cases`
- `--cache_dir`

### Outputs

- `cases/case_XXX.json`, `cases/all_cases.json`
- `weight_norms/<model>_norms.json`
- `results/case_XXX_results.json`
- `results/<model>_all_results.json`
- `results/<model>_comparison.png`
- `results/combined_comparison.png`

## Troubleshooting

- `HF_TOKEN environment variable is not set`:
  - `export HF_TOKEN=...` in the same shell session.
- Missing norms file:
  - run `extract_weight_norms.py` first.
- Missing metric file:
  - extraction will compute metrics automatically (can be slow).
- Large model memory pressure:
  - run one model at a time, reduce concurrent workloads, use smaller test runs.
- Plot/benchmark sanity check:
  - run `python benchmark/test_benchmark.py`.

## Notes

- `run_layer_selection.py` is the maintained entrypoint.
- `extract_values_from_LLMs.sh` is legacy compatibility tooling.
- `OptimizationModule_copy/` is kept as legacy/reference; use `OptimizationModule/` for active workflows.


