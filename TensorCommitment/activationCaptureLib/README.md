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
