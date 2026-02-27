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
