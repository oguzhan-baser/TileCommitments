## Theseus Tensor Commitment

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
python no_saving_time_version.py poly_interp_demo/rand10by10by10by10by10by10.npy
```

### Layout

```
theseustensorcommitment/
├── Cargo.toml / Cargo.lock         # Rust crate (theseus)
├── src/                            # Rust implementation (PST + Python bindings)
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
python no_saving_time_version.py poly_interp_demo/rand10by10by10by10by10by10.npy
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
