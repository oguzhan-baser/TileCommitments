# tensorCommitmentLib

Cryptographic tensor commitment for multivariate polynomials over the BN254 curve.

Takes the polynomial coefficients produced by `interpolationLib` and runs the full **PST (Papamanthou-Shi-Tamassia) commitment** pipeline: setup, commit, prove, and verify. Optionally cross-checks evaluations against the original hypercube ground truth.

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

## How the PST scheme works

The PST (Polynomial Structure Tensor) commitment uses:

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
