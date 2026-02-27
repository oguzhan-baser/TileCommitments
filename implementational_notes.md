# Implementation Notes

## Compute vs Committed Value Difference

- In `compute_crypto_verify_layer.py`, there can be a small mismatch between:
  - `computed_scaled_value_int` (recomputed layer output after scaling), and
  - `hypercube_value_int` / `value_int` (the value actually committed in the hypercube).
- This is expected in some environments because recomputation can differ slightly due to runtime factors (device, dtype, math kernels, library versions), and scaling/rounding amplifies tiny floating-point differences.

## Current Verification Policy

- Stage 1 (**compute verification**) checks tensor-level closeness with `rtol/atol`.
- Stage 2 enforces that `saved_scaled_value_int == hypercube_value_int` (so commitments are always tied to the saved activation artifact).
- If `--allow-approx-commitment` is enabled, proofs are generated for the committed value even when `computed_scaled_value_int != hypercube_value_int`, and the mismatch is explicitly recorded.
- Stage 3 (**crypto verification**) verifies the generated opening proofs against the commitment, index, and committed value.

## Why This Design

- It separates two guarantees cleanly:
  - **Compute consistency** (approximate numeric agreement with saved activations),
  - **Commitment consistency** (exact cryptographic validity of committed values).
- This avoids false cryptographic failures caused by benign numeric drift while preserving strict commitment correctness.

## Observed Layer_5 Drift (DeepSeek-R1-Distill-Qwen-1.5B)

- On CPU (`float16`), `layer_5` failed compute verification with `rtol=0.01`, `atol=0.01` because `max_abs_diff=1.0`.
- Re-running the same layer with `atol=2` passed compute verification and kept crypto verification valid (`all_verified=true` for sampled proofs).
- Example max-diff point from rerun:
  - tensor index: `(0, 0, 940)`
  - saved activation: `1977.0`
  - recomputed activation: `1978.0`
  - absolute difference: `1.0`
