# TileCommitments
Light version of the TensorCommitments

## Layer Drift Example

- Rerun for `layer_5` (DeepSeek-R1-Distill-Qwen-1.5B, CPU `float16`) showed:
  - `max_abs_diff = 1.0`
  - example index `(0, 0, 940)`: saved activation `1977.0`, recomputed activation `1978.0`
- With `rtol=0.01`, `atol=0.01`, compute verification fails for this layer; with `atol=2`, compute verification passes and sampled crypto verification remains valid.

## Optional Future Work

- Add a strict mode for `compute_crypto_verify_layer.py` that fails the run whenever `computed_scaled_value_int != committed_hypercube_value_int` (even when tensor-level tolerance checks pass).
- Add a per-layer tolerance profile (or auto-calibrated tolerance) and automatic logging of top activation drift indices/values for reproducible compute-verification diagnostics.
