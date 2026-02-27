# TileCommitments
Light version of the TensorCommitments

## Full-Coverage Verification Command

Run one proof + crypto check + compute check per layer (random entry per layer):

```bash
source /home/ob3942/miniconda3/etc/profile.d/conda.sh
conda activate tilecommitments
cd /home/ob3942/repos/TileCommitments
RUN_DIR=TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227
python full_coverage_verify.py \
  --activations-pt "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_activations.pt" \
  --hypercube-dir "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube" \
  --poly-dir "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_polynomial" \
  --commitment-file "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_commitment/commitment.txt" \
  --seed 42 \
  --dtype float16 \
  --rtol 1e-3 \
  --atol 5e-2 \
  --output-dir "$RUN_DIR/full_coverage_verification"
```

Outputs:
- selected entries: `$RUN_DIR/full_coverage_verification/selected_entries.json`
- proof bundles: `$RUN_DIR/full_coverage_verification/proof_bundles/`
- summary: `$RUN_DIR/full_coverage_verification/full_coverage_summary.json`

## Layer Drift Example

- Rerun for `layer_5` (DeepSeek-R1-Distill-Qwen-1.5B, CPU `float16`) showed:
  - `max_abs_diff = 1.0`
  - example index `(0, 0, 940)`: saved activation `1977.0`, recomputed activation `1978.0`
- With `rtol=0.01`, `atol=0.01`, compute verification fails for this layer; with `atol=2`, compute verification passes and sampled crypto verification remains valid.

## Optional Future Work

- Add a strict mode for `compute_crypto_verify_layer.py` that fails the run whenever `computed_scaled_value_int != committed_hypercube_value_int` (even when tensor-level tolerance checks pass).
- Add a per-layer tolerance profile (or auto-calibrated tolerance) and automatic logging of top activation drift indices/values for reproducible compute-verification diagnostics.
