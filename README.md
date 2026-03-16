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

## Multi-GPU (Large Models)

For models that do not fit on one GPU, enable Hugging Face sharding:

```bash
source /home/ob3942/miniconda3/etc/profile.d/conda.sh
conda activate tilecommitments
cd /home/ob3942/repos/TileCommitments
RUN_DIR=TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227
python compute_crypto_verify_layer.py \
  --activations-pt "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_activations.pt" \
  --hypercube-dir "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube" \
  --poly-dir "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_polynomial" \
  --commitment-file "$RUN_DIR/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_int_activations_hypercube_commitment/commitment.txt" \
  --layer 0 \
  --proof-mode sample \
  --num-proofs 1 \
  --device cuda \
  --dtype float16 \
  --device-map auto \
  --max-memory-per-gpu 700MiB \
  --max-memory-cpu 64GiB \
  --output-dir "$RUN_DIR/multigpu_probe_layer0_700mib"
```

`full_coverage_verify.py` and `capture_activations.py` support the same flags:
- `--device-map` (`none|auto|balanced|balanced_low_0|sequential`)
- `--mxfp4-mode` (`auto|native|dequantize`, where `auto` dequantizes GPT-OSS on pre-sm89 GPUs like A100)
- `--max-memory-per-gpu` (example: `20GiB`)
- `--max-memory-cpu` (example: `128GiB`)

For GPT-OSS on A100, the pipeline auto-adds GPU runtime headroom, prefers `device_map=auto`, and uses `bfloat16` in dequantized mode to reduce OOM and dtype-mismatch risk.

## Layer Drift Example

- Rerun for `layer_5` (DeepSeek-R1-Distill-Qwen-1.5B, CPU `float16`) showed:
  - `max_abs_diff = 1.0`
  - example index `(0, 0, 940)`: saved activation `1977.0`, recomputed activation `1978.0`
- With `rtol=0.01`, `atol=0.01`, compute verification fails for this layer; with `atol=2`, compute verification passes and sampled crypto verification remains valid.

## `run_metrics.json` Key Reference

`run_metrics.json` is produced at the end of `run_full_pipeline.sh` and summarizes key size/timing numbers for that run.

Top-level keys:
- `generated_at_utc`: UTC timestamp for when metrics were written.
- `run_dir`: Absolute path of the run directory these metrics describe.
- `sizes`: Byte-size related metrics for commitment/proofs/activations.
- `timings_seconds`: Time metrics (in seconds) from commitment/prove/verify benchmarking.
- `source_files`: Exact artifact files used to build the metrics.

`sizes.commitment`:
- `crypto_bytes`: Cryptographic commitment payload size in bytes (group element bytes).
- `file_bytes`: On-disk size of `commitment.txt` in bytes (includes encoding/newline overhead).

`sizes.opening_proof`:
- `avg_crypto_bytes`: Average proof payload size in bytes over sampled commitment queries.
- `proof_elements_per_query`: Average number of field/group elements per opening proof query.
- `sample_layer_bundle_crypto_bytes`: Raw crypto size (bytes) for one saved proof bundle from layer verification.
- `sample_layer_bundle_file_bytes`: Full JSON file size (bytes) of that sample proof bundle.
- `sample_layer_bundle_proof_elements`: Number of proof elements inside the sample saved bundle.
- `sample_layer_bundle_count`: Number of saved proof bundles in the sampled layer verification output folder.

`sizes.intermediate_activations_for_verification`:
- `layer_shape`: Shape of the verified layer tensor used as the size baseline.
- `dtype`: Tensor dtype used for this size estimate.
- `bytes_per_element`: Bytes per tensor element inferred from dtype (`float16/bfloat16=2`, `float32=4`, `float64=8`).
- `one_tensor_bytes`: Estimated bytes for one layer tensor (`product(layer_shape) * bytes_per_element`).
- `input_plus_output_bytes`: Estimated bytes needed when verifier uses both layer input and layer output tensors (`2 * one_tensor_bytes`).

`sizes.activations_pt_file_bytes`:
- On-disk size in bytes of the captured `<model>_activations.pt` artifact for the run.

`timings_seconds`:
- `commit`: Total time to create the commitment.
- `prove_tile_avg`: Average time to generate one opening proof (tile/query).
- `verify_tile_avg`: Average time to verify one opening proof (tile/query).
- `prove_total`: Total proof-generation time across all sampled queries.
- `verify_total`: Total verification time across all sampled queries.

`source_files`:
- `commitment_results_json`: Benchmark source for commitment/proof/verify timing and proof stats.
- `commitment_txt`: Source commitment file used for commitment size metrics.
- `layer_summary_json`: Layer compute+crypto summary used for activation-size context.
- `sample_layer_proof_bundle_json`: One proof-bundle JSON used for sample proof file-size fields (can be `null` if no bundles were saved).

## Optional Future Work

- Add a strict mode for `compute_crypto_verify_layer.py` that fails the run whenever `computed_scaled_value_int != committed_hypercube_value_int` (even when tensor-level tolerance checks pass).
- Add a per-layer tolerance profile (or auto-calibrated tolerance) and automatic logging of top activation drift indices/values for reproducible compute-verification diagnostics.
- Add an explicit CLI quantization mode (for example `int8` via bitsandbytes for non-AWQ models) in `run_full_pipeline.sh`, and document how inference precision/quantization settings map to commitment scaling (`--scale-factor`, `--quantize`).
