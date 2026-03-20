# TileCommitments Full Pipeline (Multi-GPU)

Run everything from repo root: `/home/ob3942/repos/TileCommitments`.

## Fast path (recommended): one command

`run_full_pipeline.sh` runs all stages end-to-end, auto-selects GPUs by free-memory %, auto-derives memory budgets, and writes `run_metrics.json` at the end.

```bash
cd /home/ob3942/repos/TileCommitments
./run_full_pipeline.sh \
  --model "Qwen/Qwen2.5-72B-Instruct-AWQ" \
  --prompt "Where is the capital of the world?" \
  --dtype float16 \
  --device-map balanced \
  --min-free-gpu-pct 75 \
  --gpu-memory-spread-pct 85 \
  --scale-factor 16 \
  --quantize 50 \
  --num-queries 10 \
  --layer 0 \
  --num-proofs 8 \
  --rtol 1e-3 \
  --atol 5e-2
```

Most useful CLI knobs in `run_full_pipeline.sh`:
- `--dtype`: `float16|float32|bfloat16`
- `--device-map`: `none|auto|balanced|balanced_low_0|sequential`
- `--mxfp4-mode`: `auto|native|dequantize` (`auto` switches GPT-OSS to dequantized bf16 on pre-sm89 GPUs such as A100)
- `--min-free-gpu-pct`: GPU inclusion threshold
- `--gpu-memory-spread-pct`: tighter per-GPU cap to force wider sharding
- `--build-interp`: rebuild interpolation binary instead of `--skip-build`

For GPT-OSS on A100, `run_full_pipeline.sh` also applies extra per-GPU runtime headroom, switches `device_map` from `balanced` to `auto`, and switches `dtype` from `float16` to `bfloat16` when dequantized mode is active.

## 0) Environment (one time per machine)

```bash
cd /home/ob3942/repos/TileCommitments
bash setup_a100_tilecommitments.sh
```

## 1) Session setup

```bash
source /home/ob3942/miniconda3/etc/profile.d/conda.sh
conda activate tilecommitments
cd /home/ob3942/repos/TileCommitments

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # change as needed
PROMPT="Explain verifiable inference in one short paragraph."
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="TensorCommitment/activationCaptureLib/output/${RUN_TAG}"
SAFE_MODEL="${MODEL//\//_}"

ACT_PT="$RUN_DIR/${SAFE_MODEL}_activations.pt"
INT_DIR="$RUN_DIR/${SAFE_MODEL}_int_activations"
HC_DIR="${INT_DIR}_hypercube"
POLY_DIR="${HC_DIR}_polynomial"
COMMIT_DIR="${HC_DIR}_commitment"
```

## 2) Capture activations (all visible GPUs)

```bash
python TensorCommitment/activationCaptureLib/capture_activations.py \
  --models "$MODEL" \
  --prompt "$PROMPT" \
  --max-new-tokens 16 \
  --output-dir "$RUN_DIR" \
  --device cuda \
  --dtype float16 \
  --device-map balanced \
  --mxfp4-mode auto \
  --max-memory-per-gpu 20GiB \
  --max-memory-cpu 128GiB \
  --seed 42
```

## 3) Float activations -> integer `.npy`

```bash
python TensorCommitment/activationCaptureLib/convert_to_npy.py \
  --input "$ACT_PT" \
  --scale-factor 16 \
  --quantize 50 \
  --output-dir "$INT_DIR"
```

## 4) Build hypercube

```bash
python TensorCommitment/activationCaptureLib/reshape_to_hypercube.py \
  --input-dir "$INT_DIR" \
  --output-dir "$HC_DIR"
```

## 5) Interpolate polynomial

```bash
python TensorCommitment/interpolationLib/interpolate_hypercube.py \
  --input-dir "$HC_DIR" \
  --output-dir "$POLY_DIR" \
  --skip-build
```

## 6) Commit + sample prove/verify

```bash
python TensorCommitment/tensorCommitmentLib/commit_prove_verify.py \
  --poly-dir "$POLY_DIR" \
  --hypercube-dir "$HC_DIR" \
  --output-dir "$COMMIT_DIR" \
  --num-queries 10 \
  --seed 42
```

## 7) Layer compute+crypto verification

```bash
python compute_crypto_verify_layer.py \
  --activations-pt "$ACT_PT" \
  --hypercube-dir "$HC_DIR" \
  --poly-dir "$POLY_DIR" \
  --commitment-file "$COMMIT_DIR/commitment.txt" \
  --layer 0 \
  --proof-mode sample \
  --num-proofs 8 \
  --device cuda \
  --dtype float16 \
  --device-map balanced \
  --max-memory-per-gpu 20GiB \
  --max-memory-cpu 128GiB \
  --rtol 1e-3 \
  --atol 5e-2 \
  --output-dir "$RUN_DIR/layer0_compute_crypto"
```

## 8) Full-coverage verification (one random entry per layer)

```bash
python full_coverage_verify.py \
  --activations-pt "$ACT_PT" \
  --hypercube-dir "$HC_DIR" \
  --poly-dir "$POLY_DIR" \
  --commitment-file "$COMMIT_DIR/commitment.txt" \
  --seed 42 \
  --device cuda \
  --dtype float16 \
  --device-map balanced \
  --max-memory-per-gpu 20GiB \
  --max-memory-cpu 128GiB \
  --rtol 1e-3 \
  --atol 5e-2 \
  --output-dir "$RUN_DIR/full_coverage_verification"
```

## 9) Per-run metrics (sizes + timings)

```bash
python summarize_run_metrics.py \
  --run-dir "$RUN_DIR" \
  --layer-summary "$RUN_DIR/layer0_compute_crypto/compute_crypto_verification_summary.json" \
  --output "$RUN_DIR/run_metrics.json"
```

## 10) Optional: single-index proof bundle + re-verify

```bash
python prove_verify_index.py \
  --poly-dir "$POLY_DIR" \
  --commitment-file "$COMMIT_DIR/commitment.txt" \
  --flat-index 12345 \
  --hypercube-dir "$HC_DIR" \
  --output "$RUN_DIR/proof_bundle_flat_12345.json"

python verify_saved_proof.py \
  --bundle "$RUN_DIR/proof_bundle_flat_12345.json"
```

## 11- Empirical measurements

Use this after a run is complete to get empirical (file-size + runtime) measurements for verifier/prover overhead:

```bash
source /home/ob3942/miniconda3/etc/profile.d/conda.sh
conda activate tilecommitments
cd /home/ob3942/repos/TileCommitments

python measure_empirical_pipeline.py \
  --run-dir "$RUN_DIR" \
  --model-id "$MODEL"
```

Outputs are written under `$RUN_DIR/empirical_measurements/`:
- `stage5_stage6_size_report.json`
- `stage7_stage8_compute_report.json`
- `model_param_transfer_sizes.json`
- `final_empirical_report.md`

Example (already measured on this repo):

```bash
source /home/ob3942/miniconda3/etc/profile.d/conda.sh
conda activate tilecommitments
cd /home/ob3942/repos/TileCommitments
python measure_empirical_pipeline.py \
  --run-dir TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227 \
  --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Example outputs from that run (`.../deepseek_r1d_qwen15b_run_20260227/empirical_measurements/final_empirical_report.md`):
- Verifier non-model payload: `2,312,453 B` raw (`2.21 MB`), `2,083,579 B` compressed (`1.99 MB`)
- Prover non-model payload: `265 B` raw, `437 B` compressed
- Verifier crypto verification: node `0.007140 s`, full-tile (`28` layers) `0.245146 s`
- Prover proof generation: node `3.599282 s`, full-tile (`28` layers) `94.611687 s`

## Key artifacts

- Activations: `$ACT_PT`
- Hypercube + metadata: `$HC_DIR/hypercube.npy`, `$HC_DIR/hypercube_metadata.json`
- Polynomial: `$POLY_DIR/coefficients.json`
- Commitment: `$COMMIT_DIR/commitment.txt`
- Commitment benchmark/proofs: `$COMMIT_DIR/commitment_results.json`, `$COMMIT_DIR/proofs.json`
- Layer verification summary: `$RUN_DIR/layer0_compute_crypto/compute_crypto_verification_summary.json`
- Full coverage summary: `$RUN_DIR/full_coverage_verification/full_coverage_summary.json`
- Metrics summary: `$RUN_DIR/run_metrics.json`
