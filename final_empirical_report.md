# Final Empirical Report

- Run directory: `/home/ob3942/repos/TileCommitments/TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227`
- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Generated at (UTC): `2026-03-20T15:08:49.919758+00:00`

## Data Transfer Sizes

### Verifier (non-model payload)
- Raw: **2312453 B** (2.21 MB)
- Compressed: **2083579 B** (1.99 MB)

### Prover (non-model request payload)
- Raw: **265 B** (265.00 B)
- Compressed: **437 B** (437.00 B)

### Verifier minimal model-parameter subset (compute path)
- Raw: **2620719464 B** (2.44 GB)
- Compressed: **2077652002 B** (1.93 GB)

### Prover full model parameters (inference path)
- Raw: **3554214621 B** (3.31 GB)
- Compressed: **2812964819 B** (2.62 GB)

## Verifier Compute Overhead

### Crypto verification
- Node: `0.007140 s`
- Tile total: `0.245146 s`
- Tile avg/layer: `0.008755 s`

### Compute verification (CPU)
- Node: `0.025037 s`, abs diff `0.000690721`
- Tile total: `0.454025 s`, avg/layer `0.016215 s`, max abs diff `0.00349593`

### Compute verification (GPU)
- Node: `0.335683 s`, abs diff `0.00012207`
- Tile total: `0.172855 s`, avg/layer `0.006173 s`, max abs diff `0.00195312`

## Prover Compute Overhead

- Node proof generation: `3.599282 s`
- Tile proof generation total: `94.611687 s` (avg/layer `3.378989 s`)
- Commit stage time: `2.284 s`
- Commit script total prove time: `53.32 s`
- Commit script total verify time: `0.065 s`

## Source Reports

- `/home/ob3942/repos/TileCommitments/TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227/empirical_measurements/stage5_stage6_size_report.json`
- `/home/ob3942/repos/TileCommitments/TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227/empirical_measurements/stage7_stage8_compute_report.json`
- `/home/ob3942/repos/TileCommitments/TensorCommitment/activationCaptureLib/output/deepseek_r1d_qwen15b_run_20260227/empirical_measurements/model_param_transfer_sizes.json`
