# Multi-Prover Sequential Inference Setting

This folder contains an isolated implementation of a multi-prover setting where:

1. The model layers are split into `N` contiguous ranges.
2. Each prover runs its range on a distinct GPU.
3. Prover `i` passes only its output embedding tensor to prover `i+1`.
4. Each prover commits independently to its own compute artifact.
5. Random opening proofs are generated and verified per prover.
6. Overhead is reported against a one-prover baseline.

## Main script

- `multiprover_pipeline.py`: full 5-stage pipeline.
- `run_multiprover_pipeline.sh`: convenience wrapper (activates `tilecommitments` if available).

## Example

```bash
cd /home/ob3942/repos/TileCommitments
bash multiproversetting/run_multiprover_pipeline.sh \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --prompt "Where is the capital of the world?" \
  --num-provers 3 \
  --device-ids 0,1,2 \
  --dtype bfloat16 \
  --num-queries 10 \
  --scale-factor 16 \
  --quantize 50 \
  --skip-interp-build
```

## Stage outputs

Each run writes to:

- `multiproversetting/output/run_YYYYmmdd_HHMMSS_utc/`

Key files:

- `prover_assignment.json`: layer-range/device assignment.
- `one_vs_multi_comparison.json`: one-prover vs multi-prover output match stats.
- `multiprover_compute_layout.json`: per-prover tensor shape metadata.
- `one_prover/`:
  - `activations.pt` and commitment pipeline artifacts (`int_activations`, `hypercube`, `polynomial`, `commitment`).
- `multi_prover/prover_XX/`:
  - prover-local `activations.pt` and commitment pipeline artifacts.
- `overhead_report.json`: compute/commit timing deltas and ratios (`one` vs `multi`).

## Notes

- This implementation targets decoder-only models exposing `model.layers` and `model.rotary_emb`.
- Commit/prove/verify reuse existing scripts in `TensorCommitment/*` without modifying those files.
