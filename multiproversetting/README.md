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
- `sweep_multiprover_experiments.py`: runs prover-count sweep (e.g., 1..10), saves per-run summaries.
- `run_multiprover_sweep.sh`: convenience wrapper for sweep script.
- `plot_multiprover_sweep.py`: builds 3-panel seaborn figure(s) from one or more sweep results files.
- `run_multiprover_plot.sh`: convenience wrapper for plotting script.

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
  - includes `multi_prover_compute_summary` with per-agent compute times.
  - includes `embedding_exchange_summary` with total/average exchanged embedding payload sizes.

## Sweep experiments (1..10 provers)

```bash
cd /home/ob3942/repos/TileCommitments
bash multiproversetting/run_multiprover_sweep.sh \
  --model "Qwen/Qwen2-0.5B" \
  --prompt "Where is the capital of the world?" \
  --min-provers 1 \
  --max-provers 10 \
  --skip-interp-build
```

Sweep outputs:

- `.../sweep_results.json`: full per-run status + extracted metrics.
- `.../sweep_results.csv`: tabular version of the same results.

## Plotting sweep results

```bash
cd /home/ob3942/repos/TileCommitments
bash multiproversetting/run_multiprover_plot.sh \
  --sweep-json multiproversetting/output/sweeps/<sweep_run>/sweep_results.json \
  --outdir multiproversetting/output/sweeps/<sweep_run>/plots \
  --title-prefix "Qwen/Qwen2-0.5B"
```

Generated figures:

1. `multiprover_sweep_a_avg_agent_inference_time.png`: number of provers vs average inference time across agents.
2. `multiprover_sweep_b_avg_agent_commit_time.png`: number of provers vs average commitment time across agents.
3. `multiprover_sweep_c_avg_agent_prove_time.png`: number of provers vs average proving time across agents.
4. `multiprover_sweep_d_avg_agent_verify_time.png`: number of provers vs average verification time across agents.
5. `multiprover_sweep_e_avg_embedding_exchange_size.png`: number of provers vs average embedding payload exchanged between adjacent provers.

## Notes

- This implementation targets decoder-only models exposing `model.layers` and `model.rotary_emb`.
- Commit/prove/verify reuse existing scripts in `TensorCommitment/*` without modifying those files.
- For prover count `N`, this implementation requires at least `N` distinct visible GPUs. Sweep points with insufficient GPUs are recorded as `skipped`.
