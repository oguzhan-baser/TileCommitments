# Modal Patch for SplitCompute Prover

This folder ports the **single-prover splitcompute flow** to Modal GPUs.

## What this patch does

- Keeps verifier protocol compatible with `splitcompute`:
  - `GET /health`
  - `GET /models`
  - `POST /prove`
- Moves prover execution to Modal:
  - GPU only for inference/capture stage.
  - CPU for conversion, hypercube reshape, interpolation, commitment, proving.
- Auto-selects GPU type based on model parameter count and estimated VRAM.
- Uses persistent Modal Volumes for:
  - Hugging Face model cache.
  - Prover artifacts.
- Supports model prefetch to avoid first-request download latency.

---

## Files

- `modal_splitcompute_service.py`
  - Modal app with API + GPU/CPU stage orchestration.
- `model_catalog.json`
  - Fixed model set and parameter counts (used for policy + allowlist).

---

## GPU selection policy

Policy is parameter-count based and estimates VRAM from:

- `estimated_vram_gb = params_b * dtype_bytes * runtime_overhead_factor`
- defaults:
  - `dtype_bytes = 2.0` (bf16/fp16 class weights)
  - `runtime_overhead_factor = 1.4`

Then picks the smallest GPU tier satisfying estimated VRAM:

`T4 -> L4 -> A10G -> L40S -> A100-40GB -> A100-80GB -> H100`

You can edit `GPU_MEMORY_GB`, `GPU_PRIORITY`, or `model_catalog.json` as needed.

---

## Why GPU is only used during inference

`/prove` runs in two stages:

1. **GPU stage** (`run_gpu_capture_*`)
   - Executes only `capture_activations.py`.
2. **CPU stage** (`run_cpu_commit_stage`)
   - Runs:
     - `convert_to_npy.py`
     - `reshape_to_hypercube.py`
     - `interpolate_hypercube.py`
     - `commit_prove_verify.py`

This isolates GPU billing to inference/capture.

---

## Setup-once strategy (no per-request setup)

Image build performs one-time setup:

- installs dependencies,
- installs Rust toolchain,
- builds and installs `tensorcommitments` wheel,
- builds interpolation Rust binary.

Runtime containers reuse the built image.

Model caching:

- Volume `tilecommitments-hf-cache` stores downloaded Hugging Face files.
- Use prefetch step to warm cache ahead of traffic.

---

## Deploy and run

From repo root:

```bash
cd /home/ob3942/repos/TileCommitments
modal deploy modalpatch/modal_splitcompute_service.py
```

Warm model cache (recommended):

```bash
modal run modalpatch/modal_splitcompute_service.py --models "Qwen/Qwen2-0.5B,Qwen/Qwen2-1.5B,Qwen/Qwen2-7B"
```

---

## Calling from verifier side

After deploy, Modal prints the web endpoint URL for `prover_api`.

Use existing verifier CLI from `splitcompute`:

```bash
cd /home/ob3942/repos/TileCommitments
bash splitcompute/run_verifier_client.sh \
  --prover-url "https://<your-modal-endpoint>" \
  --model-name Qwen/Qwen2-0.5B \
  --prompt "Where is the capital of the world?" \
  --num-queries 10 \
  --max-new-tokens 16
```

You can also use `splitcompute/verifier_web.py` by pointing `--prover-url` to the Modal endpoint.

---

## Known limitations / follow-up

1. Modal web endpoints may redirect while job runs; very long proofs can increase end-to-end HTTP latency.
2. Public endpoint security is not hardened in this patch (API auth/rate-limits not added).
3. Catalog is fixed allowlist by design; add models in `model_catalog.json` before serving them.
4. For very large models, consider parallel/progressive responses or async job status endpoints.

---

## Modal docs used

- Guide index: `https://modal.com/docs/guide`
- Web endpoints / ASGI patterns.
- GPU configuration (`modal.gpu` / `gpu=` strings).
- Volumes for persistent caching.
- Image build and local source inclusion (`Image.add_local_dir` / package setup).

