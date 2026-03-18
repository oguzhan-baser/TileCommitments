# SplitCompute Progress Log

Last updated: 2026-03-18 (America/Chicago)

## Scope and constraints followed

- All work was implemented **only** under `splitcompute/`.
- No code outside `splitcompute/` was modified for this feature set.
- Target setting is **single prover** (not split/multi prover).
- Existing TensorCommitment scripts were reused instead of re-implementing pipeline logic.

---

## High-level objective completed

Implemented a distributed-style prover/verifier workflow where:

1. Verifier sends `model_name + prompt` to prover over HTTP.
2. Prover runs inference + commitment/proof pipeline and returns:
   - inference output,
   - commitment metadata,
   - proof bundles.
3. Verifier verifies returned bundles cryptographically on verifier side.
4. Added both CLI and web-interface verifier flows.

---

## Files added / updated in `splitcompute/`

### Core protocol + pipeline reuse

- `splitcompute/protocol.py`
  - Shared helper module used by prover server, verifier client, and web UI.
  - Reuses existing scripts:
    - `TensorCommitment/activationCaptureLib/capture_activations.py`
    - `TensorCommitment/activationCaptureLib/convert_to_npy.py`
    - `TensorCommitment/activationCaptureLib/reshape_to_hypercube.py`
    - `TensorCommitment/interpolationLib/interpolate_hypercube.py`
    - `TensorCommitment/tensorCommitmentLib/commit_prove_verify.py`
  - Main functions:
    - `run_prover_pipeline(...)`
    - `verify_proof_bundles(...)`
    - `verify_prover_response(...)`
    - HTTP helpers: `post_json(...)`, `get_json(...)`
  - Creates normalized proof bundles per query point:
    - `num_variables`, `degree_bound`, `commitment_hex`
    - `index`, `value_int`, `proof_hex`

### Prover entity

- `splitcompute/prover_server.py`
  - HTTP server (`ThreadingHTTPServer`).
  - Endpoints:
    - `GET /health`
    - `GET /models`
    - `POST /prove`
  - `/prove` runs full prover pipeline and returns JSON payload with:
    - request metadata,
    - inference output (`generated_text`, token IDs),
    - commitment metadata,
    - proof bundles,
    - summary/timing metadata,
    - prover-side artifact paths.

### Verifier entity (CLI + offline)

- `splitcompute/verifier_client.py`
  - Verifier-side CLI that can:
    - request remote prover payload (`POST /prove`), or
    - verify local saved payload file.
  - Produces:
    - `prover_response_<tag>.json`
    - `verification_report_<tag>.json`
  - Fails with non-zero exit if any proof fails verification.

- `splitcompute/verify_payload_file.py`
  - Offline verification of previously saved prover payload.
  - Useful for “machine B verifies machine A output later”.

### Verifier web interface

- `splitcompute/verifier_web.py`
  - Browser-based verifier UI with:
    - fixed-model dropdown,
    - prompt textbox,
    - query controls,
    - local verification summary + table.
  - Theming updated to a Theseus-inspired dark neon style.
  - Loading animation added while waiting for prover response:
    - full-screen overlay + spinner,
    - submit button disabled during request.

### Runtime wrappers

- `splitcompute/run_prover_server.sh`
- `splitcompute/run_verifier_client.sh`
- `splitcompute/run_verifier_web.sh`
- `splitcompute/run_local_demo.sh`
  - One-command launcher for local demo:
    - starts prover server,
    - waits for health,
    - starts verifier UI,
    - waits for UI health,
    - prints URLs/log paths,
    - traps exit to kill both processes.

### Config + docs

- `splitcompute/fixed_models.json`
  - Current allowlist:
    - `Qwen/Qwen2-0.5B`
    - `Qwen/Qwen2-1.5B`
    - `Qwen/Qwen2-7B`

- `splitcompute/README.md`
  - Detailed runbook for:
    - machine A prover,
    - machine B verifier CLI,
    - machine B verifier web UI,
    - API request format,
    - one-command local demo.

---

## Prover API contract (implemented)

### `POST /prove` request

Fields currently supported:

- `model_name` (required)
- `prompt` (required)
- `max_new_tokens` (optional; default from server config)
- `num_queries`
- `seed`
- `scale_factor`
- `quantize`
- `min_dim`
- `max_dim`
- `skip_interp_build`

### `POST /prove` response

Top-level keys:

- `format_version`
- `generated_at_utc`
- `run_id`
- `request`
- `inference_output`
- `commitment`
- `proof_bundles`
- `commitment_results_summary`
- `artifact_paths_on_prover`

`proof_bundles` are verifier-ready and directly consumable by `tensorcommitments.verify`.

---

## Validation performed

## 1) Static/syntax checks

- `python -m py_compile` passed for:
  - `protocol.py`
  - `prover_server.py`
  - `verifier_client.py`
  - `verifier_web.py`
  - `verify_payload_file.py`
- `bash -n` passed for wrapper scripts.

## 2) Prover server smoke

- Prover server started on localhost test port.
- `GET /health` returned `{"status":"ok"}`.
- `GET /models` returned configured model list.

## 3) End-to-end remote-style verifier flow (CLI)

- Verifier client called prover with:
  - model: `Qwen/Qwen2-0.5B`
  - prompt: `"Where is the capital of the world?"`
  - reduced smoke settings (`num_queries=2`, `max_new_tokens=4`)
- Result:
  - proof count: `2`
  - `all_verified: true`
  - generated text returned and displayed.
- Artifacts created:
  - `splitcompute/output/prover_runs_smoke/.../prover_response.json`
  - `splitcompute/output/verifier_runs_smoke/prover_response_*.json`
  - `splitcompute/output/verifier_runs_smoke/verification_report_*.json`

## 4) Offline verification smoke

- `verify_payload_file.py` run on saved payload.
- Result: `all_verified: true`.

## 5) Web UI smoke

- Verifier web server served HTML correctly.
- Page includes:
  - themed UI styles,
  - loading overlay markup,
  - submit-time loading animation JS hooks.

---

## Theme + UX changes done

- Web UI visual redesign inspired by TheseusChain style:
  - dark gradient background,
  - neon accents,
  - glass-like cards,
  - styled forms/tables/details.
- Added waiting UX:
  - full-screen spinner overlay on submit,
  - rotating status text messages,
  - disabled submit button while request is in-flight.

---

## Known issues / limitations

1. **`Ctrl+C` cleanup in local demo can be inconsistent in some environments**
   - User reported demo sometimes does not fully stop.
   - Manual fallback commands were provided:
     - `pkill -f "splitcompute/prover_server.py"`
     - `pkill -f "splitcompute/verifier_web.py"`
     - or kill by port with `lsof/fuser`.
   - Local timeout-based smoke showed cleanup working in tested run, but not guaranteed everywhere.

2. **Blocking request model**
   - `/prove` is synchronous; long-running requests hold HTTP request open.
   - No background job queue / async status endpoint yet.

3. **No auth/security hardening yet**
   - No API key, no TLS termination, no rate limiting, no request signing.

4. **No streamed progress from prover**
   - Verifier UI currently shows spinner animation only (client-side).
   - No stage-by-stage progress from prover backend.

5. **Potential large payloads**
   - Returning all proof bundles inline may become large for high `num_queries`.

6. **Model allowlist is static JSON**
   - Dynamic policy management not implemented.

---

## Explicit future work (postponed)

1. **Fix demo shutdown robustness**
   - Move child process management to process groups (`setsid` / `kill -- -pgid`) or a Python supervisor to ensure all descendants terminate on `Ctrl+C`.

2. **Async prove jobs + polling**
   - `POST /prove` returns `job_id`.
   - Add `GET /jobs/<id>` for progress states:
     - capture,
     - quantize,
     - reshape,
     - interpolate,
     - commit/prove/verify.

3. **UI progress bar via polling**
   - Replace purely cosmetic spinner with real backend progress.

4. **Security hardening**
   - API auth, TLS, input limits, logging/auditing, model-level ACL.

5. **Artifact packaging**
   - Optional compressed transfer format for payload + proofs.

6. **Caching**
   - Cache identical `(model, prompt, params)` prover responses to avoid repeated heavy recomputation.

---

## Re-entry checklist for next AI

1. Read `splitcompute/protocol.py` first (central logic).
2. Confirm `tensorcommitments` binding is available in `tilecommitments` env.
3. Start prover server and verify `/health` + `/models`.
4. Run `verifier_client.py` with small smoke settings before any larger tests.
5. If debugging UI, inspect `verifier_web.py` render function and submit JS.
6. If process cleanup issues persist, prioritize process-group supervisor fix.

