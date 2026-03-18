# SplitCompute (Single-Prover, Remote Verifier)

This folder realizes a **single-prover** setting where prover and verifier run on different machines.

- Prover machine receives `{model_name, prompt}` from verifier.
- Prover computes inference + commitment pipeline and returns:
  - inference output,
  - commitment,
  - proof bundles.
- Verifier machine checks all returned proof bundles cryptographically.

No files outside `splitcompute/` are modified for this setup.

---

## 1) What is implemented

### Prover-side service
- `prover_server.py`
  - HTTP API (default `:8081`)
  - Endpoints:
    - `GET /health`
    - `GET /models`
    - `POST /prove`
  - Runs existing TensorCommitment scripts internally:
    - `activationCaptureLib/capture_activations.py`
    - `activationCaptureLib/convert_to_npy.py`
    - `activationCaptureLib/reshape_to_hypercube.py`
    - `interpolationLib/interpolate_hypercube.py`
    - `tensorCommitmentLib/commit_prove_verify.py`

### Verifier-side CLI
- `verifier_client.py`
  - Requests a remote prover job (`POST /prove`) and verifies all returned proofs locally.
- `verify_payload_file.py`
  - Verifies a previously saved prover response JSON (offline verification mode).

### Verifier-side web interface
- `verifier_web.py`
  - Serves a simple web UI (default `:8091`) with:
    - fixed-model dropdown,
    - prompt text box,
    - query button.
  - On submit:
    1. queries prover API,
    2. verifies proofs locally,
    3. displays per-proof results and summary.

### Shared helpers
- `protocol.py`
  - Pipeline orchestration, proof-bundle formatting, HTTP helpers, and verification helpers.
- `fixed_models.json`
  - Fixed model allowlist used by prover and UI fallback.

---

## 2) Prerequisites (both machines)

1. Clone repo and enter root:
   ```bash
   cd /home/ob3942/repos/TileCommitments
   ```
2. Activate environment:
   ```bash
   conda activate tilecommitments
   ```
3. Ensure `tensorcommitments` Python binding is built:
   ```bash
   cd TensorCommitment/pst_commitment_lib
   maturin develop --features python --release
   cd /home/ob3942/repos/TileCommitments
   ```

---

## 3) Machine A (Prover entity)

### Start prover server
```bash
cd /home/ob3942/repos/TileCommitments
bash splitcompute/run_prover_server.sh \
  --host 0.0.0.0 \
  --port 8081 \
  --models-file splitcompute/fixed_models.json \
  --output-root splitcompute/output/prover_runs \
  --default-max-new-tokens 16 \
  --default-num-queries 10
```

### Quick health/model checks
```bash
curl http://<PROVER_IP>:8081/health
curl http://<PROVER_IP>:8081/models
```

---

## 4) Machine B (Verifier entity) — CLI flow

### Query prover and verify immediately
```bash
cd /home/ob3942/repos/TileCommitments
bash splitcompute/run_verifier_client.sh \
  --prover-url http://<PROVER_IP>:8081 \
  --model-name Qwen/Qwen2-0.5B \
  --prompt "Where is the capital of the world?" \
  --num-queries 10 \
  --max-new-tokens 16 \
  --output-dir splitcompute/output/verifier_runs
```

This saves:
- `prover_response_*.json` (payload from prover)
- `verification_report_*.json` (verifier results)

### Verify a previously saved payload (offline)
```bash
cd /home/ob3942/repos/TileCommitments
python splitcompute/verify_payload_file.py \
  --payload splitcompute/output/verifier_runs/prover_response_<tag>.json
```

---

## 5) Machine B (Verifier entity) — Web UI flow

### Start web UI
```bash
cd /home/ob3942/repos/TileCommitments
bash splitcompute/run_verifier_web.sh \
  --host 0.0.0.0 \
  --port 8091 \
  --prover-url http://<PROVER_IP>:8081 \
  --models-file splitcompute/fixed_models.json
```

Open in browser:
```
http://<VERIFIER_IP>:8091
```

Use dropdown + prompt, then click **Query Prover and Verify**.

UI behavior:
- Uses a Theseus-inspired dark neon visual theme.
- Shows a full-screen loading animation while verifier waits for prover response.

---

## 6) API contract (prover)

### Request: `POST /prove`
```json
{
  "model_name": "Qwen/Qwen2-0.5B",
  "prompt": "Where is the capital of the world?",
  "max_new_tokens": 16,
  "num_queries": 10,
  "seed": 42,
  "scale_factor": 16,
  "quantize": 50.0,
  "min_dim": 4,
  "max_dim": 10,
  "skip_interp_build": true
}
```

### Response (high-level)
- `inference_output` (includes generated text + token IDs)
- `commitment` (`commitment_hex`, `num_variables`, `degree_bound`)
- `proof_bundles[]` each containing:
  - `index`
  - `value_int`
  - `proof_hex`
  - metadata needed for `verify()`

---

## 7) Notes

- This setup is intentionally **single-prover** (not split-prover/multi-prover).
- Fixed-model policy is enforced on prover via `fixed_models.json`.
- If prover has limited GPU resources, use a smaller allowed model first (`Qwen/Qwen2-0.5B`) for quick smoke tests.

---

## 8) One-command local demo (starts prover + verifier UI)

If you want both services started together on one machine:

```bash
cd /home/ob3942/repos/TileCommitments
bash splitcompute/run_local_demo.sh
```

Then open:
```
http://127.0.0.1:8091
```

The script:
- starts prover server (`:8081`),
- waits for health,
- starts verifier web UI (`:8091`),
- prints log paths,
- stops both services on `Ctrl+C`.

Example custom ports:

```bash
bash splitcompute/run_local_demo.sh \
  --prover-port 18081 \
  --verifier-port 18091
```
