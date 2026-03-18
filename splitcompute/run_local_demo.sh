#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROVER_HOST="127.0.0.1"
PROVER_PORT="8081"
VERIFIER_HOST="127.0.0.1"
VERIFIER_PORT="8091"
MODELS_FILE="${SCRIPT_DIR}/fixed_models.json"
PROVER_OUTPUT_ROOT="${SCRIPT_DIR}/output/prover_runs_demo"
DEFAULT_PROMPT="Where is the capital of the world?"
DEFAULT_MAX_NEW_TOKENS="16"
DEFAULT_NUM_QUERIES="10"

usage() {
  cat <<'USAGE'
Usage: bash splitcompute/run_local_demo.sh [options]

Starts both:
  1) Prover server
  2) Verifier web UI

Options:
  --prover-host <host>            Default: 127.0.0.1
  --prover-port <port>            Default: 8081
  --verifier-host <host>          Default: 127.0.0.1
  --verifier-port <port>          Default: 8091
  --models-file <path>            Default: splitcompute/fixed_models.json
  --prover-output-root <path>     Default: splitcompute/output/prover_runs_demo
  --default-prompt <text>         Default: "Where is the capital of the world?"
  --default-max-new-tokens <int>  Default: 16
  --default-num-queries <int>     Default: 10
  -h, --help                      Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prover-host) PROVER_HOST="$2"; shift 2 ;;
    --prover-port) PROVER_PORT="$2"; shift 2 ;;
    --verifier-host) VERIFIER_HOST="$2"; shift 2 ;;
    --verifier-port) VERIFIER_PORT="$2"; shift 2 ;;
    --models-file) MODELS_FILE="$2"; shift 2 ;;
    --prover-output-root) PROVER_OUTPUT_ROOT="$2"; shift 2 ;;
    --default-prompt) DEFAULT_PROMPT="$2"; shift 2 ;;
    --default-max-new-tokens) DEFAULT_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --default-num-queries) DEFAULT_NUM_QUERIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

LOG_DIR="${SCRIPT_DIR}/output/local_demo_logs"
mkdir -p "${LOG_DIR}"
RUN_TAG="$(date -u +%Y%m%d_%H%M%S_utc)"
PROVER_LOG="${LOG_DIR}/prover_${RUN_TAG}.log"
VERIFIER_LOG="${LOG_DIR}/verifier_${RUN_TAG}.log"

PROVER_PID=""
VERIFIER_PID=""

cleanup() {
  if [[ -n "${VERIFIER_PID}" ]] && kill -0 "${VERIFIER_PID}" 2>/dev/null; then
    kill "${VERIFIER_PID}" 2>/dev/null || true
    wait "${VERIFIER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${PROVER_PID}" ]] && kill -0 "${PROVER_PID}" 2>/dev/null; then
    kill "${PROVER_PID}" 2>/dev/null || true
    wait "${PROVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[demo] Starting prover server..."
bash "${SCRIPT_DIR}/run_prover_server.sh" \
  --host "${PROVER_HOST}" \
  --port "${PROVER_PORT}" \
  --models-file "${MODELS_FILE}" \
  --output-root "${PROVER_OUTPUT_ROOT}" \
  --default-max-new-tokens "${DEFAULT_MAX_NEW_TOKENS}" \
  --default-num-queries "${DEFAULT_NUM_QUERIES}" \
  >"${PROVER_LOG}" 2>&1 &
PROVER_PID=$!

echo "[demo] Waiting for prover health endpoint..."
for _ in $(seq 1 90); do
  if curl -fsS "http://${PROVER_HOST}:${PROVER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://${PROVER_HOST}:${PROVER_PORT}/health" >/dev/null 2>&1; then
  echo "[demo][error] Prover did not become healthy. Check: ${PROVER_LOG}" >&2
  exit 1
fi

echo "[demo] Starting verifier web UI..."
bash "${SCRIPT_DIR}/run_verifier_web.sh" \
  --host "${VERIFIER_HOST}" \
  --port "${VERIFIER_PORT}" \
  --prover-url "http://${PROVER_HOST}:${PROVER_PORT}" \
  --models-file "${MODELS_FILE}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --default-max-new-tokens "${DEFAULT_MAX_NEW_TOKENS}" \
  --default-num-queries "${DEFAULT_NUM_QUERIES}" \
  >"${VERIFIER_LOG}" 2>&1 &
VERIFIER_PID=$!

echo "[demo] Waiting for verifier web UI..."
for _ in $(seq 1 90); do
  if curl -fsS "http://${VERIFIER_HOST}:${VERIFIER_PORT}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://${VERIFIER_HOST}:${VERIFIER_PORT}" >/dev/null 2>&1; then
  echo "[demo][error] Verifier UI did not come up. Check: ${VERIFIER_LOG}" >&2
  exit 1
fi

echo
echo "[demo] ✅ Local splitcompute demo is running."
echo "[demo] Prover URL:   http://${PROVER_HOST}:${PROVER_PORT}"
echo "[demo] Verifier URL: http://${VERIFIER_HOST}:${VERIFIER_PORT}"
echo "[demo] Open this in browser: http://${VERIFIER_HOST}:${VERIFIER_PORT}"
echo "[demo] Logs:"
echo "       - ${PROVER_LOG}"
echo "       - ${VERIFIER_LOG}"
echo
echo "[demo] Press Ctrl+C to stop both services."

wait "${VERIFIER_PID}"

