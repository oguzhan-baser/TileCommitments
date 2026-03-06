#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="tilecommitments"
MODEL=""
PROMPT=""
RUN_TAG=""
MAX_NEW_TOKENS=16
DTYPE="float16"
DEVICE_MAP="balanced"
MXFP4_MODE="auto"
SCALE_FACTOR=16
QUANTIZE=50
MIN_FREE_GPU_PCT=75
GPU_MEMORY_SPREAD_PCT=85
MIN_DIM=4
MAX_DIM=10
NUM_QUERIES=10
LAYER=0
NUM_PROOFS=8
SEED=42
RTOL="1e-3"
ATOL="5e-2"
SKIP_INTERP_BUILD=1

log() {
  echo "[pipeline] $*"
}

die() {
  echo "[pipeline][error] $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash run_full_pipeline.sh --model <hf_model> --prompt <text> [options]

Required:
  --model                 Hugging Face model name (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
  --prompt                Prompt text for activation capture

Options:
  --env-name              Conda env name (default: tilecommitments)
  --run-tag               Output run tag (default: run_YYYYmmdd_HHMMSS_utc)
  --max-new-tokens        Generation tokens (default: 16)
  --dtype                 float16|float32|bfloat16 (default: float16)
  --device-map            none|auto|balanced|balanced_low_0|sequential (default: balanced)
  --mxfp4-mode            auto|native|dequantize (default: auto)
  --min-free-gpu-pct      Min free GPU memory percent to select a GPU (default: 75)
  --gpu-memory-spread-pct Extra per-GPU cap shrink factor to force wider multi-GPU sharding (default: 85)
  --scale-factor          Integer conversion scale factor (default: 16)
  --quantize              Quantize percent for conversion (default: 50)
  --min-dim               Hypercube min dimension size (default: 4)
  --max-dim               Hypercube max dimension size (default: 10)
  --num-queries           Commitment sample queries (default: 10)
  --layer                 Layer for compute+crypto verification (default: 0)
  --num-proofs            Number of sampled proofs for layer verification (default: 8)
  --seed                  Shared random seed (default: 42)
  --rtol                  Compute verification rtol (default: 1e-3)
  --atol                  Compute verification atol (default: 5e-2)
  --build-interp          Build interpolation binary in this run (default: skip build)
  -h, --help              Show this help
EOF
}

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "/home/ob3942/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "/home/ob3942/miniconda3/etc/profile.d/conda.sh"
    else
      die "conda not found and conda.sh could not be located"
    fi
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
}

detect_free_gpu_budget() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

  local gpu_query
  gpu_query="$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)"
  [[ -n "$gpu_query" ]] || die "Failed to read GPU information from nvidia-smi"

  local -a free_gpu_indices=()
  local -a selected_info=()
  local -a skipped_info=()
  local min_free_mib=0

  while IFS=',' read -r raw_idx raw_total raw_used; do
    local idx total used free_mib free_pct
    idx="$(echo "$raw_idx" | xargs)"
    total="$(echo "$raw_total" | xargs)"
    used="$(echo "$raw_used" | xargs)"

    [[ -z "$idx" ]] && continue
    [[ "$total" =~ ^[0-9]+$ ]] || continue
    [[ "$used" =~ ^[0-9]+$ ]] || continue
    (( total > 0 )) || continue

    free_mib=$(( total - used ))
    free_pct=$(( free_mib * 100 / total ))

    if (( free_pct >= MIN_FREE_GPU_PCT )); then
      free_gpu_indices+=("$idx")
      selected_info+=("gpu${idx}: ${free_mib}/${total}MiB free (${free_pct}%)")
      if (( min_free_mib == 0 || free_mib < min_free_mib )); then
        min_free_mib="$free_mib"
      fi
    else
      skipped_info+=("gpu${idx}: ${free_mib}/${total}MiB free (${free_pct}%)")
    fi
  done <<< "$gpu_query"

  if (( ${#free_gpu_indices[@]} == 0 )); then
    die "No GPUs found with >=${MIN_FREE_GPU_PCT}% free memory."
  fi

  FREE_GPU_CSV="$(IFS=,; echo "${free_gpu_indices[*]}")"
  FREE_GPU_COUNT="${#free_gpu_indices[@]}"
  export CUDA_VISIBLE_DEVICES="$FREE_GPU_CSV"

  local safety_margin_mib=1024
  PER_GPU_LIMIT_MIB=$(( min_free_mib - safety_margin_mib ))
  if (( FREE_GPU_COUNT > 1 )); then
    PER_GPU_LIMIT_MIB=$(( PER_GPU_LIMIT_MIB * GPU_MEMORY_SPREAD_PCT / 100 ))
  fi
  if (( PER_GPU_LIMIT_MIB < 2048 )); then
    die "Derived per-GPU memory budget (${PER_GPU_LIMIT_MIB}MiB) is too low."
  fi

  local mem_available_kib
  mem_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  [[ -n "$mem_available_kib" ]] || mem_available_kib=0
  CPU_LIMIT_GIB=$(( mem_available_kib * 80 / 100 / 1024 / 1024 ))
  if (( CPU_LIMIT_GIB < 8 )); then
    CPU_LIMIT_GIB=8
  fi

  log "Selected GPUs (>=${MIN_FREE_GPU_PCT}% free memory): ${FREE_GPU_CSV} (count=${FREE_GPU_COUNT})"
  if (( ${#selected_info[@]} > 0 )); then
    log "Selection details: ${selected_info[*]}"
  fi
  if (( ${#skipped_info[@]} > 0 )); then
    log "Skipped GPUs: ${skipped_info[*]}"
  fi
  log "Applied gpu-memory-spread-pct: ${GPU_MEMORY_SPREAD_PCT}%"
  log "Derived --max-memory-per-gpu: ${PER_GPU_LIMIT_MIB}MiB"
  log "Derived --max-memory-cpu: ${CPU_LIMIT_GIB}GiB"
}

adjust_mxfp4_mode_for_model() {
  local lower_model
  lower_model="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower_model" != *"gpt-oss"* ]] && [[ "$lower_model" != *"gpt_oss"* ]]; then
    return
  fi

  local probe_output state details
  probe_output="$(
    python - <<'PY'
import torch

if not torch.cuda.is_available():
    print("none|cuda_unavailable")
    raise SystemExit(0)

caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
details = "; ".join(f"cuda:{idx}=sm{major}{minor}" for idx, (major, minor) in enumerate(caps))
has_pre_sm89 = any(capability < (8, 9) for capability in caps)
print(("low" if has_pre_sm89 else "ok") + "|" + details)
PY
  )"

  state="${probe_output%%|*}"
  details="${probe_output#*|}"

  if [[ "$state" == "low" ]]; then
    if [[ "$MXFP4_MODE" == "auto" ]]; then
      MXFP4_MODE="dequantize"
      log "Detected pre-sm89 GPUs for GPT-OSS (${details}); forcing --mxfp4-mode=dequantize."
    elif [[ "$MXFP4_MODE" == "native" ]]; then
      die "Selected GPUs are pre-sm89 (${details}) and --mxfp4-mode=native was requested. Use --mxfp4-mode=dequantize on A100."
    fi
  elif [[ "$state" == "ok" ]]; then
    log "GPT-OSS capability check: ${details}"
  else
    log "GPT-OSS capability check unavailable (${details}); keeping --mxfp4-mode=${MXFP4_MODE}."
  fi

  if [[ "$MXFP4_MODE" == "dequantize" ]]; then
    if [[ "$DTYPE" == "float16" ]]; then
      DTYPE="bfloat16"
      log "Switched --dtype to bfloat16 for GPT-OSS dequantized MXFP4 compatibility."
    fi

    local original_limit_mib
    original_limit_mib="$PER_GPU_LIMIT_MIB"
    PER_GPU_LIMIT_MIB=$(( PER_GPU_LIMIT_MIB * 80 / 100 ))
    if (( PER_GPU_LIMIT_MIB < 12288 )); then
      PER_GPU_LIMIT_MIB=12288
    fi
    log "Applied GPT-OSS dequantized runtime headroom: --max-memory-per-gpu ${original_limit_mib}MiB -> ${PER_GPU_LIMIT_MIB}MiB"

    if [[ "$DEVICE_MAP" == "balanced" ]]; then
      DEVICE_MAP="auto"
      log "Switched --device-map to auto for GPT-OSS dequantized loading."
    fi
  fi
}

require_file() {
  [[ -f "$1" ]] || die "Expected file not found: $1"
}

run_stage() {
  local stage_name="$1"
  shift
  log "Running stage: ${stage_name}"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device-map) DEVICE_MAP="$2"; shift 2 ;;
    --mxfp4-mode) MXFP4_MODE="$2"; shift 2 ;;
    --min-free-gpu-pct) MIN_FREE_GPU_PCT="$2"; shift 2 ;;
    --gpu-memory-spread-pct) GPU_MEMORY_SPREAD_PCT="$2"; shift 2 ;;
    --scale-factor) SCALE_FACTOR="$2"; shift 2 ;;
    --quantize) QUANTIZE="$2"; shift 2 ;;
    --min-dim) MIN_DIM="$2"; shift 2 ;;
    --max-dim) MAX_DIM="$2"; shift 2 ;;
    --num-queries) NUM_QUERIES="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --num-proofs) NUM_PROOFS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --rtol) RTOL="$2"; shift 2 ;;
    --atol) ATOL="$2"; shift 2 ;;
    --build-interp) SKIP_INTERP_BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$MODEL" ]] || die "--model is required"
[[ -n "$PROMPT" ]] || die "--prompt is required"
[[ "$MIN_FREE_GPU_PCT" =~ ^[0-9]+$ ]] || die "--min-free-gpu-pct must be an integer in [1,100]"
(( MIN_FREE_GPU_PCT >= 1 && MIN_FREE_GPU_PCT <= 100 )) || die "--min-free-gpu-pct must be in [1,100]"
[[ "$GPU_MEMORY_SPREAD_PCT" =~ ^[0-9]+$ ]] || die "--gpu-memory-spread-pct must be an integer in [10,100]"
(( GPU_MEMORY_SPREAD_PCT >= 10 && GPU_MEMORY_SPREAD_PCT <= 100 )) || die "--gpu-memory-spread-pct must be in [10,100]"
case "$DEVICE_MAP" in
  none|auto|balanced|balanced_low_0|sequential) ;;
  *) die "--device-map must be one of: none|auto|balanced|balanced_low_0|sequential" ;;
esac
case "$MXFP4_MODE" in
  auto|native|dequantize) ;;
  *) die "--mxfp4-mode must be one of: auto|native|dequantize" ;;
esac

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="run_$(date -u +%Y%m%d_%H%M%S)_utc"
fi

SAFE_MODEL="${MODEL//\//_}"
SAFE_MODEL="${SAFE_MODEL//:/_}"

RUN_DIR="$REPO_ROOT/TensorCommitment/activationCaptureLib/output/${RUN_TAG}"
ACT_PT="$RUN_DIR/${SAFE_MODEL}_activations.pt"
INT_DIR="$RUN_DIR/${SAFE_MODEL}_int_activations"
HC_DIR="${INT_DIR}_hypercube"
POLY_DIR="${HC_DIR}_polynomial"
COMMIT_DIR="${HC_DIR}_commitment"
LAYER_VERIFY_DIR="$RUN_DIR/layer${LAYER}_compute_crypto"
FULL_COVERAGE_DIR="$RUN_DIR/full_coverage_verification"
METRICS_FILE="$RUN_DIR/run_metrics.json"

activate_conda_env
detect_free_gpu_budget
adjust_mxfp4_mode_for_model
mkdir -p "$RUN_DIR"

INTERP_BUILD_ARGS=()
if (( SKIP_INTERP_BUILD == 1 )); then
  INTERP_BUILD_ARGS+=(--skip-build)
fi

run_stage "capture_activations" \
  python TensorCommitment/activationCaptureLib/capture_activations.py \
    --models "$MODEL" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-dir "$RUN_DIR" \
    --device cuda \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --mxfp4-mode "$MXFP4_MODE" \
    --max-memory-per-gpu "${PER_GPU_LIMIT_MIB}MiB" \
    --max-memory-cpu "${CPU_LIMIT_GIB}GiB" \
    --seed "$SEED"
require_file "$ACT_PT"

run_stage "convert_to_npy" \
  python TensorCommitment/activationCaptureLib/convert_to_npy.py \
    --input "$ACT_PT" \
    --output-dir "$INT_DIR" \
    --scale-factor "$SCALE_FACTOR" \
    --quantize "$QUANTIZE"
require_file "$INT_DIR/embedding.npy"

run_stage "reshape_to_hypercube" \
  python TensorCommitment/activationCaptureLib/reshape_to_hypercube.py \
    --input-dir "$INT_DIR" \
    --output-dir "$HC_DIR" \
    --min-dim "$MIN_DIM" \
    --max-dim "$MAX_DIM"
require_file "$HC_DIR/hypercube.npy"
require_file "$HC_DIR/hypercube_metadata.json"

run_stage "interpolate_hypercube" \
  python TensorCommitment/interpolationLib/interpolate_hypercube.py \
    --input-dir "$HC_DIR" \
    --output-dir "$POLY_DIR" \
    "${INTERP_BUILD_ARGS[@]}"
require_file "$POLY_DIR/coefficients.json"

run_stage "commit_prove_verify" \
  python TensorCommitment/tensorCommitmentLib/commit_prove_verify.py \
    --poly-dir "$POLY_DIR" \
    --hypercube-dir "$HC_DIR" \
    --output-dir "$COMMIT_DIR" \
    --num-queries "$NUM_QUERIES" \
    --seed "$SEED"
require_file "$COMMIT_DIR/commitment.txt"

run_stage "compute_crypto_verify_layer" \
  python compute_crypto_verify_layer.py \
    --activations-pt "$ACT_PT" \
    --hypercube-dir "$HC_DIR" \
    --poly-dir "$POLY_DIR" \
    --commitment-file "$COMMIT_DIR/commitment.txt" \
    --layer "$LAYER" \
    --proof-mode sample \
    --num-proofs "$NUM_PROOFS" \
    --device cuda \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --max-memory-per-gpu "${PER_GPU_LIMIT_MIB}MiB" \
    --max-memory-cpu "${CPU_LIMIT_GIB}GiB" \
    --rtol "$RTOL" \
    --atol "$ATOL" \
    --output-dir "$LAYER_VERIFY_DIR"
require_file "$LAYER_VERIFY_DIR/compute_crypto_verification_summary.json"

run_stage "full_coverage_verify" \
  python full_coverage_verify.py \
    --activations-pt "$ACT_PT" \
    --hypercube-dir "$HC_DIR" \
    --poly-dir "$POLY_DIR" \
    --commitment-file "$COMMIT_DIR/commitment.txt" \
    --seed "$SEED" \
    --device cuda \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --max-memory-per-gpu "${PER_GPU_LIMIT_MIB}MiB" \
    --max-memory-cpu "${CPU_LIMIT_GIB}GiB" \
    --rtol "$RTOL" \
    --atol "$ATOL" \
    --output-dir "$FULL_COVERAGE_DIR"
require_file "$FULL_COVERAGE_DIR/full_coverage_summary.json"

run_stage "summarize_run_metrics" \
  python summarize_run_metrics.py \
    --run-dir "$RUN_DIR" \
    --layer-summary "$LAYER_VERIFY_DIR/compute_crypto_verification_summary.json" \
    --output "$METRICS_FILE"
require_file "$METRICS_FILE"

log "Pipeline completed successfully."
log "Run dir: $RUN_DIR"
log "Commitment: $COMMIT_DIR/commitment.txt"
log "Layer verification summary: $LAYER_VERIFY_DIR/compute_crypto_verification_summary.json"
log "Full-coverage summary: $FULL_COVERAGE_DIR/full_coverage_summary.json"
log "Run metrics: $METRICS_FILE"
