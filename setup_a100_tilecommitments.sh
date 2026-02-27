#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_DIR="$REPO_ROOT/TensorCommitment"
MAIN_ENV_FILE="$REPO_ROOT/environment.yml"
TC_ENV_FILE="$TC_DIR/environment.yml"
ENV_NAME="${ENV_NAME:-tilecommitments}"
SKIP_GPU_INSTALL="${SKIP_GPU_INSTALL:-0}"
NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-}"
CUDA_TOOLKIT_PKG="${CUDA_TOOLKIT_PKG:-}"
MIN_FREE_MB_FOR_CUDA_PKG="${MIN_FREE_MB_FOR_CUDA_PKG:-6000}"
DETECTED_OS_ID=""
DETECTED_OS_VERSION_ID=""

log() {
  echo "[setup] $*"
}

warn() {
  echo "[setup][warn] $*" >&2
}

die() {
  echo "[setup][error] $*" >&2
  exit 1
}

as_root() {
  if [[ "$EUID" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file not found: $path"
}

detect_os() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    DETECTED_OS_ID="${ID:-unknown}"
    DETECTED_OS_VERSION_ID="${VERSION_ID:-unknown}"
  else
    DETECTED_OS_ID="unknown"
    DETECTED_OS_VERSION_ID="unknown"
  fi
  log "Detected OS: ${DETECTED_OS_ID} ${DETECTED_OS_VERSION_ID}"
}

set_gpu_package_defaults() {
  if [[ -n "$NVIDIA_DRIVER_PKG" && -n "$CUDA_TOOLKIT_PKG" ]]; then
    return
  fi

  case "${DETECTED_OS_ID}:${DETECTED_OS_VERSION_ID}" in
    ubuntu:22.04|ubuntu:24.04)
      NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-nvidia-driver-550}"
      CUDA_TOOLKIT_PKG="${CUDA_TOOLKIT_PKG:-cuda-runtime-12-4}"
      ;;
    debian:12)
      NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-cuda-drivers}"
      # Default to driver-only on Debian to avoid very large toolkit installs on small boot disks.
      CUDA_TOOLKIT_PKG="${CUDA_TOOLKIT_PKG:-}"
      ;;
    *)
      warn "Unknown distro/version for GPU package defaults: ${DETECTED_OS_ID} ${DETECTED_OS_VERSION_ID}."
      NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-nvidia-driver}"
      CUDA_TOOLKIT_PKG="${CUDA_TOOLKIT_PKG:-}"
      ;;
  esac
}

install_system_packages() {
  log "Installing base Linux build dependencies..."
  export DEBIAN_FRONTEND=noninteractive
  as_root apt-get update
  as_root apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    gnupg \
    jq \
    libssl-dev \
    lsb-release \
    make \
    patchelf \
    pkg-config \
    software-properties-common \
    unzip \
    wget
}

install_cuda_repo_if_needed() {
  local repo_tag
  case "${DETECTED_OS_ID}:${DETECTED_OS_VERSION_ID}" in
    ubuntu:22.04) repo_tag="ubuntu2204" ;;
    ubuntu:24.04) repo_tag="ubuntu2404" ;;
    debian:12) repo_tag="debian12" ;;
    *)
      warn "Unsupported distro/version for automatic CUDA repo setup: ${DETECTED_OS_ID} ${DETECTED_OS_VERSION_ID}."
      return
      ;;
  esac

  local repo_list="/etc/apt/sources.list.d/cuda-${repo_tag}-x86_64.list"
  if [[ -f "$repo_list" ]]; then
    log "CUDA apt repository already configured (${repo_tag})."
    return
  fi

  log "Configuring NVIDIA CUDA apt repository (${repo_tag})..."
  local keyring_deb="/tmp/cuda-keyring.deb"
  curl -fsSL -o "$keyring_deb" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${repo_tag}/x86_64/cuda-keyring_1.1-1_all.deb"
  as_root dpkg -i "$keyring_deb"
  rm -f "$keyring_deb"
  as_root apt-get update
}

install_kernel_prereqs_if_needed() {
  case "${DETECTED_OS_ID}" in
    ubuntu)
      as_root apt-get install -y --no-install-recommends \
        "linux-headers-$(uname -r)" \
        "linux-modules-extra-$(uname -r)" || true
      ;;
    debian)
      as_root apt-get install -y --no-install-recommends \
        "linux-headers-$(uname -r)" || true
      ;;
    *)
      as_root apt-get install -y --no-install-recommends \
        "linux-headers-$(uname -r)" || true
      ;;
  esac
}

cleanup_apt_artifacts() {
  as_root apt-get clean || true
  as_root rm -rf /tmp/apt-dpkg-install-* /var/tmp/apt-dpkg-install-* || true
}

recover_dpkg_state() {
  cleanup_apt_artifacts
  as_root dpkg --configure -a || true
  as_root apt-get -f install -y || true
}

free_mb_on_rootfs() {
  df -Pm / | awk 'NR==2 {print $4}'
}

install_nvidia_cuda_packages() {
  local -a driver_attempts=()
  driver_attempts+=("${NVIDIA_DRIVER_PKG}")
  if [[ "${DETECTED_OS_ID}" == "debian" ]]; then
    driver_attempts+=("cuda-drivers")
    driver_attempts+=("nvidia-driver")
  fi

  local driver_installed=0
  local driver_pkg
  for driver_pkg in "${driver_attempts[@]}"; do
    [[ -n "$driver_pkg" ]] || continue
    log "Installing GPU driver package: ${driver_pkg}"
    recover_dpkg_state
    as_root apt-get update || true
    if as_root apt-get install -y --no-install-recommends "$driver_pkg"; then
      NVIDIA_DRIVER_PKG="$driver_pkg"
      driver_installed=1
      break
    fi
    warn "Install attempt failed for GPU driver package: ${driver_pkg}"
  done

  if [[ "$driver_installed" -ne 1 ]]; then
    die "Failed to install NVIDIA driver package on ${DETECTED_OS_ID} ${DETECTED_OS_VERSION_ID}."
  fi

  if [[ -z "$CUDA_TOOLKIT_PKG" ]]; then
    warn "CUDA_TOOLKIT_PKG is empty; continuing with driver-only setup."
    return
  fi

  local free_mb
  free_mb="$(free_mb_on_rootfs)"
  if [[ "$free_mb" -lt "$MIN_FREE_MB_FOR_CUDA_PKG" ]]; then
    warn "Only ${free_mb} MB free on '/'; skipping CUDA package '${CUDA_TOOLKIT_PKG}'."
    warn "Set CUDA_TOOLKIT_PKG explicitly and/or increase disk if you need full toolkit."
    CUDA_TOOLKIT_PKG=""
    return
  fi

  local -a cuda_attempts=()
  cuda_attempts+=("${CUDA_TOOLKIT_PKG}")
  if [[ "${DETECTED_OS_ID}" == "debian" ]]; then
    cuda_attempts+=("cuda-runtime-12-4")
    cuda_attempts+=("")
  fi

  local cuda_pkg
  for cuda_pkg in "${cuda_attempts[@]}"; do
    if [[ -z "$cuda_pkg" ]]; then
      warn "Continuing without CUDA user-space packages."
      CUDA_TOOLKIT_PKG=""
      return
    fi
    log "Installing CUDA package: ${cuda_pkg}"
    recover_dpkg_state
    as_root apt-get update || true
    if as_root apt-get install -y --no-install-recommends "$cuda_pkg"; then
      CUDA_TOOLKIT_PKG="$cuda_pkg"
      return
    fi
    warn "Install attempt failed for CUDA package: ${cuda_pkg}"
  done

  warn "Failed to install CUDA package(s); continuing with driver-only setup."
  CUDA_TOOLKIT_PKG=""
}

ensure_gpu_driver_and_cuda() {
  if [[ "$SKIP_GPU_INSTALL" == "1" ]]; then
    warn "SKIP_GPU_INSTALL=1, skipping NVIDIA driver/CUDA installation."
    return
  fi

  set_gpu_package_defaults

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver already available."
  else
    log "Installing NVIDIA driver (${NVIDIA_DRIVER_PKG}) and CUDA toolkit (${CUDA_TOOLKIT_PKG})..."
    install_cuda_repo_if_needed
    install_kernel_prereqs_if_needed
    install_nvidia_cuda_packages
    as_root modprobe nvidia || true
  fi

  export PATH="/usr/local/cuda/bin:${PATH}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || warn "nvidia-smi exists but failed. A reboot may be required."
  else
    warn "nvidia-smi not found after installation. A reboot may be required."
  fi

  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
  else
    warn "nvcc not found. CUDA toolkit might not be active yet."
  fi
}

ensure_miniconda() {
  if command -v conda >/dev/null 2>&1; then
    log "Using existing conda at $(command -v conda)"
    return
  fi

  local installer="/tmp/miniconda_installer.sh"
  log "Installing Miniconda to ${HOME}/miniconda3 ..."
  curl -fsSL -o "$installer" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "$installer" -b -p "${HOME}/miniconda3"
  rm -f "$installer"
  "${HOME}/miniconda3/bin/conda" init bash >/dev/null || true
}

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return
  fi
  if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
    return
  fi
  die "Conda executable not found."
}

ensure_rust_toolchain() {
  if ! command -v rustup >/dev/null 2>&1; then
    log "Installing Rust toolchain via rustup..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  fi
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
  rustup toolchain install stable --profile minimal >/dev/null
  rustup default stable >/dev/null
  log "Rust toolchain ready: $(rustc --version)"
}

create_or_update_env() {
  local env_file=""
  if [[ -f "$MAIN_ENV_FILE" ]]; then
    env_file="$MAIN_ENV_FILE"
    log "Using main environment file: $env_file"
  else
    env_file="$TC_ENV_FILE"
    require_file "$env_file"
    log "Main environment file not found. Falling back to: $env_file"
  fi

  local sanitized_env
  sanitized_env="$(mktemp)"
  awk '
    !/^[[:space:]]*-[[:space:]]*(theseus|multibranch-merkle|pegasus-verkle|tensorcommitments)==/ &&
    !/^[[:space:]]*prefix:[[:space:]]*/
  ' "$env_file" > "$sanitized_env"

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Updating conda env: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$sanitized_env" --prune
  else
    log "Creating conda env: $ENV_NAME"
    conda env create -n "$ENV_NAME" -f "$sanitized_env"
  fi

  rm -f "$sanitized_env"
  conda activate "$ENV_NAME"
  python --version
  python -m pip install --upgrade pip
}

build_python_bindings_and_cargo_targets() {
  conda activate "$ENV_NAME"
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"

  log "Installing build tools inside conda env..."
  python -m pip install maturin setuptools-rust

  log "Building tensorcommitments (TensorCommitment core) Python binding..."
  (
    cd "$TC_DIR/pst_commitment_lib"
    maturin develop --features python --release
  )

  log "Building multibranch_merkle binding..."
  (
    cd "$TC_DIR/merkle"
    maturin develop --release
  )

  log "Building pegasus_verkle binding..."
  (
    cd "$TC_DIR/CleanPegasus/bindings/python"
    maturin develop --release
  )

  log "Building CleanPegasus Verkle Rust crate..."
  (
    cd "$TC_DIR/CleanPegasus/verkle-tree"
    cargo build --release
  )

  log "Building polynomial interpolation binary..."
  (
    cd "$TC_DIR/pst_commitment_lib/poly_interp_demo"
    cargo build --release
  )
}

run_smoke_tests() {
  conda activate "$ENV_NAME"

  log "Running Python binding smoke tests..."
  python - <<'PY'
import datasets
import tensorcommitments
import torch
import transformers
from multibranch_merkle import MultiMerkleTree
from pegasus_verkle import KzgVerkleTree

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)

wrapper = tensorcommitments.TensorCommitmentWrapper(2, 2)
coeffs = [1, 2, 3, 4]
point = [1, 1]
commitment = wrapper.commit(coeffs)
evaluation = wrapper.evaluate_polynomial(coeffs, point)
proof = wrapper.prove(coeffs, point, evaluation)
assert wrapper.verify(commitment, point, evaluation, proof)

mtree = MultiMerkleTree([1, 2, 3, 4], arity=2)
mproof = mtree.prove(1)
assert mproof.verify(2, mtree.root_hex())

tree = KzgVerkleTree(list(range(1, 513)), width=8)
root = tree.root_hex()
proof = tree.prove_single(7)
assert proof.verify(root, 7, 8)

print("smoke_tests_ok")
PY
}

print_next_steps() {
  cat <<EOF

[setup] Complete.
[setup] Repo: ${REPO_ROOT}
[setup] Conda env: ${ENV_NAME}

To use:
  source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate ${ENV_NAME}
  cd ${REPO_ROOT}

If NVIDIA driver was newly installed and GPU commands fail, reboot once and rerun:
  nvidia-smi

EOF
}

main() {
  [[ -d "$TC_DIR" ]] || die "TensorCommitment directory not found under $REPO_ROOT"

  detect_os
  install_system_packages
  ensure_gpu_driver_and_cuda
  ensure_miniconda
  activate_conda
  ensure_rust_toolchain
  create_or_update_env
  build_python_bindings_and_cargo_targets
  run_smoke_tests
  print_next_steps
}

main "$@"
# source ~/miniconda3/etc/profile.d/conda.sh
# conda env create -n tilecommitments -f TensorCommitment/environment.yml
# conda activate tilecommitments
