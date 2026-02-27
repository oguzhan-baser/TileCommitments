#!/bin/bash
# ICML Experiment Setup Script
# Installs all Rust libraries with Python bindings for Terkle, Verkle, and Merkle experiments

set -e

echo "=========================================="
echo "ICML Experiment Installation"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for required tools
command -v cargo >/dev/null 2>&1 || { echo "Error: cargo is required. Install Rust first."; exit 1; }
command -v maturin >/dev/null 2>&1 || { echo "Installing maturin..."; pip install maturin; }

echo ""
echo "Step 1/4: Installing tensorcommitments (PST Tensor Commitment Library)"
echo "-----------------------------------------------------------------------"
cd "$SCRIPT_DIR/tensorCommitmentLib"
maturin develop --release
echo "✓ tensorcommitments installed"

echo ""
echo "Step 2/4: Installing terkle (Multivariate Verkle Tree)"
echo "-------------------------------------------------------"
cd "$SCRIPT_DIR/terkleLib"
maturin develop --release
echo "✓ terkle installed"

echo ""
echo "Step 3/4: Installing pegasus_verkle (Univariate KZG Verkle Tree)"
echo "-----------------------------------------------------------------"
cd "$SCRIPT_DIR/CleanPegasus/bindings/python"
maturin develop --release
echo "✓ pegasus_verkle installed"

echo ""
echo "Step 4/4: Installing multibranch_merkle (Multi-branch Merkle Tree)"
echo "-------------------------------------------------------------------"
cd "$SCRIPT_DIR/merkle"
maturin develop --release
echo "✓ multibranch_merkle installed"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Python packages installed:"
echo "  - tensorcommitments  : PST tensor commitment scheme"
echo "  - terkle             : Multivariate Verkle tree (Terkle)"
echo "  - pegasus_verkle     : Univariate KZG Verkle tree"
echo "  - multibranch_merkle : Configurable-arity Merkle tree"
echo ""
echo "To run experiments:"
echo "  cd experiments"
echo "  python run_all_sweeps.py"
echo ""
echo "To plot results:"
echo "  python plot_dimension_benchmark.py --input results/dimension_benchmark.jsonl"
echo ""
