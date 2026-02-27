#!/bin/bash
# Build script for Python bindings of the Theseus Tensor Commitment library

set -e

echo "Building Python bindings for Theseus Tensor Commitments..."

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: This script must be run from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo is not installed. Please install from https://rustup.rs/"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install maturin if not available
if ! python3 -c "import maturin" 2>/dev/null; then
    echo "Installing maturin..."
    pip3 install maturin
fi

# Build the library in development mode
echo "Building Rust library with Python bindings..."
maturin develop --features python

# Test the installation
echo "Testing the installation..."
python3 -c "
try:
    import tensorcommitments
    print('✅ Python bindings installed successfully!')
    
    # Quick test
    wrapper = tensorcommitments.TensorCommitmentWrapper(2, 3)
    print('✅ Tensor commitment wrapper created successfully!')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Test failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "You can now run the example scripts:"
echo "  python3 no_saving_time_version.py poly_interp_demo/rand10by10by10by10by10by10.npy"
echo "  python3 prover_verifier.py poly_interp_demo/rand10by10by10by10by10by10.npy"
echo ""
echo "Or use the library in your own code:"
echo "  import tensorcommitments"
echo "  wrapper = tensorcommitments.TensorCommitmentWrapper(num_vars, degree_bound)"
