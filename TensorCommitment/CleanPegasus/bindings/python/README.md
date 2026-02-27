# Pegasus Verkle

Python bindings for the CleanPegasus Verkle tree implementation using KZG commitments.

## Installation

```bash
cd CleanPegasus/bindings/python
maturin develop --release
```

## Usage

```python
from pegasus_verkle import KzgVerkleTree

# Create tree from values
values = [1, 2, 3, 4, 5, 6, 7, 8]
tree = KzgVerkleTree(values, width=8)

# Get root
root = tree.root()

# Generate proof
proof = tree.prove_single(3)

# Verify proof
result = proof.verify(root, 3, values[3])
```
