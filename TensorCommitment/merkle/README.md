# Multi-branch Merkle Tree

Configurable-arity Merkle tree implementation with Python bindings.

## Installation

```bash
cd merkle
maturin develop --release
```

## Usage

```python
from multibranch_merkle import MultiMerkleTree

# Create tree from values
values = [b"value1", b"value2", b"value3", b"value4"]
tree = MultiMerkleTree(values, arity=2)

# Get root
root = tree.root_hex()

# Generate proof
proof = tree.proof(0)

# Verify proof
result = proof.verify(values[0], root)
```
