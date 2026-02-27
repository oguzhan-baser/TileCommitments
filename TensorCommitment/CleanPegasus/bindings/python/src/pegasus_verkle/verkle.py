"""High-level Python helpers around the Rust Verkle bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ._verkle import PyBatchProof, PyKzgVerkleTree


class KzgVerkleTree:
    """Wrapper that mirrors the Rust API but with Python conveniences."""

    def __init__(self, values: Sequence[int], width: int) -> None:
        self._tree = PyKzgVerkleTree(values, width)

    @property
    def width(self) -> int:
        return self._tree.width

    @property
    def depth(self) -> int:
        return self._tree.depth

    @property
    def value_count(self) -> int:
        return self._tree.value_count

    def dataset(self) -> List[int]:
        return list(self._tree.dataset())

    def root_hex(self) -> str:
        return self._tree.root_hex()

    def prove_indices(self, indices: Iterable[int]) -> "BatchProof":
        proof = self._tree.prove_indices(list(indices))
        return BatchProof(proof)

    def prove_single(self, index: int) -> "BatchProof":
        return self.prove_indices([index])


@dataclass
class BatchProof:
    """Light wrapper that exposes verification directly in Python."""

    _inner: PyBatchProof

    def verify(self, root_hex: str, indices: Sequence[int], values: Sequence[int]) -> bool:
        return self._inner.verify(root_hex, list(indices), list(values))

    @property
    def node_count(self) -> int:
        return self._inner.node_count()

