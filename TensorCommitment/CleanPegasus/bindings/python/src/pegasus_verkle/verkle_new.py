"""High-level Python helpers around the Rust Verkle bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

from ._verkle import PyBatchProof, PyKzgVerkleTree, PySingleProof


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

    def prove(
        self, indices: Iterable[int], *, batch: bool = True
    ) -> Union["BatchProof", "SingleProof"]:
        """Generate either a batch or single proof depending on requested mode."""
        idx_list = list(indices)
        if not idx_list:
            raise ValueError("indices must not be empty")
        if batch or len(idx_list) > 1:
            return self.prove_indices(idx_list)
        return self.prove_single(idx_list[0])

    def prove_indices(self, indices: Iterable[int]) -> "BatchProof":
        proof = self._tree.prove_indices(list(indices))
        return BatchProof(proof)

    def prove_single(self, index: int, *, as_batch: bool = False) -> Union["BatchProof", "SingleProof"]:
        if as_batch:
            return self.prove_indices([index])
        proof = self._tree.prove_single(index)
        return SingleProof(proof)


@dataclass
class BatchProof:
    """Light wrapper that exposes verification directly in Python."""

    _inner: PyBatchProof

    def verify(self, root_hex: str, indices: Sequence[int], values: Sequence[int]) -> bool:
        return self._inner.verify(root_hex, list(indices), list(values))

    @property
    def node_count(self) -> int:
        return self._inner.node_count()


@dataclass
class SingleProof:
    """Wrapper for single-path Verkle proofs."""

    _inner: PySingleProof

    def verify(self, root_hex: str, index: int, value: int) -> bool:
        return self._inner.verify(root_hex, int(index), int(value))

    @property
    def index(self) -> int:
        return self._inner.index

    @property
    def value(self) -> int:
        return self._inner.value

    @property
    def node_count(self) -> int:
        return self._inner.node_count()
