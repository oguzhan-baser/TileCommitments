use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Size of every hash produced by the tree (SHA-256).
pub type Hash = [u8; 32];

const LEAF_PREFIX: u8 = 0x00;
const NODE_PREFIX: u8 = 0x01;

/// Multi-branch Merkle tree whose arity can be >= 2.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiMerkleTree {
    arity: usize,
    levels: Vec<Vec<Hash>>,
    leaves: Vec<Vec<u8>>,
}

impl MultiMerkleTree {
    /// Build a tree from the provided `leaves`.
    ///
    /// # Panics
    ///
    /// Panics when `arity < 2`.
    pub fn new(arity: usize, leaves: Vec<Vec<u8>>) -> Self {
        assert!(arity >= 2, "arity must be at least 2");

        if leaves.is_empty() {
            return Self {
                arity,
                levels: vec![vec![hash_empty()]],
                leaves,
            };
        }

        let hashed_leaves: Vec<Hash> = leaves.iter().map(|leaf| hash_leaf(leaf)).collect();
        let levels = build_levels(hashed_leaves, arity);

        Self {
            arity,
            levels,
            leaves,
        }
    }

    /// Convenience helper to build the tree from integers.
    pub fn from_integers<I: IntoIterator<Item = u64>>(arity: usize, values: I) -> Self {
        let leaves = values
            .into_iter()
            .map(|value| value.to_le_bytes().to_vec())
            .collect();
        Self::new(arity, leaves)
    }

    /// Returns the number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Returns `true` if no leaves were provided.
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Returns the arity used for this tree.
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Returns the root hash.
    pub fn root(&self) -> Hash {
        *self
            .levels
            .last()
            .and_then(|lvl| lvl.first())
            .expect("tree always contains a root hash")
    }

    /// Root hash formatted as hex.
    pub fn root_hex(&self) -> String {
        hex::encode(self.root())
    }

    /// Generate the inclusion proof for the leaf at `index`.
    pub fn proof(&self, index: usize) -> Option<Proof> {
        if index >= self.leaves.len() {
            return None;
        }

        let mut idx = index;
        let mut steps = Vec::new();

        for level in 0..self.levels.len().saturating_sub(1) {
            let nodes = &self.levels[level];
            if nodes.len() <= 1 {
                idx /= self.arity;
                continue;
            }

            let chunk_start = (idx / self.arity) * self.arity;
            let chunk_end = (chunk_start + self.arity).min(nodes.len());
            let child_count = chunk_end - chunk_start;
            let position = idx - chunk_start;

            if child_count <= 1 {
                idx /= self.arity;
                continue;
            }

            let mut siblings = Vec::with_capacity(child_count - 1);
            for offset in 0..child_count {
                let child_idx = chunk_start + offset;
                if child_idx == idx {
                    continue;
                }

                siblings.push(IndexedHash {
                    index: offset,
                    hash: nodes[child_idx],
                });
            }

            steps.push(ProofStep {
                position,
                child_count,
                siblings,
            });

            idx /= self.arity;
        }

        Some(Proof { steps })
    }

    /// Convenience helper that generates a proof internally and validates it.
    pub fn verify(&self, index: usize, value: &[u8]) -> bool {
        match self.proof(index) {
            Some(proof) => proof.verify(value, &self.root()),
            None => false,
        }
    }

    /// Returns the leaves as stored internally.
    pub fn leaves(&self) -> &[Vec<u8>] {
        &self.leaves
    }
}

/// Position tagged hash of a sibling subtree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedHash {
    pub index: usize,
    pub hash: Hash,
}

/// One level of a multi-branch proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofStep {
    pub position: usize,
    pub child_count: usize,
    pub siblings: Vec<IndexedHash>,
}

/// Inclusion proof for a leaf.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    steps: Vec<ProofStep>,
}

impl Proof {
    /// Returns the steps that make up this proof.
    pub fn steps(&self) -> &[ProofStep] {
        &self.steps
    }

    /// Validates the proof for the provided `value` against `root`.
    pub fn verify(&self, value: &[u8], root: &Hash) -> bool {
        let mut current = hash_leaf(value);

        if self.steps.is_empty() {
            return &current == root;
        }

        for step in &self.steps {
            if step.child_count < 2 || step.position >= step.child_count {
                return false;
            }

            if step.siblings.len() + 1 != step.child_count {
                return false;
            }

            let mut children: Vec<Option<Hash>> = vec![None; step.child_count];
            children[step.position] = Some(current);

            for sibling in &step.siblings {
                if sibling.index >= step.child_count {
                    return false;
                }

                if children[sibling.index].is_some() {
                    return false;
                }

                children[sibling.index] = Some(sibling.hash);
            }

            if children.iter().any(|entry| entry.is_none()) {
                return false;
            }

            let mut hasher = Sha256::new();
            hasher.update(&[NODE_PREFIX]);
            for child in children {
                hasher.update(child.unwrap());
            }
            current = hasher.finalize().into();
        }

        &current == root
    }
}

fn build_levels(mut current: Vec<Hash>, arity: usize) -> Vec<Vec<Hash>> {
    let mut levels = Vec::new();
    levels.push(current.clone());

    while current.len() > 1 {
        let mut next = Vec::new();

        for chunk in current.chunks(arity) {
            if chunk.len() == 1 {
                next.push(chunk[0]);
            } else {
                next.push(hash_branch(chunk));
            }
        }

        current = next.clone();
        levels.push(next);
    }

    levels
}

fn hash_leaf(value: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(&[LEAF_PREFIX]);
    hasher.update(value);
    hasher.finalize().into()
}

fn hash_branch(children: &[Hash]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(&[NODE_PREFIX]);
    for child in children {
        hasher.update(child);
    }
    hasher.finalize().into()
}

fn hash_empty() -> Hash {
    Sha256::digest(&[]).into()
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::*;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn root_matches_manual_construction() {
        let leaves = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"d".to_vec(),
            b"e".to_vec(),
        ];

        let tree = MultiMerkleTree::new(3, leaves.clone());

        let hashed: Vec<Hash> = leaves.iter().map(|leaf| hash_leaf(leaf)).collect();
        let left = hash_branch(&hashed[0..3]);
        let right = hash_branch(&hashed[3..5]);
        let root = hash_branch(&[left, right]);

        assert_eq!(tree.root(), root);
    }

    #[test]
    fn proofs_verify_for_all_leaves() {
        let values = (0u64..15)
            .map(|v| v.to_le_bytes().to_vec())
            .collect::<Vec<_>>();
        let tree = MultiMerkleTree::new(4, values.clone());
        let root = tree.root();

        for (idx, value) in values.iter().enumerate() {
            let proof = tree.proof(idx).expect("proof exists");
            assert!(proof.verify(value, &root), "proof failed for index {idx}");
        }
    }

    #[test]
    fn verify_fails_for_wrong_value() {
        let values = (0u64..8)
            .map(|v| v.to_le_bytes().to_vec())
            .collect::<Vec<_>>();
        let tree = MultiMerkleTree::new(3, values.clone());
        let root = tree.root();

        let proof = tree.proof(3).expect("proof exists");
        let mut tampered = values[3].clone();
        tampered[0] ^= 0xFF;
        assert!(!proof.verify(&tampered, &root));
    }

    #[test]
    fn verify_helper_works() {
        let values = (0u64..32)
            .map(|v| v.to_le_bytes().to_vec())
            .collect::<Vec<_>>();
        let tree = MultiMerkleTree::new(5, values.clone());

        for (idx, value) in values.iter().enumerate() {
            assert!(tree.verify(idx, value));
        }

        assert!(!tree.verify(values.len(), b"nope"));
    }

    #[test]
    fn handles_randomized_inputs() {
        let mut rng = rand::thread_rng();
        let mut leaves = Vec::new();
        for _ in 0..100 {
            let value = rng.gen::<u64>().to_le_bytes().to_vec();
            leaves.push(value);
        }

        let tree = MultiMerkleTree::new(6, leaves.clone());
        let root = tree.root();

        for (idx, value) in leaves.iter().enumerate().take(20) {
            let proof = tree.proof(idx).expect("proof exists");
            assert!(proof.verify(value, &root));
        }
    }
}
