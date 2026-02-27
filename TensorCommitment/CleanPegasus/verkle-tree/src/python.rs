#![cfg(feature = "python")]

use std::fmt::Write;

use ark_bls12_381::{Fr as F, G1Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::verkle_tree::{ProofNode, VerkleProof, VerkleTree, VerkleTreeError};

#[pyclass(module = "pegasus_verkle._verkle")]
pub struct PyKzgVerkleTree {
    tree: VerkleTree,
    width: usize,
    depth: usize,
    data: Vec<F>,
    raw_values: Vec<u128>,
}

#[pymethods]
impl PyKzgVerkleTree {
    #[new]
    pub fn new(py_values: Bound<'_, PyAny>, width: usize) -> PyResult<Self> {
        if width < 2 {
            return Err(PyValueError::new_err(
                "width must be at least 2 for a Verkle tree",
            ));
        }
        let (values_fr, raw_values) = extract_values(&py_values)?;
        ensure_full_subtree(raw_values.len(), width)?;
        let tree = VerkleTree::new(&values_fr, width).map_err(to_py_err)?;
        let depth = tree.depth();
        Ok(Self {
            tree,
            width,
            depth,
            data: values_fr,
            raw_values,
        })
    }

    #[getter]
    pub fn width(&self) -> usize {
        self.width
    }

    #[getter]
    pub fn depth(&self) -> usize {
        self.depth
    }

    #[getter]
    pub fn value_count(&self) -> usize {
        self.raw_values.len()
    }

    pub fn dataset(&self) -> Vec<u128> {
        self.raw_values.clone()
    }

    pub fn root_hex(&self) -> PyResult<String> {
        let root = self
            .tree
            .root_commitment()
            .ok_or_else(|| PyValueError::new_err("tree has no root commitment"))?;
        g1_to_hex(&root)
    }

    pub fn prove_indices(&self, indices: Vec<usize>) -> PyResult<PyBatchProof> {
        if indices.is_empty() {
            return Err(PyValueError::new_err(
                "cannot generate a proof for an empty set of indices",
            ));
        }
        for &idx in &indices {
            if idx >= self.raw_values.len() {
                return Err(PyValueError::new_err(format!(
                    "index {} exceeds dataset size {}",
                    idx,
                    self.raw_values.len()
                )));
            }
        }
        let proofs = self.tree.generate_batch_proof(indices.clone(), &self.data);
        Ok(PyBatchProof {
            proofs,
            width: self.width,
            depth: self.depth,
        })
    }

    pub fn prove_single(&self, index: usize) -> PyResult<PySingleProof> {
        if index >= self.raw_values.len() {
            return Err(PyValueError::new_err(format!(
                "index {} exceeds dataset size {}",
                index,
                self.raw_values.len()
            )));
        }
        let claimed_value = self.data[index];
        let raw_value = self.raw_values[index];
        let proof = self
            .tree
            .generate_proof(index, &claimed_value)
            .map_err(to_py_err)?;
        Ok(PySingleProof::new(
            proof,
            self.width,
            index,
            claimed_value,
            raw_value,
        ))
    }
}

#[pyclass(module = "pegasus_verkle._verkle")]
#[derive(Clone)]
pub struct PyBatchProof {
    proofs: Vec<Option<ProofNode>>,
    width: usize,
    depth: usize,
}

#[pymethods]
impl PyBatchProof {
    pub fn verify(&self, root_hex: &str, indices: Vec<usize>, values: Vec<u128>) -> PyResult<bool> {
        if indices.is_empty() {
            return Err(PyValueError::new_err(
                "indices for verification cannot be empty",
            ));
        }
        if indices.len() != values.len() {
            return Err(PyValueError::new_err(
                "indices and values must have matching length",
            ));
        }
        let root = hex_to_g1(root_hex)?;
        let data: Vec<F> = values.into_iter().map(F::from).collect();
        Ok(VerkleTree::batch_proof_verify(
            root,
            self.proofs.clone(),
            self.width,
            indices,
            self.depth,
            data,
        ))
    }

    pub fn node_count(&self) -> usize {
        self.proofs.len()
    }
}

#[pyclass(module = "pegasus_verkle._verkle")]
#[derive(Clone)]
pub struct PySingleProof {
    proof: VerkleProof,
    width: usize,
    index: usize,
    claimed_value: F,
    raw_value: u128,
}

impl PySingleProof {
    fn new(
        proof: VerkleProof,
        width: usize,
        index: usize,
        claimed_value: F,
        raw_value: u128,
    ) -> Self {
        Self {
            proof,
            width,
            index,
            claimed_value,
            raw_value,
        }
    }
}

#[pymethods]
impl PySingleProof {
    pub fn verify(&self, root_hex: &str, index: usize, value: u128) -> PyResult<bool> {
        if index != self.index {
            return Ok(false);
        }
        if value != self.raw_value {
            return Ok(false);
        }
        if let Some(last) = self.proof.proofs.last() {
            if last.point.len() != 1 || last.point[0].1 != self.claimed_value {
                return Ok(false);
            }
        }
        let root = hex_to_g1(root_hex)?;
        Ok(VerkleTree::verify_proof(root, &self.proof, self.width))
    }

    #[getter]
    pub fn index(&self) -> usize {
        self.index
    }

    #[getter]
    pub fn value(&self) -> u128 {
        self.raw_value
    }

    pub fn node_count(&self) -> usize {
        self.proof.proofs.len()
    }
}

#[pyfunction]
fn commitment_from_hex(root_hex: &str) -> PyResult<Vec<u8>> {
    let point = hex_to_g1(root_hex)?;
    let mut bytes = vec![];
    point
        .serialize_compressed(&mut bytes)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize point: {}", e)))?;
    Ok(bytes)
}

fn extract_values(values: &Bound<'_, PyAny>) -> PyResult<(Vec<F>, Vec<u128>)> {
    if let Ok(vec_u128) = values.extract::<Vec<u128>>() {
        let scalars = vec_u128.iter().copied().map(F::from).collect();
        return Ok((scalars, vec_u128));
    }

    if values.hasattr("tolist")? {
        let py_list = values.call_method0("tolist")?;
        if let Ok(vec_u128) = py_list.extract::<Vec<u128>>() {
            let scalars = vec_u128.iter().copied().map(F::from).collect();
            return Ok((scalars, vec_u128));
        }
    }

    Err(PyValueError::new_err(
        "values must be a sequence of non-negative integers that fit in 128 bits",
    ))
}

fn ensure_full_subtree(len: usize, width: usize) -> PyResult<()> {
    if len == 0 {
        return Err(PyValueError::new_err(
            "cannot build a tree with zero data elements",
        ));
    }
    let mut capacity = 1usize;
    while capacity < len {
        capacity *= width;
    }
    if capacity != len {
        return Err(PyValueError::new_err(format!(
            "data length {} must equal width^depth for some integer depth (next capacity: {})",
            len, capacity
        )));
    }
    Ok(())
}

fn g1_to_hex(point: &G1Affine) -> PyResult<String> {
    let mut bytes = vec![];
    point
        .serialize_compressed(&mut bytes)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize point: {}", e)))?;
    let mut hex_string = String::with_capacity(2 + bytes.len() * 2);
    hex_string.push_str("0x");
    for byte in bytes {
        write!(&mut hex_string, "{:02x}", byte).unwrap();
    }
    Ok(hex_string)
}

fn hex_to_g1(hex_str: &str) -> PyResult<G1Affine> {
    let clean = hex_str.trim_start_matches("0x").trim_start_matches("0X");
    let bytes = hex::decode(clean)
        .map_err(|e| PyValueError::new_err(format!("invalid hex encoding: {}", e)))?;
    G1Affine::deserialize_compressed(bytes.as_slice())
        .map_err(|e| PyValueError::new_err(format!("failed to decode commitment: {}", e)))
}

fn to_py_err(err: VerkleTreeError) -> PyErr {
    match err {
        VerkleTreeError::BuildError => PyValueError::new_err("failed to build Verkle tree"),
        VerkleTreeError::ProofGenerateError => {
            PyValueError::new_err("failed to generate Verkle proof")
        }
        VerkleTreeError::EmptyTree => PyValueError::new_err("tree is empty"),
    }
}

#[pymodule]
pub fn _verkle(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKzgVerkleTree>()?;
    m.add_class::<PyBatchProof>()?;
    m.add_class::<PySingleProof>()?;
    m.add_function(wrap_pyfunction!(commitment_from_hex, m)?)?;
    pyo3::prepare_freethreaded_python();
    Ok(())
}
