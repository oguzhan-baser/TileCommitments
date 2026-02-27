use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::{MultiMerkleTree, Proof};

fn value_to_bytes(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[pyclass(name = "MultiMerkleTree")]
pub struct PyMultiMerkleTree {
    inner: MultiMerkleTree,
}

#[pymethods]
impl PyMultiMerkleTree {
    #[new]
    fn new(values: Vec<u64>, arity: usize) -> PyResult<Self> {
        if arity < 2 {
            return Err(PyValueError::new_err("arity must be >= 2"));
        }
        let leaves = values.into_iter().map(value_to_bytes).collect();
        Ok(Self {
            inner: MultiMerkleTree::new(arity, leaves),
        })
    }

    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn root_hex(&self) -> String {
        self.inner.root_hex()
    }

    fn prove(&self, index: usize) -> PyResult<PyProof> {
        self.inner
            .proof(index)
            .map(|proof| PyProof { inner: proof })
            .ok_or_else(|| PyValueError::new_err("index out of range"))
    }

    fn verify(&self, index: usize, value: u64) -> PyResult<bool> {
        Ok(self.inner.verify(index, &value_to_bytes(value)))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self.inner)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let reader = BufReader::new(file);
        let tree: MultiMerkleTree = serde_json::from_reader(reader)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner: tree })
    }
}

#[pyclass(name = "Proof")]
pub struct PyProof {
    inner: Proof,
}

#[pymethods]
impl PyProof {
    fn verify(&self, value: u64, root_hex: &str) -> PyResult<bool> {
        let mut root = [0u8; 32];
        let bytes = hex::decode(root_hex).map_err(|err| PyValueError::new_err(format!("{err}")))?;
        if bytes.len() != 32 {
            return Err(PyValueError::new_err(
                "root hex string must encode 32 bytes",
            ));
        }
        root.copy_from_slice(&bytes);

        Ok(self.inner.verify(&value_to_bytes(value), &root))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self.inner)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let reader = BufReader::new(file);
        let proof: Proof = serde_json::from_reader(reader)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner: proof })
    }
}

#[pymodule]
pub fn multibranch_merkle(_py: Python, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<PyMultiMerkleTree>()?;
    module.add_class::<PyProof>()?;
    Ok(())
}
