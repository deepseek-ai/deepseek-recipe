//! Bindings for tokenizers used to decode token-id inference chunks.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokenizers::{FromPretrainedParameters, Tokenizer};

/// A HuggingFace tokenizer loaded from a `tokenizer.json` file.
///
/// A stream processor uses it to decode token-id inference chunks.
#[pyclass(name = "Tokenizer", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyTokenizer {
    pub(crate) inner: Arc<Tokenizer>,
}

impl PyTokenizer {
    fn from_tokenizer(inner: Tokenizer) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        Tokenizer::from_file(path)
            .map(Self::from_tokenizer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Load a tokenizer from the contents of a `tokenizer.json` file.
    #[staticmethod]
    fn from_str(text: &str) -> PyResult<Self> {
        Tokenizer::from_bytes(text)
            .map(Self::from_tokenizer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Download and cache `tokenizer.json` from the HuggingFace Hub.
    #[staticmethod]
    #[pyo3(signature = (identifier, revision = None, token = None))]
    #[allow(deprecated)]
    fn from_pretrained(
        py: Python<'_>,
        identifier: &str,
        revision: Option<String>,
        token: Option<String>,
    ) -> PyResult<Self> {
        let mut params = FromPretrainedParameters::default();
        if let Some(revision) = revision {
            params.revision = revision;
        }
        params.token = token;
        py.detach(|| Tokenizer::from_pretrained(identifier, Some(params)))
            .map(Self::from_tokenizer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Encode text into token ids without added special tokens.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.detach(|| self.inner.encode(text, false))
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}
