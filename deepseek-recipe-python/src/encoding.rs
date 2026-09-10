//! Direct bindings for the Rust model-specific prompt renderers.

use std::sync::Arc;

use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::conversation::{PyConversation, PyRenderedPrompt};
use crate::tokenizer::PyTokenizer;

macro_rules! prompt_encoding {
    ($python:ident, $rust:ty, $name:literal) => {
        /// A Rust prompt renderer. Token encoding requires an attached tokenizer.
        #[pyclass(name = $name, module = "deepseek_recipe._native", frozen)]
        pub(crate) struct $python {
            inner: $rust,
        }

        #[pymethods]
        impl $python {
            #[new]
            fn new() -> Self {
                Self {
                    inner: <$rust>::new(),
                }
            }

            /// Attach a tokenizer for `encode`, returning a new encoding.
            fn with_tokenizer(&self, tokenizer: PyRef<'_, PyTokenizer>) -> Self {
                Self {
                    inner: <$rust>::new().with_tokenizer(Arc::clone(&tokenizer.inner)),
                }
            }

            /// Render a conversation through the Rust model's prompt template.
            fn render_conversation(
                &self,
                py: Python<'_>,
                conversation: &PyConversation,
            ) -> PyRenderedPrompt {
                let rendered = py.detach(|| self.inner.render_conversation(&conversation.inner));
                PyRenderedPrompt::from_core(rendered)
            }

            /// Encode a conversation into model token IDs using the attached tokenizer.
            fn encode(&self, py: Python<'_>, conversation: &PyConversation) -> PyResult<Vec<u32>> {
                py.detach(|| self.inner.encode(&conversation.inner))
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
            }
        }
    };
}

prompt_encoding!(
    PyDeepseekV4Encoding,
    DeepseekV4Encoding,
    "DeepseekV4Encoding"
);
prompt_encoding!(
    PyDeepseekV41Encoding,
    DeepseekV41Encoding,
    "DeepseekV41Encoding"
);
