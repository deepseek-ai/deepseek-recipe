//! Bindings for the model-specific prompt encodings.

use std::sync::Arc;

use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::conversation::{PyConversation, PyRenderedPrompt};
use crate::special_tokens::PySpecialTokens;
use crate::tokenizer::PyTokenizer;

macro_rules! prompt_encoding {
    ($python:ident, $rust:ty, $name:literal) => {
        /// A model-specific prompt encoding: it renders a conversation and,
        /// with an attached tokenizer, encodes it into token IDs.
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

            /// Return a new prompt encoding that uses `tokenizer` for `encode`.
            fn with_tokenizer(&self, tokenizer: PyRef<'_, PyTokenizer>) -> Self {
                Self {
                    inner: self
                        .inner
                        .clone()
                        .with_tokenizer(PromptTokenizer::new(&tokenizer.inner)),
                }
            }

            /// Return a new prompt encoding that renders prompts with
            /// `special_tokens`.
            fn with_special_tokens(&self, special_tokens: PyRef<'_, PySpecialTokens>) -> Self {
                Self {
                    inner: self
                        .inner
                        .clone()
                        .with_special_tokens(Arc::clone(&special_tokens.inner)),
                }
            }

            /// Render `conversation` through this model's prompt template.
            fn render_conversation(
                &self,
                py: Python<'_>,
                conversation: &PyConversation,
            ) -> PyRenderedPrompt {
                let rendered = py.detach(|| self.inner.render_conversation(&conversation.inner));
                PyRenderedPrompt::from_core(rendered)
            }

            /// Encode a conversation into model token IDs using the attached
            /// tokenizer. Raises `RuntimeError` when no tokenizer is attached
            /// or encoding fails.
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
