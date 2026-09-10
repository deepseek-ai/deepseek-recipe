//! Python representations of Rust protocol requests and conversion results.

use std::collections::HashSet;

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{
    ConversationRequest, ConversionOptions, InferenceOptions, ProtocolRequest, WebSearchBehavior,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::conversation::PyConversation;
use crate::error::request_error;
use crate::json::python_to_json;
use crate::parsing::PyParsingOptions;
use crate::response::{
    PyChatCompletionChunkGenerator, PyMessagesChunkGenerator, PyResponsesChunkGenerator,
};

/// Rust conversion behavior for supported web-search declarations and content.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(
    name = "WebSearchBehavior",
    module = "deepseek_recipe._native",
    eq,
    frozen
)]
pub(crate) enum PyWebSearchBehavior {
    Ignore,
    Reject,
}

impl PyWebSearchBehavior {
    fn to_core(self) -> WebSearchBehavior {
        match self {
            Self::Ignore => WebSearchBehavior::Ignore,
            Self::Reject => WebSearchBehavior::Reject,
        }
    }

    fn from_core(behavior: WebSearchBehavior) -> PyResult<Self> {
        match behavior {
            WebSearchBehavior::Ignore => Ok(Self::Ignore),
            WebSearchBehavior::Reject => Ok(Self::Reject),
            other => Err(PyValueError::new_err(format!(
                "unsupported web search behavior: {other:?}"
            ))),
        }
    }
}

/// Defaults passed to Rust protocol request conversion.
#[pyclass(name = "ConversionOptions", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyConversionOptions {
    inner: ConversionOptions,
}

#[pymethods]
impl PyConversionOptions {
    /// Construct conversion defaults.
    ///
    /// Every argument defaults to the Rust default: thinking enabled, Responses
    /// `web_search` ignored, and Messages `web_search` rejected.
    #[new]
    #[pyo3(signature = (
        *,
        default_thinking_mode=true,
        responses_web_search=PyWebSearchBehavior::Ignore,
        messages_web_search=PyWebSearchBehavior::Reject
    ))]
    fn new(
        default_thinking_mode: bool,
        responses_web_search: PyWebSearchBehavior,
        messages_web_search: PyWebSearchBehavior,
    ) -> Self {
        Self {
            inner: ConversionOptions::new()
                .with_default_thinking_mode(default_thinking_mode)
                .with_responses_web_search(responses_web_search.to_core())
                .with_messages_web_search(messages_web_search.to_core()),
        }
    }

    #[getter]
    fn default_thinking_mode(&self) -> bool {
        self.inner.default_thinking_mode
    }

    fn with_default_thinking_mode(&self, default_thinking_mode: bool) -> Self {
        Self {
            inner: self.inner.with_default_thinking_mode(default_thinking_mode),
        }
    }

    /// Responses web-search behavior, defaulting to `Ignore`.
    #[getter]
    fn responses_web_search(&self) -> PyResult<PyWebSearchBehavior> {
        PyWebSearchBehavior::from_core(self.inner.responses_web_search)
    }

    /// Messages web-search behavior, defaulting to `Reject`.
    #[getter]
    fn messages_web_search(&self) -> PyResult<PyWebSearchBehavior> {
        PyWebSearchBehavior::from_core(self.inner.messages_web_search)
    }

    /// Construct options with the supplied Responses web-search behavior.
    fn with_responses_web_search(&self, responses_web_search: PyWebSearchBehavior) -> Self {
        Self {
            inner: self
                .inner
                .with_responses_web_search(responses_web_search.to_core()),
        }
    }

    /// Construct options with the supplied Messages web-search behavior.
    fn with_messages_web_search(&self, messages_web_search: PyWebSearchBehavior) -> Self {
        Self {
            inner: self
                .inner
                .with_messages_web_search(messages_web_search.to_core()),
        }
    }
}

/// Inference parameters extracted by Rust protocol conversion.
#[pyclass(
    name = "InferenceOptions",
    module = "deepseek_recipe._native",
    frozen,
    get_all
)]
pub(crate) struct PyInferenceOptions {
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    thinking_budget_tokens: Option<u64>,
    disable_parallel_tool_use: Option<bool>,
}

impl PyInferenceOptions {
    fn from_core(options: &InferenceOptions) -> Self {
        Self {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            thinking_budget_tokens: options.thinking_budget_tokens,
            disable_parallel_tool_use: options.disable_parallel_tool_use,
        }
    }
}

/// A Rust conversation request with inference and parsing options.
#[pyclass(
    name = "ConversationRequest",
    module = "deepseek_recipe._native",
    frozen
)]
pub(crate) struct PyConversationRequest {
    pub(crate) inner: ConversationRequest,
}

#[pymethods]
impl PyConversationRequest {
    #[new]
    fn new(conversation: &PyConversation) -> Self {
        Self {
            inner: ConversationRequest::new(conversation.inner.clone()),
        }
    }

    #[getter]
    fn conversation(&self) -> PyConversation {
        PyConversation::from_core(self.inner.conversation.clone())
    }

    #[getter]
    fn inference_options(&self) -> PyInferenceOptions {
        PyInferenceOptions::from_core(&self.inner.inference_options)
    }

    #[getter]
    fn parsing_options(&self) -> PyParsingOptions {
        PyParsingOptions::from_core(self.inner.parsing_options.clone())
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }
}

macro_rules! protocol_request {
    ($python:ident, $rust:ty, $name:literal, $generator:ty, $($extra:item)*) => {
        /// A Rust protocol request deserialized from a Python dictionary or JSON.
        ///
        /// Conversion consumes the request even if conversion fails. Keep caller
        /// metadata from the original body; `include_usage` is exposed separately
        /// for Chat Completions and must be read before calling `convert`.
        #[pyclass(name = $name, module = "deepseek_recipe._native")]
        pub(crate) struct $python {
            inner: Option<$rust>,
        }

        #[pymethods]
        impl $python {
            #[new]
            fn new(body: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self {
                    inner: Some(deserialize(body)?),
                })
            }

            fn convert(
                &mut self,
                py: Python<'_>,
                options: &PyConversionOptions,
            ) -> PyResult<PyConversationRequest> {
                let request = self.inner.take().ok_or_else(|| {
                    PyRuntimeError::new_err("protocol request has been consumed")
                })?;
                let options = options.inner;
                let inner = py
                    .detach(|| request.convert(options))
                    .map_err(|error| request_error(py, error))?;
                Ok(PyConversationRequest { inner })
            }

            #[staticmethod]
            fn chunk_generator(
                request: &PyConversationRequest,
                id: String,
                model: String,
            ) -> $generator {
                <$generator>::from_core(<$rust>::chunk_generator(&request.inner, id, model))
            }

            $($extra)*
        }
    };
}

protocol_request!(
    PyChatCompletionRequest,
    ChatCompletionRequest,
    "ChatCompletionRequest",
    PyChatCompletionChunkGenerator,
    fn include_usage(&self) -> PyResult<bool> {
        self.inner
            .as_ref()
            .map(ChatCompletionRequest::include_usage)
            .ok_or_else(|| PyRuntimeError::new_err("protocol request has been consumed"))
    }
);
protocol_request!(
    PyResponsesRequest,
    ResponsesRequest,
    "ResponsesRequest",
    PyResponsesChunkGenerator,
    fn custom_tool_names(&self) -> PyResult<HashSet<String>> {
        self.inner
            .as_ref()
            .map(ResponsesRequest::custom_tool_names)
            .ok_or_else(|| PyRuntimeError::new_err("protocol request has been consumed"))
    }
);
protocol_request!(
    PyMessagesRequest,
    MessagesRequest,
    "MessagesRequest",
    PyMessagesChunkGenerator,
);

/// Deserialize the Rust request using Python-to-JSON representation conversion.
fn deserialize<T: DeserializeOwned>(body: &Bound<'_, PyAny>) -> PyResult<T> {
    let value: Value = if let Ok(text) = body.cast::<PyString>() {
        serde_json::from_str(text.to_str()?)
            .map_err(|error| PyValueError::new_err(format!("invalid JSON request: {error}")))?
    } else if let Ok(bytes) = body.cast::<PyBytes>() {
        serde_json::from_slice(bytes.as_bytes())
            .map_err(|error| PyValueError::new_err(format!("invalid JSON request: {error}")))?
    } else {
        python_to_json(body)?
    };
    serde_json::from_value(value)
        .map_err(|error| PyValueError::new_err(format!("invalid request body: {error}")))
}
