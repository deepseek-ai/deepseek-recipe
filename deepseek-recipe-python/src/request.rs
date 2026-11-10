//! Python representations of Rust protocol requests and conversion results.

use std::collections::HashMap;
use std::sync::Arc;

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{
    ConversationRequest, ConversionOptions, DEFAULT_MAX_TOKENS_UPPER_BOUND, InferenceOptions,
    ModelVersion, OutputConstraint, ProtocolRequest, ReasoningContentBehavior, WebSearchBehavior,
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
    PyResponsesEcho,
};
use crate::special_tokens::PySpecialTokens;

/// Rust conversion behavior for supported web-search declarations and content.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(
    name = "WebSearchBehavior",
    module = "deepseek_recipe._native",
    eq,
    frozen
)]
pub(crate) enum PyWebSearchBehavior {
    Drop,
    Reject,
}

impl PyWebSearchBehavior {
    fn to_core(self) -> WebSearchBehavior {
        match self {
            Self::Drop => WebSearchBehavior::Drop,
            Self::Reject => WebSearchBehavior::Reject,
        }
    }

    fn from_core(behavior: WebSearchBehavior) -> PyResult<Self> {
        match behavior {
            WebSearchBehavior::Drop => Ok(Self::Drop),
            WebSearchBehavior::Reject => Ok(Self::Reject),
            other => Err(PyValueError::new_err(format!(
                "unsupported web search behavior: {other:?}"
            ))),
        }
    }
}

/// Handling of assistant history missing `reasoning_content` in thinking mode.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(
    name = "ReasoningContentBehavior",
    module = "deepseek_recipe._native",
    eq,
    frozen
)]
pub(crate) enum PyReasoningContentBehavior {
    FillEmpty,
    Reject,
}

impl PyReasoningContentBehavior {
    fn to_core(self) -> ReasoningContentBehavior {
        match self {
            Self::FillEmpty => ReasoningContentBehavior::FillEmpty,
            Self::Reject => ReasoningContentBehavior::Reject,
        }
    }

    fn from_core(behavior: ReasoningContentBehavior) -> PyResult<Self> {
        match behavior {
            ReasoningContentBehavior::FillEmpty => Ok(Self::FillEmpty),
            ReasoningContentBehavior::Reject => Ok(Self::Reject),
            other => Err(PyValueError::new_err(format!(
                "unsupported reasoning content behavior: {other:?}"
            ))),
        }
    }
}

/// The model version a converted conversation targets.
///
/// Conversion resolves it from the request's model name through
/// `ConversionOptions.model_versions_by_name`, and it selects the prompt
/// encoding: `DeepseekV4Encoding` or `DeepseekV41Encoding`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(
    name = "ModelVersion",
    module = "deepseek_recipe._native",
    eq,
    frozen,
    hash
)]
pub(crate) enum PyModelVersion {
    /// DeepSeek V4 Pro, rendered by `DeepseekV4Encoding`.
    DeepseekV4Pro,
    /// DeepSeek V4.1 Flash, rendered by `DeepseekV41Encoding`.
    DeepseekV41Flash,
}

impl PyModelVersion {
    fn to_core(self) -> ModelVersion {
        match self {
            Self::DeepseekV4Pro => ModelVersion::DeepseekV4Pro,
            Self::DeepseekV41Flash => ModelVersion::DeepseekV41Flash,
        }
    }

    fn from_core(version: ModelVersion) -> PyResult<Self> {
        match version {
            ModelVersion::DeepseekV4Pro => Ok(Self::DeepseekV4Pro),
            ModelVersion::DeepseekV41Flash => Ok(Self::DeepseekV41Flash),
            other => Err(PyValueError::new_err(format!(
                "unsupported model version: {other:?}"
            ))),
        }
    }
}

/// Options for protocol request conversion.
#[pyclass(name = "ConversionOptions", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyConversionOptions {
    inner: ConversionOptions,
}

#[pymethods]
impl PyConversionOptions {
    /// Construct conversion options.
    ///
    /// Every argument defaults to the Rust default: thinking mode enabled,
    /// missing reasoning content filled with an empty string, Responses
    /// `web_search` dropped, Messages `web_search` rejected, and a `max_tokens`
    /// upper bound of 393216 in thinking mode and outside it. Conversion
    /// resolves the model version from the request's model name; the accepted
    /// names default to `DeepSeek-V4-Pro` and `DeepSeek-V4.1-Flash`.
    #[new]
    #[pyo3(signature = (
        *,
        default_thinking_mode=true,
        reasoning_content_behavior=PyReasoningContentBehavior::FillEmpty,
        responses_web_search=PyWebSearchBehavior::Drop,
        messages_web_search=PyWebSearchBehavior::Reject,
        thinking_max_tokens_upper_bound=DEFAULT_MAX_TOKENS_UPPER_BOUND,
        non_thinking_max_tokens_upper_bound=DEFAULT_MAX_TOKENS_UPPER_BOUND
    ))]
    fn new(
        default_thinking_mode: bool,
        reasoning_content_behavior: PyReasoningContentBehavior,
        responses_web_search: PyWebSearchBehavior,
        messages_web_search: PyWebSearchBehavior,
        thinking_max_tokens_upper_bound: u32,
        non_thinking_max_tokens_upper_bound: u32,
    ) -> Self {
        Self {
            inner: ConversionOptions::new()
                .with_default_thinking_mode(default_thinking_mode)
                .with_reasoning_content_behavior(reasoning_content_behavior.to_core())
                .with_responses_web_search(responses_web_search.to_core())
                .with_messages_web_search(messages_web_search.to_core())
                .with_thinking_max_tokens_upper_bound(thinking_max_tokens_upper_bound)
                .with_non_thinking_max_tokens_upper_bound(non_thinking_max_tokens_upper_bound),
        }
    }

    /// The target model version of each accepted request model name. The
    /// table defaults to `DeepSeek-V4-Pro` and `DeepSeek-V4.1-Flash`;
    /// conversion rejects a request whose model is not a key.
    #[getter]
    fn model_versions_by_name(&self) -> PyResult<HashMap<String, PyModelVersion>> {
        self.inner
            .model_versions_by_name
            .iter()
            .map(|(model, version)| Ok((model.clone(), PyModelVersion::from_core(*version)?)))
            .collect()
    }

    /// Accept `model` as a name of `model_version`.
    ///
    /// Adds the name, or replaces the version of an existing one.
    fn with_model_alias(&self, model: &str, model_version: PyModelVersion) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_model_alias(model, model_version.to_core()),
        }
    }

    /// Replace the whole table of accepted model names.
    ///
    /// Drops the default names unless the mapping includes them.
    fn with_model_versions_by_name(
        &self,
        model_versions_by_name: HashMap<String, PyModelVersion>,
    ) -> Self {
        Self {
            inner: self.inner.clone().with_model_versions_by_name(
                model_versions_by_name
                    .into_iter()
                    .map(|(model, model_version)| (model, model_version.to_core())),
            ),
        }
    }

    /// The largest accepted `max_tokens` (`max_output_tokens` in Responses) of
    /// a request in thinking mode. Chat Completions and Responses reject a
    /// larger value; Messages lowers it to this bound.
    #[getter]
    fn thinking_max_tokens_upper_bound(&self) -> u32 {
        self.inner.thinking_max_tokens_upper_bound
    }

    /// Construct options with the supplied `max_tokens` upper bound for
    /// thinking mode.
    fn with_thinking_max_tokens_upper_bound(&self, thinking_max_tokens_upper_bound: u32) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_thinking_max_tokens_upper_bound(thinking_max_tokens_upper_bound),
        }
    }

    /// The largest accepted `max_tokens` (`max_output_tokens` in Responses) of
    /// a request outside thinking mode. Chat Completions and Responses reject a
    /// larger value; Messages lowers it to this bound.
    #[getter]
    fn non_thinking_max_tokens_upper_bound(&self) -> u32 {
        self.inner.non_thinking_max_tokens_upper_bound
    }

    /// Construct options with the supplied `max_tokens` upper bound for
    /// requests outside thinking mode.
    fn with_non_thinking_max_tokens_upper_bound(
        &self,
        non_thinking_max_tokens_upper_bound: u32,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_non_thinking_max_tokens_upper_bound(non_thinking_max_tokens_upper_bound),
        }
    }

    /// Whether requests that do not specify a thinking mode use thinking mode.
    /// Defaults to `True`.
    #[getter]
    fn default_thinking_mode(&self) -> bool {
        self.inner.default_thinking_mode
    }

    /// Construct options with the supplied default thinking mode.
    fn with_default_thinking_mode(&self, default_thinking_mode: bool) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_default_thinking_mode(default_thinking_mode),
        }
    }

    /// Missing-reasoning-content behavior, defaulting to `FillEmpty`.
    #[getter]
    fn reasoning_content_behavior(&self) -> PyResult<PyReasoningContentBehavior> {
        PyReasoningContentBehavior::from_core(self.inner.reasoning_content_behavior)
    }

    /// Construct options with the supplied missing-reasoning-content behavior.
    fn with_reasoning_content_behavior(
        &self,
        reasoning_content_behavior: PyReasoningContentBehavior,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_reasoning_content_behavior(reasoning_content_behavior.to_core()),
        }
    }

    /// Responses web-search behavior, defaulting to `Drop`.
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
                .clone()
                .with_responses_web_search(responses_web_search.to_core()),
        }
    }

    /// Construct options with the supplied Messages web-search behavior.
    fn with_messages_web_search(&self, messages_web_search: PyWebSearchBehavior) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_messages_web_search(messages_web_search.to_core()),
        }
    }

    /// Special token mapping, defaulting to the built-in mapping.
    #[getter]
    fn special_tokens(&self) -> PySpecialTokens {
        PySpecialTokens::from_core(Arc::clone(&self.inner.special_tokens))
    }

    /// Construct options with the supplied special token mapping.
    fn with_special_tokens(&self, special_tokens: PyRef<'_, PySpecialTokens>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_special_tokens(Arc::clone(&special_tokens.inner)),
        }
    }
}

/// Inference parameters extracted by protocol conversion.
///
/// Conversion always sets `temperature` and `top_p` to their resolved values.
/// Other unspecified parameters are `None`;
/// `ConversationRequest.resolve_max_tokens` supplies the default `max_tokens`,
/// and the caller applies the other model-specific defaults.
#[pyclass(
    name = "InferenceOptions",
    module = "deepseek_recipe._native",
    frozen,
    get_all
)]
pub(crate) struct PyInferenceOptions {
    /// Maximum number of output tokens, or `None` when the request sets none;
    /// `ConversationRequest.resolve_max_tokens` supplies the default after the
    /// prompt is encoded.
    max_tokens: Option<u32>,
    /// Sampling temperature: 1.0 in thinking mode, otherwise the requested
    /// value or 1.0.
    temperature: Option<f32>,
    /// Final `top_p` for the target model version.
    top_p: Option<f32>,
    /// Whether the request asks for the log probability of each output token.
    logprobs: bool,
    /// Number of most likely alternatives reported per output token, in
    /// `[0, 20]`.
    top_logprobs: Option<u8>,
    /// Sampling seed in `[0, 2^63)`. The caller chooses an unspecified seed.
    seed: Option<u64>,
    /// The constraint that constrained decoding would enforce on the output,
    /// or `None`; whether the backend applies it is the caller's decision.
    output_constraint: Option<PyOutputConstraint>,
}

impl PyInferenceOptions {
    fn from_core(options: &InferenceOptions) -> Self {
        Self {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            logprobs: options.logprobs,
            top_logprobs: options.top_logprobs,
            seed: options.seed,
            output_constraint: options
                .output_constraint
                .as_ref()
                .map(PyOutputConstraint::from_core),
        }
    }
}

/// A constraint on the model output that the backend can enforce by
/// constrained decoding, as a kind name and a constraint string.
#[derive(Clone)]
#[pyclass(
    name = "OutputConstraint",
    module = "deepseek_recipe._native",
    frozen,
    get_all
)]
pub(crate) struct PyOutputConstraint {
    /// The constraint kind: `json`, `regex`, `tool_call`, or
    /// `tool_call_required`.
    kind: String,
    /// The constraint string: the schema as compact JSON, the regular
    /// expression, the tool list as a compact JSON array of Chat Completions
    /// function tools with `defer_loading: true` on deferred tools, or an
    /// empty string for any JSON value.
    schema: String,
}

impl PyOutputConstraint {
    fn from_core(constraint: &OutputConstraint) -> Self {
        Self {
            kind: constraint.kind().to_string(),
            schema: constraint.schema(),
        }
    }
}

#[pymethods]
impl PyOutputConstraint {
    fn __repr__(&self) -> String {
        format!(
            "OutputConstraint(kind={:?}, schema={:?})",
            self.kind, self.schema
        )
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
    /// Construct a request with default inference, parsing, and transport
    /// options.
    #[new]
    fn new(conversation: &PyConversation, model_version: PyModelVersion) -> Self {
        Self {
            inner: ConversationRequest::new(conversation.inner.clone(), model_version.to_core()),
        }
    }

    /// The conversation to render.
    #[getter]
    fn conversation(&self) -> PyConversation {
        PyConversation::from_core(self.inner.conversation.clone())
    }

    /// Inference parameters for the backend.
    #[getter]
    fn inference_options(&self) -> PyInferenceOptions {
        PyInferenceOptions::from_core(&self.inner.inference_options)
    }

    /// Settings for parsing the model output.
    #[getter]
    fn parsing_options(&self) -> PyParsingOptions {
        PyParsingOptions::from_core(self.inner.parsing_options.clone())
    }

    /// The target model version: resolved by conversion from the request's
    /// model name, or supplied to the constructor.
    #[getter]
    fn model_version(&self) -> PyResult<PyModelVersion> {
        PyModelVersion::from_core(self.inner.model_version)
    }

    /// Whether the protocol request asks for a streamed response. `False` for
    /// a request constructed directly.
    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }

    /// Resolve the `max_tokens` to send to the backend.
    ///
    /// `prompt_tokens` is the number of tokens in the encoded prompt, with each
    /// image counted at its image token count instead of as one placeholder,
    /// and `context_limit` is the number of tokens the model's context holds.
    /// A requested `max_tokens` is kept; an unspecified one becomes 8192
    /// outside thinking mode, 131072 in thinking mode with `max` effort, and
    /// 65536 in thinking mode otherwise. The result is lowered to the context
    /// left after the prompt and raised to at least 1. Raises `ConversionError`
    /// when `prompt_tokens` plus the requested `max_tokens` exceeds
    /// `context_limit`, or, without a requested `max_tokens`, when
    /// `prompt_tokens` alone exceeds it.
    fn resolve_max_tokens(
        &self,
        py: Python<'_>,
        prompt_tokens: u32,
        context_limit: u32,
    ) -> PyResult<u32> {
        self.inner
            .resolve_max_tokens(prompt_tokens, context_limit)
            .map_err(|error| request_error(py, error))
    }
}

macro_rules! protocol_request {
    ($python:ident, $rust:ty, $name:literal, $generator:ty, { $($extra:item)* }) => {
        /// A Rust protocol request deserialized from a Python dictionary or JSON.
        ///
        /// Conversion consumes the request even if conversion fails.
        #[pyclass(name = $name, module = "deepseek_recipe._native")]
        pub(crate) struct $python {
            inner: Option<$rust>,
        }

        #[pymethods]
        impl $python {
            /// Deserialize `body`: a `dict`, a JSON `str`, or JSON `bytes`.
            /// Raises `ValueError` for invalid JSON or a body that does not
            /// match the protocol request format.
            #[new]
            fn new(body: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self {
                    inner: Some(deserialize(body)?),
                })
            }

            /// The requested model name. The caller must read it before calling
            /// `convert`, which consumes the request.
            #[getter]
            fn model(&self) -> PyResult<&str> {
                self.inner
                    .as_ref()
                    .map(|request| request.model.as_str())
                    .ok_or_else(|| PyRuntimeError::new_err("protocol request has been consumed"))
            }

            /// Convert this request with `options`, consuming it even on
            /// failure. Raises `ConversionError` when conversion rejects the
            /// request, and `RuntimeError` when the request was already
            /// consumed.
            fn convert(
                &mut self,
                py: Python<'_>,
                options: &PyConversionOptions,
            ) -> PyResult<PyConversationRequest> {
                let request = self.inner.take().ok_or_else(|| {
                    PyRuntimeError::new_err("protocol request has been consumed")
                })?;
                let options = options.inner.clone();
                let inner = py
                    .detach(|| request.convert(options))
                    .map_err(|error| request_error(py, error))?;
                Ok(PyConversationRequest { inner })
            }

            /// Construct this protocol's chunk generator from the resolved
            /// settings of `request`. `id` identifies the response, and
            /// `model` is the model name that the response reports.
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
    {}
);
protocol_request!(
    PyResponsesRequest,
    ResponsesRequest,
    "ResponsesRequest",
    PyResponsesChunkGenerator,
    {
        /// Return the request settings that response objects report.
        /// Conversion preserves them for the chunk generator. Raises
        /// `RuntimeError` after `convert`.
        fn echo(&self) -> PyResult<PyResponsesEcho> {
            self.inner
                .as_ref()
                .map(|request| PyResponsesEcho {
                    inner: request.echo(),
                })
                .ok_or_else(|| PyRuntimeError::new_err("protocol request has been consumed"))
        }

        /// Return the function names of the declared namespace tools mapped
        /// to `(namespace, local name)`, as conversion with `options` assigns
        /// them. Conversion supplies the mapping to the chunk generator.
        /// Raises `ConversionError` for an invalid tool declaration and
        /// `RuntimeError` after `convert`.
        fn namespace_tool_map(
            &self,
            py: Python<'_>,
            options: &PyConversionOptions,
        ) -> PyResult<HashMap<String, (String, String)>> {
            let request = self
                .inner
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("protocol request has been consumed"))?;
            request
                .namespace_tool_map(&options.inner)
                .map_err(|error| request_error(py, error))
        }
    }
);
protocol_request!(
    PyMessagesRequest,
    MessagesRequest,
    "MessagesRequest",
    PyMessagesChunkGenerator,
    {}
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
