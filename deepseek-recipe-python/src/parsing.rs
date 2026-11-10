//! Python representations of Rust stream parsing options.

use std::sync::Arc;

use deepseek_recipe::stream::{ParsingOptions, ThinkingInitialStage};
use pyo3::prelude::*;

use crate::special_tokens::PySpecialTokens;

/// The stage where parsing begins in thinking mode.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(
    name = "ThinkingInitialStage",
    module = "deepseek_recipe._native",
    eq,
    frozen
)]
pub(crate) enum PyThinkingInitialStage {
    /// Begin the reasoning content of a new assistant turn, discarding leading
    /// newlines.
    ReasoningStart,
    /// Continue the reasoning content of a prefix continuation, preserving
    /// leading newlines.
    ReasoningContinuation,
    /// Continue the answer content of a prefix continuation whose reasoning
    /// content has ended, preserving leading newlines.
    ContentContinuation,
}

impl PyThinkingInitialStage {
    fn to_core(self) -> ThinkingInitialStage {
        match self {
            Self::ReasoningStart => ThinkingInitialStage::ReasoningStart,
            Self::ReasoningContinuation => ThinkingInitialStage::ReasoningContinuation,
            Self::ContentContinuation => ThinkingInitialStage::ContentContinuation,
        }
    }

    fn from_core(stage: ThinkingInitialStage) -> Self {
        match stage {
            ThinkingInitialStage::ReasoningStart => Self::ReasoningStart,
            ThinkingInitialStage::ReasoningContinuation => Self::ReasoningContinuation,
            ThinkingInitialStage::ContentContinuation => Self::ContentContinuation,
        }
    }
}

#[pymethods]
impl PyThinkingInitialStage {
    /// Return whether parsing begins in reasoning content.
    fn starts_in_reasoning(&self) -> bool {
        self.to_core().starts_in_reasoning()
    }
}

/// Settings passed directly to the Rust stream parser.
#[pyclass(name = "ParsingOptions", module = "deepseek_recipe._native")]
pub(crate) struct PyParsingOptions {
    pub(crate) inner: ParsingOptions,
}

impl PyParsingOptions {
    pub(crate) fn from_core(inner: ParsingOptions) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyParsingOptions {
    #[new]
    #[pyo3(signature = (*, parse_tool_calls = true, starts_in_tool_calls = false,
        parse_json_output = false, thinking_initial_stage = None, stop_sequences = None,
        special_tokens = None))]
    fn new(
        parse_tool_calls: bool,
        starts_in_tool_calls: bool,
        parse_json_output: bool,
        thinking_initial_stage: Option<PyRef<'_, PyThinkingInitialStage>>,
        stop_sequences: Option<Vec<String>>,
        special_tokens: Option<PyRef<'_, PySpecialTokens>>,
    ) -> Self {
        Self::from_core(ParsingOptions {
            parse_tool_calls,
            starts_in_tool_calls,
            parse_json_output,
            thinking_initial_stage: thinking_initial_stage.map(|stage| (*stage).to_core()),
            stop_sequences: stop_sequences.unwrap_or_default(),
            special_tokens: special_tokens
                .map(|tokens| Arc::clone(&tokens.inner))
                .unwrap_or_else(deepseek_recipe_core::special_tokens::default_special_tokens),
        })
    }

    /// Whether to parse model-specific tool-call markup.
    #[getter]
    fn parse_tool_calls(&self) -> bool {
        self.inner.parse_tool_calls
    }

    #[setter]
    fn set_parse_tool_calls(&mut self, value: bool) {
        self.inner.parse_tool_calls = value;
    }

    /// Whether parsing begins inside the tool-call block that the prompt
    /// opened, as for a required tool choice. Takes precedence over
    /// `thinking_initial_stage`.
    #[getter]
    fn starts_in_tool_calls(&self) -> bool {
        self.inner.starts_in_tool_calls
    }

    #[setter]
    fn set_starts_in_tool_calls(&mut self, value: bool) {
        self.inner.starts_in_tool_calls = value;
    }

    /// Whether to recognize fenced or bare JSON output and replace the text
    /// around it with spaces. JSON syntax is not validated.
    #[getter]
    fn parse_json_output(&self) -> bool {
        self.inner.parse_json_output
    }

    #[setter]
    fn set_parse_json_output(&mut self, value: bool) {
        self.inner.parse_json_output = value;
    }

    /// The stage where parsing begins in thinking mode. `None` means thinking
    /// mode is off: parsing begins in answer content and discards leading
    /// newlines. A stage also makes the processor report reasoning tokens.
    #[getter]
    fn thinking_initial_stage(&self) -> Option<PyThinkingInitialStage> {
        self.inner
            .thinking_initial_stage
            .map(PyThinkingInitialStage::from_core)
    }

    #[setter]
    fn set_thinking_initial_stage(&mut self, value: Option<PyRef<'_, PyThinkingInitialStage>>) {
        self.inner.thinking_initial_stage = value.map(|stage| (*stage).to_core());
    }

    /// Stop sequences matched in answer content and JSON output.
    #[getter]
    fn stop_sequences(&self) -> Vec<String> {
        self.inner.stop_sequences.clone()
    }

    #[setter]
    fn set_stop_sequences(&mut self, value: Vec<String>) {
        self.inner.stop_sequences = value;
    }

    /// Special token mapping used to recognize the thinking-end token and
    /// tool-call markup.
    #[getter]
    fn special_tokens(&self) -> PySpecialTokens {
        PySpecialTokens::from_core(Arc::clone(&self.inner.special_tokens))
    }

    #[setter]
    fn set_special_tokens(&mut self, value: PyRef<'_, PySpecialTokens>) {
        self.inner.special_tokens = Arc::clone(&value.inner);
    }
}
