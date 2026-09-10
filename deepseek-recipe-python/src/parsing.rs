//! Python representations of Rust stream parsing options.

use deepseek_recipe::stream::state_machine::{ParsingOptions, ReasoningStage};
use pyo3::prelude::*;

/// Position within reasoning content when parsing begins.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(
    name = "ReasoningStage",
    module = "deepseek_recipe._native",
    eq,
    frozen
)]
pub(crate) enum PyReasoningStage {
    Start,
    Reasoning,
    Content,
}

impl PyReasoningStage {
    fn to_core(self) -> ReasoningStage {
        match self {
            Self::Start => ReasoningStage::Start,
            Self::Reasoning => ReasoningStage::Reasoning,
            Self::Content => ReasoningStage::Content,
        }
    }

    fn from_core(stage: ReasoningStage) -> Self {
        match stage {
            ReasoningStage::Start => Self::Start,
            ReasoningStage::Reasoning => Self::Reasoning,
            ReasoningStage::Content => Self::Content,
        }
    }
}

#[pymethods]
impl PyReasoningStage {
    /// Whether the initial stage contains reasoning content.
    fn start_from_reasoning(&self) -> bool {
        self.to_core().start_from_reasoning()
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
    #[pyo3(signature = (*, parse_tool_calls = true, tool_call_initial_stage = false,
        parse_json_output = false, reasoning_initial_stage = None, stop_sequences = None))]
    fn new(
        parse_tool_calls: bool,
        tool_call_initial_stage: bool,
        parse_json_output: bool,
        reasoning_initial_stage: Option<PyRef<'_, PyReasoningStage>>,
        stop_sequences: Option<Vec<String>>,
    ) -> Self {
        Self::from_core(ParsingOptions {
            parse_tool_calls,
            tool_call_initial_stage,
            parse_json_output,
            reasoning_initial_stage: reasoning_initial_stage.map(|stage| (*stage).to_core()),
            stop_sequences: stop_sequences.unwrap_or_default(),
        })
    }

    #[getter]
    fn parse_tool_calls(&self) -> bool {
        self.inner.parse_tool_calls
    }

    #[setter]
    fn set_parse_tool_calls(&mut self, value: bool) {
        self.inner.parse_tool_calls = value;
    }

    #[getter]
    fn tool_call_initial_stage(&self) -> bool {
        self.inner.tool_call_initial_stage
    }

    #[setter]
    fn set_tool_call_initial_stage(&mut self, value: bool) {
        self.inner.tool_call_initial_stage = value;
    }

    #[getter]
    fn parse_json_output(&self) -> bool {
        self.inner.parse_json_output
    }

    #[setter]
    fn set_parse_json_output(&mut self, value: bool) {
        self.inner.parse_json_output = value;
    }

    #[getter]
    fn reasoning_initial_stage(&self) -> Option<PyReasoningStage> {
        self.inner
            .reasoning_initial_stage
            .map(PyReasoningStage::from_core)
    }

    #[setter]
    fn set_reasoning_initial_stage(&mut self, value: Option<PyRef<'_, PyReasoningStage>>) {
        self.inner.reasoning_initial_stage = value.map(|stage| (*stage).to_core());
    }

    #[getter]
    fn stop_sequences(&self) -> Vec<String> {
        self.inner.stop_sequences.clone()
    }

    #[setter]
    fn set_stop_sequences(&mut self, value: Vec<String>) {
        self.inner.stop_sequences = value;
    }
}
