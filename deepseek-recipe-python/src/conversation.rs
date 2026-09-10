//! Python bindings for the conversation types that prompt rendering consumes.

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use deepseek_recipe_encoding::RenderedPrompt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::json::{json_to_python, python_to_json};

/// Parse a `tool_choice` string.
fn parse_tool_choice(value: &str) -> PyResult<ToolChoice> {
    match value {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "required" => Ok(ToolChoice::Required),
        other => Err(PyValueError::new_err(format!(
            "invalid tool_choice {other:?}; expected \"auto\", \"none\", or \"required\""
        ))),
    }
}

/// Parse a `reasoning_effort` string.
fn parse_reasoning_effort(value: &str) -> PyResult<ReasoningEffort> {
    match value {
        "low" => Ok(ReasoningEffort::Low),
        "high" => Ok(ReasoningEffort::High),
        "xhigh" => Ok(ReasoningEffort::Xhigh),
        "max" => Ok(ReasoningEffort::Max),
        other => Err(PyValueError::new_err(format!(
            "invalid reasoning_effort {other:?}; expected \"low\", \"high\", \"xhigh\", or \"max\""
        ))),
    }
}

/// Parse a `response_format` string.
fn parse_response_format(value: &str) -> PyResult<ResponseFormat> {
    match value {
        "text" => Ok(ResponseFormat::Text),
        "json_object" => Ok(ResponseFormat::JsonObject),
        other => Err(PyValueError::new_err(format!(
            "invalid response_format {other:?}; expected \"text\" or \"json_object\""
        ))),
    }
}

/// Parse an image `detail` string.
pub(crate) fn parse_image_detail(value: &str) -> PyResult<ImageDetail> {
    match value {
        "low" => Ok(ImageDetail::Low),
        "high" => Ok(ImageDetail::High),
        "original" => Ok(ImageDetail::Original),
        "auto" => Ok(ImageDetail::Auto),
        other => Err(PyValueError::new_err(format!(
            "invalid detail {other:?}; expected \"low\", \"high\", \"original\", or \"auto\""
        ))),
    }
}

/// Return the detail level string of an image source.
pub(crate) fn image_detail_str(detail: ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Original => "original",
        ImageDetail::Auto => "auto",
    }
}

/// Base class of the messages in a conversation.
///
/// Every message has a role and content. Fields that belong to one role are
/// `None` for the other roles.
#[pyclass(name = "Message", module = "deepseek_recipe._native", subclass, frozen)]
pub(crate) struct PyMessage {
    pub(crate) inner: InputMessage,
}

#[pymethods]
impl PyMessage {
    /// Role of this message: `system`, `user`, `assistant`, `tool`, or
    /// `latest_reminder`.
    #[getter]
    fn role(&self) -> &'static str {
        match &self.inner {
            InputMessage::System { .. } => "system",
            InputMessage::User { .. } => "user",
            InputMessage::Assistant { .. } => "assistant",
            InputMessage::Tool { .. } => "tool",
            InputMessage::LatestReminder { .. } => "latest_reminder",
        }
    }

    /// Message text.
    #[getter]
    fn content(&self) -> &str {
        match &self.inner {
            InputMessage::System { content }
            | InputMessage::User { content, .. }
            | InputMessage::Assistant { content, .. }
            | InputMessage::Tool { content, .. }
            | InputMessage::LatestReminder { content } => content,
        }
    }

    /// Reasoning content of an assistant message.
    #[getter]
    fn reasoning_content(&self) -> Option<&str> {
        match &self.inner {
            InputMessage::Assistant {
                reasoning_content, ..
            } => reasoning_content.as_deref(),
            _ => None,
        }
    }

    /// Tool calls of an assistant message.
    #[getter]
    fn tool_calls(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyToolCall>>>> {
        let tool_calls = match &self.inner {
            InputMessage::Assistant { tool_calls, .. } => tool_calls.as_deref(),
            _ => None,
        };
        tool_calls
            .map(|tool_calls| {
                tool_calls
                    .iter()
                    .map(|tool_call| Py::new(py, PyToolCall::from_core(tool_call.clone())))
                    .collect()
            })
            .transpose()
    }

    /// Call ID of a tool message.
    #[getter]
    fn tool_call_id(&self) -> Option<&str> {
        match &self.inner {
            InputMessage::Tool { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }

    /// Image sources of a user or tool message.
    #[getter]
    fn image_sources(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let image_sources = match &self.inner {
            InputMessage::User { image_sources, .. } | InputMessage::Tool { image_sources, .. } => {
                image_sources.as_slice()
            }
            _ => &[],
        };
        image_sources
            .iter()
            .map(|source| image_source_to_py(py, source))
            .collect()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            InputMessage::System { content } => format!("SystemMessage({content:?})"),
            InputMessage::User {
                content,
                image_sources,
            } => format!(
                "UserMessage({content:?}, image_sources={})",
                image_sources.len()
            ),
            InputMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => format!(
                "AssistantMessage({content:?}, reasoning_content={}, tool_calls={})",
                reasoning_content.is_some(),
                tool_calls.as_ref().map_or(0, Vec::len),
            ),
            InputMessage::Tool {
                content,
                image_sources,
                tool_call_id,
            } => format!(
                "ToolMessage({content:?}, tool_call_id={tool_call_id:?}, image_sources={})",
                image_sources.len()
            ),
            InputMessage::LatestReminder { content } => {
                format!("LatestReminderMessage({content:?})")
            }
        }
    }
}

/// A system message.
#[pyclass(
    name = "SystemMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PySystemMessage;

impl PySystemMessage {
    fn from_core(content: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMessage {
            inner: InputMessage::System { content },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PySystemMessage {
    #[new]
    fn new(content: String) -> PyClassInitializer<Self> {
        Self::from_core(content)
    }
}

/// A user message.
#[pyclass(
    name = "UserMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PyUserMessage;

impl PyUserMessage {
    fn from_core(content: String, image_sources: Vec<ImageSource>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMessage {
            inner: InputMessage::User {
                content,
                image_sources,
            },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyUserMessage {
    #[new]
    #[pyo3(signature = (content, image_sources=None))]
    fn new(
        content: String,
        image_sources: Option<Vec<Py<PyImageSource>>>,
    ) -> PyClassInitializer<Self> {
        Self::from_core(content, image_sources_from_py(image_sources))
    }
}

/// An assistant message.
#[pyclass(
    name = "AssistantMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PyAssistantMessage;

impl PyAssistantMessage {
    fn from_core(
        content: String,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMessage {
            inner: InputMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyAssistantMessage {
    #[new]
    #[pyo3(signature = (content, reasoning_content=None, tool_calls=None))]
    fn new(
        content: String,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<Py<PyToolCall>>>,
    ) -> PyClassInitializer<Self> {
        let tool_calls = tool_calls.map(|tool_calls| {
            tool_calls
                .iter()
                .map(|tool_call| tool_call.get().inner.clone())
                .collect()
        });
        Self::from_core(content, reasoning_content, tool_calls)
    }
}

/// A tool result message.
#[pyclass(
    name = "ToolMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PyToolMessage;

impl PyToolMessage {
    fn from_core(
        content: String,
        tool_call_id: String,
        image_sources: Vec<ImageSource>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMessage {
            inner: InputMessage::Tool {
                content,
                image_sources,
                tool_call_id,
            },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyToolMessage {
    #[new]
    #[pyo3(signature = (content, tool_call_id, image_sources=None))]
    fn new(
        content: String,
        tool_call_id: String,
        image_sources: Option<Vec<Py<PyImageSource>>>,
    ) -> PyClassInitializer<Self> {
        Self::from_core(content, tool_call_id, image_sources_from_py(image_sources))
    }
}

/// A reminder message that carries additional instructions.
#[pyclass(
    name = "LatestReminderMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PyLatestReminderMessage;

impl PyLatestReminderMessage {
    fn from_core(content: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMessage {
            inner: InputMessage::LatestReminder { content },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyLatestReminderMessage {
    #[new]
    fn new(content: String) -> PyClassInitializer<Self> {
        Self::from_core(content)
    }
}

/// Collect the core image sources of Python image source objects.
fn image_sources_from_py(image_sources: Option<Vec<Py<PyImageSource>>>) -> Vec<ImageSource> {
    image_sources
        .unwrap_or_default()
        .iter()
        .map(|source| source.get().inner.clone())
        .collect()
}

/// A historical tool call and its arguments.
#[pyclass(name = "ToolCall", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyToolCall {
    pub(crate) inner: ToolCall,
}

impl PyToolCall {
    fn from_core(inner: ToolCall) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyToolCall {
    #[new]
    fn new(id: String, name: String, arguments: String) -> Self {
        Self::from_core(ToolCall {
            id,
            name,
            arguments,
        })
    }

    /// Call ID that a tool result refers to.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Name of the called tool.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Caller-supplied arguments, expected to encode a JSON object.
    /// Construction does not validate the string.
    #[getter]
    fn arguments(&self) -> &str {
        &self.inner.arguments
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolCall(id={:?}, name={:?}, arguments={:?})",
            self.inner.id, self.inner.name, self.inner.arguments
        )
    }
}

/// A client tool definition.
#[pyclass(name = "ToolDefinition", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyToolDefinition {
    pub(crate) inner: ToolDefinition,
}

#[pymethods]
impl PyToolDefinition {
    /// Construct a tool definition.
    ///
    /// `parameters` is the input JSON Schema. Python `None` maps to JSON `null`.
    /// `strict` requests strict argument validation and is left to the caller.
    #[new]
    #[pyo3(signature = (name, description=None, parameters=None, strict=None))]
    fn new(
        py: Python<'_>,
        name: String,
        description: Option<String>,
        parameters: Option<Py<PyAny>>,
        strict: Option<bool>,
    ) -> PyResult<Self> {
        let parameters = match parameters {
            Some(parameters) => python_to_json(parameters.bind(py))?,
            None => Value::Null,
        };
        Ok(Self {
            inner: ToolDefinition {
                name,
                description,
                parameters,
                strict,
            },
        })
    }

    /// Tool name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Human-readable purpose of the tool.
    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    /// Input JSON Schema of the tool.
    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_python(py, &self.inner.parameters)
    }

    /// Requested strict argument validation.
    #[getter]
    fn strict(&self) -> Option<bool> {
        self.inner.strict
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolDefinition(name={:?}, description={:?}, strict={:?})",
            self.inner.name, self.inner.description, self.inner.strict
        )
    }
}

/// Base class of the images supplied with a conversation.
#[pyclass(
    name = "ImageSource",
    module = "deepseek_recipe._native",
    subclass,
    frozen
)]
pub(crate) struct PyImageSource {
    pub(crate) inner: ImageSource,
}

#[pymethods]
impl PyImageSource {
    /// Kind of image source: `data_url`, `url`, or `bytes`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            ImageSource::DataUrl { .. } => "data_url",
            ImageSource::Url { .. } => "url",
            ImageSource::Bytes { .. } => "bytes",
        }
    }

    /// Requested image detail level.
    #[getter]
    fn detail(&self) -> &'static str {
        image_detail_str(self.inner.detail())
    }

    /// Data URL of a data URL source, with a base64 or percent-encoded body.
    #[getter]
    fn data_url(&self) -> Option<&str> {
        match &self.inner {
            ImageSource::DataUrl { data_url, .. } => Some(data_url),
            _ => None,
        }
    }

    /// External URL of a URL source.
    #[getter]
    fn url(&self) -> Option<&str> {
        match &self.inner {
            ImageSource::Url { url, .. } => Some(url),
            _ => None,
        }
    }

    /// Encoded bytes of a bytes source.
    #[getter]
    fn data(&self) -> Option<&[u8]> {
        match &self.inner {
            ImageSource::Bytes { data, .. } => Some(data),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ImageSource::DataUrl { data_url, detail } => format!(
                "DataUrlImageSource({data_url:?}, detail={:?})",
                image_detail_str(*detail)
            ),
            ImageSource::Url { url, detail } => {
                format!(
                    "UrlImageSource({url:?}, detail={:?})",
                    image_detail_str(*detail)
                )
            }
            ImageSource::Bytes { data, detail } => format!(
                "BytesImageSource(<{} bytes>, detail={:?})",
                data.len(),
                image_detail_str(*detail)
            ),
        }
    }
}

/// An image referenced by a data URL with a base64 or percent-encoded body.
#[pyclass(
    name = "DataUrlImageSource",
    module = "deepseek_recipe._native",
    extends = PyImageSource,
    frozen
)]
pub(crate) struct PyDataUrlImageSource;

impl PyDataUrlImageSource {
    fn from_core(data_url: String, detail: ImageDetail) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyImageSource {
            inner: ImageSource::DataUrl { data_url, detail },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyDataUrlImageSource {
    #[new]
    #[pyo3(signature = (data_url, detail="high"))]
    fn new(data_url: String, detail: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(Self::from_core(data_url, parse_image_detail(detail)?))
    }
}

/// An image referenced by an external URL.
#[pyclass(
    name = "UrlImageSource",
    module = "deepseek_recipe._native",
    extends = PyImageSource,
    frozen
)]
pub(crate) struct PyUrlImageSource;

impl PyUrlImageSource {
    fn from_core(url: String, detail: ImageDetail) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyImageSource {
            inner: ImageSource::Url { url, detail },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyUrlImageSource {
    #[new]
    #[pyo3(signature = (url, detail="high"))]
    fn new(url: String, detail: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(Self::from_core(url, parse_image_detail(detail)?))
    }
}

/// An image supplied as encoded bytes.
#[pyclass(
    name = "BytesImageSource",
    module = "deepseek_recipe._native",
    extends = PyImageSource,
    frozen
)]
pub(crate) struct PyBytesImageSource;

impl PyBytesImageSource {
    fn from_core(data: Vec<u8>, detail: ImageDetail) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyImageSource {
            inner: ImageSource::Bytes { data, detail },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyBytesImageSource {
    #[new]
    #[pyo3(signature = (data, detail="high"))]
    fn new(data: Vec<u8>, detail: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(Self::from_core(data, parse_image_detail(detail)?))
    }
}

/// Construct the Python image source object for a core image source.
fn image_source_to_py(py: Python<'_>, source: &ImageSource) -> PyResult<Py<PyAny>> {
    Ok(match source {
        ImageSource::DataUrl { data_url, detail } => Py::new(
            py,
            PyDataUrlImageSource::from_core(data_url.clone(), *detail),
        )?
        .into_any(),
        ImageSource::Url { url, detail } => {
            Py::new(py, PyUrlImageSource::from_core(url.clone(), *detail))?.into_any()
        }
        ImageSource::Bytes { data, detail } => {
            Py::new(py, PyBytesImageSource::from_core(data.clone(), *detail))?.into_any()
        }
    })
}

/// Construct the Python message object for a core message.
fn message_to_py(py: Python<'_>, message: &InputMessage) -> PyResult<Py<PyAny>> {
    Ok(match message {
        InputMessage::System { content } => {
            Py::new(py, PySystemMessage::from_core(content.clone()))?.into_any()
        }
        InputMessage::User {
            content,
            image_sources,
        } => Py::new(
            py,
            PyUserMessage::from_core(content.clone(), image_sources.clone()),
        )?
        .into_any(),
        InputMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => Py::new(
            py,
            PyAssistantMessage::from_core(
                content.clone(),
                reasoning_content.clone(),
                tool_calls.clone(),
            ),
        )?
        .into_any(),
        InputMessage::Tool {
            content,
            image_sources,
            tool_call_id,
        } => Py::new(
            py,
            PyToolMessage::from_core(content.clone(), tool_call_id.clone(), image_sources.clone()),
        )?
        .into_any(),
        InputMessage::LatestReminder { content } => {
            Py::new(py, PyLatestReminderMessage::from_core(content.clone()))?.into_any()
        }
    })
}

/// A conversation and its prompt configuration.
#[pyclass(name = "Conversation", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyConversation {
    pub(crate) inner: Conversation,
}

impl PyConversation {
    pub(crate) fn from_core(inner: Conversation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyConversation {
    /// Construct a conversation.
    ///
    /// Thinking mode is enabled by default, tool selection defaults to `auto`,
    /// and the response format defaults to `text`.
    #[new]
    #[pyo3(signature = (
        messages=None,
        *,
        thinking_mode=true,
        tools=None,
        tool_choice="auto",
        reasoning_effort=None,
        response_format="text"
    ))]
    fn new(
        messages: Option<Vec<Py<PyMessage>>>,
        thinking_mode: bool,
        tools: Option<Vec<Py<PyToolDefinition>>>,
        tool_choice: &str,
        reasoning_effort: Option<&str>,
        response_format: &str,
    ) -> PyResult<Self> {
        let messages = messages
            .unwrap_or_default()
            .iter()
            .map(|message| message.get().inner.clone())
            .collect();
        let tools = tools
            .unwrap_or_default()
            .iter()
            .map(|tool| tool.get().inner.clone())
            .collect();
        let reasoning_effort = reasoning_effort.map(parse_reasoning_effort).transpose()?;
        Ok(Self {
            inner: Conversation {
                messages,
                thinking_mode,
                tools,
                tool_choice: parse_tool_choice(tool_choice)?,
                reasoning_effort,
                response_format: parse_response_format(response_format)?,
            },
        })
    }

    /// Messages in conversation order.
    #[getter]
    fn messages(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .messages
            .iter()
            .map(|message| message_to_py(py, message))
            .collect()
    }

    /// Whether thinking mode is enabled for the next assistant turn.
    #[getter]
    fn thinking_mode(&self) -> bool {
        self.inner.thinking_mode
    }

    /// Available client tools, in declaration order.
    #[getter]
    fn tools(&self, py: Python<'_>) -> PyResult<Vec<Py<PyToolDefinition>>> {
        self.inner
            .tools
            .iter()
            .map(|tool| {
                Py::new(
                    py,
                    PyToolDefinition {
                        inner: tool.clone(),
                    },
                )
            })
            .collect()
    }

    /// Tool selection: `auto`, `none`, or `required`.
    #[getter]
    fn tool_choice(&self) -> &'static str {
        match self.inner.tool_choice {
            ToolChoice::Auto => "auto",
            ToolChoice::None => "none",
            ToolChoice::Required => "required",
        }
    }

    /// Reasoning effort used when thinking mode is enabled.
    #[getter]
    fn reasoning_effort(&self) -> Option<&'static str> {
        self.inner.reasoning_effort.map(|effort| match effort {
            ReasoningEffort::Low => "low",
            ReasoningEffort::High => "high",
            ReasoningEffort::Xhigh => "xhigh",
            ReasoningEffort::Max => "max",
        })
    }

    /// Requested answer format: `text` or `json_object`.
    #[getter]
    fn response_format(&self) -> &'static str {
        match self.inner.response_format {
            ResponseFormat::Text => "text",
            ResponseFormat::JsonObject => "json_object",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Conversation(messages={}, thinking_mode={}, tools={}, tool_choice={:?}, \
             reasoning_effort={:?}, response_format={:?})",
            self.inner.messages.len(),
            self.inner.thinking_mode,
            self.inner.tools.len(),
            self.tool_choice(),
            self.reasoning_effort(),
            self.response_format(),
        )
    }
}

/// A rendered prompt and the images its placeholders refer to.
#[pyclass(name = "RenderedPrompt", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyRenderedPrompt {
    inner: RenderedPrompt,
}

impl PyRenderedPrompt {
    pub(crate) fn from_core(inner: RenderedPrompt) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRenderedPrompt {
    /// Model input string.
    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    /// Image sources in the order their placeholders appear in the prompt.
    #[getter]
    fn image_sources(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .image_sources
            .iter()
            .map(|source| image_source_to_py(py, source))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderedPrompt(prompt=<{} characters>, image_sources={})",
            self.inner.prompt.chars().count(),
            self.inner.image_sources.len(),
        )
    }
}
