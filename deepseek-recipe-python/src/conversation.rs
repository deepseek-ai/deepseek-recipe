//! Python bindings for the conversation types that prompt rendering consumes.

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource};
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_core::special_tokens::SpecialToken;
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

/// Resolve the response format of a constructed conversation.
///
/// `response_schema` selects `json_schema` and `response_regex` selects
/// `regex`; without either, `response_format` names `text` or `json_object`.
/// A schema or regex combined with the other, or with `json_object`, is
/// rejected.
fn resolve_response_format(
    response_format: &str,
    response_schema: Option<&Bound<'_, PyAny>>,
    response_regex: Option<String>,
) -> PyResult<ResponseFormat> {
    let format = parse_response_format(response_format)?;
    match (format, response_schema, response_regex) {
        (_, Some(_), Some(_)) => Err(PyValueError::new_err(
            "response_schema and response_regex cannot be combined",
        )),
        (ResponseFormat::Text, Some(schema), None) => {
            Ok(ResponseFormat::JsonSchema(python_to_json(schema)?))
        }
        (ResponseFormat::Text, None, Some(regex)) => Ok(ResponseFormat::Regex(regex)),
        (format, None, None) => Ok(format),
        (_, _, _) => Err(PyValueError::new_err(format!(
            "response_format {response_format:?} cannot be combined with \
             response_schema or response_regex"
        ))),
    }
}

/// Return the Python spelling of a boolean for `__repr__`.
fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
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

/// Return the string of an image detail level.
pub(crate) fn image_detail_str(detail: ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Original => "original",
        ImageDetail::Auto => "auto",
    }
}

/// Build the system, user, or tool message `Prompt` from text.
///
/// The first `image_count` occurrences of `image_token`, which defaults to the
/// default image token spelling, become explicit special tokens: the
/// placeholders of the message's image sources. Later occurrences and the rest
/// of the text are plain text. When `text` holds fewer than `image_count`
/// occurrences, every occurrence becomes a placeholder and no error is
/// reported. An empty `image_token` is rejected with `ValueError`.
fn prompt_with_image_placeholders(
    text: &str,
    image_count: usize,
    image_token: Option<&str>,
) -> PyResult<Prompt> {
    let placeholder = image_token.unwrap_or_else(|| SpecialToken::Image.default_str());
    if placeholder.is_empty() {
        return Err(PyValueError::new_err("image_token must not be empty"));
    }
    let mut content = Prompt::new();
    let mut cursor = 0;
    for _ in 0..image_count {
        let Some(start) = text[cursor..].find(placeholder).map(|at| cursor + at) else {
            break;
        };
        content.push_plain(&text[cursor..start]);
        content.push_special(placeholder);
        cursor = start + placeholder.len();
    }
    content.push_plain(&text[cursor..]);
    Ok(content)
}

/// Base class of the messages in a conversation.
///
/// Every message has a role and content. On a message of another role, the
/// role-specific properties have fixed values: `reasoning_content`,
/// `tool_calls`, and `tool_call_id` are `None`, `prefix` is `False`, and
/// `image_sources` is an empty list.
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

    /// Message text. For a system, user, or tool message, it includes the
    /// spellings of the image placeholders.
    #[getter]
    fn content(&self) -> &str {
        match &self.inner {
            InputMessage::System { content, .. }
            | InputMessage::User { content, .. }
            | InputMessage::Tool { content, .. } => content.as_str(),
            InputMessage::Assistant { content, .. } | InputMessage::LatestReminder { content } => {
                content
            }
        }
    }

    /// Whether an assistant message is continued instead of starting a new
    /// assistant turn.
    #[getter]
    fn prefix(&self) -> bool {
        matches!(self.inner, InputMessage::Assistant { prefix: true, .. })
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
            InputMessage::Tool { tool_call_id, .. } => tool_call_id.as_deref(),
            _ => None,
        }
    }

    /// Image sources of a system, user, or tool message.
    #[getter]
    fn image_sources(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let image_sources = match &self.inner {
            InputMessage::System { image_sources, .. }
            | InputMessage::User { image_sources, .. }
            | InputMessage::Tool { image_sources, .. } => image_sources.as_slice(),
            _ => &[],
        };
        image_sources
            .iter()
            .map(|source| image_source_to_py(py, source))
            .collect()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            InputMessage::System { content, .. } => {
                format!("SystemMessage({:?})", content.as_str())
            }
            InputMessage::User {
                content,
                image_sources,
            } => format!(
                "UserMessage({:?}, image_sources={})",
                content.as_str(),
                image_sources.len()
            ),
            InputMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
                prefix,
            } => format!(
                "AssistantMessage({content:?}, reasoning_content={}, tool_calls={}, prefix={})",
                py_bool(reasoning_content.is_some()),
                tool_calls.as_ref().map_or(0, Vec::len),
                py_bool(*prefix),
            ),
            InputMessage::Tool {
                content,
                image_sources,
                tool_call_id,
            } => format!(
                "ToolMessage({:?}, tool_call_id={tool_call_id:?}, image_sources={})",
                content.as_str(),
                image_sources.len()
            ),
            InputMessage::LatestReminder { content } => {
                format!("LatestReminderMessage({content:?})")
            }
        }
    }
}

/// A system message.
///
/// See `UserMessage` for image placeholder rules.
#[pyclass(
    name = "SystemMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PySystemMessage;

#[pymethods]
impl PySystemMessage {
    #[new]
    #[pyo3(signature = (content, image_sources=None, tools=None, *, image_token=None))]
    fn new(
        content: String,
        image_sources: Option<Vec<Py<PyImageSource>>>,
        tools: Option<Vec<PyRef<'_, PyToolDefinition>>>,
        image_token: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let image_sources = image_sources_from_py(image_sources);
        let content = prompt_with_image_placeholders(&content, image_sources.len(), image_token)?;
        Ok(message_initializer(
            Self,
            InputMessage::System {
                content,
                image_sources,
                tools: tools.map(|tools| tools.iter().map(|tool| tool.inner.clone()).collect()),
            },
        ))
    }

    /// Tool definitions introduced by this message. V4.1 renders them after
    /// the content; V4 ignores them.
    #[getter]
    fn tools(slf: PyRef<'_, Self>) -> Option<Vec<PyToolDefinition>> {
        match &slf.as_super().inner {
            InputMessage::System { tools, .. } => tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .cloned()
                    .map(|inner| PyToolDefinition { inner })
                    .collect()
            }),
            _ => None,
        }
    }
}

/// A user message.
///
/// The first occurrences of `image_token` in `content`, one per image source,
/// become the placeholders of the image sources; later occurrences are plain
/// text. `image_token` defaults to the default image token spelling; a caller
/// that renders with a `SpecialTokens` mapping whose image token differs
/// passes that mapping's spelling. When `content` holds fewer occurrences than
/// there are image sources, the message keeps every image source, so a
/// rendered prompt holds fewer placeholders than image sources; construction
/// does not report the mismatch. An empty `image_token` raises `ValueError`.
#[pyclass(
    name = "UserMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PyUserMessage;

#[pymethods]
impl PyUserMessage {
    #[new]
    #[pyo3(signature = (content, image_sources=None, *, image_token=None))]
    fn new(
        content: String,
        image_sources: Option<Vec<Py<PyImageSource>>>,
        image_token: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let image_sources = image_sources_from_py(image_sources);
        let content = prompt_with_image_placeholders(&content, image_sources.len(), image_token)?;
        Ok(message_initializer(
            Self,
            InputMessage::User {
                content,
                image_sources,
            },
        ))
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

#[pymethods]
impl PyAssistantMessage {
    #[new]
    #[pyo3(signature = (content, reasoning_content=None, tool_calls=None, prefix=false))]
    fn new(
        content: String,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<Py<PyToolCall>>>,
        prefix: bool,
    ) -> PyClassInitializer<Self> {
        let tool_calls = tool_calls.map(|tool_calls| {
            tool_calls
                .iter()
                .map(|tool_call| tool_call.get().inner.clone())
                .collect()
        });
        message_initializer(
            Self,
            InputMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
                prefix,
            },
        )
    }
}

/// A tool result message.
///
/// The first occurrences of `image_token` in `content`, one per image source,
/// become the placeholders of the image sources; later occurrences are plain
/// text. `image_token` defaults to the default image token spelling; a caller
/// that renders with a `SpecialTokens` mapping whose image token differs
/// passes that mapping's spelling. When `content` holds fewer occurrences than
/// there are image sources, the message keeps every image source, so a
/// rendered prompt holds fewer placeholders than image sources; construction
/// does not report the mismatch. An empty `image_token` raises `ValueError`.
#[pyclass(
    name = "ToolMessage",
    module = "deepseek_recipe._native",
    extends = PyMessage,
    frozen
)]
pub(crate) struct PyToolMessage;

#[pymethods]
impl PyToolMessage {
    #[new]
    #[pyo3(signature = (content, tool_call_id=None, image_sources=None, *, image_token=None))]
    fn new(
        content: String,
        tool_call_id: Option<String>,
        image_sources: Option<Vec<Py<PyImageSource>>>,
        image_token: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let image_sources = image_sources_from_py(image_sources);
        let content = prompt_with_image_placeholders(&content, image_sources.len(), image_token)?;
        Ok(message_initializer(
            Self,
            InputMessage::Tool {
                content,
                image_sources,
                tool_call_id,
            },
        ))
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

#[pymethods]
impl PyLatestReminderMessage {
    #[new]
    fn new(content: String) -> PyClassInitializer<Self> {
        message_initializer(Self, InputMessage::LatestReminder { content })
    }
}

/// Return the initializer of a message subclass wrapping `inner`.
fn message_initializer<T: pyo3::PyClass<BaseType = PyMessage>>(
    subclass: T,
    inner: InputMessage,
) -> PyClassInitializer<T> {
    PyClassInitializer::from(PyMessage { inner }).add_subclass(subclass)
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
    /// `strict` is passed through. Direct construction does not normalize
    /// `parameters`, unlike conversion, which normalizes the parameters of a
    /// tool with `strict: true`.
    #[new]
    #[pyo3(signature = (name, description=None, parameters=None, strict=None, defer_loading=false))]
    fn new(
        py: Python<'_>,
        name: String,
        description: Option<String>,
        parameters: Option<Py<PyAny>>,
        strict: Option<bool>,
        defer_loading: bool,
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
                defer_loading,
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

    /// The `strict` setting, passed through without normalizing `parameters`.
    #[getter]
    fn strict(&self) -> Option<bool> {
        self.inner.strict
    }

    /// Whether the initial tool prompt omits this definition.
    /// Loading it through a system message requires the V4.1 prompt encoding.
    #[getter]
    fn defer_loading(&self) -> bool {
        self.inner.defer_loading
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolDefinition(name={:?}, description={:?}, strict={:?})",
            self.inner.name, self.inner.description, self.inner.strict
        )
    }
}

/// Base class of the image sources supplied with a conversation.
///
/// Every source records a `path`: the request location of the image as error
/// messages report it, such as `messages.0.content[1]`. Conversion fills it;
/// a source constructed directly has an empty path unless the `path` keyword
/// supplies one.
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

    /// Request location of the image as error messages report it; empty when
    /// the source was constructed without one.
    #[getter]
    fn path(&self) -> &str {
        self.inner.path()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ImageSource::DataUrl {
                data_url,
                detail,
                path,
            } => format!(
                "DataUrlImageSource({data_url:?}, detail={:?}, path={path:?})",
                image_detail_str(*detail)
            ),
            ImageSource::Url { url, detail, path } => {
                format!(
                    "UrlImageSource({url:?}, detail={:?}, path={path:?})",
                    image_detail_str(*detail)
                )
            }
            ImageSource::Bytes { data, detail, path } => format!(
                "BytesImageSource(<{} bytes>, detail={:?}, path={path:?})",
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
    fn from_core(data_url: String, detail: ImageDetail, path: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyImageSource {
            inner: ImageSource::DataUrl {
                data_url,
                detail,
                path,
            },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyDataUrlImageSource {
    #[new]
    #[pyo3(signature = (data_url, detail="high", *, path=String::new()))]
    fn new(data_url: String, detail: &str, path: String) -> PyResult<PyClassInitializer<Self>> {
        Ok(Self::from_core(data_url, parse_image_detail(detail)?, path))
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
    fn from_core(url: String, detail: ImageDetail, path: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyImageSource {
            inner: ImageSource::Url { url, detail, path },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyUrlImageSource {
    #[new]
    #[pyo3(signature = (url, detail="high", *, path=String::new()))]
    fn new(url: String, detail: &str, path: String) -> PyResult<PyClassInitializer<Self>> {
        Ok(Self::from_core(url, parse_image_detail(detail)?, path))
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
    fn from_core(data: Vec<u8>, detail: ImageDetail, path: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyImageSource {
            inner: ImageSource::Bytes { data, detail, path },
        })
        .add_subclass(Self)
    }
}

#[pymethods]
impl PyBytesImageSource {
    #[new]
    #[pyo3(signature = (data, detail="high", *, path=String::new()))]
    fn new(data: Vec<u8>, detail: &str, path: String) -> PyResult<PyClassInitializer<Self>> {
        Ok(Self::from_core(data, parse_image_detail(detail)?, path))
    }
}

/// Construct the Python image source object for a core image source.
fn image_source_to_py(py: Python<'_>, source: &ImageSource) -> PyResult<Py<PyAny>> {
    Ok(match source {
        ImageSource::DataUrl {
            data_url,
            detail,
            path,
        } => Py::new(
            py,
            PyDataUrlImageSource::from_core(data_url.clone(), *detail, path.clone()),
        )?
        .into_any(),
        ImageSource::Url { url, detail, path } => Py::new(
            py,
            PyUrlImageSource::from_core(url.clone(), *detail, path.clone()),
        )?
        .into_any(),
        ImageSource::Bytes { data, detail, path } => Py::new(
            py,
            PyBytesImageSource::from_core(data.clone(), *detail, path.clone()),
        )?
        .into_any(),
    })
}

/// Construct the Python message object for a core message.
fn message_to_py(py: Python<'_>, message: &InputMessage) -> PyResult<Py<PyAny>> {
    let inner = message.clone();
    Ok(match message {
        InputMessage::System { .. } => {
            Py::new(py, message_initializer(PySystemMessage, inner))?.into_any()
        }
        InputMessage::User { .. } => {
            Py::new(py, message_initializer(PyUserMessage, inner))?.into_any()
        }
        InputMessage::Assistant { .. } => {
            Py::new(py, message_initializer(PyAssistantMessage, inner))?.into_any()
        }
        InputMessage::Tool { .. } => {
            Py::new(py, message_initializer(PyToolMessage, inner))?.into_any()
        }
        InputMessage::LatestReminder { .. } => {
            Py::new(py, message_initializer(PyLatestReminderMessage, inner))?.into_any()
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
    /// and the response format defaults to `text`. `response_format` accepts
    /// `text` or `json_object`; `response_schema` selects the `json_schema`
    /// format with that JSON Schema, and `response_regex` selects the `regex`
    /// format with that regular expression. Supplying both, or either with
    /// `json_object`, raises `ValueError`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        messages=None,
        *,
        thinking_mode=true,
        tools=None,
        tool_choice="auto",
        reasoning_effort=None,
        response_format="text",
        response_schema=None,
        response_regex=None
    ))]
    fn new(
        messages: Option<Vec<Py<PyMessage>>>,
        thinking_mode: bool,
        tools: Option<Vec<Py<PyToolDefinition>>>,
        tool_choice: &str,
        reasoning_effort: Option<&str>,
        response_format: &str,
        response_schema: Option<&Bound<'_, PyAny>>,
        response_regex: Option<String>,
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
                response_format: resolve_response_format(
                    response_format,
                    response_schema,
                    response_regex,
                )?,
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

    /// Requested answer format: `text`, `json_object`, `json_schema`, or
    /// `regex`.
    ///
    /// The constructor selects `json_schema` through `response_schema` and
    /// `regex` through `response_regex`. The caller enforces the format during
    /// inference; see `response_schema` and `response_regex`.
    #[getter]
    fn response_format(&self) -> &'static str {
        match self.inner.response_format {
            ResponseFormat::Text => "text",
            ResponseFormat::JsonObject => "json_object",
            ResponseFormat::JsonSchema(_) => "json_schema",
            ResponseFormat::Regex(_) => "regex",
        }
    }

    /// The JSON Schema of a `json_schema` response format, or `None`.
    #[getter]
    fn response_schema(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.response_format {
            ResponseFormat::JsonSchema(schema) => json_to_python(py, schema).map(Some),
            _ => Ok(None),
        }
    }

    /// The regular expression of a `regex` response format, or `None`.
    #[getter]
    fn response_regex(&self) -> Option<&str> {
        match &self.inner.response_format {
            ResponseFormat::Regex(regex) => Some(regex),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Conversation(messages={}, thinking_mode={}, tools={}, tool_choice={:?}, \
             reasoning_effort={:?}, response_format={:?})",
            self.inner.messages.len(),
            py_bool(self.inner.thinking_mode),
            self.inner.tools.len(),
            self.tool_choice(),
            self.reasoning_effort(),
            self.response_format(),
        )
    }
}

/// A rendered prompt and the image sources its placeholders refer to.
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

    /// Ascending, disjoint ranges of `prompt` that hold explicit special
    /// tokens, as half-open `(start, end)` code point offsets into `prompt`
    /// (Python `str` indices), so `prompt[start:end]` is the spelling of one
    /// token. The Rust `RenderedPrompt::special_ranges` field holds byte
    /// offsets instead.
    ///
    /// The text between the ranges is plain text, which prompt encoding
    /// encodes without the tokenizer's added vocabulary. `Tokenizer.encode`
    /// uses the full vocabulary, so it encodes a special-token spelling in
    /// plain text as that token's ID. `DeepseekV4Encoding.encode` and
    /// `DeepseekV41Encoding.encode` perform prompt encoding.
    #[getter]
    fn special_ranges(&self) -> Vec<(usize, usize)> {
        // Range boundaries ascend, so one pass over the prompt converts them.
        let mut boundaries = self
            .inner
            .special_ranges
            .iter()
            .flat_map(|range| [range.start, range.end])
            .peekable();
        let mut char_offsets = Vec::new();
        for (char_offset, (byte_offset, _)) in self
            .inner
            .prompt
            .char_indices()
            .chain([(self.inner.prompt.len(), ' ')])
            .enumerate()
        {
            while boundaries.next_if_eq(&byte_offset).is_some() {
                char_offsets.push(char_offset);
            }
        }
        char_offsets
            .chunks_exact(2)
            .map(|range| (range[0], range[1]))
            .collect()
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
