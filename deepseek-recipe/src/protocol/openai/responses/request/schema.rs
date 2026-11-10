use deepseek_recipe_core::multimodal::ImageDetail;
use deepseek_recipe_macros::UntaggedDeserialize;
use serde::{Deserialize, Serialize, Serializer};

pub use crate::protocol::reasoning::RequestReasoningEffort;

/// A Responses request with text and image input and client tools.
///
/// Unknown top-level fields, including `include`, are ignored. Conversion
/// resolves the model version through
/// [`ConversionOptions::model_versions_by_name`]; inference execution, tool
/// execution, server tools, and encrypted reasoning recovery belong to the
/// caller.
///
/// [`ConversionOptions::model_versions_by_name`]:
///     crate::request::ConversionOptions::model_versions_by_name
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Option<ResponsesInput>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<ResponsesReasoningConfig>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub text: Option<ResponsesTextConfig>,
    pub tool_choice: Option<ResponsesToolChoice>,
    pub tools: Option<Vec<ResponsesTool>>,
    /// Number of most likely alternatives per output token, in [0, 20]. Any
    /// value enables output token log probabilities.
    pub top_logprobs: Option<u8>,
    pub top_p: Option<f32>,
    /// Ignored by conversion; [`ResponsesRequest::echo`] reports it.
    pub user: Option<String>,
}

#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "input should be a string or a list of input items")]
pub enum ResponsesInput {
    #[untagged_deserialize(pattern = String(_))]
    String(String),
    #[untagged_deserialize(pattern = Array(_))]
    Items(Vec<ResponsesInputItem>),
}

/// An explicitly typed item, or a message without a `type` field.
#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "invalid input item")]
pub enum ResponsesInputItem {
    #[untagged_deserialize(field = "type")]
    Typed(ResponsesTypedInputItem),
    #[untagged_deserialize(field = "role")]
    Message(ResponsesInputMessage),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTypedInputItem {
    Message(ResponsesInputMessage),
    FunctionCall(ResponsesFunctionCall),
    FunctionCallOutput(ResponsesFunctionCallOutput),
    Reasoning(ResponsesReasoningItem),
    CustomToolCall(ResponsesCustomToolCall),
    CustomToolCallOutput(ResponsesCustomToolCallOutput),
    ToolSearchCall(ResponsesToolSearchCall),
    ToolSearchOutput(ResponsesToolSearchOutput),
    /// A historical web search call. Conversion drops it and rejects the
    /// request when an earlier tool call has no output before it.
    WebSearchCall,
    /// Other unsupported item types are ignored.
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesInputMessage {
    pub role: ResponsesMessageRole,
    pub content: ResponsesMessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesMessageRole {
    User,
    Assistant,
    System,
    Developer,
}

#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "content should be a string or a list of content blocks")]
pub enum ResponsesMessageContent {
    #[untagged_deserialize(pattern = String(_))]
    String(String),
    #[untagged_deserialize(pattern = Array(_))]
    List(Vec<ResponsesContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentBlock {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    InputImage(ResponsesInputImage),
    /// Conversion writes the unsupported-document text in place of a
    /// document. Every field is ignored.
    InputFile {},
}

/// One image referenced by URL or by a file identifier. A file identifier
/// requires a file service and is rejected during conversion.
#[derive(Debug, Deserialize)]
pub struct ResponsesInputImage {
    /// HTTP URL or base64 data URL. An empty string is treated as absent, and a
    /// value that does not start with `http` is a data URL, which user,
    /// developer, and tool content require to carry a `base64` body.
    pub image_url: Option<String>,
    #[serde(default)]
    pub detail: Option<ImageDetail>,
    pub file_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesFunctionCall {
    pub arguments: String,
    pub call_id: String,
    pub name: String,
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesFunctionCallOutput {
    /// The answered call. An output without a call ID becomes a tool result
    /// that answers no call.
    pub call_id: Option<String>,
    pub output: ResponsesToolCallOutputContent,
}

#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "output should be a string or a list of output content parts")]
pub enum ResponsesToolCallOutputContent {
    #[untagged_deserialize(pattern = String(_))]
    String(String),
    #[untagged_deserialize(pattern = Array(_))]
    List(Vec<ResponsesToolCallOutputItem>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesToolCallOutputItem {
    InputText {
        text: String,
    },
    InputImage(ResponsesInputImage),
    /// Conversion writes the unsupported-document text in place of a
    /// document. Every field is ignored.
    InputFile {},
}

/// A historical reasoning item. Conversion joins the `text` of its `content`
/// entries with newlines into the reasoning content of an assistant message;
/// `summary` and other unlisted fields are ignored.
#[derive(Debug, Deserialize)]
pub struct ResponsesReasoningItem {
    pub content: Option<Vec<ResponsesReasoningContent>>,
    /// Validated during deserialization and ignored by conversion.
    pub encrypted_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesReasoningContent {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesCustomToolCall {
    pub call_id: String,
    pub name: String,
    pub input: String,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesCustomToolCallOutput {
    pub call_id: String,
    pub output: Option<ResponsesToolCallOutputContent>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesToolSearchCall {
    pub call_id: String,
    pub execution: Option<ResponsesToolSearchExecution>,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesToolSearchOutput {
    pub call_id: String,
    pub execution: Option<ResponsesToolSearchExecution>,
    pub tools: Vec<ResponsesTool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesToolSearchExecution {
    Client,
    Server,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResponsesReasoningConfig {
    /// Requested reasoning effort. `none` disables thinking.
    pub effort: Option<RequestReasoningEffort>,
    /// The requested reasoning summary. Ignored by conversion;
    /// [`ResponsesRequest::echo`] reports it.
    pub summary: Option<ResponsesReasoningSummary>,
    /// Another field for the requested reasoning summary. Ignored by
    /// conversion; [`ResponsesRequest::echo`] reports it as `summary` when
    /// `summary` is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<ResponsesReasoningSummary>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResponsesTextConfig {
    #[serde(default)]
    pub format: ResponsesTextFormat,
    /// The requested verbosity. Ignored by conversion;
    /// [`ResponsesRequest::echo`] reports it in `text`.
    pub verbosity: Option<ResponsesVerbosity>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTextFormat {
    #[default]
    Text,
    /// A JSON Schema rendered into the prompt as written. With `strict: true`,
    /// conversion also checks the schema with the normalization of strict tool
    /// parameters and rejects it on failure. `name` and `description` are
    /// ignored.
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        description: Option<String>,
        strict: Option<bool>,
    },
    JsonObject,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesVerbosity {
    Low,
    Medium,
    High,
}

/// A tool declaration. Serialization writes the fields that responses report.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function(ResponsesFunctionTool),
    /// The hosted web search tool; dropped or rejected per
    /// [`crate::request::ConversionOptions::responses_web_search`].
    WebSearch(ResponsesWebSearchTool),
    #[serde(rename = "web_search_2025_08_26")]
    WebSearch20250826(ResponsesWebSearchTool),
    Namespace {
        name: String,
        description: Option<String>,
        tools: Vec<ResponsesNamespaceTool>,
    },
    /// A custom tool. Conversion accepts only `apply_patch`, which it converts
    /// to a function tool, and rejects any other custom tool name.
    Custom {
        name: String,
    },
    ToolSearch {
        execution: Option<ResponsesToolSearchExecution>,
        description: Option<String>,
        parameters: Option<serde_json::Value>,
    },
    /// Other tool types are ignored. Serializing this variant fails, so
    /// [`ResponsesRequest::echo`] omits it.
    #[serde(other, skip_serializing)]
    Unsupported,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesWebSearchTool {
    pub search_context_size: Option<ResponsesSearchContextSize>,
    pub user_location: Option<ResponsesUserLocation>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesSearchContextSize {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesUserLocation {
    pub city: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
    /// With `true`, conversion normalizes [`parameters`](Self::parameters) as
    /// strict tool parameters. Conversion passes the flag through in the tool
    /// definition; enforcement during inference belongs to the caller.
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesNamespaceTool {
    Function(ResponsesFunctionTool),
    /// Custom tools inside a namespace are rejected during conversion.
    Custom {
        name: String,
    },
}

#[derive(Debug, Clone, Default, UntaggedDeserialize)]
#[untagged_deserialize(
    error = "expected one of `none`, `auto`, `required` or a tool choice object"
)]
pub enum ResponsesToolChoice {
    #[untagged_deserialize(value = "none")]
    None,
    #[default]
    #[untagged_deserialize(value = "auto")]
    Auto,
    #[untagged_deserialize(value = "required")]
    Required,
    #[untagged_deserialize(field = "type")]
    Named(ResponsesNamedToolChoice),
}

impl Serialize for ResponsesToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::Named(named) => named.serialize(serializer),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesNamedToolChoice {
    Function {
        name: String,
    },
    WebSearch,
    #[serde(rename = "web_search_2025_08_26")]
    WebSearch20250826,
    Custom {
        name: String,
    },
}
