use deepseek_recipe_core::multimodal::ImageMediaType;
use deepseek_recipe_macros::UntaggedDeserialize;
use serde::Deserialize;

pub use crate::protocol::reasoning::{RequestReasoningEffort, ThinkingType};

/// An Anthropic Messages request. Unknown top-level fields are ignored.
///
/// Conversion resolves the model version through
/// [`ConversionOptions::model_versions_by_name`]; inference execution, tool
/// execution, and usage accounting belong to the caller. Images are supported.
/// Document blocks become the text
/// `[Unsupported document type: {media_type}]`; their contents are discarded.
/// `web_search` server tools are rejected by default; see
/// [`crate::request::WebSearchBehavior`].
///
/// [`ConversionOptions::model_versions_by_name`]:
///     crate::request::ConversionOptions::model_versions_by_name
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<MessagesMessage>,
    pub system: Option<MessagesSystemContent>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<MessagesTool>>,
    pub tool_choice: Option<MessagesToolChoice>,
    pub thinking: Option<MessagesThinking>,
    pub output_config: Option<MessagesOutputConfig>,
    /// Validated during deserialization and ignored by conversion.
    pub metadata: Option<MessagesMetadata>,
}

/// Request metadata.
#[derive(Debug, Deserialize)]
pub struct MessagesMetadata {
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", content = "content", rename_all = "snake_case")]
pub enum MessagesMessage {
    User(MessagesContent),
    Assistant(MessagesContent),
    System(MessagesContent),
}

#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "expected a string or a list")]
pub enum MessagesContent {
    #[untagged_deserialize(pattern = String(_))]
    Raw(String),
    #[untagged_deserialize(pattern = Array(_))]
    Blocks(Vec<MessagesContentBlock>),
}

/// The top-level `system` field: a string or a list of text blocks.
#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "expected a string or a list")]
pub enum MessagesSystemContent {
    #[untagged_deserialize(pattern = String(_))]
    Raw(String),
    #[untagged_deserialize(pattern = Array(_))]
    Blocks(Vec<MessagesSystemBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesSystemBlock {
    Text { text: String },
}

/// The content of a `tool_result` block.
#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "expected a string or a list")]
pub enum MessagesToolResultContent {
    #[untagged_deserialize(pattern = String(_))]
    Raw(String),
    #[untagged_deserialize(pattern = Array(_))]
    Blocks(Vec<MessagesToolResultBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesToolResultBlock {
    Text {
        text: String,
    },
    Image {
        source: MessagesImageSource,
    },
    /// Conversion writes the unsupported-document text in place of a document.
    Document {
        source: MessagesDocumentSource,
    },
    /// Conversion writes the referenced tool name as plain text.
    ToolReference {
        tool_name: String,
    },
}

/// One image referenced by inline data, URL, or file identifier. A file
/// identifier requires a file service and is rejected during conversion.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesImageSource {
    Base64 {
        media_type: MessagesImageMediaType,
        data: String,
    },
    Url {
        url: String,
    },
    File {
        file_id: String,
    },
}

/// A supported image media type, matched exactly. `image/jpg` is an alias of
/// `image/jpeg`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum MessagesImageMediaType {
    #[serde(rename = "image/jpeg", alias = "image/jpg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/gif")]
    Gif,
    #[serde(rename = "image/webp")]
    Webp,
}

impl From<MessagesImageMediaType> for ImageMediaType {
    fn from(media_type: MessagesImageMediaType) -> Self {
        match media_type {
            MessagesImageMediaType::Jpeg => Self::Jpeg,
            MessagesImageMediaType::Png => Self::Png,
            MessagesImageMediaType::Gif => Self::Gif,
            MessagesImageMediaType::Webp => Self::Webp,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MessagesDocumentSource {
    pub media_type: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesContentBlock {
    Text {
        text: String,
    },
    Image {
        source: MessagesImageSource,
    },
    /// Conversion writes the unsupported-document text in place of a document.
    Document {
        source: MessagesDocumentSource,
    },
    Thinking {
        /// Signature data available to the caller before conversion.
        signature: Option<String>,
        thinking: String,
    },
    /// A server tool invocation. Assistant messages only; dropped or rejected
    /// per [`crate::request::ConversionOptions::messages_web_search`].
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<MessagesToolResultContent>,
        /// Accepted for compatibility; the result text is preserved either way.
        is_error: Option<bool>,
    },
    /// A server tool result. Assistant messages only; dropped or rejected per
    /// [`crate::request::ConversionOptions::messages_web_search`].
    WebSearchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    ToolAddition {
        tool: MessagesToolReference,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesToolReference {
    ToolReference { name: String },
}

/// A tool declaration. An object with a `type` field is a server tool; any
/// other object is a client tool and requires `name`.
#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "missing field `name`")]
pub enum MessagesTool {
    /// A `type` starting with `web_search` is dropped or rejected per
    /// [`crate::request::ConversionOptions::messages_web_search`]; other types
    /// are rejected as unsupported.
    #[untagged_deserialize(field = "type")]
    Server(MessagesServerTool),
    #[untagged_deserialize(field = "name")]
    Client {
        name: String,
        description: Option<String>,
        /// The input schema; a missing schema is `null` and fails validation.
        input_schema: serde_json::Value,
        defer_loading: Option<bool>,
    },
}

/// A server tool declaration.
#[derive(Debug, Deserialize)]
pub struct MessagesServerTool {
    pub r#type: String,
    pub name: Option<String>,
}

/// The tool choice. Deserialization validates `disable_parallel_tool_use` as a
/// boolean, and conversion ignores it.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesToolChoice {
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    /// Converted to the `auto` tool choice, so the model can answer with text or
    /// call tools.
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
    None,
}

/// The `thinking` setting. When present, its `type` decides thinking mode
/// before `output_config.effort` does.
#[derive(Debug, Deserialize)]
pub struct MessagesThinking {
    pub r#type: ThinkingType,
    /// Validated during deserialization and ignored by conversion.
    pub budget_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct MessagesOutputConfig {
    /// Requested reasoning effort. `none` disables thinking unless `thinking`
    /// enables it.
    pub effort: Option<RequestReasoningEffort>,
}
