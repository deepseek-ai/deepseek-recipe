use serde::Deserialize;

/// An Anthropic Messages request. Unknown top-level fields are ignored.
///
/// Model selection, execution and usage accounting are the caller's
/// responsibility. Images are supported. Document blocks become the text
/// `[Unsupported Document]`; their contents are discarded. `web_search` server
/// tools are rejected by default; see
/// [`crate::request::WebSearchBehavior`].
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<MessagesMessage>,
    pub system: Option<MessagesTextOrTextBlocks>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<MessagesTool>>,
    pub tool_choice: Option<MessagesToolChoice>,
    pub thinking: Option<MessagesThinking>,
    pub output_config: Option<MessagesOutputConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", content = "content", rename_all = "snake_case")]
pub enum MessagesMessage {
    User(MessagesContent),
    Assistant(MessagesContent),
    /// Compatibility extension, rendered as a user system-reminder.
    System(MessagesTextOrTextBlocks),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessagesContent {
    Raw(String),
    Blocks(Vec<MessagesContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessagesTextOrTextBlocks {
    Raw(String),
    Blocks(Vec<MessagesTextBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesTextBlock {
    Text {
        text: String,
    },
    ToolReference {
        tool_name: String,
    },
    Image {
        source: MessagesImageSource,
    },
    /// Documents are rendered as an unsupported-document placeholder.
    Document,
    /// Recognized during deserialization, rejected during conversion.
    #[serde(other)]
    Unsupported,
}

/// One image referenced by inline data, URL, or file identifier. A file
/// identifier requires a file service and is rejected during conversion.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesImageSource {
    Base64 {
        /// Declared media type, validated against the supported image types.
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
    File {
        file_id: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesContentBlock {
    Text {
        text: String,
    },
    ToolReference {
        tool_name: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<MessagesTextOrTextBlocks>,
        /// Accepted for compatibility; the result text is preserved either way.
        is_error: Option<bool>,
    },
    Thinking {
        /// Signature data available to the caller before conversion.
        signature: Option<String>,
        thinking: String,
    },
    Image {
        source: MessagesImageSource,
    },
    /// A server tool invocation. Assistant messages only; dropped or rejected
    /// per [`crate::request::ConversionOptions::messages_web_search`].
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A server tool result. Assistant messages only; dropped or rejected per
    /// [`crate::request::ConversionOptions::messages_web_search`].
    WebSearchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    /// Documents are rendered as an unsupported-document placeholder.
    Document,
    #[serde(other)]
    Unsupported,
}

/// A client tool declaration. A `type` starting with `web_search` is dropped or
/// rejected per [`crate::request::ConversionOptions::messages_web_search`];
/// other `type` discriminators are rejected as unsupported.
#[derive(Debug, Deserialize)]
pub struct MessagesTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<serde_json::Value>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesToolChoice {
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    /// For compatibility, `any` permits both text replies and tool calls.
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
    None,
}

#[derive(Debug, Deserialize)]
pub struct MessagesThinking {
    #[serde(rename = "type")]
    pub kind: MessagesThinkingType,
    /// Thinking budget passed through for the caller to enforce.
    pub budget_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagesThinkingType {
    #[serde(alias = "adaptive")]
    Enabled,
    Disabled,
}

#[derive(Debug, Deserialize)]
pub struct MessagesOutputConfig {
    pub effort: Option<MessagesReasoningEffort>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagesReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Ultra,
    Max,
}
