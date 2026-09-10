use serde::{Deserialize, Deserializer};

use deepseek_recipe_core::multimodal::ImageDetail;

/// A Responses request with text and image input and client tools.
///
/// Unknown top-level fields, including `logprobs` and `top_logprobs`, are
/// ignored. Model resolution, server tools, and encrypted reasoning recovery
/// belong to the caller.
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
    pub top_p: Option<f32>,
    /// Caller metadata. Extract this before consuming the request.
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    String(String),
    Items(Vec<ResponsesInputItem>),
}

/// An explicitly typed item or a message without a `type` field.
#[derive(Debug)]
pub enum ResponsesInputItem {
    Typed(ResponsesTypedInputItem),
    Message(ResponsesInputMessage),
}

impl<'de> Deserialize<'de> for ResponsesInputItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        // The presence of `type` selects typed-item deserialization.
        if value.get("type").is_some() {
            serde_json::from_value(value)
                .map(Self::Typed)
                .map_err(serde::de::Error::custom)
        } else {
            serde_json::from_value(value)
                .map(Self::Message)
                .map_err(serde::de::Error::custom)
        }
    }
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
    /// Unsupported item types, including `web_search_call`, are ignored.
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesMessageContent {
    String(String),
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
    /// Documents are rendered as an unsupported-document placeholder. Every
    /// field is accepted and ignored.
    InputFile {},
    /// Content block types not listed above are rejected during conversion.
    #[serde(other)]
    Unsupported,
}

/// One image referenced by URL or by a file identifier. A file identifier
/// requires a file service and is rejected during conversion.
#[derive(Debug, Deserialize)]
pub struct ResponsesInputImage {
    /// HTTP URL or base64 data URL. An empty string is treated as absent, and a
    /// value that does not start with `http` must carry a `base64` body.
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
    pub call_id: String,
    pub output: ResponsesToolCallOutputContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolCallOutputContent {
    String(String),
    List(Vec<ResponsesToolCallOutputItem>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesToolCallOutputItem {
    InputText {
        text: String,
    },
    InputImage(ResponsesInputImage),
    /// Documents are rendered as an unsupported-document placeholder. Every
    /// field is accepted and ignored.
    InputFile {},
    /// Content block types not listed above are rejected during conversion.
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesReasoningItem {
    pub content: Option<Vec<ResponsesReasoningContent>>,
    /// Encrypted reasoning is ignored; plaintext reasoning is preserved.
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
pub struct ResponsesReasoningConfig {
    pub effort: Option<ResponsesReasoningEffort>,
    /// Summary setting available to the caller before conversion.
    pub summary: Option<ResponsesReasoningSummary>,
    /// Alternate summary setting available to the caller before conversion.
    pub generate_summary: Option<ResponsesReasoningSummary>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesTextConfig {
    #[serde(default)]
    pub format: ResponsesTextFormat,
    /// Verbosity setting available to the caller before conversion.
    pub verbosity: Option<ResponsesVerbosity>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTextFormat {
    #[default]
    Text,
    JsonObject,
    /// Accepted and ignored, including all schema fields. Uses plain text output.
    JsonSchema,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesVerbosity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function(ResponsesFunctionTool),
    Namespace {
        name: String,
        description: Option<String>,
        tools: Vec<ResponsesNamespaceTool>,
    },
    /// The `apply_patch` custom tool is converted to a function tool.
    Custom {
        name: String,
    },
    #[serde(alias = "web_search_2025_08_26")]
    WebSearch,
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesFunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
    /// Passed through; backend enforcement belongs to the caller.
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesNamespaceTool {
    Function(ResponsesFunctionTool),
    /// Custom tools inside a namespace are rejected during conversion.
    Custom {
        name: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(ResponsesToolChoiceMode),
    Named(ResponsesNamedToolChoice),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesNamedToolChoice {
    Function {
        name: String,
    },
    Custom {
        name: String,
    },
    #[serde(alias = "web_search_2025_08_26")]
    WebSearch,
}
