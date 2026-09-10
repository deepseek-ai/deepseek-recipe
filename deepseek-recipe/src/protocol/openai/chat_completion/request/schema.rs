use serde::Deserialize;

use deepseek_recipe_core::multimodal::ImageDetail;

/// A Chat Completions request. Unknown fields are ignored.
///
/// Model resolution and inference execution belong to the caller. Client tools,
/// historical tool calls, and image blocks are supported. Server tools are
/// rejected by this adapter. Constrained decoding and model-specific sampling
/// belong to the caller. JSON Schema output, `logprobs`, and `top_logprobs` are
/// accepted and ignored.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,

    pub model: String,

    /// Validated in [-2, 2], then omitted from the converted request. Extract
    /// before conversion if the backend supports it.
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Only one completion per request is supported.
    pub n: Option<u8>,
    /// Validated in [-2, 2], then omitted from the converted request. Extract
    /// before conversion if the backend supports it.
    pub presence_penalty: Option<f32>,
    /// Validated in [0, 2^63), then omitted from the converted request. Extract
    /// before conversion if the backend supports it.
    pub seed: Option<u64>,
    pub stop: Option<ChatCompletionStopSequences>,
    pub stream: Option<bool>,
    pub stream_options: Option<ChatCompletionStreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,

    /// Text or JSON object output. JSON Schema settings are accepted and ignored.
    pub response_format: Option<ChatCompletionResponseFormat>,

    pub tools: Option<Vec<ChatCompletionTool>>,
    /// With tools present, required and named choices require thinking off.
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,

    pub reasoning_effort: Option<ChatCompletionReasoningEffort>,
    /// Explicit thinking overrides reasoning effort and the conversion default.
    pub thinking: Option<ChatCompletionThinking>,
    /// Deserialized for the caller; conversion leaves this setting unused.
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatCompletionRequestMessage {
    System {
        content: ChatCompletionRequestContent,
    },
    User {
        content: ChatCompletionRequestContent,
    },
    Assistant {
        content: Option<ChatCompletionRequestContent>,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<ChatCompletionRequestToolCall>>,
    },
    Tool {
        content: ChatCompletionRequestContent,
        tool_call_id: String,
    },
    LatestReminder {
        content: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequestToolCall {
    pub id: String,
    pub function: ChatCompletionRequestFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequestFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionRequestContent {
    String(String),
    List(Vec<ChatCompletionRequestContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionImageUrl {
    /// HTTP URL or base64 data URL. A value that does not start with `http`
    /// must carry a `base64` body.
    pub url: String,
    #[serde(default)]
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionRequestContentBlock {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ChatCompletionImageUrl,
    },
    /// One image carried as a data URL. The resolver accepts base64 and
    /// percent-encoded bodies. `file_id` is rejected during conversion; the
    /// filename is accepted and ignored.
    File {
        file_id: Option<String>,
        file_data: Option<String>,
        filename: Option<String>,
    },
}

/// One stop sequence or a list of stop sequences.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionStopSequences {
    Array(Vec<String>),
    String(String),
}

impl From<ChatCompletionStopSequences> for Vec<String> {
    fn from(value: ChatCompletionStopSequences) -> Self {
        match value {
            ChatCompletionStopSequences::Array(values) => values,
            ChatCompletionStopSequences::String(value) => vec![value],
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionStreamOptions {
    /// Include a null usage field before the final chunk. The final chunk
    /// carries usage regardless of this setting.
    pub include_usage: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionResponseFormat {
    JsonObject,
    /// Accepted and ignored, including nested schema fields.
    JsonSchema,
    Regex {
        regex: String,
    },
    Text,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionTool {
    pub r#type: ChatCompletionToolType,
    pub function: ChatCompletionFunctionDefinition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionToolType {
    Function,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionFunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    /// A JSON Schema of type object. Omission renders an empty schema.
    pub parameters: Option<serde_json::Value>,
    /// Passed through for the caller to enforce during inference.
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionToolChoiceOption {
    /// One of `"none"`, `"auto"`, or `"required"`.
    Mode(ChatCompletionToolChoiceMode),
    Named(ChatCompletionNamedToolChoice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionNamedToolChoice {
    pub r#type: ChatCompletionToolType,
    pub function: ChatCompletionToolChoiceFunction,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionToolChoiceFunction {
    pub name: String,
}

/// Optional thinking control for DeepSeek-compatible requests.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionThinking {
    #[serde(rename = "type")]
    pub kind: ChatCompletionThinkingType,
    /// Deserialized for the caller; conversion leaves this setting unused.
    pub budget_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionThinkingType {
    #[serde(alias = "adaptive")]
    Enabled,
    Disabled,
}

/// Reasoning effort accepted by this adapter. Unless overridden by `thinking`,
/// `none` disables thinking and other values enable it. Enabled thinking uses
/// high effort by default. Supplied values map to the shared reasoning levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}
