use deepseek_recipe_core::multimodal::ImageDetail;
use deepseek_recipe_macros::UntaggedDeserialize;
use serde::Deserialize;

pub use crate::protocol::reasoning::{RequestReasoningEffort, ThinkingType};

/// A Chat Completions request. Unknown fields are ignored.
///
/// Conversion resolves the model version through
/// [`ConversionOptions::model_versions_by_name`]; inference execution and tool
/// execution belong to the caller. Client tools, historical tool calls, and
/// image blocks are supported. Server tools are rejected by this adapter.
/// Constrained decoding belongs to the caller.
///
/// [`ConversionOptions::model_versions_by_name`]:
///     crate::request::ConversionOptions::model_versions_by_name
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,

    pub model: String,

    /// Validated in [-2, 2], then omitted from the converted request. Extract
    /// before conversion if the backend supports it.
    pub frequency_penalty: Option<f32>,
    /// Whether the response reports the log probability of each output token.
    pub logprobs: Option<bool>,
    /// Number of most likely alternatives per output token, in [0, 20].
    /// Requires `logprobs`.
    pub top_logprobs: Option<u8>,
    pub max_tokens: Option<u32>,
    /// Only one completion per request is supported.
    pub n: Option<u8>,
    /// Validated in [-2, 2], then omitted from the converted request. Extract
    /// before conversion if the backend supports it.
    pub presence_penalty: Option<f32>,
    /// Sampling seed in [0, 2^63), passed through to the caller.
    pub seed: Option<u64>,
    pub stop: Option<ChatCompletionStopSequences>,
    pub stream: Option<bool>,
    pub stream_options: Option<ChatCompletionStreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,

    /// Text, JSON object, JSON Schema, or regex output; see
    /// [`ChatCompletionResponseFormat`].
    pub response_format: Option<ChatCompletionResponseFormat>,

    pub tools: Option<Vec<ChatCompletionTool>>,
    /// With tools present, required and named choices require thinking off.
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,

    /// Requested reasoning effort. `none` disables thinking unless `thinking`
    /// enables it.
    pub reasoning_effort: Option<RequestReasoningEffort>,
    /// Explicit thinking overrides reasoning effort and the conversion default.
    pub thinking: Option<ChatCompletionThinking>,
    /// Validated during deserialization and ignored by conversion.
    pub parallel_tool_calls: Option<bool>,
}

/// A request message. `name` is accepted and ignored.
#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatCompletionRequestMessage {
    System {
        content: ChatCompletionRequestContent,
        name: Option<String>,
    },
    User {
        content: ChatCompletionRequestContent,
        name: Option<String>,
    },
    Assistant {
        content: Option<ChatCompletionRequestContent>,
        reasoning_content: Option<String>,
        name: Option<String>,
        tool_calls: Option<Vec<ChatCompletionRequestToolCall>>,
        /// Whether conversion continues this final assistant message as a
        /// prefix continuation instead of starting a new assistant turn.
        prefix: Option<bool>,
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
    pub r#type: ChatCompletionToolType,
    pub function: ChatCompletionRequestFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequestFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "content should be a string or a list")]
pub enum ChatCompletionRequestContent {
    #[untagged_deserialize(pattern = String(_))]
    String(String),
    #[untagged_deserialize(pattern = Array(_))]
    List(Vec<ChatCompletionRequestContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionImageUrl {
    /// HTTP URL or base64 data URL. A value that does not start with `http`
    /// is a data URL, which system, user, and tool content require to carry a
    /// `base64` body.
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
    /// One image carried as a base64 data URL in `file_data`. `file_id` is
    /// rejected during conversion; the filename is accepted and ignored.
    File {
        file_id: Option<String>,
        file_data: Option<String>,
        filename: Option<String>,
    },
}

/// One stop sequence or a list of stop sequences.
#[derive(Debug, Clone, UntaggedDeserialize)]
#[untagged_deserialize(error = "expected a string or a list")]
pub enum ChatCompletionStopSequences {
    #[untagged_deserialize(pattern = Array(_))]
    Array(Vec<String>),
    #[untagged_deserialize(pattern = String(_))]
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

/// The requested answer format. Conversion validates the schema of
/// `json_schema` and passes `regex` through; the caller enforces either one
/// during inference.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionResponseFormat {
    JsonObject,
    JsonSchema {
        json_schema: ChatCompletionResponseFormatJsonSchema,
    },
    Regex {
        regex: String,
    },
    Text,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponseFormatJsonSchema {
    pub schema: serde_json::Value,
    pub name: Option<String>,
    pub strict: Option<bool>,
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
    /// With `true`, conversion normalizes [`parameters`](Self::parameters) as
    /// strict tool parameters. Conversion passes the flag through in the tool
    /// definition; enforcement during inference belongs to the caller.
    pub strict: Option<bool>,
}

#[derive(Debug, UntaggedDeserialize)]
#[untagged_deserialize(error = "expected one of `none`, `auto`, `required` or a tool")]
pub enum ChatCompletionToolChoiceOption {
    #[untagged_deserialize(value = "none")]
    None,
    #[untagged_deserialize(value = "auto")]
    Auto,
    #[untagged_deserialize(value = "required")]
    Required,
    #[untagged_deserialize(field = "type")]
    Named {
        r#type: ChatCompletionToolType,
        function: ChatCompletionToolChoiceFunction,
    },
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionToolChoiceFunction {
    pub name: String,
}

/// Optional thinking control for DeepSeek-compatible requests.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionThinking {
    pub r#type: ThinkingType,
    /// Deserialized for the caller; conversion leaves this setting unused.
    pub budget_tokens: Option<u64>,
}
