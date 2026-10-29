use serde::Serialize;
use strum_macros::IntoStaticStr;

use crate::openai::responses::request::{
    ResponsesReasoningConfig, ResponsesTextConfig, ResponsesTool, ResponsesToolChoice,
    ResponsesToolSearchExecution,
};
use crate::response::ProtocolResponse;
use crate::stream::{CompletionUsage, PromptUsage, TokenLogprob};

use super::chunk_generator::ResponsesChunkGenerator;

/// A complete Responses response, also usable as an accumulator for generated
/// events.
/// Deltas append, done events replace values, and full responses replace state.
/// Incremental events with mismatched indices, IDs, or types are not applied.
///
/// Request settings come from [`ResponsesEcho`]. Settings for unsupported
/// features (background mode, stored responses, penalties, tool call limits,
/// moderation, prompt caching, service tiers, truncation, and metadata) report
/// fixed defaults.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: ResponsesStatus,
    pub background: bool,
    pub completed_at: Option<u64>,
    pub content_filters: Option<serde_json::Value>,
    pub error: Option<ResponsesError>,
    pub frequency_penalty: f32,
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub model: String,
    pub moderation: Option<String>,
    pub output: Vec<ResponsesOutputItem>,
    pub parallel_tool_calls: bool,
    pub presence_penalty: f32,
    pub previous_response_id: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<String>,
    pub reasoning: ResponsesReasoningConfig,
    pub safety_identifier: Option<String>,
    pub service_tier: &'static str,
    pub store: bool,
    pub temperature: f32,
    pub text: ResponsesTextConfig,
    pub tool_choice: ResponsesToolChoice,
    pub tools: Vec<ResponsesTool>,
    pub top_logprobs: u8,
    pub top_p: f32,
    pub truncation: &'static str,
    /// Null until the chunk generator handles [`OutputChunk::Finish`]. The
    /// token counts passed to [`ProtocolResponse::new`] are ignored.
    ///
    /// [`OutputChunk::Finish`]: crate::stream::OutputChunk::Finish
    pub usage: Option<ResponsesUsage>,
    pub user: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Request settings that a response reports, as the request specified them.
///
/// Conversion preserves these settings for the chunk generator.
/// [`ResponsesRequest::echo`] reads them before conversion, and
/// [`ResponsesChunkGenerator::with_echo`] overrides them. The default reports
/// an unspecified request.
///
/// [`ResponsesRequest::echo`]: crate::openai::ResponsesRequest::echo
#[derive(Debug, Clone)]
pub struct ResponsesEcho {
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    /// The requested reasoning settings, with `generate_summary` merged into
    /// an unset `summary`.
    pub reasoning: ResponsesReasoningConfig,
    /// The requested temperature or 1.0, not the sampling temperature.
    pub temperature: f32,
    pub text: ResponsesTextConfig,
    pub tool_choice: ResponsesToolChoice,
    /// Declared tools, without unsupported tool types.
    pub tools: Vec<ResponsesTool>,
    pub top_logprobs: u8,
    /// The requested `top_p` or 1.0, not the sampling `top_p`.
    pub top_p: f32,
    pub user: Option<String>,
}

impl Default for ResponsesEcho {
    fn default() -> Self {
        Self {
            instructions: None,
            max_output_tokens: None,
            reasoning: ResponsesReasoningConfig::default(),
            temperature: 1.0,
            text: ResponsesTextConfig::default(),
            tool_choice: ResponsesToolChoice::default(),
            tools: Vec::new(),
            top_logprobs: 0,
            top_p: 1.0,
            user: None,
        }
    }
}

impl ResponsesResponse {
    /// Set the request settings that the response reports to `echo`.
    pub(super) fn set_echo(&mut self, echo: ResponsesEcho) {
        let ResponsesEcho {
            instructions,
            max_output_tokens,
            reasoning,
            temperature,
            text,
            tool_choice,
            tools,
            top_logprobs,
            top_p,
            user,
        } = echo;
        self.instructions = instructions;
        self.max_output_tokens = max_output_tokens;
        self.reasoning = reasoning;
        self.temperature = temperature;
        self.text = text;
        self.tool_choice = tool_choice;
        self.tools = tools;
        self.top_logprobs = top_logprobs;
        self.top_p = top_p;
        self.user = user;
    }
}

/// Response and output item states generated from parsed output chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// The `error` fields of a response object. Generated responses set `error`
/// to null.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesIncompleteDetails {
    pub reason: ResponsesIncompleteReason,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesIncompleteReason {
    MaxOutputTokens,
    ContentFilter,
}

/// An item in the `output` of a response.
///
/// Output items retain their position in the response throughout streaming.
/// Item IDs identify output items in events. Call IDs associate subsequent
/// tool results with their calls.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        status: ResponsesStatus,
        role: ResponsesRole,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<ResponsesPhase>,
        content: Vec<ResponsesContentPart>,
    },
    Reasoning {
        id: String,
        status: ResponsesStatus,
        content: Vec<ResponsesContentPart>,
        summary: Vec<()>,
        /// An identifier of the reasoning, `{response_id}-0`.
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    FunctionCall {
        id: String,
        status: ResponsesStatus,
        call_id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        arguments: String,
    },
    CustomToolCall {
        id: String,
        status: ResponsesStatus,
        call_id: String,
        name: String,
        input: String,
    },
    ToolSearchCall {
        id: String,
        status: ResponsesStatus,
        call_id: String,
        execution: ResponsesToolSearchExecution,
        arguments: serde_json::Map<String, serde_json::Value>,
        #[serde(skip)]
        raw_arguments: String,
    },
}

/// Content produced from parsed model output.
/// Annotations are emitted as empty arrays. Log probabilities are empty unless
/// the inference output carries them.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        annotations: Vec<()>,
        logprobs: Vec<ResponsesTokenLogprob>,
        text: String,
    },
    ReasoningText {
        text: String,
    },
}

/// The log probability of one output text token.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<ResponsesTopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

impl From<&TokenLogprob> for ResponsesTokenLogprob {
    fn from(logprob: &TokenLogprob) -> Self {
        Self {
            token: logprob.token.clone(),
            logprob: logprob.logprob,
            bytes: Some(logprob.bytes.clone()),
            top_logprobs: logprob
                .top_logprobs
                .iter()
                .map(|top_logprob| ResponsesTopLogprob {
                    token: top_logprob.token.clone(),
                    logprob: top_logprob.logprob,
                    bytes: Some(top_logprob.bytes.clone()),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesRole {
    Assistant,
}

/// The phase of a completed message output item. A message whose answer
/// content immediately precedes a tool call has `commentary`; other messages
/// have `final_answer`. In-progress messages omit the phase until the next
/// output item or response completion determines it.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesPhase {
    Commentary,
    FinalAnswer,
}

/// Backend prompt counts combined with completion counts accumulated by the
/// stream processor.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub input_tokens_details: ResponsesInputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: ResponsesOutputTokensDetails,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ResponsesInputTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ResponsesOutputTokensDetails {
    /// The reasoning token count, or zero without one.
    pub reasoning_tokens: usize,
}

impl From<(PromptUsage, CompletionUsage)> for ResponsesUsage {
    fn from((prompt, completion): (PromptUsage, CompletionUsage)) -> Self {
        Self {
            input_tokens: prompt.prompt_tokens,
            input_tokens_details: ResponsesInputTokensDetails {
                cached_tokens: prompt.prompt_cache_hit_tokens,
            },
            output_tokens: completion.completion_tokens,
            output_tokens_details: ResponsesOutputTokensDetails {
                reasoning_tokens: completion.reasoning_tokens.unwrap_or(0),
            },
            total_tokens: prompt.prompt_tokens + completion.completion_tokens,
        }
    }
}

/// A typed Responses SSE event. The `type` field is also the SSE event name.
/// Sequence numbers increase across all event kinds within one response.
#[derive(Debug, Clone, Serialize, IntoStaticStr)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.created")]
    #[strum(serialize = "response.created")]
    Created {
        response: ResponsesResponse,
        sequence_number: usize,
    },
    #[serde(rename = "response.in_progress")]
    #[strum(serialize = "response.in_progress")]
    InProgress {
        response: ResponsesResponse,
        sequence_number: usize,
    },
    #[serde(rename = "response.output_item.added")]
    #[strum(serialize = "response.output_item.added")]
    OutputItemAdded {
        item: ResponsesOutputItem,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.content_part.added")]
    #[strum(serialize = "response.content_part.added")]
    ContentPartAdded {
        content_index: usize,
        item_id: String,
        output_index: usize,
        part: ResponsesContentPart,
        sequence_number: usize,
    },
    #[serde(rename = "response.output_text.delta")]
    #[strum(serialize = "response.output_text.delta")]
    OutputTextDelta {
        content_index: usize,
        delta: String,
        item_id: String,
        logprobs: Vec<ResponsesTokenLogprob>,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.output_text.done")]
    #[strum(serialize = "response.output_text.done")]
    OutputTextDone {
        content_index: usize,
        item_id: String,
        logprobs: Vec<ResponsesTokenLogprob>,
        output_index: usize,
        sequence_number: usize,
        text: String,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    #[strum(serialize = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        content_index: usize,
        delta: String,
        item_id: String,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.reasoning_text.done")]
    #[strum(serialize = "response.reasoning_text.done")]
    ReasoningTextDone {
        content_index: usize,
        item_id: String,
        output_index: usize,
        sequence_number: usize,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    #[strum(serialize = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        delta: String,
        item_id: String,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    #[strum(serialize = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        arguments: String,
        item_id: String,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.custom_tool_call_input.delta")]
    #[strum(serialize = "response.custom_tool_call_input.delta")]
    CustomToolCallInputDelta {
        delta: String,
        item_id: String,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.custom_tool_call_input.done")]
    #[strum(serialize = "response.custom_tool_call_input.done")]
    CustomToolCallInputDone {
        input: String,
        item_id: String,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.content_part.done")]
    #[strum(serialize = "response.content_part.done")]
    ContentPartDone {
        content_index: usize,
        item_id: String,
        output_index: usize,
        part: ResponsesContentPart,
        sequence_number: usize,
    },
    #[serde(rename = "response.output_item.done")]
    #[strum(serialize = "response.output_item.done")]
    OutputItemDone {
        item: ResponsesOutputItem,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.completed")]
    #[strum(serialize = "response.completed")]
    Completed {
        response: ResponsesResponse,
        sequence_number: usize,
    },
    #[serde(rename = "response.incomplete")]
    #[strum(serialize = "response.incomplete")]
    Incomplete {
        response: ResponsesResponse,
        sequence_number: usize,
    },
}

impl ProtocolResponse for ResponsesResponse {
    type ChunkGenerator = ResponsesChunkGenerator;

    fn new(
        id: String,
        model: String,
        created: u64,
        _prompt_tokens: usize,
        _prompt_cache_hit_tokens: usize,
    ) -> Self {
        let ResponsesEcho {
            instructions,
            max_output_tokens,
            reasoning,
            temperature,
            text,
            tool_choice,
            tools,
            top_logprobs,
            top_p,
            user,
        } = ResponsesEcho::default();
        Self {
            id,
            object: "response",
            created_at: created,
            status: ResponsesStatus::InProgress,
            background: false,
            completed_at: None,
            content_filters: None,
            error: None,
            frequency_penalty: 0.0,
            incomplete_details: None,
            instructions,
            max_output_tokens,
            max_tool_calls: None,
            model,
            moderation: None,
            output: vec![],
            parallel_tool_calls: true,
            presence_penalty: 0.0,
            previous_response_id: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning,
            safety_identifier: None,
            service_tier: "default",
            store: false,
            temperature,
            text,
            tool_choice,
            tools,
            top_logprobs,
            top_p,
            truncation: "disabled",
            usage: None,
            user,
            metadata: serde_json::Map::new(),
        }
    }

    fn sse_event_name(chunk: &ResponsesStreamEvent) -> Option<&'static str> {
        Some(chunk.into())
    }
}
