use serde::Serialize;
use strum_macros::IntoStaticStr;

use crate::response::ProtocolResponse;
use crate::stream::{CompletionUsage, PromptUsage};

use super::chunk_generator::ResponsesChunkGenerator;

/// A complete Responses response with event accumulation.
/// Deltas append, done events replace values, and full responses replace state.
/// Incremental events with mismatched indices, IDs, or types are ignored.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: ResponsesStatus,
    pub completed_at: Option<u64>,
    pub error: Option<ResponsesError>,
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
    pub output: Vec<ResponsesOutputItem>,
    /// Usage is null until `Finish`; constructor token counts are ignored.
    pub usage: Option<ResponsesUsage>,
}

/// Response and output item states generated from parsed output chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// Error fields for a complete response. Parsed output chunks contain no error
/// information, so generated responses set `error` to null.
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
        phase: ResponsesPhase,
        content: Vec<ResponsesContentPart>,
    },
    Reasoning {
        id: String,
        status: ResponsesStatus,
        content: Vec<ResponsesContentPart>,
        summary: Vec<()>,
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
}

/// Content produced from parsed model output.
/// Annotations and log probabilities are emitted as empty arrays.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        annotations: Vec<()>,
        logprobs: Vec<()>,
        text: String,
    },
    ReasoningText {
        text: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesRole {
    Assistant,
}

/// Answer content immediately before a tool call uses `commentary` in the
/// completed output item.
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
    /// Always zero because the backend reports no reasoning token count.
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
                reasoning_tokens: 0,
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
        logprobs: Vec<()>,
        output_index: usize,
        sequence_number: usize,
    },
    #[serde(rename = "response.output_text.done")]
    #[strum(serialize = "response.output_text.done")]
    OutputTextDone {
        content_index: usize,
        item_id: String,
        logprobs: Vec<()>,
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
        Self {
            id,
            object: "response",
            created_at: created,
            model,
            status: ResponsesStatus::InProgress,
            completed_at: None,
            error: None,
            incomplete_details: None,
            output: vec![],
            usage: None,
        }
    }

    fn chunk_event_type(chunk: &ResponsesStreamEvent) -> Option<&'static str> {
        Some(chunk.into())
    }
}
