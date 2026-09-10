use std::collections::BTreeMap;

use serde::Serialize;

use crate::response::ProtocolResponse;
use crate::stream::{CompletionUsage, FinishReason, PromptUsage};
use crate::util::append_delta::AppendDelta;

use super::chunk_generator::MessagesChunkGenerator;

/// A complete Messages response, also usable as an accumulator for generated
/// events.
///
/// Tool argument fragments are parsed when their content block stops. Invalid
/// or truncated JSON produces `{}`. Events with invalid indices or incompatible
/// delta types leave the response unchanged.
#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: MessagesResponseType,
    pub role: MessagesRole,
    pub model: String,
    pub content: Vec<MessagesResponseContent>,
    pub stop_reason: Option<MessagesStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: MessagesUsage,
    #[serde(skip)]
    partial_tool_inputs: BTreeMap<usize, String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagesResponseType {
    Message,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagesRole {
    Assistant,
}

/// A complete content block, distinct from the delta applied to it.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesResponseContent {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagesStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    Refusal,
}

impl From<FinishReason> for MessagesStopReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::Stop | FinishReason::EndOfStream => Self::EndTurn,
            FinishReason::Length => Self::MaxTokens,
            FinishReason::StopSequence => Self::StopSequence,
            FinishReason::ToolCalls => Self::ToolUse,
            FinishReason::ContentFilter => Self::Refusal,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MessagesStopInfo {
    pub stop_reason: Option<MessagesStopReason>,
    pub stop_sequence: Option<String>,
}

/// Protocol representation of caller-supplied accounting.
#[derive(Debug, Serialize)]
pub struct MessagesUsage {
    pub input_tokens: usize,
    pub cache_creation_input_tokens: usize,
    pub cache_read_input_tokens: usize,
    pub output_tokens: usize,
    pub service_tier: MessagesServiceTier,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagesServiceTier {
    Standard,
}

impl From<(PromptUsage, CompletionUsage)> for MessagesUsage {
    fn from((prompt, completion): (PromptUsage, CompletionUsage)) -> Self {
        Self {
            input_tokens: prompt
                .prompt_tokens
                .saturating_sub(prompt.prompt_cache_hit_tokens),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: prompt.prompt_cache_hit_tokens,
            output_tokens: completion.completion_tokens,
            service_tier: MessagesServiceTier::Standard,
        }
    }
}

/// A Messages stream event. Serialize it as JSON and obtain its SSE event name
/// with [`Self::event_type`]. The caller supplies SSE framing and transport.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesStreamEvent {
    MessageStart {
        message: MessagesResponse,
    },
    ContentBlockStart {
        index: usize,
        content_block: MessagesResponseContent,
    },
    ContentBlockDelta {
        index: usize,
        delta: MessagesDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessagesStopInfo,
        usage: MessagesUsage,
    },
    MessageStop,
    Ping,
}

impl MessagesStreamEvent {
    /// Event name used by the SSE transport.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop => "message_stop",
            Self::Ping => "ping",
        }
    }
}

impl ProtocolResponse for MessagesResponse {
    type ChunkGenerator = MessagesChunkGenerator;

    fn new(
        id: String,
        model: String,
        _created: u64,
        prompt_tokens: usize,
        prompt_cache_hit_tokens: usize,
    ) -> Self {
        Self {
            id,
            response_type: MessagesResponseType::Message,
            role: MessagesRole::Assistant,
            model,
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: MessagesUsage::from((
                PromptUsage {
                    prompt_tokens,
                    prompt_cache_hit_tokens,
                },
                CompletionUsage::default(),
            )),
            partial_tool_inputs: BTreeMap::new(),
        }
    }

    fn chunk_event_type(chunk: &MessagesStreamEvent) -> Option<&'static str> {
        Some(chunk.event_type())
    }
}

impl AppendDelta<MessagesStreamEvent> for MessagesResponse {
    fn append(&mut self, event: MessagesStreamEvent) {
        match event {
            MessagesStreamEvent::MessageStart { message } => *self = message,
            MessagesStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if index == self.content.len() {
                    self.content.push(content_block);
                }
            }
            MessagesStreamEvent::ContentBlockDelta { index, delta } => {
                let Some(block) = self.content.get_mut(index) else {
                    return;
                };
                match (block, delta) {
                    (
                        MessagesResponseContent::Text { text },
                        MessagesDelta::TextDelta { text: value },
                    ) => text.push_str(&value),
                    (
                        MessagesResponseContent::Thinking { thinking, .. },
                        MessagesDelta::ThinkingDelta { thinking: value },
                    ) => thinking.push_str(&value),
                    (
                        MessagesResponseContent::Thinking { signature, .. },
                        MessagesDelta::SignatureDelta { signature: value },
                    ) => signature.push_str(&value),
                    (
                        MessagesResponseContent::ToolUse { .. },
                        MessagesDelta::InputJsonDelta { partial_json },
                    ) => {
                        self.partial_tool_inputs
                            .entry(index)
                            .or_default()
                            .push_str(&partial_json);
                    }
                    _ => {}
                }
            }
            MessagesStreamEvent::ContentBlockStop { index } => {
                if let Some(json) = self.partial_tool_inputs.remove(&index)
                    && let Some(MessagesResponseContent::ToolUse { input, .. }) =
                        self.content.get_mut(index)
                {
                    *input = serde_json::from_str(&json).unwrap_or_else(|_| serde_json::json!({}));
                }
            }
            MessagesStreamEvent::MessageDelta { delta, usage } => {
                self.stop_reason = delta.stop_reason;
                self.stop_sequence = delta.stop_sequence;
                self.usage = usage;
            }
            MessagesStreamEvent::MessageStop | MessagesStreamEvent::Ping => {}
        }
    }
}
