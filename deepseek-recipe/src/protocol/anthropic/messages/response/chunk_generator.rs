use crate::response::ProtocolResponse;
use crate::stream::{ChunkGenerator, OutputChunk, PromptUsage};

use super::schema::{
    MessagesDelta, MessagesResponse, MessagesResponseContent, MessagesStopInfo,
    MessagesStreamEvent, MessagesUsage,
};

/// Converts parsed inference output to Messages events.
#[derive(Debug)]
pub struct MessagesChunkGenerator {
    id: String,
    model: String,
    thinking_mode: bool,
    signature: String,
    content_index: usize,
    tool_call_index: usize,
    active: Option<MessagesContentBlockKind>,
    prompt_usage: PromptUsage,
    started: bool,
    finished: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum MessagesContentBlockKind {
    Text,
    Thinking,
    ToolUse,
}

impl MessagesChunkGenerator {
    /// The caller supplies a unique response ID and the public response model name.
    /// Tool IDs are derived from that response ID and the per-response tool index.
    /// The default opaque thinking signature is the response ID.
    pub fn new(id: impl Into<String>, model: impl Into<String>, thinking_mode: bool) -> Self {
        let id = id.into();
        Self {
            signature: id.clone(),
            id,
            model: model.into(),
            thinking_mode,
            content_index: 0,
            tool_call_index: 0,
            active: None,
            prompt_usage: PromptUsage::default(),
            started: false,
            finished: false,
        }
    }

    /// Set the signature emitted at the end of a thinking block.
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = signature.into();
        self
    }

    fn close_block(&mut self, events: &mut Vec<MessagesStreamEvent>) {
        if let Some(kind) = self.active.take() {
            if kind == MessagesContentBlockKind::Thinking {
                events.push(MessagesStreamEvent::ContentBlockDelta {
                    index: self.content_index,
                    delta: MessagesDelta::SignatureDelta {
                        signature: self.signature.clone(),
                    },
                });
            }
            events.push(MessagesStreamEvent::ContentBlockStop {
                index: self.content_index,
            });
            self.content_index += 1;
        }
    }

    fn open_block(
        &mut self,
        kind: MessagesContentBlockKind,
        content_block: MessagesResponseContent,
        events: &mut Vec<MessagesStreamEvent>,
    ) {
        self.close_block(events);
        events.push(MessagesStreamEvent::ContentBlockStart {
            index: self.content_index,
            content_block,
        });
        if self.content_index == 0 {
            events.push(MessagesStreamEvent::Ping);
        }
        self.active = Some(kind);
    }

    fn delta(&self, delta: MessagesDelta) -> MessagesStreamEvent {
        MessagesStreamEvent::ContentBlockDelta {
            index: self.content_index,
            delta,
        }
    }
}

impl ChunkGenerator for MessagesChunkGenerator {
    type Chunk = MessagesStreamEvent;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        if self.finished {
            return Vec::new();
        }
        let mut events = Vec::new();
        match chunk {
            OutputChunk::Start { usage, .. } => {
                if self.started {
                    return events;
                }
                self.started = true;
                self.prompt_usage = usage;
                events.push(MessagesStreamEvent::MessageStart {
                    message: MessagesResponse::new(
                        self.id.clone(),
                        self.model.clone(),
                        0,
                        usage.prompt_tokens,
                        usage.prompt_cache_hit_tokens,
                    ),
                });
                if self.thinking_mode {
                    self.open_block(
                        MessagesContentBlockKind::Thinking,
                        MessagesResponseContent::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                        &mut events,
                    );
                }
            }
            OutputChunk::Raw { content } => {
                if self.active != Some(MessagesContentBlockKind::Text) {
                    self.open_block(
                        MessagesContentBlockKind::Text,
                        MessagesResponseContent::Text {
                            text: String::new(),
                        },
                        &mut events,
                    );
                }
                events.push(self.delta(MessagesDelta::TextDelta { text: content }));
            }
            OutputChunk::Reasoning { content } => {
                if self.active != Some(MessagesContentBlockKind::Thinking) {
                    self.open_block(
                        MessagesContentBlockKind::Thinking,
                        MessagesResponseContent::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                        &mut events,
                    );
                }
                events.push(self.delta(MessagesDelta::ThinkingDelta { thinking: content }));
            }
            OutputChunk::ToolCall {
                tool_name,
                arguments,
            } => {
                let id = format!("toolu_{}_{}", self.id, self.tool_call_index);
                self.tool_call_index += 1;
                let input = if arguments.is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&arguments).unwrap_or_else(|_| serde_json::json!({}))
                };
                self.open_block(
                    MessagesContentBlockKind::ToolUse,
                    MessagesResponseContent::ToolUse {
                        id,
                        name: tool_name,
                        input,
                    },
                    &mut events,
                );
            }
            OutputChunk::ToolArgumentsDelta { content } => {
                if self.active == Some(MessagesContentBlockKind::ToolUse) {
                    events.push(self.delta(MessagesDelta::InputJsonDelta {
                        partial_json: content,
                    }));
                }
            }
            OutputChunk::Finish {
                reason,
                stop_sequence,
                usage,
            } => {
                self.close_block(&mut events);
                events.push(MessagesStreamEvent::MessageDelta {
                    delta: MessagesStopInfo {
                        stop_reason: Some(reason.into()),
                        stop_sequence,
                    },
                    usage: MessagesUsage::from((self.prompt_usage, usage)),
                });
                events.push(MessagesStreamEvent::MessageStop);
                self.finished = true;
            }
            OutputChunk::ToolCallBegin => {}
        }
        events
    }
}
