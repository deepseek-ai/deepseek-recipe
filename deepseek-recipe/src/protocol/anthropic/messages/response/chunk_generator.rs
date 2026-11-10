use crate::protocol::tool_call_id::new_tool_call_id;
use crate::response::ProtocolResponse;
use crate::stream::{ChunkGenerator, OutputChunk, PromptUsage};

use super::schema::{
    MessagesContentBlockDelta, MessagesMessageDelta, MessagesResponse,
    MessagesResponseContentBlock, MessagesStreamEvent, MessagesUsage,
};

/// A chunk generator that converts parsed output into Messages events.
#[derive(Debug)]
pub struct MessagesChunkGenerator {
    response_id: String,
    model: String,
    thinking_mode: bool,
    signature: String,
    content_index: usize,
    next_tool_call_index: usize,
    reasoning_content: String,
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
    /// Tool IDs combine the per-response tool index, random characters, and a
    /// checksum. The default opaque thinking signature is the response ID.
    pub fn new(
        response_id: impl Into<String>,
        model: impl Into<String>,
        thinking_mode: bool,
    ) -> Self {
        let response_id = response_id.into();
        Self {
            signature: response_id.clone(),
            response_id,
            model: model.into(),
            thinking_mode,
            content_index: 0,
            next_tool_call_index: 0,
            reasoning_content: String::new(),
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
                    delta: MessagesContentBlockDelta::SignatureDelta {
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
        content_block: MessagesResponseContentBlock,
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

    fn delta(&self, delta: MessagesContentBlockDelta) -> MessagesStreamEvent {
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
            OutputChunk::Start { prompt_usage, .. } => {
                if self.started {
                    return events;
                }
                self.started = true;
                self.prompt_usage = prompt_usage;
                events.push(MessagesStreamEvent::MessageStart {
                    message: MessagesResponse::new(
                        self.response_id.clone(),
                        self.model.clone(),
                        0,
                        prompt_usage.prompt_tokens,
                        prompt_usage.prompt_cache_hit_tokens,
                    ),
                });
                if self.thinking_mode {
                    self.open_block(
                        MessagesContentBlockKind::Thinking,
                        MessagesResponseContentBlock::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                        &mut events,
                    );
                }
            }
            OutputChunk::Content { content, .. } => {
                if self.active != Some(MessagesContentBlockKind::Text) {
                    self.open_block(
                        MessagesContentBlockKind::Text,
                        MessagesResponseContentBlock::Text {
                            text: String::new(),
                        },
                        &mut events,
                    );
                }
                events.push(self.delta(MessagesContentBlockDelta::TextDelta { text: content }));
            }
            OutputChunk::Reasoning { content, .. } => {
                if self.active != Some(MessagesContentBlockKind::Thinking) {
                    self.open_block(
                        MessagesContentBlockKind::Thinking,
                        MessagesResponseContentBlock::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                        &mut events,
                    );
                }
                self.reasoning_content.push_str(&content);
                events.push(
                    self.delta(MessagesContentBlockDelta::ThinkingDelta { thinking: content }),
                );
            }
            OutputChunk::ToolCallNamed { tool_name } => {
                let empty_thinking = self.thinking_mode && self.reasoning_content.is_empty();
                let id = new_tool_call_id(self.next_tool_call_index, empty_thinking);
                self.next_tool_call_index += 1;
                self.open_block(
                    MessagesContentBlockKind::ToolUse,
                    MessagesResponseContentBlock::ToolUse {
                        id,
                        name: tool_name,
                        input: serde_json::json!({}),
                    },
                    &mut events,
                );
            }
            OutputChunk::ToolCallArgumentsDelta { arguments } => {
                if self.active == Some(MessagesContentBlockKind::ToolUse) {
                    events.push(self.delta(MessagesContentBlockDelta::InputJsonDelta {
                        partial_json: arguments,
                    }));
                }
            }
            OutputChunk::Finish {
                finish_reason,
                stop_sequence,
                completion_usage,
            } => {
                self.close_block(&mut events);
                events.push(MessagesStreamEvent::MessageDelta {
                    delta: MessagesMessageDelta {
                        stop_reason: Some(finish_reason.into()),
                        stop_sequence,
                    },
                    usage: MessagesUsage::from((self.prompt_usage, completion_usage)),
                });
                events.push(MessagesStreamEvent::MessageStop);
                self.finished = true;
            }
            OutputChunk::ToolCallBegin => {}
        }
        events
    }
}
