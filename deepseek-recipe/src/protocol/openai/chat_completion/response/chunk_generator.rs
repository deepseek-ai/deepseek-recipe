use std::time::{SystemTime, UNIX_EPOCH};

use crate::stream::{ChunkGenerator, OutputChunk, PromptUsage};

use super::schema::{
    CHAT_COMPLETION_CHUNK_OBJECT, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionFinishReason, ChatCompletionFunctionArgumentsDelta, ChatCompletionMessageDelta,
    ChatCompletionResponseFunctionCall, ChatCompletionResponseToolCall, ChatCompletionRole,
    ChatCompletionToolCallDelta, ChatCompletionToolCallType, ChatCompletionUsage,
};

/// Converts parsed output into Chat Completions streaming chunks.
///
/// The caller supplies an ordered stream for one inference. Tool argument
/// deltas follow the tool call whose arguments they extend.
#[derive(Debug)]
pub struct ChatCompletionChunkGenerator {
    id: String,
    model: String,
    created: u64,
    thinking_mode: bool,
    include_usage: bool,
    tool_call_index: u32,
    prompt_usage: PromptUsage,
    system_fingerprint: Option<String>,
}

impl ChatCompletionChunkGenerator {
    /// The caller supplies a unique response ID and public model name. Tool call
    /// IDs combine the response ID and per-response tool index.
    pub fn new(id: String, model: String, include_usage: bool, thinking_mode: bool) -> Self {
        Self {
            id,
            model,
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0),
            thinking_mode,
            include_usage,
            tool_call_index: 0,
            prompt_usage: PromptUsage::default(),
            system_fingerprint: None,
        }
    }

    /// Apply usage settings extracted from the original Chat Completions request.
    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = include_usage;
        self
    }

    fn new_chunk(
        &self,
        delta: ChatCompletionMessageDelta,
        finish_reason: Option<ChatCompletionFinishReason>,
        usage: Option<ChatCompletionUsage>,
    ) -> Vec<ChatCompletionChunk> {
        // Include null usage before the final snapshot when requested. Otherwise the
        // final chunk includes usage.
        let usage = if self.include_usage {
            Some(usage)
        } else {
            usage.map(Some)
        };
        vec![ChatCompletionChunk {
            id: self.id.clone(),
            object: CHAT_COMPLETION_CHUNK_OBJECT,
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta,
                logprobs: None,
                finish_reason,
            }],
            usage,
        }]
    }
}

impl ChunkGenerator for ChatCompletionChunkGenerator {
    type Chunk = ChatCompletionChunk;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        match chunk {
            OutputChunk::Start {
                system_fingerprint,
                usage,
            } => {
                self.prompt_usage = usage;
                self.system_fingerprint = system_fingerprint;
                let delta = ChatCompletionMessageDelta {
                    role: Some(ChatCompletionRole::Assistant),
                    content: if self.thinking_mode {
                        Some(None)
                    } else {
                        Some(Some(String::new()))
                    },
                    reasoning_content: self.thinking_mode.then(|| Some(String::new())),
                    tool_calls: None,
                };
                self.new_chunk(delta, None, None)
            }
            OutputChunk::Raw { content } => {
                let delta = ChatCompletionMessageDelta {
                    content: Some(Some(content)),
                    reasoning_content: self.thinking_mode.then_some(None),
                    ..Default::default()
                };
                self.new_chunk(delta, None, None)
            }
            OutputChunk::Reasoning { content } => {
                let delta = ChatCompletionMessageDelta {
                    content: Some(None),
                    reasoning_content: Some(Some(content)),
                    ..Default::default()
                };
                self.new_chunk(delta, None, None)
            }
            OutputChunk::ToolCallBegin => vec![],
            OutputChunk::ToolCall {
                tool_name,
                arguments,
            } => {
                let index = self.tool_call_index;
                self.tool_call_index += 1;
                let delta = ChatCompletionMessageDelta {
                    tool_calls: Some(vec![ChatCompletionToolCallDelta::Start {
                        index,
                        call: ChatCompletionResponseToolCall {
                            id: format!("call_{}_{index}", self.id),
                            r#type: ChatCompletionToolCallType::Function,
                            function: ChatCompletionResponseFunctionCall {
                                name: tool_name,
                                arguments,
                            },
                        },
                    }]),
                    ..Default::default()
                };
                self.new_chunk(delta, None, None)
            }
            OutputChunk::ToolArgumentsDelta { content } => {
                let delta = ChatCompletionMessageDelta {
                    tool_calls: Some(vec![ChatCompletionToolCallDelta::Arguments {
                        index: self.tool_call_index - 1,
                        function: ChatCompletionFunctionArgumentsDelta { arguments: content },
                    }]),
                    ..Default::default()
                };
                self.new_chunk(delta, None, None)
            }
            OutputChunk::Finish { reason, usage, .. } => {
                let delta = ChatCompletionMessageDelta {
                    content: Some(Some(String::new())),
                    reasoning_content: self.thinking_mode.then_some(None),
                    ..Default::default()
                };
                self.new_chunk(
                    delta,
                    Some(reason.into()),
                    Some(ChatCompletionUsage::from((self.prompt_usage, usage))),
                )
            }
        }
    }
}
