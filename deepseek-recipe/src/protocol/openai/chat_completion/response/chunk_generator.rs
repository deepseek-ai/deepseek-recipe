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
    /// Which of the request's `n` parallel completions this generator's
    /// output belongs to. Tagged on every emitted chunk's `choices[].index`.
    choice_index: u32,
    tool_call_index: u32,
    prompt_usage: PromptUsage,
    system_fingerprint: Option<String>,
}

impl ChatCompletionChunkGenerator {
    /// The caller supplies a unique response ID and public model name. Tool call
    /// IDs combine the response ID, choice index, and per-response tool index.
    ///
    /// Defaults to choice index `0`; use [`Self::with_choice_index`] for
    /// requests with `n > 1`.
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
            choice_index: 0,
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

    /// Tag this generator's output as belonging to a specific completion.
    ///
    /// The caller creates one generator per completion when `n > 1`, each with
    /// a distinct `choice_index` (`0..n`), and drives each through its own
    /// [`crate::stream::StreamProcessor`].
    pub fn with_choice_index(mut self, choice_index: u32) -> Self {
        self.choice_index = choice_index;
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
                index: self.choice_index,
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
                            id: format!("call_{}_{}_{index}", self.id, self.choice_index),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A generator built for one choice of an `n > 1` request must tag every
    /// chunk it produces with that choice's index, not the default `0`.
    #[tokio::test]
    async fn with_choice_index_tags_every_emitted_chunk() {
        let mut generator =
            ChatCompletionChunkGenerator::new("resp_1".into(), "deepseek-chat".into(), true, false)
                .with_choice_index(2);

        let start_chunks = generator
            .generate(OutputChunk::Start {
                system_fingerprint: None,
                usage: PromptUsage::default(),
            })
            .await;
        let text_chunks = generator
            .generate(OutputChunk::Raw {
                content: "hi".into(),
            })
            .await;

        for chunk in start_chunks.iter().chain(text_chunks.iter()) {
            for choice in &chunk.choices {
                assert_eq!(choice.index, 2);
            }
        }
    }

    /// Two generators for different choices of the same response must not
    /// mint colliding tool call ids, since both start from the same response
    /// id and the same per-generator tool call counter.
    #[tokio::test]
    async fn tool_call_ids_are_unique_across_choices() {
        let mut choice0 =
            ChatCompletionChunkGenerator::new("resp_1".into(), "deepseek-chat".into(), true, false)
                .with_choice_index(0);
        let mut choice1 =
            ChatCompletionChunkGenerator::new("resp_1".into(), "deepseek-chat".into(), true, false)
                .with_choice_index(1);

        let chunks0 = choice0
            .generate(OutputChunk::ToolCall {
                tool_name: "search".into(),
                arguments: String::new(),
            })
            .await;
        let chunks1 = choice1
            .generate(OutputChunk::ToolCall {
                tool_name: "search".into(),
                arguments: String::new(),
            })
            .await;

        assert_ne!(tool_call_id(&chunks0), tool_call_id(&chunks1));
    }

    fn tool_call_id(chunks: &[ChatCompletionChunk]) -> String {
        for chunk in chunks {
            for choice in &chunk.choices {
                let Some(tool_calls) = &choice.delta.tool_calls else {
                    continue;
                };
                for call in tool_calls {
                    if let ChatCompletionToolCallDelta::Start { call, .. } = call {
                        return call.id.clone();
                    }
                }
            }
        }
        panic!("no tool call start delta found");
    }
}
