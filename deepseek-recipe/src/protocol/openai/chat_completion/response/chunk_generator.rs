use std::time::{SystemTime, UNIX_EPOCH};

use crate::protocol::tool_call_id::new_tool_call_id;
use crate::stream::{ChunkGenerator, OutputChunk, PromptUsage, TokenLogprob};

use super::schema::{
    CHAT_COMPLETION_CHUNK_OBJECT, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionFinishReason, ChatCompletionFunctionArgumentsDelta, ChatCompletionLogprobs,
    ChatCompletionMessageDelta, ChatCompletionResponseFunctionCall, ChatCompletionResponseToolCall,
    ChatCompletionRole, ChatCompletionTokenLogprob, ChatCompletionToolCallDelta,
    ChatCompletionToolCallType, ChatCompletionUsage,
};

/// Convert output token log probabilities to protocol entries.
fn token_logprobs(logprobs: &[TokenLogprob]) -> Vec<ChatCompletionTokenLogprob> {
    logprobs.iter().map(Into::into).collect()
}

/// A chunk generator that converts parsed output into Chat Completions
/// streaming chunks.
///
/// The caller supplies an ordered stream for one inference. Tool argument
/// deltas follow the tool call whose arguments they extend.
/// Created by [`ChatCompletionRequest::chunk_generator`].
///
/// [`ChatCompletionRequest::chunk_generator`]:
///     crate::request::ProtocolRequest::chunk_generator
#[derive(Debug)]
pub struct ChatCompletionChunkGenerator {
    response_id: String,
    model: String,
    created: u64,
    thinking_mode: bool,
    starts_in_reasoning: bool,
    include_usage: bool,
    next_tool_call_index: u32,
    reasoning_content: String,
    prompt_usage: PromptUsage,
    system_fingerprint: Option<String>,
}

impl ChatCompletionChunkGenerator {
    /// The caller supplies a unique response ID and public model name. Tool call
    /// IDs combine the per-response tool index, random characters, and a
    /// checksum.
    pub(crate) fn new(
        response_id: String,
        model: String,
        include_usage: bool,
        thinking_mode: bool,
    ) -> Self {
        Self {
            response_id,
            model,
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0),
            thinking_mode,
            starts_in_reasoning: thinking_mode,
            include_usage,
            next_tool_call_index: 0,
            reasoning_content: String::new(),
            prompt_usage: PromptUsage::default(),
            system_fingerprint: None,
        }
    }

    /// Set whether output starts in reasoning content. A continued prefix
    /// message with content starts in answer content. Defaults to the thinking
    /// mode.
    pub fn with_starts_in_reasoning(mut self, starts_in_reasoning: bool) -> Self {
        self.starts_in_reasoning = starts_in_reasoning;
        self
    }

    fn new_chunk(
        &self,
        delta: ChatCompletionMessageDelta,
        logprobs: Option<ChatCompletionLogprobs>,
        finish_reason: Option<ChatCompletionFinishReason>,
        usage: Option<ChatCompletionUsage>,
    ) -> Vec<ChatCompletionChunk> {
        let final_usage = usage;
        let usage = self.include_usage.then_some(None);
        let mut chunks = vec![ChatCompletionChunk {
            id: self.response_id.clone(),
            object: CHAT_COMPLETION_CHUNK_OBJECT,
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta,
                logprobs,
                finish_reason,
            }],
            usage,
        }];
        if self.include_usage
            && let Some(usage) = final_usage
        {
            chunks.push(ChatCompletionChunk {
                id: self.response_id.clone(),
                object: CHAT_COMPLETION_CHUNK_OBJECT,
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: self.system_fingerprint.clone(),
                choices: Vec::new(),
                usage: Some(Some(usage)),
            });
        }
        chunks
    }
}

impl ChunkGenerator for ChatCompletionChunkGenerator {
    type Chunk = ChatCompletionChunk;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        match chunk {
            OutputChunk::Start {
                system_fingerprint,
                prompt_usage,
            } => {
                self.prompt_usage = prompt_usage;
                self.system_fingerprint = system_fingerprint;
                let delta = ChatCompletionMessageDelta {
                    role: Some(ChatCompletionRole::Assistant),
                    content: if self.starts_in_reasoning {
                        Some(None)
                    } else {
                        Some(Some(String::new()))
                    },
                    reasoning_content: if self.starts_in_reasoning {
                        Some(Some(String::new()))
                    } else {
                        self.thinking_mode.then_some(None)
                    },
                    tool_calls: None,
                };
                self.new_chunk(delta, None, None, None)
            }
            OutputChunk::Content { content, logprobs } => {
                let delta = ChatCompletionMessageDelta {
                    content: Some(Some(content)),
                    reasoning_content: self.thinking_mode.then_some(None),
                    ..Default::default()
                };
                let logprobs = logprobs.map(|logprobs| ChatCompletionLogprobs {
                    content: Some(token_logprobs(&logprobs)),
                    reasoning_content: None,
                });
                self.new_chunk(delta, logprobs, None, None)
            }
            OutputChunk::Reasoning { content, logprobs } => {
                self.reasoning_content.push_str(&content);
                let delta = ChatCompletionMessageDelta {
                    content: Some(None),
                    reasoning_content: Some(Some(content)),
                    ..Default::default()
                };
                let logprobs = logprobs.map(|logprobs| ChatCompletionLogprobs {
                    content: None,
                    reasoning_content: Some(token_logprobs(&logprobs)),
                });
                self.new_chunk(delta, logprobs, None, None)
            }
            OutputChunk::ToolCallBegin => vec![],
            OutputChunk::ToolCallNamed { tool_name } => {
                let index = self.next_tool_call_index;
                self.next_tool_call_index += 1;
                let empty_thinking = self.thinking_mode && self.reasoning_content.is_empty();
                let delta = ChatCompletionMessageDelta {
                    tool_calls: Some(vec![ChatCompletionToolCallDelta::Start {
                        index,
                        call: ChatCompletionResponseToolCall {
                            id: new_tool_call_id(index as usize, empty_thinking),
                            r#type: ChatCompletionToolCallType::Function,
                            function: ChatCompletionResponseFunctionCall {
                                name: tool_name,
                                arguments: String::new(),
                            },
                        },
                    }]),
                    ..Default::default()
                };
                self.new_chunk(delta, None, None, None)
            }
            OutputChunk::ToolCallArgumentsDelta { arguments } => {
                // Arguments without a preceding tool call have no index.
                let Some(index) = self.next_tool_call_index.checked_sub(1) else {
                    return Vec::new();
                };
                let delta = ChatCompletionMessageDelta {
                    tool_calls: Some(vec![ChatCompletionToolCallDelta::Arguments {
                        index,
                        function: ChatCompletionFunctionArgumentsDelta { arguments },
                    }]),
                    ..Default::default()
                };
                self.new_chunk(delta, None, None, None)
            }
            OutputChunk::Finish {
                finish_reason,
                completion_usage,
                ..
            } => {
                let delta = ChatCompletionMessageDelta {
                    content: Some(Some(String::new())),
                    reasoning_content: self.thinking_mode.then_some(None),
                    ..Default::default()
                };
                let mut usage = ChatCompletionUsage::from((self.prompt_usage, completion_usage));
                if !self.thinking_mode {
                    usage.completion_tokens_details = None;
                }
                self.new_chunk(delta, None, Some(finish_reason.into()), Some(usage))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::protocol::tool_call_id::is_empty_thinking_tool_call_id;

    use super::*;

    async fn first_tool_call_id(thinking_mode: bool, reasoning: Option<&str>) -> String {
        let mut generator = ChatCompletionChunkGenerator::new(
            "response".to_string(),
            "model".to_string(),
            false,
            thinking_mode,
        );
        generator
            .generate(OutputChunk::Start {
                system_fingerprint: None,
                prompt_usage: PromptUsage::default(),
            })
            .await;
        if let Some(content) = reasoning {
            generator
                .generate(OutputChunk::Reasoning {
                    content: content.to_string(),
                    logprobs: None,
                })
                .await;
        }
        let chunks = generator
            .generate(OutputChunk::ToolCallNamed {
                tool_name: "f".to_string(),
            })
            .await;
        match chunks[0].choices[0].delta.tool_calls.as_deref() {
            Some([ChatCompletionToolCallDelta::Start { call, .. }]) => call.id.clone(),
            _ => panic!("expected a tool call start delta"),
        }
    }

    #[tokio::test]
    async fn tool_call_without_reasoning_is_marked_empty_thinking() {
        let id = first_tool_call_id(true, None).await;
        assert!(is_empty_thinking_tool_call_id(&id));
    }

    #[tokio::test]
    async fn tool_call_after_reasoning_is_not_marked() {
        let id = first_tool_call_id(true, Some("thinking")).await;
        assert!(!is_empty_thinking_tool_call_id(&id));
    }

    #[tokio::test]
    async fn non_thinking_tool_call_is_not_marked() {
        let id = first_tool_call_id(false, None).await;
        assert!(!is_empty_thinking_tool_call_id(&id));
    }
}

#[cfg(test)]
mod contribution_regression {
    use super::*;

    #[tokio::test]
    async fn requested_usage_has_a_separate_empty_choice_chunk() {
        let mut generator =
            ChatCompletionChunkGenerator::new("id".into(), "model".into(), true, false);
        let chunks = generator
            .generate(OutputChunk::Finish {
                finish_reason: crate::stream::FinishReason::Stop,
                stop_sequence: None,
                completion_usage: crate::stream::CompletionUsage {
                    completion_tokens: 7,
                    reasoning_tokens: None,
                },
            })
            .await;
        assert_eq!(chunks.len(), 2);
        assert!(matches!(chunks[0].usage, Some(None)));
        assert_eq!(
            chunks[0].choices[0].finish_reason,
            Some(ChatCompletionFinishReason::Stop)
        );
        assert!(chunks[1].choices.is_empty());
        assert_eq!(
            chunks[1]
                .usage
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap()
                .completion_tokens,
            7
        );
    }

    #[tokio::test]
    async fn usage_is_omitted_when_not_requested() {
        let mut generator =
            ChatCompletionChunkGenerator::new("id".into(), "model".into(), false, false);
        let chunks = generator
            .generate(OutputChunk::Finish {
                finish_reason: crate::stream::FinishReason::Stop,
                stop_sequence: None,
                completion_usage: crate::stream::CompletionUsage::default(),
            })
            .await;
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].usage.is_none());
    }
}
