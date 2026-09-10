use serde::Serialize;

use crate::response::ProtocolResponse;
use crate::stream::{CompletionUsage, FinishReason, PromptUsage};
use crate::util::append_delta::AppendDelta;

use super::chunk_generator::ChatCompletionChunkGenerator;

const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
pub(crate) const CHAT_COMPLETION_CHUNK_OBJECT: &str = "chat.completion.chunk";

/// A complete response, also usable as an accumulator for generated chunks.
/// Chunks are accumulated in inference order. Tool argument deltas extend the
/// tool call identified by their index.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionUsage>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    pub logprobs: Option<ChatCompletionLogprobs>,
    pub finish_reason: Option<ChatCompletionFinishReason>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<ChatCompletionUsage>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionMessageDelta,
    pub logprobs: Option<ChatCompletionLogprobs>,
    pub finish_reason: Option<ChatCompletionFinishReason>,
}

/// A complete assistant message accumulated from response deltas.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponseMessage {
    pub role: ChatCompletionRole,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionResponseToolCall>>,
}

/// An incremental update to an assistant message.
#[derive(Debug, Default, Serialize)]
pub struct ChatCompletionMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatCompletionRole>,
    /// `None` omits the field; `Some(None)` serializes an explicit null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionToolCallDelta>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRole {
    Assistant,
}

/// A complete tool call in an assistant response message.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponseToolCall {
    pub id: String,
    pub r#type: ChatCompletionToolCallType,
    pub function: ChatCompletionResponseFunctionCall,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// An initial tool call or an update to its arguments.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ChatCompletionToolCallDelta {
    Start {
        index: u32,
        #[serde(flatten)]
        call: ChatCompletionResponseToolCall,
    },
    Arguments {
        index: u32,
        function: ChatCompletionFunctionArgumentsDelta,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionToolCallType {
    Function,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionFunctionArgumentsDelta {
    pub arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionFinishReason {
    Stop,
    Length,
    ContentFilter,
    InsufficientSystemResource,
    ToolCalls,
    Aborted,
}

impl From<FinishReason> for ChatCompletionFinishReason {
    fn from(value: FinishReason) -> Self {
        match value {
            FinishReason::Stop | FinishReason::StopSequence | FinishReason::EndOfStream => {
                Self::Stop
            }
            FinishReason::ToolCalls => Self::ToolCalls,
            FinishReason::ContentFilter => Self::ContentFilter,
            FinishReason::Length => Self::Length,
        }
    }
}

/// Log probabilities are not supplied by the current inference contract.
#[derive(Debug, Serialize)]
pub struct ChatCompletionLogprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ChatCompletionTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Vec<ChatCompletionTokenLogprob>>,
}

impl AppendDelta<ChatCompletionLogprobs> for ChatCompletionLogprobs {
    fn append(&mut self, delta: ChatCompletionLogprobs) {
        self.content.append(delta.content);
        self.reasoning_content.append(delta.reasoning_content);
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<ChatCompletionTopLogprob>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: ChatCompletionInputTokenUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<ChatCompletionOutputTokenUsage>,
    pub prompt_cache_hit_tokens: usize,
    pub prompt_cache_miss_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionInputTokenUsage {
    pub cached_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionOutputTokenUsage {
    pub reasoning_tokens: usize,
}

impl From<(PromptUsage, CompletionUsage)> for ChatCompletionUsage {
    fn from((prompt, completion): (PromptUsage, CompletionUsage)) -> Self {
        Self {
            prompt_tokens: prompt.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            total_tokens: prompt.prompt_tokens + completion.completion_tokens,
            prompt_tokens_details: ChatCompletionInputTokenUsage {
                cached_tokens: prompt.prompt_cache_hit_tokens,
            },
            completion_tokens_details: None,
            prompt_cache_hit_tokens: prompt.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens: prompt
                .prompt_tokens
                .saturating_sub(prompt.prompt_cache_hit_tokens),
        }
    }
}

impl ProtocolResponse for ChatCompletionResponse {
    type ChunkGenerator = ChatCompletionChunkGenerator;

    fn new(
        id: String,
        model: String,
        created: u64,
        _prompt_tokens: usize,
        _prompt_cache_hit_tokens: usize,
    ) -> Self {
        Self {
            id,
            object: CHAT_COMPLETION_OBJECT,
            created,
            model,
            system_fingerprint: None,
            choices: vec![],
            usage: None,
        }
    }

    fn done_message() -> Option<String> {
        Some("[DONE]".to_string())
    }
}

impl AppendDelta<ChatCompletionChunk> for ChatCompletionResponse {
    fn append(&mut self, chunk: ChatCompletionChunk) {
        if let Some(Some(usage)) = chunk.usage {
            self.usage = Some(usage);
        }
        if self.system_fingerprint.is_none() {
            self.system_fingerprint = chunk.system_fingerprint;
        }

        for chunk_choice in chunk.choices {
            let choice = match self
                .choices
                .iter_mut()
                .find(|choice| choice.index == chunk_choice.index)
            {
                Some(choice) => choice,
                None => {
                    self.choices.push(ChatCompletionChoice {
                        index: chunk_choice.index,
                        message: ChatCompletionResponseMessage {
                            role: ChatCompletionRole::Assistant,
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        logprobs: None,
                        finish_reason: None,
                    });
                    self.choices.last_mut().expect("the choice was inserted")
                }
            };
            choice.message.append(chunk_choice.delta);
            choice.logprobs.append(chunk_choice.logprobs);
            if chunk_choice.finish_reason.is_some() {
                choice.finish_reason = chunk_choice.finish_reason;
            }
        }
    }
}

impl AppendDelta<ChatCompletionMessageDelta> for ChatCompletionResponseMessage {
    fn append(&mut self, delta: ChatCompletionMessageDelta) {
        if let Some(role) = delta.role {
            self.role = role;
        }
        self.content.append(delta.content.flatten());
        self.reasoning_content
            .append(delta.reasoning_content.flatten());
        if let Some(tool_deltas) = delta.tool_calls {
            let tool_calls = self.tool_calls.get_or_insert_with(Vec::new);
            for tool_delta in tool_deltas {
                match tool_delta {
                    ChatCompletionToolCallDelta::Start { call, .. } => tool_calls.push(call),
                    ChatCompletionToolCallDelta::Arguments { index, function } => {
                        if let Some(call) = tool_calls.get_mut(index as usize) {
                            call.function.arguments.push_str(&function.arguments);
                        }
                    }
                }
            }
        }
    }
}
