use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::protocol::openai::responses::request::ResponsesToolSearchExecution;
use crate::protocol::tool_call_id::new_tool_call_id;
use crate::request::ResponseOptions;
use crate::response::ProtocolResponse;
use crate::stream::{ChunkGenerator, FinishReason, OutputChunk, PromptUsage, TokenLogprob};

use super::custom_tool_input::ResponsesCustomToolInputParser;
use super::schema::{
    ResponsesContentPart, ResponsesEcho, ResponsesIncompleteDetails, ResponsesIncompleteReason,
    ResponsesOutputItem, ResponsesPhase, ResponsesResponse, ResponsesRole, ResponsesStatus,
    ResponsesStreamEvent, ResponsesTokenLogprob, ResponsesUsage,
};

/// A chunk generator that converts one [`OutputChunk::Start`] to
/// [`OutputChunk::Finish`] sequence into Responses events.
///
/// Created by [`ResponsesRequest::chunk_generator`].
///
/// The generator ignores a repeated `Start` and chunks outside this sequence.
/// Argument deltas extend only the active tool call.
/// Namespace tool names and reported request settings come from conversion;
/// [`Self::with_namespace_tool_map`] and [`Self::with_echo`] override them.
///
/// [`ResponsesRequest::chunk_generator`]:
///     crate::request::ProtocolRequest::chunk_generator
#[derive(Debug)]
pub struct ResponsesChunkGenerator {
    response: ResponsesResponse,
    active: Option<usize>,
    sequence_number: usize,
    next_tool_call_index: usize,
    thinking_mode: bool,
    reasoning_content: String,
    custom_tool_names: HashSet<String>,
    namespace_tool_map: Option<HashMap<String, (String, String)>>,
    client_tool_search: bool,
    custom_tool_input: Option<ResponsesCustomToolInputParser>,
    prompt_usage: PromptUsage,
    started: bool,
    finished: bool,
}

enum ResponsesTextKind {
    Output,
    Reasoning,
}

impl ResponsesChunkGenerator {
    /// Construct a generator for one response.
    ///
    /// The caller supplies a unique response ID, the public model name, and
    /// the thinking mode. Item IDs derive from the response ID and output
    /// indices, and the generator creates call IDs. In thinking mode, the ID
    /// of a tool call from a turn without reasoning content carries the
    /// empty-thinking marker.
    pub(crate) fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        thinking_mode: bool,
        response_options: ResponseOptions,
    ) -> Self {
        let mut response = ResponsesResponse::new(id.into(), model.into(), timestamp(), 0, 0);
        if let Some(echo) = response_options.responses_echo {
            response.set_echo(echo);
        }
        Self {
            response,
            active: None,
            sequence_number: 0,
            next_tool_call_index: 0,
            thinking_mode,
            reasoning_content: String::new(),
            custom_tool_names: response_options.custom_tool_names,
            namespace_tool_map: response_options.namespace_tool_map,
            client_tool_search: response_options.client_tool_search,
            custom_tool_input: None,
            prompt_usage: PromptUsage::default(),
            started: false,
            finished: false,
        }
    }

    /// Set the function names of namespace tools mapped to their namespace and
    /// local name, from [`crate::openai::ResponsesRequest::namespace_tool_map`].
    /// A tool call whose name is not in the map has no namespace.
    pub fn with_namespace_tool_map(
        mut self,
        namespace_tool_map: HashMap<String, (String, String)>,
    ) -> Self {
        self.namespace_tool_map = Some(namespace_tool_map);
        self
    }

    /// Set the request settings that response objects report, from
    /// [`crate::openai::ResponsesRequest::echo`].
    pub fn with_echo(mut self, echo: ResponsesEcho) -> Self {
        self.response.set_echo(echo);
        self
    }

    /// Resolve a tool call name to its local name and namespace.
    fn resolve_namespace(&self, tool_name: String) -> (String, Option<String>) {
        match &self.namespace_tool_map {
            Some(names) => match names.get(&tool_name) {
                Some((namespace, name)) => (name.clone(), Some(namespace.clone())),
                None => (tool_name, None),
            },
            None => match tool_name.split_once("::") {
                Some((namespace, name)) => (name.to_string(), Some(namespace.to_string())),
                None => (tool_name, None),
            },
        }
    }

    fn next_sequence_number(&mut self) -> usize {
        let sequence_number = self.sequence_number;
        self.sequence_number += 1;
        sequence_number
    }

    fn open_item(&mut self, mut item: ResponsesOutputItem, events: &mut Vec<ResponsesStreamEvent>) {
        let output_index = self.response.output.len();
        self.response.output.push(item.clone());
        let content = match &mut item {
            ResponsesOutputItem::Message { id, content, .. }
            | ResponsesOutputItem::Reasoning { id, content, .. } => {
                content.pop().map(|part| (id.clone(), part))
            }
            ResponsesOutputItem::FunctionCall { .. }
            | ResponsesOutputItem::CustomToolCall { .. }
            | ResponsesOutputItem::ToolSearchCall { .. } => None,
        };
        self.active = Some(output_index);
        events.push(ResponsesStreamEvent::OutputItemAdded {
            item,
            output_index,
            sequence_number: self.next_sequence_number(),
        });
        if let Some((item_id, part)) = content {
            events.push(ResponsesStreamEvent::ContentPartAdded {
                content_index: 0,
                item_id,
                output_index,
                part,
                sequence_number: self.next_sequence_number(),
            });
        }
    }

    fn close_item(&mut self, status: ResponsesStatus, events: &mut Vec<ResponsesStreamEvent>) {
        let Some(output_index) = self.active.take() else {
            return;
        };
        if let Some(mut parser) = self.custom_tool_input.take() {
            self.append_custom_input(output_index, parser.finish(), events);
        }
        let item = &mut self.response.output[output_index];
        if let ResponsesOutputItem::Message { phase, .. } = item
            && phase.is_none()
        {
            *phase = Some(ResponsesPhase::FinalAnswer);
        }
        match item {
            ResponsesOutputItem::Message {
                status: item_status,
                ..
            }
            | ResponsesOutputItem::Reasoning {
                status: item_status,
                ..
            }
            | ResponsesOutputItem::FunctionCall {
                status: item_status,
                ..
            }
            | ResponsesOutputItem::CustomToolCall {
                status: item_status,
                ..
            } => *item_status = status,
            ResponsesOutputItem::ToolSearchCall {
                status: item_status,
                arguments,
                raw_arguments,
                ..
            } => {
                if status == ResponsesStatus::Completed
                    && let Ok(parsed) = serde_json::from_str(raw_arguments)
                {
                    *arguments = parsed;
                    *item_status = ResponsesStatus::Completed;
                } else {
                    *item_status = ResponsesStatus::Incomplete;
                }
            }
        }
        let item = item.clone();
        match &item {
            ResponsesOutputItem::Message { id, content, .. }
            | ResponsesOutputItem::Reasoning { id, content, .. } => {
                for (content_index, part) in content.iter().enumerate() {
                    let sequence_number = self.next_sequence_number();
                    events.push(match part {
                        ResponsesContentPart::OutputText { text, logprobs, .. } => {
                            ResponsesStreamEvent::OutputTextDone {
                                content_index,
                                item_id: id.clone(),
                                output_index,
                                text: text.clone(),
                                logprobs: logprobs.clone(),
                                sequence_number,
                            }
                        }
                        ResponsesContentPart::ReasoningText { text } => {
                            ResponsesStreamEvent::ReasoningTextDone {
                                content_index,
                                item_id: id.clone(),
                                output_index,
                                text: text.clone(),
                                sequence_number,
                            }
                        }
                    });
                    events.push(ResponsesStreamEvent::ContentPartDone {
                        content_index,
                        item_id: id.clone(),
                        output_index,
                        part: part.clone(),
                        sequence_number: self.next_sequence_number(),
                    });
                }
            }
            ResponsesOutputItem::FunctionCall { id, arguments, .. } => {
                events.push(ResponsesStreamEvent::FunctionCallArgumentsDone {
                    arguments: arguments.clone(),
                    item_id: id.clone(),
                    output_index,
                    sequence_number: self.next_sequence_number(),
                });
            }
            ResponsesOutputItem::CustomToolCall { id, input, .. } => {
                events.push(ResponsesStreamEvent::CustomToolCallInputDone {
                    input: input.clone(),
                    item_id: id.clone(),
                    output_index,
                    sequence_number: self.next_sequence_number(),
                });
            }
            ResponsesOutputItem::ToolSearchCall { .. } => {}
        }
        events.push(ResponsesStreamEvent::OutputItemDone {
            item,
            output_index,
            sequence_number: self.next_sequence_number(),
        });
    }

    /// Append text to the active message or reasoning item, opening one when
    /// needed. Output text keeps its token log probabilities; reasoning text
    /// drops them.
    fn append_text(
        &mut self,
        delta: String,
        logprobs: Option<&[TokenLogprob]>,
        kind: ResponsesTextKind,
        events: &mut Vec<ResponsesStreamEvent>,
    ) {
        let reasoning = matches!(kind, ResponsesTextKind::Reasoning);
        let current = self
            .active
            .and_then(|index| self.response.output.get(index));
        let continues_item = matches!(
            (reasoning, current),
            (true, Some(ResponsesOutputItem::Reasoning { .. }))
                | (false, Some(ResponsesOutputItem::Message { .. }))
        );
        if !continues_item {
            self.close_item(ResponsesStatus::Completed, events);
            let output_index = self.response.output.len();
            let item = if reasoning {
                ResponsesOutputItem::Reasoning {
                    id: format!("rs_{}_{output_index}", self.response.id),
                    status: ResponsesStatus::InProgress,
                    content: vec![ResponsesContentPart::ReasoningText {
                        text: String::new(),
                    }],
                    summary: Vec::new(),
                    encrypted_content: Some(format!("{}-0", self.response.id)),
                }
            } else {
                ResponsesOutputItem::Message {
                    id: format!("msg_{}_{output_index}", self.response.id),
                    status: ResponsesStatus::InProgress,
                    role: ResponsesRole::Assistant,
                    phase: None,
                    content: vec![ResponsesContentPart::OutputText {
                        annotations: Vec::new(),
                        logprobs: Vec::new(),
                        text: String::new(),
                    }],
                }
            };
            self.open_item(item, events);
        }
        let Some(output_index) = self.active else {
            return;
        };
        let (item_id, content) = match &mut self.response.output[output_index] {
            ResponsesOutputItem::Message { id, content, .. }
            | ResponsesOutputItem::Reasoning { id, content, .. } => (id.clone(), content),
            ResponsesOutputItem::FunctionCall { .. }
            | ResponsesOutputItem::CustomToolCall { .. }
            | ResponsesOutputItem::ToolSearchCall { .. } => return,
        };
        let Some(part) = content.first_mut() else {
            return;
        };
        let delta_logprobs: Vec<ResponsesTokenLogprob> = logprobs
            .filter(|_| !reasoning)
            .map(|logprobs| logprobs.iter().map(Into::into).collect())
            .unwrap_or_default();
        match part {
            ResponsesContentPart::OutputText { text, logprobs, .. } => {
                text.push_str(&delta);
                logprobs.extend(delta_logprobs.iter().cloned());
            }
            ResponsesContentPart::ReasoningText { text } => text.push_str(&delta),
        }
        let sequence_number = self.next_sequence_number();
        events.push(if reasoning {
            ResponsesStreamEvent::ReasoningTextDelta {
                content_index: 0,
                delta,
                item_id,
                output_index,
                sequence_number,
            }
        } else {
            ResponsesStreamEvent::OutputTextDelta {
                content_index: 0,
                delta,
                item_id,
                logprobs: delta_logprobs,
                output_index,
                sequence_number,
            }
        });
    }

    fn append_custom_input(
        &mut self,
        output_index: usize,
        delta: String,
        events: &mut Vec<ResponsesStreamEvent>,
    ) {
        if delta.is_empty() {
            return;
        }
        if let ResponsesOutputItem::CustomToolCall { id, input, .. } =
            &mut self.response.output[output_index]
        {
            input.push_str(&delta);
            let item_id = id.clone();
            events.push(ResponsesStreamEvent::CustomToolCallInputDelta {
                delta,
                item_id,
                output_index,
                sequence_number: self.next_sequence_number(),
            });
        }
    }
}

impl ChunkGenerator for ResponsesChunkGenerator {
    type Chunk = ResponsesStreamEvent;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        if self.finished || (!self.started && !matches!(chunk, OutputChunk::Start { .. })) {
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
                events.push(ResponsesStreamEvent::Created {
                    response: self.response.clone(),
                    sequence_number: self.next_sequence_number(),
                });
                events.push(ResponsesStreamEvent::InProgress {
                    response: self.response.clone(),
                    sequence_number: self.next_sequence_number(),
                });
            }
            OutputChunk::Content { content, logprobs } => {
                self.append_text(
                    content,
                    logprobs.as_deref(),
                    ResponsesTextKind::Output,
                    &mut events,
                );
            }
            OutputChunk::Reasoning { content, .. } => {
                self.reasoning_content.push_str(&content);
                self.append_text(content, None, ResponsesTextKind::Reasoning, &mut events);
            }
            OutputChunk::ToolCallNamed { tool_name } => {
                if let Some(output_index) = self.active
                    && let ResponsesOutputItem::Message { phase, .. } =
                        &mut self.response.output[output_index]
                {
                    *phase = Some(ResponsesPhase::Commentary);
                }
                self.close_item(ResponsesStatus::Completed, &mut events);
                let output_index = self.response.output.len();
                let empty_thinking = self.thinking_mode && self.reasoning_content.is_empty();
                let call_id = new_tool_call_id(self.next_tool_call_index, empty_thinking);
                let item = if self.custom_tool_names.contains(&tool_name) {
                    self.custom_tool_input = Some(ResponsesCustomToolInputParser::default());
                    ResponsesOutputItem::CustomToolCall {
                        id: format!("ctc_{}_{output_index}", self.response.id),
                        status: ResponsesStatus::InProgress,
                        call_id,
                        name: tool_name,
                        input: String::new(),
                    }
                } else if self.client_tool_search && tool_name == "tool_search" {
                    ResponsesOutputItem::ToolSearchCall {
                        id: format!("tsc_{}_{output_index}", self.response.id),
                        status: ResponsesStatus::InProgress,
                        call_id,
                        execution: ResponsesToolSearchExecution::Client,
                        arguments: serde_json::Map::new(),
                        raw_arguments: String::new(),
                    }
                } else {
                    let (name, namespace) = self.resolve_namespace(tool_name);
                    ResponsesOutputItem::FunctionCall {
                        id: format!("fc_{}_{output_index}", self.response.id),
                        status: ResponsesStatus::InProgress,
                        call_id,
                        name,
                        namespace,
                        arguments: String::new(),
                    }
                };
                self.next_tool_call_index += 1;
                self.open_item(item, &mut events);
            }
            OutputChunk::ToolCallArgumentsDelta { arguments: delta } => {
                if let Some(output_index) = self.active
                    && let ResponsesOutputItem::FunctionCall { id, arguments, .. } =
                        &mut self.response.output[output_index]
                {
                    arguments.push_str(&delta);
                    let item_id = id.clone();
                    events.push(ResponsesStreamEvent::FunctionCallArgumentsDelta {
                        delta,
                        item_id,
                        output_index,
                        sequence_number: self.next_sequence_number(),
                    });
                } else if let Some(output_index) = self.active
                    && let Some(parser) = &mut self.custom_tool_input
                {
                    let delta = parser.feed(&delta);
                    self.append_custom_input(output_index, delta, &mut events);
                } else if let Some(output_index) = self.active
                    && let ResponsesOutputItem::ToolSearchCall { raw_arguments, .. } =
                        &mut self.response.output[output_index]
                {
                    raw_arguments.push_str(&delta);
                }
            }
            OutputChunk::Finish {
                finish_reason,
                completion_usage,
                ..
            } => {
                let incomplete_reason = match finish_reason {
                    FinishReason::Length => Some(ResponsesIncompleteReason::MaxOutputTokens),
                    FinishReason::ContentFilter => Some(ResponsesIncompleteReason::ContentFilter),
                    FinishReason::Stop
                    | FinishReason::ToolCalls
                    | FinishReason::StopSequence
                    | FinishReason::EndOfStream => None,
                };
                self.response.incomplete_details =
                    incomplete_reason.map(|reason| ResponsesIncompleteDetails { reason });
                self.response.status = if self.response.incomplete_details.is_some() {
                    ResponsesStatus::Incomplete
                } else {
                    ResponsesStatus::Completed
                };
                self.response.usage =
                    Some(ResponsesUsage::from((self.prompt_usage, completion_usage)));
                self.response.completed_at = Some(timestamp());
                self.close_item(self.response.status, &mut events);
                let response = self.response.clone();
                let sequence_number = self.next_sequence_number();
                events.push(match self.response.status {
                    ResponsesStatus::Incomplete => ResponsesStreamEvent::Incomplete {
                        response,
                        sequence_number,
                    },
                    ResponsesStatus::InProgress | ResponsesStatus::Completed => {
                        ResponsesStreamEvent::Completed {
                            response,
                            sequence_number,
                        }
                    }
                });
                self.finished = true;
            }
            OutputChunk::ToolCallBegin => {}
        }
        events
    }
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    async fn started_generator() -> ResponsesChunkGenerator {
        let mut generator = ResponsesChunkGenerator::new(
            "phase-repro",
            "DeepSeek-V4.1-Flash",
            false,
            ResponseOptions::default(),
        );
        generator
            .generate(OutputChunk::Start {
                system_fingerprint: None,
                prompt_usage: PromptUsage::default(),
            })
            .await;
        generator
    }

    fn event_json(event: &ResponsesStreamEvent) -> Value {
        serde_json::to_value(event).expect("response event must serialize")
    }

    #[tokio::test]
    async fn streaming_message_phase_is_omitted_until_tool_call_classifies_it() {
        let mut generator = started_generator().await;
        let text_events = generator
            .generate(OutputChunk::Content {
                content: "I will check now.".to_string(),
                logprobs: None,
            })
            .await;
        let added = event_json(&text_events[0]);
        assert_eq!(added["type"], "response.output_item.added");
        assert!(added["item"].get("phase").is_none());

        let tool_events = generator
            .generate(OutputChunk::ToolCallNamed {
                tool_name: "get_temperature".to_string(),
            })
            .await;
        let done = tool_events
            .iter()
            .map(event_json)
            .find(|event| event["type"] == "response.output_item.done")
            .expect("message completion event");
        assert_eq!(done["item"]["phase"], "commentary");
    }

    #[tokio::test]
    async fn streaming_message_phase_is_final_answer_when_response_finishes() {
        let mut generator = started_generator().await;
        let text_events = generator
            .generate(OutputChunk::Content {
                content: "The answer is 42.".to_string(),
                logprobs: None,
            })
            .await;
        let added = event_json(&text_events[0]);
        assert!(added["item"].get("phase").is_none());

        let finish_events = generator
            .generate(OutputChunk::Finish {
                finish_reason: FinishReason::Stop,
                stop_sequence: None,
                completion_usage: Default::default(),
            })
            .await;
        let done = finish_events
            .iter()
            .map(event_json)
            .find(|event| event["type"] == "response.output_item.done")
            .expect("message completion event");
        assert_eq!(done["item"]["phase"], "final_answer");
    }
}
