use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::response::ProtocolResponse;
use crate::stream::{ChunkGenerator, FinishReason, OutputChunk, PromptUsage};

use super::custom_tool_input::ResponsesCustomToolInputParser;
use super::schema::{
    ResponsesContentPart, ResponsesIncompleteDetails, ResponsesIncompleteReason,
    ResponsesOutputItem, ResponsesPhase, ResponsesResponse, ResponsesRole, ResponsesStatus,
    ResponsesStreamEvent, ResponsesUsage,
};

/// Generate Responses events from one `Start` to `Finish` sequence.
/// Repeated starts and chunks outside this sequence are ignored.
/// Argument deltas extend only the active tool call.
/// Custom tool output requires `with_custom_tool_names`.
/// Other tools retain their model names, including flattened namespaces.
#[derive(Debug)]
pub struct ResponsesChunkGenerator {
    response: ResponsesResponse,
    active: Option<usize>,
    sequence_number: usize,
    tool_call_index: usize,
    custom_tool_names: HashSet<String>,
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
    /// Use a unique response ID and public model name.
    /// Item and call IDs derive from the response ID and indices.
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            response: ResponsesResponse::new(id.into(), model.into(), timestamp(), 0, 0),
            active: None,
            sequence_number: 0,
            tool_call_index: 0,
            custom_tool_names: HashSet::new(),
            custom_tool_input: None,
            prompt_usage: PromptUsage::default(),
            started: false,
            finished: false,
        }
    }

    /// Set declared custom tool names whose JSON `input` strings are decoded.
    /// Defaults to an empty set; other tools retain their function arguments.
    pub fn with_custom_tool_names(mut self, custom_tool_names: HashSet<String>) -> Self {
        self.custom_tool_names = custom_tool_names;
        self
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
            | ResponsesOutputItem::CustomToolCall { .. } => None,
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
        }
        events.push(ResponsesStreamEvent::OutputItemDone {
            item,
            output_index,
            sequence_number: self.next_sequence_number(),
        });
    }

    fn append_text(
        &mut self,
        delta: String,
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
                }
            } else {
                ResponsesOutputItem::Message {
                    id: format!("msg_{}_{output_index}", self.response.id),
                    status: ResponsesStatus::InProgress,
                    role: ResponsesRole::Assistant,
                    phase: ResponsesPhase::FinalAnswer,
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
            | ResponsesOutputItem::CustomToolCall { .. } => return,
        };
        let Some(part) = content.first_mut() else {
            return;
        };
        match part {
            ResponsesContentPart::OutputText { text, .. }
            | ResponsesContentPart::ReasoningText { text } => text.push_str(&delta),
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
                logprobs: Vec::new(),
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
            OutputChunk::Start { usage, .. } => {
                if self.started {
                    return events;
                }
                self.started = true;
                self.prompt_usage = usage;
                events.push(ResponsesStreamEvent::Created {
                    response: self.response.clone(),
                    sequence_number: self.next_sequence_number(),
                });
                events.push(ResponsesStreamEvent::InProgress {
                    response: self.response.clone(),
                    sequence_number: self.next_sequence_number(),
                });
            }
            OutputChunk::Raw { content } => {
                self.append_text(content, ResponsesTextKind::Output, &mut events);
            }
            OutputChunk::Reasoning { content } => {
                self.append_text(content, ResponsesTextKind::Reasoning, &mut events);
            }
            OutputChunk::ToolCall {
                tool_name,
                arguments,
            } => {
                if let Some(output_index) = self.active
                    && let ResponsesOutputItem::Message { phase, .. } =
                        &mut self.response.output[output_index]
                {
                    *phase = ResponsesPhase::Commentary;
                }
                self.close_item(ResponsesStatus::Completed, &mut events);
                let output_index = self.response.output.len();
                let call_id = format!("call_{}_{}", self.response.id, self.tool_call_index);
                let item = if self.custom_tool_names.contains(&tool_name) {
                    let mut parser = ResponsesCustomToolInputParser::default();
                    let input = parser.feed(&arguments);
                    self.custom_tool_input = Some(parser);
                    ResponsesOutputItem::CustomToolCall {
                        id: format!("ctc_{}_{output_index}", self.response.id),
                        status: ResponsesStatus::InProgress,
                        call_id,
                        name: tool_name,
                        input,
                    }
                } else {
                    let (name, namespace) = match tool_name.split_once("::") {
                        Some((namespace, name)) => (name.to_string(), Some(namespace.to_string())),
                        None => (tool_name, None),
                    };
                    ResponsesOutputItem::FunctionCall {
                        id: format!("fc_{}_{output_index}", self.response.id),
                        status: ResponsesStatus::InProgress,
                        call_id,
                        name,
                        namespace,
                        arguments,
                    }
                };
                self.tool_call_index += 1;
                self.open_item(item, &mut events);
            }
            OutputChunk::ToolArgumentsDelta { content } => {
                if let Some(output_index) = self.active
                    && let ResponsesOutputItem::FunctionCall { id, arguments, .. } =
                        &mut self.response.output[output_index]
                {
                    arguments.push_str(&content);
                    let item_id = id.clone();
                    events.push(ResponsesStreamEvent::FunctionCallArgumentsDelta {
                        delta: content,
                        item_id,
                        output_index,
                        sequence_number: self.next_sequence_number(),
                    });
                } else if let Some(output_index) = self.active
                    && let Some(parser) = &mut self.custom_tool_input
                {
                    let delta = parser.feed(&content);
                    self.append_custom_input(output_index, delta, &mut events);
                }
            }
            OutputChunk::Finish { reason, usage, .. } => {
                let incomplete_reason = match reason {
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
                self.response.usage = Some(ResponsesUsage::from((self.prompt_usage, usage)));
                self.response.completed_at = Some(timestamp());
                self.close_item(self.response.status, &mut events);
                let response = self.response.clone();
                let sequence_number = self.next_sequence_number();
                events.push(if self.response.status == ResponsesStatus::Incomplete {
                    ResponsesStreamEvent::Incomplete {
                        response,
                        sequence_number,
                    }
                } else {
                    ResponsesStreamEvent::Completed {
                        response,
                        sequence_number,
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
