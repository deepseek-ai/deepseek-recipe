use std::collections::{HashMap, HashSet};

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{IMAGE_SPECIAL_TOKEN, ImageSource};
use deepseek_recipe_core::tools::ToolChoice;
use itertools::Itertools;

use crate::protocol::image::{
    UNSUPPORTED_DOCUMENT_PLACEHOLDER, file_id_unsupported, image_not_allowed, image_url_source,
};
use crate::protocol::openai::responses::response::{ResponsesChunkGenerator, ResponsesResponse};
use crate::protocol::validation::{
    contains_json_instruction, image_special_token_not_allowed, validate_max_tokens,
    validate_sampling, validate_tool_text,
};
use crate::request::{
    ConversationRequest, ConversionError, ConversionOptions, InferenceOptions, ProtocolRequest,
};
use crate::stream::state_machine::{ParsingOptions, ReasoningStage};

use super::schema::{
    ResponsesContentBlock, ResponsesInput, ResponsesInputImage, ResponsesInputItem,
    ResponsesInputMessage, ResponsesMessageContent, ResponsesMessageRole, ResponsesReasoningEffort,
    ResponsesReasoningItem, ResponsesRequest, ResponsesTextConfig, ResponsesTextFormat,
    ResponsesTool, ResponsesToolCallOutputContent, ResponsesToolCallOutputItem,
    ResponsesTypedInputItem,
};
use super::tools::ResponsesToolSet;

impl ResponsesMessageContent {
    /// Convert text, substitute document placeholders, and collect images with
    /// one [`IMAGE_SPECIAL_TOKEN`] placeholder per image.
    fn into_content(self, path: &str) -> Result<(String, Vec<ImageSource>), ConversionError> {
        match self {
            ResponsesMessageContent::String(s) => Ok((s, Vec::new())),
            ResponsesMessageContent::List(blocks) => {
                let mut texts = Vec::new();
                let mut image_sources = Vec::new();
                for (block_index, block) in blocks.into_iter().enumerate() {
                    let block_path = format!("{path}.content[{block_index}]");
                    match block {
                        ResponsesContentBlock::InputText { text }
                        | ResponsesContentBlock::OutputText { text } => texts.push(text),
                        ResponsesContentBlock::InputImage(image) => {
                            image_sources.push(input_image_source(image, &block_path)?);
                            texts.push(IMAGE_SPECIAL_TOKEN.to_string());
                        }
                        ResponsesContentBlock::InputFile {} => {
                            texts.push(UNSUPPORTED_DOCUMENT_PLACEHOLDER.to_string());
                        }
                        ResponsesContentBlock::Unsupported => return Err(unsupported_content()),
                    }
                }
                Ok((texts.join("\n\n"), image_sources))
            }
        }
    }

    /// Convert text and document placeholders, rejecting images.
    fn into_text(self, path: &str, role: &str) -> Result<String, ConversionError> {
        let (content, image_sources) = self.into_content(path)?;
        if !image_sources.is_empty() {
            return Err(image_not_allowed(role));
        }
        Ok(content)
    }
}

impl ResponsesToolCallOutputContent {
    /// Convert text and document placeholders, collecting image sources.
    fn into_content(self, path: &str) -> Result<(String, Vec<ImageSource>), ConversionError> {
        match self {
            ResponsesToolCallOutputContent::String(s) => Ok((s, Vec::new())),
            ResponsesToolCallOutputContent::List(items) => {
                let mut texts = Vec::new();
                let mut image_sources = Vec::new();
                for (item_index, item) in items.into_iter().enumerate() {
                    let item_path = format!("{path}[{item_index}]");
                    match item {
                        ResponsesToolCallOutputItem::InputText { text } => texts.push(text),
                        ResponsesToolCallOutputItem::InputImage(image) => {
                            image_sources.push(input_image_source(image, &item_path)?);
                            texts.push(IMAGE_SPECIAL_TOKEN.to_string());
                        }
                        ResponsesToolCallOutputItem::InputFile {} => {
                            texts.push(UNSUPPORTED_DOCUMENT_PLACEHOLDER.to_string());
                        }
                        ResponsesToolCallOutputItem::Unsupported => {
                            return Err(unsupported_content());
                        }
                    }
                }
                Ok((texts.join("\n\n"), image_sources))
            }
        }
    }
}

/// Return whether the input or instructions carry [`IMAGE_SPECIAL_TOKEN`].
///
/// Message text, tool output text, and reasoning text are checked. Tool call
/// names and arguments are checked after conversion by [`validate_tool_text`].
fn contains_image_special_token(
    input: Option<&ResponsesInput>,
    instructions: Option<&str>,
) -> bool {
    if instructions.is_some_and(|text| text.contains(IMAGE_SPECIAL_TOKEN)) {
        return true;
    }
    match input {
        Some(ResponsesInput::String(content)) => content.contains(IMAGE_SPECIAL_TOKEN),
        Some(ResponsesInput::Items(items)) => items.iter().any(item_contains_image_special_token),
        None => false,
    }
}

/// Return whether one input item carries [`IMAGE_SPECIAL_TOKEN`].
fn item_contains_image_special_token(item: &ResponsesInputItem) -> bool {
    match item {
        ResponsesInputItem::Message(message) => {
            message_content_contains_image_special_token(&message.content)
        }
        ResponsesInputItem::Typed(typed) => match typed {
            ResponsesTypedInputItem::Message(message) => {
                message_content_contains_image_special_token(&message.content)
            }
            ResponsesTypedInputItem::FunctionCallOutput(output) => {
                output_content_contains_image_special_token(&output.output)
            }
            ResponsesTypedInputItem::CustomToolCallOutput(output) => output
                .output
                .as_ref()
                .is_some_and(output_content_contains_image_special_token),
            ResponsesTypedInputItem::Reasoning(reasoning) => {
                reasoning.content.as_ref().is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|block| block.text.contains(IMAGE_SPECIAL_TOKEN))
                })
            }
            _ => false,
        },
    }
}

/// Return whether one message content carries [`IMAGE_SPECIAL_TOKEN`].
fn message_content_contains_image_special_token(content: &ResponsesMessageContent) -> bool {
    match content {
        ResponsesMessageContent::String(content) => content.contains(IMAGE_SPECIAL_TOKEN),
        ResponsesMessageContent::List(blocks) => blocks.iter().any(|block| match block {
            ResponsesContentBlock::InputText { text }
            | ResponsesContentBlock::OutputText { text } => text.contains(IMAGE_SPECIAL_TOKEN),
            _ => false,
        }),
    }
}

/// Return whether one tool call output carries [`IMAGE_SPECIAL_TOKEN`].
fn output_content_contains_image_special_token(content: &ResponsesToolCallOutputContent) -> bool {
    match content {
        ResponsesToolCallOutputContent::String(content) => content.contains(IMAGE_SPECIAL_TOKEN),
        ResponsesToolCallOutputContent::List(items) => items.iter().any(|item| match item {
            ResponsesToolCallOutputItem::InputText { text } => text.contains(IMAGE_SPECIAL_TOKEN),
            _ => false,
        }),
    }
}

/// Convert one Responses image reference.
fn input_image_source(
    image: ResponsesInputImage,
    path: &str,
) -> Result<ImageSource, ConversionError> {
    let image_url = image.image_url.filter(|url| !url.is_empty());
    match (image_url, image.file_id) {
        (None, None) => Err(ConversionError::bad_request(format!(
            "{path}: input_image must have image_url or file_id"
        ))),
        (Some(_), Some(_)) => Err(ConversionError::bad_request(format!(
            "{path}: input_image cannot have both image_url and file_id"
        ))),
        (None, Some(_)) => Err(file_id_unsupported(path)),
        (Some(url), None) => image_url_source(url, image.detail, &format!("{path}.image_url")),
    }
}

fn check_tool_call_id(
    item_idx: usize,
    call_id: &str,
    pending: &HashSet<String>,
    resolved: &HashSet<String>,
) -> Result<(), ConversionError> {
    if call_id.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "Invalid 'input[{item_idx}].call_id': empty string. Expected a string with minimum length 1, but got an empty string \
             instead."
        )));
    }
    if pending.contains(call_id) || resolved.contains(call_id) {
        return Err(ConversionError::bad_request(format!(
            "Duplicate 'call_id': {call_id}."
        )));
    }
    Ok(())
}

fn check_tool_output_id(
    item_idx: usize,
    call_id: &str,
    pending: &mut HashSet<String>,
    resolved: &mut HashSet<String>,
) -> Result<(), ConversionError> {
    if call_id.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "Invalid 'input[{item_idx}].call_id': empty string. Expected a string with minimum length 1, but got an empty string \
             instead."
        )));
    }
    if !pending.contains(call_id) {
        if resolved.contains(call_id) {
            return Err(ConversionError::bad_request(format!(
                "Duplicate tool output for call_id: {call_id}."
            )));
        }
        return Err(ConversionError::bad_request(format!(
            "No tool call found for tool output with call_id {call_id}."
        )));
    }
    pending.remove(call_id);
    resolved.insert(call_id.to_string());
    Ok(())
}

fn check_pending_calls(pending_call_ids: &HashSet<String>) -> Result<(), ConversionError> {
    if let Some(call_id) = pending_call_ids.iter().next() {
        return Err(ConversionError::bad_request(format!(
            "No tool output found for tool call {call_id}."
        )));
    }
    Ok(())
}

#[derive(Default)]
struct ResponsesToolResult {
    content: String,
    image_sources: Vec<ImageSource>,
}

#[derive(Default)]
struct ResponsesToolCallGroup {
    calls: Vec<ToolCall>,
    outputs: HashMap<String, ResponsesToolResult>,
}

impl ResponsesToolCallGroup {
    fn is_complete(&self) -> bool {
        self.outputs.len() >= self.calls.len()
    }

    fn append_messages(self, messages: &mut Vec<InputMessage>) {
        if self.calls.is_empty() {
            return;
        }
        let mut calls = self.calls;
        calls.sort_by(|a, b| a.id.cmp(&b.id));
        let mut outputs = self.outputs;
        let tool_results: Vec<InputMessage> = calls
            .iter()
            .filter_map(|call| {
                outputs.remove(&call.id).map(|output| InputMessage::Tool {
                    content: output.content,
                    image_sources: output.image_sources,
                    tool_call_id: call.id.clone(),
                })
            })
            .collect();

        if !matches!(messages.last(), Some(InputMessage::Assistant { .. })) {
            messages.push(InputMessage::Assistant {
                content: String::new(),
                reasoning_content: None,
                tool_calls: None,
            });
        }
        if let Some(InputMessage::Assistant { tool_calls, .. }) = messages.last_mut() {
            match tool_calls {
                Some(existing) => existing.extend(calls),
                None => *tool_calls = Some(calls),
            }
        }
        messages.extend(tool_results);
    }
}

fn push_call_to_group(pending_group: &mut Option<ResponsesToolCallGroup>, tool_call: ToolCall) {
    pending_group
        .get_or_insert_with(ResponsesToolCallGroup::default)
        .calls
        .push(tool_call);
}

fn push_output_to_group(
    pending_group: &mut Option<ResponsesToolCallGroup>,
    call_id: String,
    output: ResponsesToolResult,
    messages: &mut Vec<InputMessage>,
) {
    let Some(group) = pending_group.as_mut() else {
        messages.push(InputMessage::Tool {
            content: output.content,
            image_sources: output.image_sources,
            tool_call_id: call_id,
        });
        return;
    };
    group.outputs.insert(call_id, output);
    if group.is_complete() {
        append_tool_call_group(pending_group, messages);
    }
}

fn append_tool_call_group(
    pending_group: &mut Option<ResponsesToolCallGroup>,
    messages: &mut Vec<InputMessage>,
) {
    if let Some(group) = pending_group.take() {
        group.append_messages(messages);
    }
}

fn push_message(
    msg: ResponsesInputMessage,
    item_idx: usize,
    messages: &mut Vec<InputMessage>,
) -> Result<(), ConversionError> {
    let path = format!("input[{item_idx}]");
    match msg.role {
        ResponsesMessageRole::System => {
            let content = msg.content.into_text(&path, "system")?;
            messages.push(InputMessage::System { content });
        }
        ResponsesMessageRole::User | ResponsesMessageRole::Developer => {
            let (content, image_sources) = msg.content.into_content(&path)?;
            messages.push(InputMessage::User {
                content,
                image_sources,
            });
        }
        ResponsesMessageRole::Assistant => {
            let content = msg.content.into_text(&path, "assistant")?;
            if let Some(InputMessage::Assistant {
                content: last_content,
                tool_calls: None,
                ..
            }) = messages.last_mut()
            {
                if !content.is_empty() {
                    if !last_content.is_empty() {
                        last_content.push_str("\n\n");
                    }
                    last_content.push_str(&content);
                }
            } else {
                messages.push(InputMessage::Assistant {
                    content,
                    reasoning_content: None,
                    tool_calls: None,
                });
            }
        }
    }
    Ok(())
}

fn push_reasoning(reasoning: ResponsesReasoningItem, messages: &mut Vec<InputMessage>) {
    let text = reasoning
        .content
        .unwrap_or_default()
        .into_iter()
        .map(|c| c.text)
        .join("\n");

    if text.is_empty() {
        return;
    }

    if !matches!(
        messages.last(),
        Some(InputMessage::Assistant { content, .. }) if content.is_empty()
    ) {
        messages.push(InputMessage::Assistant {
            content: String::new(),
            reasoning_content: None,
            tool_calls: None,
        });
    }
    if let Some(InputMessage::Assistant {
        reasoning_content, ..
    }) = messages.last_mut()
    {
        match reasoning_content {
            Some(existing) => {
                existing.push('\n');
                existing.push_str(&text);
            }
            None => *reasoning_content = Some(text),
        }
    }
}

fn transform_input_items(
    items: Vec<ResponsesInputItem>,
    messages: &mut Vec<InputMessage>,
    tools: &ResponsesToolSet,
) -> Result<(), ConversionError> {
    if items.is_empty() {
        return Err(ConversionError::bad_request(
            "Input items array must not be empty",
        ));
    }

    let mut pending_call_ids: HashSet<String> = HashSet::new();
    let mut resolved_call_ids: HashSet<String> = HashSet::new();
    let mut pending_group: Option<ResponsesToolCallGroup> = None;
    let mut historical_names = HashMap::new();

    for (item_idx, item) in items.into_iter().enumerate() {
        match item {
            ResponsesInputItem::Typed(typed) => match typed {
                ResponsesTypedInputItem::Message(msg) => {
                    check_pending_calls(&pending_call_ids)?;
                    append_tool_call_group(&mut pending_group, messages);
                    push_message(msg, item_idx, messages)?;
                }
                ResponsesTypedInputItem::Reasoning(reasoning) => {
                    check_pending_calls(&pending_call_ids)?;
                    append_tool_call_group(&mut pending_group, messages);
                    push_reasoning(reasoning, messages);
                }
                ResponsesTypedInputItem::FunctionCall(fc) => {
                    check_tool_call_id(
                        item_idx,
                        &fc.call_id,
                        &pending_call_ids,
                        &resolved_call_ids,
                    )?;
                    pending_call_ids.insert(fc.call_id.clone());
                    push_call_to_group(
                        &mut pending_group,
                        tools.function_call(fc, &mut historical_names),
                    );
                }
                ResponsesTypedInputItem::FunctionCallOutput(fco) => {
                    check_tool_output_id(
                        item_idx,
                        &fco.call_id,
                        &mut pending_call_ids,
                        &mut resolved_call_ids,
                    )?;
                    let (content, image_sources) = fco
                        .output
                        .into_content(&format!("input[{item_idx}].output"))?;
                    push_output_to_group(
                        &mut pending_group,
                        fco.call_id,
                        ResponsesToolResult {
                            content,
                            image_sources,
                        },
                        messages,
                    );
                }
                ResponsesTypedInputItem::CustomToolCall(call) => {
                    check_tool_call_id(
                        item_idx,
                        &call.call_id,
                        &pending_call_ids,
                        &resolved_call_ids,
                    )?;
                    pending_call_ids.insert(call.call_id.clone());
                    push_call_to_group(
                        &mut pending_group,
                        ToolCall {
                            id: call.call_id,
                            name: call.name,
                            arguments: serde_json::json!({"input": call.input}).to_string(),
                        },
                    );
                }
                ResponsesTypedInputItem::CustomToolCallOutput(output) => {
                    check_tool_output_id(
                        item_idx,
                        &output.call_id,
                        &mut pending_call_ids,
                        &mut resolved_call_ids,
                    )?;
                    let (content, image_sources) = output
                        .output
                        .map(|content| content.into_content(&format!("input[{item_idx}].output")))
                        .transpose()?
                        .unwrap_or_default();
                    push_output_to_group(
                        &mut pending_group,
                        output.call_id,
                        ResponsesToolResult {
                            content,
                            image_sources,
                        },
                        messages,
                    );
                }
                ResponsesTypedInputItem::Unsupported => {}
            },
            ResponsesInputItem::Message(msg) => {
                check_pending_calls(&pending_call_ids)?;
                append_tool_call_group(&mut pending_group, messages);
                push_message(msg, item_idx, messages)?;
            }
        }
    }
    append_tool_call_group(&mut pending_group, messages);
    check_pending_calls(&pending_call_ids)?;

    Ok(())
}

impl ProtocolRequest for ResponsesRequest {
    type Response = ResponsesResponse;

    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError> {
        if contains_image_special_token(self.input.as_ref(), self.instructions.as_deref()) {
            return Err(image_special_token_not_allowed());
        }
        let (thinking_mode, reasoning_effort) = self.resolve_thinking(options);
        let inference_options = self.inference_options()?;
        let converted_tools = ResponsesToolSet::convert(self.tools, &options)?;
        let mut messages: Vec<InputMessage> = Vec::new();

        if let Some(instructions) = self.instructions
            && !instructions.is_empty()
        {
            messages.push(InputMessage::System {
                content: instructions,
            });
        }

        match self.input {
            Some(ResponsesInput::String(s)) => messages.push(InputMessage::User {
                content: s,
                image_sources: Vec::new(),
            }),
            Some(ResponsesInput::Items(items)) => {
                transform_input_items(items, &mut messages, &converted_tools)?
            }
            None if messages.is_empty() => {
                return Err(ConversionError::bad_request(
                    "Either input or instructions must be provided",
                ));
            }
            None => {}
        }

        let (tools, tool_choice) =
            converted_tools.select(self.tool_choice, thinking_mode, &options)?;
        validate_tool_text(&tools, &messages)?;
        let response_format = convert_text_format(self.text, &messages)?;
        let parsing_options = ParsingOptions {
            parse_tool_calls: !tools.is_empty(),
            tool_call_initial_stage: tool_choice == ToolChoice::Required,
            parse_json_output: matches!(response_format, ResponseFormat::JsonObject),
            reasoning_initial_stage: thinking_mode.then_some(ReasoningStage::Start),
            ..ParsingOptions::default()
        };
        Ok(ConversationRequest {
            conversation: Conversation {
                messages,
                thinking_mode,
                reasoning_effort,
                tools,
                tool_choice,
                response_format,
            },
            inference_options,
            parsing_options,
            model: Some(self.model),
            stream: self.stream.unwrap_or(false),
        })
    }

    fn chunk_generator(
        _request: &ConversationRequest,
        id: String,
        model: String,
    ) -> ResponsesChunkGenerator {
        ResponsesChunkGenerator::new(id, model)
    }
}

impl ResponsesRequest {
    /// Return the custom tool names declared in `tools`.
    /// Read before `convert` and pass to
    /// [`ResponsesChunkGenerator::with_custom_tool_names`].
    pub fn custom_tool_names(&self) -> HashSet<String> {
        self.tools
            .iter()
            .flatten()
            .filter_map(|tool| match tool {
                ResponsesTool::Custom { name } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    fn resolve_thinking(&self, options: ConversionOptions) -> (bool, Option<ReasoningEffort>) {
        let effort = self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort);
        let effort = match effort {
            Some(ResponsesReasoningEffort::None) => return (false, None),
            Some(ResponsesReasoningEffort::Minimal | ResponsesReasoningEffort::Low) => {
                ReasoningEffort::Low
            }
            Some(ResponsesReasoningEffort::Medium | ResponsesReasoningEffort::High) => {
                ReasoningEffort::High
            }
            Some(ResponsesReasoningEffort::Xhigh) => ReasoningEffort::Xhigh,
            Some(ResponsesReasoningEffort::Max) => ReasoningEffort::Max,
            None if !options.default_thinking_mode => return (false, None),
            None => ReasoningEffort::High,
        };
        (true, Some(effort))
    }

    fn inference_options(&self) -> Result<InferenceOptions, ConversionError> {
        validate_max_tokens(self.max_output_tokens, "max_output_tokens")?;
        validate_sampling(self.temperature, self.top_p)?;
        Ok(InferenceOptions {
            max_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            ..InferenceOptions::default()
        })
    }
}

fn convert_text_format(
    text: Option<ResponsesTextConfig>,
    messages: &[InputMessage],
) -> Result<ResponseFormat, ConversionError> {
    match text.map(|text| text.format).unwrap_or_default() {
        ResponsesTextFormat::Text | ResponsesTextFormat::JsonSchema => Ok(ResponseFormat::Text),
        ResponsesTextFormat::JsonObject => {
            if !contains_json_instruction(messages) {
                return Err(ConversionError::bad_request(
                    "Response input messages must contain the word 'json' to use text.format of type json_object",
                ));
            }
            Ok(ResponseFormat::JsonObject)
        }
    }
}

fn unsupported_content() -> ConversionError {
    ConversionError::bad_request("Unsupported content block")
}
