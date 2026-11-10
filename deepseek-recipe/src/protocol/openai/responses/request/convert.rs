use std::collections::{HashMap, HashSet};
use std::iter::Peekable;
use std::sync::Arc;

use deepseek_recipe_core::conversation::{Conversation, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{ImageSource, MAX_IMAGE_URL_LEN};
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_core::special_tokens::SpecialToken;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use itertools::Itertools;

use crate::protocol::client_tools::normalize_message_tools;
use crate::protocol::image::{
    UNSUPPORTED_DOCUMENT_PLACEHOLDER, UNSUPPORTED_IMAGE_PLACEHOLDER, file_id_unsupported,
    image_url_source, validate_data_url_sources,
};
use crate::protocol::openai::responses::response::{
    ResponsesChunkGenerator, ResponsesEcho, ResponsesResponse,
};
use crate::protocol::output_constraint::resolve_output_constraint;
use crate::protocol::reasoning::resolve_thinking_mode;
use crate::protocol::reasoning_content::normalize_reasoning_content;
use crate::protocol::validation::{
    RawInferenceOptions, contains_json_instruction, validate_response_schema,
};
use crate::request::{
    ConversationRequest, ConversionError, ConversionOptions, ModelVersion, ProtocolRequest,
    ResponseOptions,
};
use crate::stream::{ParsingOptions, ThinkingInitialStage};

use super::schema::{
    ResponsesContentBlock, ResponsesInput, ResponsesInputImage, ResponsesInputItem,
    ResponsesInputMessage, ResponsesMessageContent, ResponsesMessageRole, ResponsesReasoningItem,
    ResponsesRequest, ResponsesTextConfig, ResponsesTextFormat, ResponsesTool,
    ResponsesToolCallOutputContent, ResponsesToolCallOutputItem, ResponsesTypedInputItem,
};
use super::tools::ResponsesToolSet;

/// A converter for the content of input items, with the settings that every
/// item shares.
struct ResponsesContentConverter<'a> {
    image_token: &'a str,
    model_version: ModelVersion,
}

impl ResponsesContentConverter<'_> {
    /// Convert one `input_image` block and append its prompt part and image
    /// source. `path` locates the block in error messages, and the image
    /// source records `item_path`, the location of the input item. A model
    /// without image input receives the unsupported-image text, and the block
    /// is not validated.
    fn push_input_image(
        &self,
        image: ResponsesInputImage,
        path: &str,
        item_path: &str,
        contents: &mut Vec<Prompt>,
        image_sources: &mut Vec<ImageSource>,
    ) -> Result<(), ConversionError> {
        if !self.model_version.supports_image() {
            contents.push(Prompt::plain(UNSUPPORTED_IMAGE_PLACEHOLDER));
            return Ok(());
        }
        contents.push(Prompt::special(self.image_token));
        image_sources.push(input_image_source(image, path, item_path)?);
        Ok(())
    }
}

impl ResponsesMessageContent {
    /// Convert the blocks to prompt parts joined with blank lines: text as
    /// plain text, `input_file` as the unsupported-document text, and each
    /// `input_image` as one image placeholder, or as the unsupported-image text
    /// for a model without image input. Data URL images are not validated
    /// here.
    fn into_content(
        self,
        item_path: &str,
        converter: &ResponsesContentConverter<'_>,
    ) -> Result<(Prompt, Vec<ImageSource>), ConversionError> {
        match self {
            ResponsesMessageContent::String(s) => Ok((Prompt::plain(s), Vec::new())),
            ResponsesMessageContent::List(blocks) => {
                let mut contents = Vec::new();
                let mut image_sources = Vec::new();
                for (block_index, block) in blocks.into_iter().enumerate() {
                    match block {
                        ResponsesContentBlock::InputText { text }
                        | ResponsesContentBlock::OutputText { text } => {
                            contents.push(Prompt::plain(text));
                        }
                        ResponsesContentBlock::InputImage(image) => converter.push_input_image(
                            image,
                            &format!("{item_path}.content[{block_index}]"),
                            item_path,
                            &mut contents,
                            &mut image_sources,
                        )?,
                        ResponsesContentBlock::InputFile {} => {
                            contents.push(Prompt::plain(UNSUPPORTED_DOCUMENT_PLACEHOLDER));
                        }
                    }
                }
                Ok((Prompt::join(&contents, "\n\n"), image_sources))
            }
        }
    }
}

impl ResponsesToolCallOutputContent {
    /// Convert the blocks as [`ResponsesMessageContent::into_content`] does,
    /// and validate the data URL images.
    fn into_content(
        self,
        item_path: &str,
        converter: &ResponsesContentConverter<'_>,
    ) -> Result<(Prompt, Vec<ImageSource>), ConversionError> {
        let (content, image_sources) = match self {
            ResponsesToolCallOutputContent::String(s) => (Prompt::plain(s), Vec::new()),
            ResponsesToolCallOutputContent::List(items) => {
                let mut contents = Vec::new();
                let mut image_sources = Vec::new();
                for (output_index, item) in items.into_iter().enumerate() {
                    match item {
                        ResponsesToolCallOutputItem::InputText { text } => {
                            contents.push(Prompt::plain(text));
                        }
                        ResponsesToolCallOutputItem::InputImage(image) => converter
                            .push_input_image(
                                image,
                                &format!("{item_path}.output[{output_index}]"),
                                item_path,
                                &mut contents,
                                &mut image_sources,
                            )?,
                        ResponsesToolCallOutputItem::InputFile {} => {
                            contents.push(Prompt::plain(UNSUPPORTED_DOCUMENT_PLACEHOLDER));
                        }
                    }
                }
                (Prompt::join(&contents, "\n\n"), image_sources)
            }
        };
        validate_data_url_sources(&image_sources, item_path)?;
        Ok((content, image_sources))
    }
}

/// Serialize discovered tools as plain text, flattening namespace tools and
/// preserving their namespace name and description.
fn tool_search_output_content(tools: Vec<ResponsesTool>) -> serde_json::Result<Prompt> {
    let mut flat_tools = Vec::new();
    for tool in tools {
        match tool {
            ResponsesTool::Namespace {
                name,
                description,
                tools,
            } => {
                let namespace = serde_json::json!({
                    "name": name,
                    "description": description.unwrap_or_default(),
                });
                for tool in tools {
                    let mut tool = serde_json::to_value(tool)?;
                    tool["namespace"] = namespace.clone();
                    flat_tools.push(tool);
                }
            }
            tool => flat_tools.push(serde_json::to_value(tool)?),
        }
    }
    serde_json::to_string(&flat_tools).map(Prompt::plain)
}

/// Convert one Responses image reference. `path` locates the block in error
/// messages, and the image source records `item_path`.
fn input_image_source(
    image: ResponsesInputImage,
    path: &str,
    item_path: &str,
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
        (Some(url), None) => image_url_source(url, image.detail, item_path, |length| {
            format!(
                "{path}: image_url external link length {length} too long, max link length \
                 {MAX_IMAGE_URL_LEN}. Consider passing file data directly with base64 data url instead."
            )
        }),
    }
}

/// Validate the call ID of a tool call: it must be non-empty and differ from
/// every earlier call ID.
fn validate_new_call_id(
    item_index: usize,
    call_id: &str,
    pending_call_ids: &HashSet<String>,
    resolved_call_ids: &HashSet<String>,
) -> Result<(), ConversionError> {
    if call_id.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "Invalid 'input[{item_index}].call_id': empty string. Expected a string with minimum length 1, but got an empty string \
             instead."
        )));
    }
    if pending_call_ids.contains(call_id) || resolved_call_ids.contains(call_id) {
        return Err(ConversionError::bad_request(format!(
            "Duplicate 'call_id': {call_id}."
        )));
    }
    Ok(())
}

/// Move the call ID of a tool output from `pending_call_ids` to
/// `resolved_call_ids`.
///
/// # Errors
///
/// Returns an error when `call_id` is empty, already resolved, or matches no
/// earlier tool call.
fn mark_call_resolved(
    item_index: usize,
    call_id: &str,
    pending_call_ids: &mut HashSet<String>,
    resolved_call_ids: &mut HashSet<String>,
) -> Result<(), ConversionError> {
    if call_id.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "Invalid 'input[{item_index}].call_id': empty string. Expected a string with minimum length 1, but got an empty string \
             instead."
        )));
    }
    if !pending_call_ids.contains(call_id) {
        if resolved_call_ids.contains(call_id) {
            return Err(ConversionError::bad_request(format!(
                "Duplicate tool output for call_id: {call_id}."
            )));
        }
        return Err(ConversionError::bad_request(format!(
            "No tool call found for tool output with call_id {call_id}."
        )));
    }
    pending_call_ids.remove(call_id);
    resolved_call_ids.insert(call_id.to_string());
    Ok(())
}

/// Require every earlier tool call to have an output.
fn ensure_no_pending_calls(pending_call_ids: &HashSet<String>) -> Result<(), ConversionError> {
    if let Some(call_id) = pending_call_ids.iter().next() {
        return Err(ConversionError::bad_request(format!(
            "No tool output found for tool call {call_id}."
        )));
    }
    Ok(())
}

/// The tool calls of one assistant turn and their outputs, collected until
/// every call has an output.
#[derive(Default)]
struct ResponsesToolCallGroup {
    calls: Vec<ToolCall>,
    /// The messages of each call's output: the tool result, optionally
    /// followed by an image resize notice.
    outputs: HashMap<String, Vec<InputMessage>>,
    declarations: Vec<Vec<ToolDefinition>>,
}

impl ResponsesToolCallGroup {
    fn is_complete(&self) -> bool {
        self.outputs.len() >= self.calls.len()
    }

    /// Sort the calls by call ID and append them to the tool calls of the last
    /// message when it is an assistant message, or of a new empty assistant
    /// message otherwise. Then append the output messages of each call in the
    /// same order.
    fn append_messages(self, messages: &mut Vec<InputMessage>) {
        if self.calls.is_empty() {
            return;
        }
        let mut calls = self.calls;
        calls.sort_by(|a, b| a.id.cmp(&b.id));
        let mut outputs = self.outputs;
        let tool_results: Vec<InputMessage> = calls
            .iter()
            .filter_map(|call| outputs.remove(&call.id))
            .flatten()
            .collect();

        if !matches!(messages.last(), Some(InputMessage::Assistant { .. })) {
            messages.push(empty_assistant());
        }
        if let Some(InputMessage::Assistant { tool_calls, .. }) = messages.last_mut() {
            match tool_calls {
                Some(existing) => existing.extend(calls),
                None => *tool_calls = Some(calls),
            }
        }
        messages.extend(tool_results);
        messages.extend(
            self.declarations
                .into_iter()
                .map(|tools| InputMessage::System {
                    content: Prompt::new(),
                    image_sources: Vec::new(),
                    tools: Some(tools),
                }),
        );
    }
}

fn empty_assistant() -> InputMessage {
    InputMessage::Assistant {
        content: String::new(),
        reasoning_content: None,
        tool_calls: None,
        prefix: false,
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
    output: Vec<InputMessage>,
    messages: &mut Vec<InputMessage>,
) {
    let Some(group) = pending_group.as_mut() else {
        messages.extend(output);
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

/// Build the messages of one tool output: the tool result, and for V4.1 a
/// following image resize notice.
///
/// For V4.1, this takes from `items` a developer message that directly
/// follows the output and holds exactly one `input_text` block wrapped in
/// `<image_resize_notice>` tags, and places it right after the tool result as a
/// system message.
fn take_tool_output_messages(
    items: &mut Peekable<impl Iterator<Item = (usize, ResponsesInputItem)>>,
    model_version: ModelVersion,
    call_id: &str,
    content: Prompt,
    image_sources: Vec<ImageSource>,
) -> Vec<InputMessage> {
    let mut output = vec![InputMessage::Tool {
        content,
        image_sources,
        tool_call_id: Some(call_id.to_owned()),
    }];
    if model_version == ModelVersion::DeepseekV41Flash
        && let Some((_, item)) = items.next_if(|(_, item)| image_resize_notice(item).is_some())
        && let Some(notice) = image_resize_notice(&item)
    {
        output.push(InputMessage::System {
            content: Prompt::plain(notice),
            image_sources: Vec::new(),
            tools: None,
        });
    }
    output
}

/// Return the text of a developer message that holds exactly one `input_text`
/// block wrapped in `<image_resize_notice>` tags.
fn image_resize_notice(item: &ResponsesInputItem) -> Option<&str> {
    let (ResponsesInputItem::Typed(ResponsesTypedInputItem::Message(message))
    | ResponsesInputItem::Message(message)) = item
    else {
        return None;
    };
    if !matches!(message.role, ResponsesMessageRole::Developer) {
        return None;
    }
    let ResponsesMessageContent::List(blocks) = &message.content else {
        return None;
    };
    let [ResponsesContentBlock::InputText { text }] = blocks.as_slice() else {
        return None;
    };
    (text.starts_with("<image_resize_notice>") && text.ends_with("</image_resize_notice>"))
        .then_some(text.as_str())
}

/// Convert one message item and append it to `messages`. `item_index`
/// locates the item in error messages.
///
/// A developer message becomes a system message in V4.1 or a user message in
/// V4. Conversion validates data URL images in system and user messages. For
/// V4.1, an assistant message with `input_image` blocks is rejected. When the
/// last message is an assistant message without tool calls, such as one
/// that a reasoning item added, an assistant message item appends its
/// non-empty content to that message's content, separated by a blank line
/// when that message's content is non-empty; otherwise the item becomes a new
/// assistant message.
fn push_message(
    msg: ResponsesInputMessage,
    item_index: usize,
    messages: &mut Vec<InputMessage>,
    converter: &ResponsesContentConverter<'_>,
) -> Result<(), ConversionError> {
    let item_path = format!("input[{item_index}]");
    let (content, image_sources) = msg.content.into_content(&item_path, converter)?;
    let role = match msg.role {
        ResponsesMessageRole::Developer
            if !converter.model_version.supports_mid_conversation_system() =>
        {
            ResponsesMessageRole::User
        }
        role => role,
    };
    match role {
        ResponsesMessageRole::System | ResponsesMessageRole::Developer => {
            validate_data_url_sources(&image_sources, &item_path)?;
            messages.push(InputMessage::System {
                content,
                image_sources,
                tools: None,
            });
        }
        ResponsesMessageRole::User => {
            validate_data_url_sources(&image_sources, &item_path)?;
            messages.push(InputMessage::User {
                content,
                image_sources,
            });
        }
        ResponsesMessageRole::Assistant => {
            if !image_sources.is_empty() {
                return Err(ConversionError::bad_request(format!(
                    "{item_path}: Image in assistant message is unsupported"
                )));
            }
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
                    last_content.push_str(content.as_str());
                }
            } else {
                messages.push(InputMessage::Assistant {
                    content: content.as_str().to_owned(),
                    reasoning_content: None,
                    tool_calls: None,
                    prefix: false,
                });
            }
        }
    }
    Ok(())
}

/// Append the reasoning text of a reasoning item, the `text` of its `content`
/// entries joined with newlines, to the reasoning content of the last message
/// when it is an assistant message with empty content, or of a new empty
/// assistant message otherwise. An item without reasoning text adds nothing;
/// `encrypted_content` is ignored.
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
        messages.push(empty_assistant());
    }
    if !text.is_empty()
        && let Some(InputMessage::Assistant {
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

/// Convert the input items in order and append the resulting messages to
/// `messages`.
///
/// Tool calls and tool outputs collect in a pending [`ResponsesToolCallGroup`]
/// until every call in the group has an output; the group is then appended
/// through [`ResponsesToolCallGroup::append_messages`]. A message, a reasoning
/// item, a `web_search_call` item, or a `function_call_output` item without a
/// call ID requires every earlier call to have an output and appends the
/// pending group before its own messages. A `web_search_call` item and items
/// of unsupported types add no messages.
///
/// # Errors
///
/// Returns an error when `items` is empty, when a call ID is empty or
/// repeated, when an output answers no pending call, when a call has no output
/// before one of the items above or at the end of the input, or when content
/// conversion fails.
fn transform_input_items(
    items: Vec<ResponsesInputItem>,
    messages: &mut Vec<InputMessage>,
    tools: &ResponsesToolSet,
    converter: &ResponsesContentConverter<'_>,
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

    let mut items = items.into_iter().enumerate().peekable();
    while let Some((item_index, item)) = items.next() {
        match item {
            ResponsesInputItem::Typed(typed) => match typed {
                ResponsesTypedInputItem::Message(msg) => {
                    ensure_no_pending_calls(&pending_call_ids)?;
                    append_tool_call_group(&mut pending_group, messages);
                    push_message(msg, item_index, messages, converter)?;
                }
                ResponsesTypedInputItem::Reasoning(reasoning) => {
                    ensure_no_pending_calls(&pending_call_ids)?;
                    append_tool_call_group(&mut pending_group, messages);
                    push_reasoning(reasoning, messages);
                }
                ResponsesTypedInputItem::FunctionCall(fc) => {
                    validate_new_call_id(
                        item_index,
                        &fc.call_id,
                        &pending_call_ids,
                        &resolved_call_ids,
                    )?;
                    pending_call_ids.insert(fc.call_id.clone());
                    push_call_to_group(
                        &mut pending_group,
                        tools.convert_function_call(fc, &mut historical_names),
                    );
                }
                ResponsesTypedInputItem::FunctionCallOutput(fco) => {
                    let (content, image_sources) = fco
                        .output
                        .into_content(&format!("input[{item_index}]"), converter)?;
                    if let Some(call_id) = fco.call_id {
                        mark_call_resolved(
                            item_index,
                            &call_id,
                            &mut pending_call_ids,
                            &mut resolved_call_ids,
                        )?;
                        let output = take_tool_output_messages(
                            &mut items,
                            converter.model_version,
                            &call_id,
                            content,
                            image_sources,
                        );
                        push_output_to_group(&mut pending_group, call_id, output, messages);
                    } else {
                        // An output without a call ID answers no call.
                        ensure_no_pending_calls(&pending_call_ids)?;
                        append_tool_call_group(&mut pending_group, messages);
                        messages.push(InputMessage::Tool {
                            content,
                            image_sources,
                            tool_call_id: None,
                        });
                    }
                }
                ResponsesTypedInputItem::CustomToolCall(call) => {
                    validate_new_call_id(
                        item_index,
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
                    mark_call_resolved(
                        item_index,
                        &output.call_id,
                        &mut pending_call_ids,
                        &mut resolved_call_ids,
                    )?;
                    let (content, image_sources) = output
                        .output
                        .map(|content| {
                            content.into_content(&format!("input[{item_index}]"), converter)
                        })
                        .transpose()?
                        .unwrap_or_default();
                    let tool_output = take_tool_output_messages(
                        &mut items,
                        converter.model_version,
                        &output.call_id,
                        content,
                        image_sources,
                    );
                    push_output_to_group(&mut pending_group, output.call_id, tool_output, messages);
                }
                ResponsesTypedInputItem::WebSearchCall => {
                    ensure_no_pending_calls(&pending_call_ids)?;
                    append_tool_call_group(&mut pending_group, messages);
                }
                ResponsesTypedInputItem::ToolSearchCall(call) => {
                    if !converter.model_version.supports_mid_conversation_system() {
                        continue;
                    }
                    validate_new_call_id(
                        item_index,
                        &call.call_id,
                        &pending_call_ids,
                        &resolved_call_ids,
                    )?;
                    pending_call_ids.insert(call.call_id.clone());
                    push_call_to_group(
                        &mut pending_group,
                        ToolCall {
                            id: call.call_id,
                            name: "tool_search".to_string(),
                            arguments: serde_json::Value::Object(call.arguments).to_string(),
                        },
                    );
                }
                ResponsesTypedInputItem::ToolSearchOutput(output) => {
                    if !converter.model_version.supports_mid_conversation_system() {
                        continue;
                    }
                    mark_call_resolved(
                        item_index,
                        &output.call_id,
                        &mut pending_call_ids,
                        &mut resolved_call_ids,
                    )?;
                    let content = tool_search_output_content(output.tools).map_err(|error| {
                        ConversionError::bad_request(format!("input[{item_index}]: {error}"))
                    })?;
                    if let Some(declarations) = tools
                        .discovered_tools
                        .get(&item_index)
                        .filter(|tools| !tools.is_empty())
                        && let Some(group) = pending_group.as_mut()
                    {
                        group.declarations.push(declarations.clone());
                    }
                    push_output_to_group(
                        &mut pending_group,
                        output.call_id.clone(),
                        vec![InputMessage::Tool {
                            content,
                            image_sources: Vec::new(),
                            tool_call_id: Some(output.call_id),
                        }],
                        messages,
                    );
                }
                ResponsesTypedInputItem::Unsupported => {}
            },
            ResponsesInputItem::Message(msg) => {
                ensure_no_pending_calls(&pending_call_ids)?;
                append_tool_call_group(&mut pending_group, messages);
                push_message(msg, item_index, messages, converter)?;
            }
        }
    }
    append_tool_call_group(&mut pending_group, messages);
    ensure_no_pending_calls(&pending_call_ids)?;

    Ok(())
}

impl ProtocolRequest for ResponsesRequest {
    type Response = ResponsesResponse;

    /// Validate and convert Responses input into a conversation request.
    ///
    /// Checks run in the order model name, tool definitions, input items,
    /// reasoning content, tool schemas and tool choice, text format, and
    /// sampling parameters.
    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError> {
        let model_version = options.resolve_model_version(&self.model)?;
        // Responses has no `thinking` setting; `reasoning.effort` alone
        // overrides the default thinking mode.
        let (thinking_mode, reasoning_effort) = resolve_thinking_mode(
            options.default_thinking_mode,
            None,
            self.reasoning
                .as_ref()
                .and_then(|reasoning| reasoning.effort),
        );
        let echo = self.echo();
        let tool_set = ResponsesToolSet::from_declarations(
            self.tools,
            &options,
            model_version,
            self.input.as_ref(),
        )?;
        let converter = ResponsesContentConverter {
            image_token: options.special_tokens.token(SpecialToken::Image),
            model_version,
        };
        let mut messages: Vec<InputMessage> = Vec::new();

        if let Some(instructions) = self.instructions
            && !instructions.is_empty()
        {
            messages.push(InputMessage::System {
                content: Prompt::plain(instructions),
                image_sources: Vec::new(),
                tools: None,
            });
        }

        match self.input {
            Some(ResponsesInput::String(s)) => messages.push(InputMessage::User {
                content: Prompt::plain(s),
                image_sources: Vec::new(),
            }),
            Some(ResponsesInput::Items(items)) => {
                transform_input_items(items, &mut messages, &tool_set, &converter)?
            }
            None if messages.is_empty() => {
                return Err(ConversionError::bad_request(
                    "Either input or instructions must be provided",
                ));
            }
            None => {}
        }

        normalize_reasoning_content(
            &mut messages,
            thinking_mode,
            tool_set.has_tools(),
            options.reasoning_content_behavior,
        )
        .map_err(|()| {
            ConversionError::bad_request(
                "The `reasoning_text` in the thinking mode must be passed back to the API.",
            )
        })?;
        let response_options = ResponseOptions {
            namespace_tool_map: Some(tool_set.namespace_tool_map()),
            responses_echo: Some(echo),
            ..tool_set.response_options.clone()
        };
        let (tools, tool_choice) =
            tool_set.convert_tools(self.tool_choice, thinking_mode, &options)?;
        normalize_message_tools(&mut messages, &tools);
        let (response_format, constraint_format) = convert_text_format(self.text, &messages)?;
        let output_constraint = resolve_output_constraint(
            &tools,
            tool_choice,
            &constraint_format,
            !matches!(response_format, ResponseFormat::Text),
        )?;
        let mut inference_options = RawInferenceOptions {
            thinking_mode,
            temperature: self.temperature,
            top_p: self.top_p,
            logprobs: self.top_logprobs.map(|_| true),
            top_logprobs: self.top_logprobs,
            max_tokens: self.max_output_tokens,
            ..RawInferenceOptions::default()
        }
        .into_inference_options(options.resolve_max_tokens_upper_bound(thinking_mode))?;
        let parse_tool_calls = !tools.is_empty() && tool_choice != ToolChoice::None;
        inference_options.top_p =
            Some(model_version.resolve_top_p(self.top_p, thinking_mode, parse_tool_calls));
        inference_options.output_constraint = output_constraint;
        let parsing_options = ParsingOptions {
            parse_tool_calls,
            starts_in_tool_calls: tool_choice == ToolChoice::Required,
            parse_json_output: matches!(response_format, ResponseFormat::JsonObject),
            thinking_initial_stage: thinking_mode.then_some(ThinkingInitialStage::ReasoningStart),
            special_tokens: Arc::clone(&options.special_tokens),
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
            model_version,
            stream: self.stream.unwrap_or(false),
            response_options,
        })
    }

    fn chunk_generator(
        request: &ConversationRequest,
        response_id: String,
        model: String,
    ) -> ResponsesChunkGenerator {
        ResponsesChunkGenerator::new(
            response_id,
            model,
            request.conversation.thinking_mode,
            request.response_options.clone(),
        )
    }
}

impl ResponsesRequest {
    /// Return the request settings that response objects report.
    ///
    /// Conversion preserves these settings for
    /// [`ProtocolRequest::chunk_generator`].
    pub fn echo(&self) -> ResponsesEcho {
        let mut reasoning = self.reasoning.clone().unwrap_or_default();
        if reasoning.summary.is_none() {
            reasoning.summary = reasoning.generate_summary.take();
        }
        reasoning.generate_summary = None;
        ResponsesEcho {
            instructions: self.instructions.clone(),
            max_output_tokens: self.max_output_tokens,
            reasoning,
            temperature: self.temperature.unwrap_or(1.0),
            text: self.text.clone().unwrap_or_default(),
            tool_choice: self.tool_choice.clone().unwrap_or_default(),
            tools: self
                .tools
                .iter()
                .flatten()
                .filter(|tool| !matches!(tool, ResponsesTool::Unsupported))
                .cloned()
                .collect(),
            top_logprobs: self.top_logprobs.unwrap_or(0),
            top_p: self.top_p.unwrap_or(1.0),
            user: self.user.clone(),
        }
    }

    /// Return the function names of the declared namespace tools mapped to
    /// their namespace and local name, as conversion with `options` assigns
    /// them.
    ///
    /// Conversion supplies the namespace mapping to
    /// [`ProtocolRequest::chunk_generator`].
    ///
    /// # Errors
    ///
    /// Returns the error of an unknown model name, or the conversion error of
    /// an invalid function or namespace declaration.
    pub fn namespace_tool_map(
        &self,
        options: &ConversionOptions,
    ) -> Result<HashMap<String, (String, String)>, ConversionError> {
        let model_version = options.resolve_model_version(&self.model)?;
        // Only function and namespace declarations determine the names of
        // namespace tools. Replacing the other declarations with ignored ones
        // keeps their errors, such as a rejected `web_search` declaration or
        // an unsupported custom tool name, out of this method while keeping
        // the `tools[N]` positions of the error messages.
        let tools: Option<Vec<ResponsesTool>> = self.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| match tool {
                    ResponsesTool::Function(_) | ResponsesTool::Namespace { .. } => tool.clone(),
                    _ => ResponsesTool::Unsupported,
                })
                .collect()
        });
        Ok(
            ResponsesToolSet::from_declarations(tools, options, model_version, None)?
                .namespace_tool_map(),
        )
    }
}

/// Convert the text format into the format the conversation renders and the
/// format for the output constraint. Conversion validates a `json_schema`
/// schema; rendering writes it into the prompt as written, and the constraint
/// takes the normalized schema of a strict format.
///
/// # Errors
///
/// Returns [`ConversionError::BadRequest`] for an invalid `json_schema`
/// schema, or for `json_object` when no input message contains the word
/// `json`.
fn convert_text_format(
    text: Option<ResponsesTextConfig>,
    messages: &[InputMessage],
) -> Result<(ResponseFormat, ResponseFormat), ConversionError> {
    match text.map(|text| text.format).unwrap_or_default() {
        ResponsesTextFormat::Text => Ok((ResponseFormat::Text, ResponseFormat::Text)),
        ResponsesTextFormat::JsonSchema { schema, strict, .. } => {
            let constraint_schema = validate_response_schema(&schema, strict)?;
            Ok((
                ResponseFormat::JsonSchema(schema),
                ResponseFormat::JsonSchema(constraint_schema),
            ))
        }
        ResponsesTextFormat::JsonObject => {
            if !contains_json_instruction(messages) {
                return Err(ConversionError::bad_request(
                    "Response input messages must contain the word 'json' in some form to use 'text.format' of type \
                     'json_object'.",
                ));
            }
            Ok((ResponseFormat::JsonObject, ResponseFormat::JsonObject))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ResponsesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "DeepSeek-V4.1-Flash",
            "input": "hi",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "bad name"},
            ],
        }))
        .expect("the request deserializes")
    }

    #[test]
    fn tool_errors_locate_the_declaration_by_request_position() {
        let expected = "Invalid 'tools[1].name': string does not match pattern. Expected a string \
                        that matches the pattern '^[a-zA-Z0-9_-]+$'.";
        let options = ConversionOptions::new();
        assert_eq!(
            request()
                .namespace_tool_map(&options)
                .expect_err("the declaration is rejected")
                .to_string(),
            expected
        );
        assert_eq!(
            request()
                .convert(options)
                .expect_err("the declaration is rejected")
                .to_string(),
            expected
        );
    }
}
