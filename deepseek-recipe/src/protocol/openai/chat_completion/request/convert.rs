use std::collections::HashSet;
use std::sync::Arc;

use deepseek_recipe_core::conversation::{Conversation, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{ImageSource, MAX_IMAGE_URL_LEN};
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::protocol::client_tools::{
    ClientToolChoice, ClientToolDeclaration, convert_client_tools,
};
use crate::protocol::image::{
    UNSUPPORTED_IMAGE_PLACEHOLDER, file_data_image_source, file_id_unsupported, image_url_source,
    validate_data_url_sources,
};
use crate::protocol::output_constraint::resolve_output_constraint;
use crate::protocol::reasoning::resolve_thinking_mode;
use crate::protocol::reasoning_content::normalize_reasoning_content;
use crate::protocol::validation::{
    RawInferenceOptions, contains_json_instruction, validate_response_schema,
    validate_stop_sequence, validate_stop_sequences, validate_tool_name,
};
use crate::request::{
    ConversationRequest, ConversionError, ConversionOptions, ModelVersion, ProtocolRequest,
    ResponseOptions,
};
use crate::stream::{ParsingOptions, ThinkingInitialStage};

use super::super::response::{ChatCompletionChunkGenerator, ChatCompletionResponse};
use super::schema::{
    ChatCompletionRequest, ChatCompletionRequestContent, ChatCompletionRequestContentBlock,
    ChatCompletionRequestMessage, ChatCompletionResponseFormat, ChatCompletionStopSequences,
    ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionToolChoiceOption,
};

impl ProtocolRequest for ChatCompletionRequest {
    type Response = ChatCompletionResponse;

    /// Validate and convert Chat Completions input into a conversation request.
    ///
    /// Checks run in the order model name, tool names, messages, reasoning
    /// content, prefix continuation, JSON instruction, tool schemas and tool
    /// choice, prefix combinations, stop sequences and stream options, response
    /// format, and sampling parameters.
    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError> {
        let model_version = options.resolve_model_version(&self.model)?;
        let (thinking_mode, reasoning_effort) = resolve_thinking_mode(
            options.default_thinking_mode,
            self.thinking.as_ref().map(|thinking| thinking.r#type),
            self.reasoning_effort,
        );
        let tools = self.tools.unwrap_or_default();
        for (index, tool) in tools.iter().enumerate() {
            validate_tool_name(&tool.function.name, &format!("tools[{index}].function"))?;
        }
        let has_tools = !tools.is_empty();
        let has_tool_context = has_tools
            || self.messages.iter().any(|message| {
                matches!(
                    message,
                    ChatCompletionRequestMessage::Assistant {
                        tool_calls: Some(_),
                        ..
                    } | ChatCompletionRequestMessage::Tool { .. }
                )
            });
        let image_token = options.special_tokens.token(SpecialToken::Image);
        let mut messages = transform_messages(self.messages, image_token, model_version)?;
        normalize_reasoning_content(
            &mut messages,
            thinking_mode,
            has_tools,
            options.reasoning_content_behavior,
        )
        .map_err(|()| {
            ConversionError::bad_request(
                "The `reasoning_content` in the thinking mode must be passed back to the API.",
            )
        })?;
        let is_prefix_continuation = validate_prefix(&messages)?;
        let json_mode = matches!(
            self.response_format,
            Some(ChatCompletionResponseFormat::JsonObject)
        );
        if json_mode && !contains_json_instruction(&messages) {
            return Err(ConversionError::bad_request(
                "Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'.",
            ));
        }
        let (tools, tool_choice) = convert_tools(tools, self.tool_choice, thinking_mode)?;
        if json_mode && is_prefix_continuation {
            return Err(ConversionError::bad_request(
                "response_format json_object should not be used with prefix",
            ));
        }
        if has_tool_context && is_prefix_continuation {
            return Err(ConversionError::bad_request(
                "Function call should not be used with prefix",
            ));
        }
        validate_api_params(
            self.stop.as_ref(),
            self.stream,
            self.stream_options.as_ref(),
        )?;
        let has_response_format = self.response_format.is_some();
        let (response_format, constraint_format) = match self.response_format {
            Some(ChatCompletionResponseFormat::JsonObject) => {
                (ResponseFormat::JsonObject, ResponseFormat::JsonObject)
            }
            Some(ChatCompletionResponseFormat::JsonSchema { json_schema }) => {
                let constraint_schema =
                    validate_response_schema(&json_schema.schema, json_schema.strict)?;
                (
                    ResponseFormat::JsonSchema(json_schema.schema),
                    ResponseFormat::JsonSchema(constraint_schema),
                )
            }
            Some(ChatCompletionResponseFormat::Regex { regex }) => (
                ResponseFormat::Regex(regex.clone()),
                ResponseFormat::Regex(regex),
            ),
            Some(ChatCompletionResponseFormat::Text) | None => {
                (ResponseFormat::Text, ResponseFormat::Text)
            }
        };
        let output_constraint = resolve_output_constraint(
            &tools,
            tool_choice,
            &constraint_format,
            has_response_format,
        )?;
        let mut inference_options = RawInferenceOptions {
            thinking_mode,
            n: self.n,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            temperature: self.temperature,
            top_p: self.top_p,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            max_tokens: self.max_tokens,
            clamp_max_tokens: false,
            seed: self.seed,
        }
        .into_inference_options(options.resolve_max_tokens_upper_bound(thinking_mode))?;

        let parse_tool_calls = !tools.is_empty() && tool_choice != ToolChoice::None;
        inference_options.top_p =
            Some(model_version.resolve_top_p(self.top_p, thinking_mode, parse_tool_calls));
        inference_options.output_constraint = output_constraint;
        let parsing_options = ParsingOptions {
            parse_tool_calls,
            starts_in_tool_calls: tool_choice == ToolChoice::Required,
            parse_json_output: json_mode,
            thinking_initial_stage: thinking_mode.then(|| thinking_initial_stage(&messages)),
            stop_sequences: self.stop.map(Vec::<String>::from).unwrap_or_default(),
            special_tokens: Arc::clone(&options.special_tokens),
        };
        Ok(ConversationRequest {
            conversation: Conversation {
                messages,
                thinking_mode,
                tools,
                tool_choice,
                reasoning_effort,
                response_format,
            },
            inference_options,
            parsing_options,
            model_version,
            stream: self.stream.unwrap_or(false),
            response_options: ResponseOptions {
                include_usage: self
                    .stream_options
                    .and_then(|options| options.include_usage)
                    .unwrap_or(false),
                ..ResponseOptions::default()
            },
        })
    }

    fn chunk_generator(
        request: &ConversationRequest,
        response_id: String,
        model: String,
    ) -> ChatCompletionChunkGenerator {
        ChatCompletionChunkGenerator::new(
            response_id,
            model,
            !request.stream || request.response_options.include_usage,
            request.conversation.thinking_mode,
        )
        .with_starts_in_reasoning(
            request
                .parsing_options
                .thinking_initial_stage
                .is_some_and(|stage| stage.starts_in_reasoning()),
        )
    }
}

/// Return the initial reasoning stage of a thinking-mode request.
///
/// A final prefix message with content continues the answer; one with only
/// reasoning content continues the reasoning. Otherwise reasoning starts.
fn thinking_initial_stage(messages: &[InputMessage]) -> ThinkingInitialStage {
    match messages.last() {
        Some(InputMessage::Assistant {
            content,
            reasoning_content,
            prefix: true,
            ..
        }) => {
            if !content.is_empty() {
                ThinkingInitialStage::ContentContinuation
            } else if reasoning_content
                .as_ref()
                .is_some_and(|reasoning_content| !reasoning_content.is_empty())
            {
                ThinkingInitialStage::ReasoningContinuation
            } else {
                ThinkingInitialStage::ReasoningStart
            }
        }
        _ => ThinkingInitialStage::ReasoningStart,
    }
}

/// Validate prefix continuation, returning whether it is requested. Only the
/// final message can request prefix continuation.
fn validate_prefix(messages: &[InputMessage]) -> Result<bool, ConversionError> {
    let Some(index) = messages
        .iter()
        .position(|message| matches!(message, InputMessage::Assistant { prefix: true, .. }))
    else {
        return Ok(false);
    };
    if index != messages.len() - 1 {
        return Err(ConversionError::bad_request(
            "Invalid completion message: prefix set true with message which is not final message or message role is not assistant",
        ));
    }
    Ok(true)
}

/// Validate tool schemas and resolve the tool choice.
///
/// A `none` choice or a request without tools skips the schema checks and
/// returns no tool definitions; [`convert_client_tools`] resolves the other
/// choices.
fn convert_tools(
    tools: Vec<ChatCompletionTool>,
    choice: Option<ChatCompletionToolChoiceOption>,
    thinking_mode: bool,
) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
    if matches!(choice, Some(ChatCompletionToolChoiceOption::None)) {
        return Ok((Vec::new(), ToolChoice::None));
    }
    if tools.is_empty() {
        return Ok((Vec::new(), ToolChoice::Auto));
    }
    let tools = tools
        .into_iter()
        .map(|tool| ClientToolDeclaration {
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters,
            strict: tool.function.strict,
            defer_loading: false,
        })
        .collect();
    let choice = match choice {
        Some(ChatCompletionToolChoiceOption::Named { function, .. }) => {
            ClientToolChoice::Named(function.name)
        }
        Some(ChatCompletionToolChoiceOption::Required) => ClientToolChoice::Required,
        _ => ClientToolChoice::Auto,
    };
    convert_client_tools(tools, choice, thinking_mode)
}

impl ChatCompletionRequestContent {
    /// Convert text and image blocks of message `message_index`, joined with
    /// blank lines. Each image block contributes one image placeholder special
    /// token; a model without image input receives the unsupported-image text
    /// instead. The image sources record the message's location,
    /// `.messages[{message_index}]`.
    fn into_content(
        self,
        message_index: usize,
        image_token: &str,
        model_version: ModelVersion,
    ) -> Result<(Prompt, Vec<ImageSource>), ConversionError> {
        let blocks = match self {
            Self::String(content) => return Ok((Prompt::plain(content), Vec::new())),
            Self::List(blocks) => blocks,
        };
        let path = format!(".messages[{message_index}]");
        let supports_image = model_version.supports_image();
        let mut contents = Vec::new();
        let mut image_sources = Vec::new();
        for (block_index, block) in blocks.into_iter().enumerate() {
            match block {
                ChatCompletionRequestContentBlock::Text { text } => {
                    contents.push(Prompt::plain(text));
                }
                ChatCompletionRequestContentBlock::ImageUrl { .. }
                | ChatCompletionRequestContentBlock::File { .. }
                    if !supports_image =>
                {
                    contents.push(Prompt::plain(UNSUPPORTED_IMAGE_PLACEHOLDER));
                }
                ChatCompletionRequestContentBlock::ImageUrl { image_url } => {
                    image_sources.push(image_url_source(
                        image_url.url,
                        image_url.detail,
                        &path,
                        |length| {
                            format!(
                                "{path}.image_url.url[{block_index}]: image_url.url external link \
                                 length {length} too long, max link length {MAX_IMAGE_URL_LEN}. \
                                 Consider passing file data directly with base64 data url \
                                 instead."
                            )
                        },
                    )?);
                    contents.push(Prompt::special(image_token));
                }
                ChatCompletionRequestContentBlock::File {
                    file_id,
                    file_data,
                    filename,
                } => {
                    let error =
                        |detail: &str| ConversionError::bad_request(format!("{path}: {detail}"));
                    match (file_id, file_data) {
                        (None, None) => return Err(error("file must have a file_id or file_data")),
                        (Some(_), Some(_)) => {
                            return Err(error("file cannot have both file_id and file_data"));
                        }
                        (Some(_), None) if filename.is_some() => {
                            return Err(error("file cannot have filename when file_id is present"));
                        }
                        (Some(_), None) => return Err(file_id_unsupported(&path)),
                        (None, Some(file_data)) => {
                            image_sources.push(file_data_image_source(file_data, &path)?);
                        }
                    }
                    contents.push(Prompt::special(image_token));
                }
            }
        }
        Ok((Prompt::join(&contents, "\n\n"), image_sources))
    }
}

/// Convert user or tool content and validate its data URL images.
fn convert_user_content(
    content: ChatCompletionRequestContent,
    message_index: usize,
    image_token: &str,
    model_version: ModelVersion,
) -> Result<(Prompt, Vec<ImageSource>), ConversionError> {
    let (content, image_sources) =
        content.into_content(message_index, image_token, model_version)?;
    validate_data_url_sources(&image_sources, &format!(".messages[{message_index}]"))?;
    Ok((content, image_sources))
}

/// Remove an Anthropic billing header from system content.
///
/// Content that starts with `x-anthropic-billing-header` keeps the text from
/// the first newline or `You are`, whichever comes first, without leading
/// whitespace. Without either, the content becomes empty.
fn strip_billing_header(mut content: Prompt) -> Prompt {
    let text = content.as_str();
    if !text.starts_with("x-anthropic-billing-header") {
        return content;
    }
    let start = ["\n", "You are"]
        .iter()
        .filter_map(|pattern| text.find(pattern))
        .min();
    match start {
        Some(start) => {
            let start = text.len() - text[start..].trim_start().len();
            content.split_off(start)
        }
        None => Prompt::new(),
    }
}

/// Convert the request messages into conversation messages.
///
/// A `system` message becomes a system message without an Anthropic billing
/// header, or nothing when its content is then empty. A `user` message becomes
/// a user message, and a `latest_reminder` message a reminder. An `assistant`
/// message becomes an assistant message with its tool calls, followed by one
/// tool message per `tool` message that answers those calls.
///
/// # Errors
///
/// Returns an error when `messages` is empty; when an `assistant`
/// message of a model with image input holds an image block; when an
/// `assistant` message has neither `content` nor `tool_calls`, has an empty
/// `tool_calls` list, or repeats a tool call ID; when the messages following
/// an `assistant` message with `tool_calls` are not one `tool` message per
/// call, each answering a distinct call of that message; when a `tool` message
/// does not follow such an `assistant` message; or when, for a model with
/// image input, an `image_url` or `file` block is invalid, including a system,
/// user, or tool data URL that fails [`validate_data_url_sources`].
fn transform_messages(
    messages: Vec<ChatCompletionRequestMessage>,
    image_token: &str,
    model_version: ModelVersion,
) -> Result<Vec<InputMessage>, ConversionError> {
    if messages.is_empty() {
        return Err(ConversionError::bad_request("Empty input messages"));
    }

    let mut result = Vec::new();
    let mut message_iter = messages.into_iter().enumerate();
    while let Some((index, message)) = message_iter.next() {
        match message {
            ChatCompletionRequestMessage::System { content, .. } => {
                let (content, image_sources) =
                    content.into_content(index, image_token, model_version)?;
                let content = strip_billing_header(content);
                if !content.is_empty() {
                    validate_data_url_sources(&image_sources, &format!(".messages[{index}]"))?;
                    result.push(InputMessage::System {
                        content,
                        image_sources,
                        tools: None,
                    });
                }
            }
            ChatCompletionRequestMessage::User { content, .. } => {
                let (content, image_sources) =
                    convert_user_content(content, index, image_token, model_version)?;
                result.push(InputMessage::User {
                    content,
                    image_sources,
                });
            }
            ChatCompletionRequestMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
                prefix,
                ..
            } => {
                if content.is_none() && tool_calls.is_none() {
                    return Err(ConversionError::bad_request(
                        "Invalid assistant message: content or tool_calls must be set",
                    ));
                }
                let (content, image_sources) = content
                    .map(|content| content.into_content(index, image_token, model_version))
                    .transpose()?
                    .unwrap_or_default();
                if !image_sources.is_empty() {
                    return Err(ConversionError::bad_request(
                        "Image in assistant message is not supported",
                    ));
                }

                let mut tool_results = Vec::new();
                let mut input_tool_calls = None;
                if let Some(tool_calls) = &tool_calls {
                    let count = tool_calls.len();
                    if count == 0 {
                        return Err(ConversionError::bad_request(format!(
                            "Invalid 'messages[{index}].tool_calls': empty array. Expected an array with minimum length 1, but got an empty \
                             array instead."
                        )));
                    }
                    let mut tool_ids: HashSet<String> = HashSet::new();
                    for tool_call in tool_calls {
                        if tool_ids.contains(&tool_call.id) {
                            return Err(ConversionError::bad_request(format!(
                                "Duplicate value for 'tool_call_id' of {} in message[{}]",
                                tool_call.id, index
                            )));
                        }
                        tool_ids.insert(tool_call.id.clone());
                    }

                    // The assistant message is followed by one tool message per call.
                    let mut checked_tool_call_ids: HashSet<String> = HashSet::new();
                    for _ in 0..count {
                        let Some((
                            tool_result_index,
                            ChatCompletionRequestMessage::Tool {
                                content,
                                tool_call_id,
                            },
                        )) = message_iter.next()
                        else {
                            return Err(ConversionError::bad_request(
                                "An assistant message with 'tool_calls' must be followed by tool messages responding to each \
                                 'tool_call_id'. (insufficient tool messages following tool_calls message)",
                            ));
                        };
                        if !tool_ids.contains(&tool_call_id) {
                            return Err(ConversionError::bad_request(format!(
                                "An assistant message with 'tool_calls' must be followed by tool messages responding to each \
                                 'tool_call_id', The following tool_call_ids did not have response messages: {tool_call_id}"
                            )));
                        }
                        if checked_tool_call_ids.contains(&tool_call_id) {
                            return Err(ConversionError::bad_request(format!(
                                "Duplicate value for 'tool_call_id' of {tool_call_id} in message[{tool_result_index}]",
                            )));
                        }
                        checked_tool_call_ids.insert(tool_call_id.clone());

                        let (content, image_sources) = convert_user_content(
                            content,
                            tool_result_index,
                            image_token,
                            model_version,
                        )?;
                        tool_results.push(InputMessage::Tool {
                            content,
                            image_sources,
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                    input_tool_calls = Some(
                        tool_calls
                            .iter()
                            .map(|tool_call| ToolCall {
                                id: tool_call.id.clone(),
                                name: tool_call.function.name.clone(),
                                arguments: tool_call.function.arguments.clone(),
                            })
                            .collect(),
                    );
                }

                result.push(InputMessage::Assistant {
                    content: content.as_str().to_owned(),
                    reasoning_content,
                    tool_calls: input_tool_calls,
                    prefix: prefix == Some(true),
                });
                result.extend(tool_results);
            }
            ChatCompletionRequestMessage::Tool { .. } => {
                // Tool results must follow an assistant tool call.
                return Err(ConversionError::bad_request(
                    "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'",
                ));
            }
            ChatCompletionRequestMessage::LatestReminder { content } => {
                result.push(InputMessage::LatestReminder { content });
            }
        }
    }

    Ok(result)
}

/// Validate stop sequences and streaming options.
fn validate_api_params(
    stop: Option<&ChatCompletionStopSequences>,
    stream: Option<bool>,
    stream_options: Option<&ChatCompletionStreamOptions>,
) -> Result<(), ConversionError> {
    match stop {
        Some(ChatCompletionStopSequences::Array(stop_sequences)) => {
            validate_stop_sequences(stop_sequences)?;
        }
        Some(ChatCompletionStopSequences::String(stop)) => validate_stop_sequence(stop)?,
        None => {}
    }
    if stream != Some(true) && stream_options.is_some() {
        return Err(ConversionError::bad_request(
            "stream_options should be set along with stream = true",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn billing_headers_are_stripped_from_system_content() {
        let strip = |text: &str| {
            strip_billing_header(Prompt::plain(text))
                .as_str()
                .to_owned()
        };
        assert_eq!(
            strip("x-anthropic-billing-header: a\n  You are b"),
            "You are b"
        );
        assert_eq!(
            strip("x-anthropic-billing-header: aYou are b\nc"),
            "You are b\nc"
        );
        assert_eq!(strip("x-anthropic-billing-header: a"), "");
        assert_eq!(strip("system"), "system");
    }
}
