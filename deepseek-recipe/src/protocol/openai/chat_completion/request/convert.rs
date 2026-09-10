use std::collections::HashSet;

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{IMAGE_SPECIAL_TOKEN, ImageSource};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::protocol::image::{
    data_url_image_source, file_id_unsupported, image_not_allowed, image_url_source,
};
use crate::protocol::validation::{
    contains_json_instruction, image_special_token_not_allowed, validate_max_tokens,
    validate_sampling, validate_tool_name, validate_tool_parameters, validate_tool_text,
};
use crate::request::{
    ConversationRequest, ConversionError, ConversionOptions, InferenceOptions, ProtocolRequest,
};
use crate::stream::state_machine::{ParsingOptions, ReasoningStage};

use super::super::response::{ChatCompletionChunkGenerator, ChatCompletionResponse};
use super::schema::{
    ChatCompletionReasoningEffort, ChatCompletionRequest, ChatCompletionRequestContent,
    ChatCompletionRequestContentBlock, ChatCompletionRequestMessage, ChatCompletionResponseFormat,
    ChatCompletionStopSequences, ChatCompletionStreamOptions, ChatCompletionThinkingType,
    ChatCompletionTool, ChatCompletionToolChoiceMode, ChatCompletionToolChoiceOption,
};

/// Maximum number of stop sequences accepted by the adapter.
const MAX_STOP_SEQUENCES: usize = 16;

impl ProtocolRequest for ChatCompletionRequest {
    type Response = ChatCompletionResponse;

    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError> {
        let (thinking_mode, reasoning_effort) = self.resolve_thinking(options);
        let mut parsing_options = self.parsing_options(thinking_mode);
        let inference_options = self.inference_options()?;
        validate_api_params(
            self.stop.as_ref(),
            self.stream,
            self.stream_options.as_ref(),
        )?;
        let (tools, tool_choice) = convert_tools(self.tools, self.tool_choice, thinking_mode)?;
        parsing_options.parse_tool_calls = !tools.is_empty() && tool_choice != ToolChoice::None;
        parsing_options.tool_call_initial_stage = tool_choice == ToolChoice::Required;
        let messages = transform_messages(self.messages)?;
        validate_tool_text(&tools, &messages)?;
        let response_format = convert_response_format(&messages, self.response_format)?;
        parsing_options.parse_json_output = matches!(response_format, ResponseFormat::JsonObject);
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
            model: Some(self.model),
            stream: self.stream.unwrap_or(false),
        })
    }

    fn chunk_generator(
        request: &ConversationRequest,
        id: String,
        model: String,
    ) -> ChatCompletionChunkGenerator {
        ChatCompletionChunkGenerator::new(
            id,
            model,
            !request.stream,
            request.conversation.thinking_mode,
        )
    }
}

impl ChatCompletionRequest {
    fn resolve_thinking(&self, options: ConversionOptions) -> (bool, Option<ReasoningEffort>) {
        let effort = self
            .reasoning_effort
            .and_then(ChatCompletionReasoningEffort::to_reasoning_effort);
        let thinking = match self.thinking.as_ref().map(|thinking| thinking.kind) {
            Some(ChatCompletionThinkingType::Enabled) => true,
            Some(ChatCompletionThinkingType::Disabled) => false,
            None => match self.reasoning_effort {
                Some(ChatCompletionReasoningEffort::None) => false,
                Some(_) => true,
                None => options.default_thinking_mode,
            },
        };
        (
            thinking,
            thinking.then_some(effort.unwrap_or(ReasoningEffort::High)),
        )
    }

    fn inference_options(&self) -> Result<InferenceOptions, ConversionError> {
        if self.n.is_some_and(|value| value != 1) {
            return Err(ConversionError::bad_request(
                "Invalid n value (currently only n = 1 is supported)",
            ));
        }
        for (name, value) in [
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            if value.is_some_and(|value| !(-2.0..=2.0).contains(&value)) {
                return Err(ConversionError::bad_request(format!(
                    "{name} must be in [-2, 2]"
                )));
            }
        }
        validate_max_tokens(self.max_tokens, "max_tokens")?;
        validate_sampling(self.temperature, self.top_p)?;
        if self.seed.is_some_and(|value| value >= 1u64 << 63) {
            return Err(ConversionError::bad_request("seed must be in [0, 2^63)"));
        }
        Ok(InferenceOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            ..InferenceOptions::default()
        })
    }

    fn parsing_options(&self, thinking: bool) -> ParsingOptions {
        ParsingOptions {
            reasoning_initial_stage: thinking.then_some(ReasoningStage::Start),
            stop_sequences: self
                .stop
                .clone()
                .map(Vec::<String>::from)
                .unwrap_or_default(),
            ..ParsingOptions::default()
        }
    }

    /// Return the setting for usage inclusion in response chunks.
    ///
    /// Read this before consuming the request with `convert`, then pass the value
    /// to `ChatCompletionChunkGenerator::with_include_usage`.
    pub fn include_usage(&self) -> bool {
        self.stream != Some(true)
            || self
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false)
    }
}

fn convert_tools(
    tools: Option<Vec<ChatCompletionTool>>,
    choice: Option<ChatCompletionToolChoiceOption>,
    thinking: bool,
) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
    let tools = tools.unwrap_or_default();
    for (index, tool) in tools.iter().enumerate() {
        validate_tool_name(&tool.function.name, &format!("tools.{index}.function.name"))?;
    }
    if matches!(
        choice,
        Some(ChatCompletionToolChoiceOption::Mode(
            ChatCompletionToolChoiceMode::None
        ))
    ) {
        return Ok((Vec::new(), ToolChoice::None));
    }
    if tools.is_empty() {
        return Ok((Vec::new(), ToolChoice::Auto));
    }

    let mut names = HashSet::new();
    let mut definitions = Vec::new();
    for (index, tool) in tools.into_iter().enumerate() {
        let tool = tool.function;
        if !names.insert(tool.name.clone()) {
            return Err(ConversionError::bad_request("Tool names must be unique"));
        }
        if let Some(parameters) = &tool.parameters {
            validate_tool_parameters(parameters, &format!("tools.{index}.function.parameters"))?;
        }
        definitions.push(ToolDefinition {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters.unwrap_or_else(|| serde_json::json!({})),
            strict: tool.strict,
        });
    }
    let choice = match choice {
        Some(ChatCompletionToolChoiceOption::Named(named)) => {
            let name = named.function.name;
            if !names.contains(&name) {
                return Err(ConversionError::bad_request(format!(
                    "tool_choice: no tool named '{name}' was specified"
                )));
            }
            definitions.retain(|tool| tool.name == name);
            ToolChoice::Required
        }
        Some(ChatCompletionToolChoiceOption::Mode(ChatCompletionToolChoiceMode::Required)) => {
            ToolChoice::Required
        }
        _ => ToolChoice::Auto,
    };
    if choice == ToolChoice::Required && thinking {
        return Err(ConversionError::bad_request(
            "Thinking mode does not support this tool_choice",
        ));
    }
    Ok((definitions, choice))
}

impl ChatCompletionReasoningEffort {
    fn to_reasoning_effort(self) -> Option<ReasoningEffort> {
        match self {
            Self::None => None,
            Self::Minimal | Self::Low => Some(ReasoningEffort::Low),
            Self::Medium | Self::High => Some(ReasoningEffort::High),
            Self::Xhigh => Some(ReasoningEffort::Xhigh),
            Self::Max => Some(ReasoningEffort::Max),
        }
    }
}

impl ChatCompletionRequestContent {
    /// Return whether text carries [`IMAGE_SPECIAL_TOKEN`].
    fn contains_image_special_token(&self) -> bool {
        match self {
            Self::String(content) => content.contains(IMAGE_SPECIAL_TOKEN),
            Self::List(blocks) => blocks.iter().any(|block| match block {
                ChatCompletionRequestContentBlock::Text { text } => {
                    text.contains(IMAGE_SPECIAL_TOKEN)
                }
                _ => false,
            }),
        }
    }

    /// Convert text and image blocks. Each image block contributes one
    /// [`IMAGE_SPECIAL_TOKEN`] placeholder in the returned text.
    fn into_content(self, path: &str) -> Result<(String, Vec<ImageSource>), ConversionError> {
        match self {
            Self::String(content) => Ok((content, Vec::new())),
            Self::List(blocks) => {
                let mut texts = Vec::new();
                let mut image_sources = Vec::new();
                for (block_index, block) in blocks.into_iter().enumerate() {
                    match block {
                        ChatCompletionRequestContentBlock::Text { text } => texts.push(text),
                        ChatCompletionRequestContentBlock::ImageUrl { image_url } => {
                            let block_path = format!("{path}.content[{block_index}].image_url.url");
                            image_sources.push(image_url_source(
                                image_url.url,
                                image_url.detail,
                                &block_path,
                            )?);
                            texts.push(IMAGE_SPECIAL_TOKEN.to_string());
                        }
                        ChatCompletionRequestContentBlock::File {
                            file_id,
                            file_data,
                            filename: _,
                        } => {
                            if let Some(_file_id) = file_id {
                                return Err(file_id_unsupported(&format!(
                                    "{path}.content[{block_index}]"
                                )));
                            }
                            let Some(file_data) = file_data else {
                                return Err(ConversionError::bad_request(format!(
                                    "{path}.content[{block_index}]: file must have file_data"
                                )));
                            };
                            image_sources.push(data_url_image_source(file_data, None));
                            texts.push(IMAGE_SPECIAL_TOKEN.to_string());
                        }
                    }
                }
                Ok((texts.join("\n\n"), image_sources))
            }
        }
    }

    /// Convert text blocks and reject image blocks.
    fn into_text(self, path: &str, role: &str) -> Result<String, ConversionError> {
        let (content, image_sources) = self.into_content(path)?;
        if !image_sources.is_empty() {
            return Err(image_not_allowed(role));
        }
        Ok(content)
    }
}

/// Return whether any message carries [`IMAGE_SPECIAL_TOKEN`] in text that
/// reaches the prompt.
///
/// Message content and assistant reasoning content are checked. Tool call names
/// and arguments are checked after conversion by [`validate_tool_text`].
fn messages_contain_image_special_token(messages: &[ChatCompletionRequestMessage]) -> bool {
    messages.iter().any(|message| match message {
        ChatCompletionRequestMessage::System { content }
        | ChatCompletionRequestMessage::User { content }
        | ChatCompletionRequestMessage::Tool { content, .. } => {
            content.contains_image_special_token()
        }
        ChatCompletionRequestMessage::Assistant {
            content,
            reasoning_content,
            ..
        } => {
            content
                .as_ref()
                .is_some_and(ChatCompletionRequestContent::contains_image_special_token)
                || reasoning_content
                    .as_ref()
                    .is_some_and(|text| text.contains(IMAGE_SPECIAL_TOKEN))
        }
        ChatCompletionRequestMessage::LatestReminder { content } => {
            content.contains(IMAGE_SPECIAL_TOKEN)
        }
    })
}

fn transform_messages(
    messages: Vec<ChatCompletionRequestMessage>,
) -> Result<Vec<InputMessage>, ConversionError> {
    if messages.is_empty() {
        return Err(ConversionError::bad_request("Empty input messages"));
    }
    if messages_contain_image_special_token(&messages) {
        return Err(image_special_token_not_allowed());
    }

    let mut result = Vec::new();
    let mut message_iter = messages.into_iter().enumerate();
    while let Some((index, message)) = message_iter.next() {
        let path = format!("messages[{index}]");
        match message {
            ChatCompletionRequestMessage::System { content } => {
                let content = content.into_text(&path, "system")?;
                if !content.is_empty() {
                    result.push(InputMessage::System { content });
                }
            }
            ChatCompletionRequestMessage::User { content } => {
                let (content, image_sources) = content.into_content(&path)?;
                result.push(InputMessage::User {
                    content,
                    image_sources,
                });
            }
            ChatCompletionRequestMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                if content.is_none() && tool_calls.is_none() {
                    return Err(ConversionError::bad_request(
                        "Invalid assistant message: content or tool_calls must be set",
                    ));
                }
                let content = content
                    .map(|content| content.into_text(&path, "assistant"))
                    .transpose()?
                    .unwrap_or_default();

                let mut tool_results = Vec::new();
                let mut input_tool_calls = None;
                if let Some(tool_calls) = &tool_calls {
                    let count = tool_calls.len();
                    if count == 0 {
                        return Err(ConversionError::bad_request(format!(
                            "Invalid '{path}.tool_calls': empty array. Expected an array with minimum length 1, but got an empty \
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
                                 'tool_call_id'. Unexpected tool_call_id: {tool_call_id}"
                            )));
                        }
                        if checked_tool_call_ids.contains(&tool_call_id) {
                            return Err(ConversionError::bad_request(format!(
                                "Duplicate value for 'tool_call_id' of {tool_call_id} in message[{tool_result_index}]",
                            )));
                        }
                        checked_tool_call_ids.insert(tool_call_id.clone());

                        let (content, image_sources) = content
                            .into_content(&format!("messages[{tool_result_index}].content"))?;
                        tool_results.push(InputMessage::Tool {
                            content,
                            image_sources,
                            tool_call_id,
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
                    content,
                    reasoning_content,
                    tool_calls: input_tool_calls,
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

fn convert_response_format(
    messages: &[InputMessage],
    response_format: Option<ChatCompletionResponseFormat>,
) -> Result<ResponseFormat, ConversionError> {
    match response_format {
        Some(ChatCompletionResponseFormat::JsonObject) => {
            if !contains_json_instruction(messages) {
                return Err(ConversionError::bad_request(
                    "Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'.",
                ));
            }
            Ok(ResponseFormat::JsonObject)
        }
        Some(ChatCompletionResponseFormat::Regex { .. }) => Err(ConversionError::bad_request(
            "response_format: only text and json_object are supported",
        )),
        Some(ChatCompletionResponseFormat::Text | ChatCompletionResponseFormat::JsonSchema)
        | None => Ok(ResponseFormat::Text),
    }
}

/// Validate stop sequences and streaming options.
fn validate_api_params(
    stop: Option<&ChatCompletionStopSequences>,
    stream: Option<bool>,
    stream_options: Option<&ChatCompletionStreamOptions>,
) -> Result<(), ConversionError> {
    if let Some(ChatCompletionStopSequences::Array(stop_sequences)) = stop
        && stop_sequences.len() > MAX_STOP_SEQUENCES
    {
        return Err(ConversionError::bad_request(format!(
            "ChatCompletionStopSequences string array too long: {}",
            stop_sequences.len()
        )));
    }
    if stream != Some(true) && stream_options.is_some() {
        return Err(ConversionError::bad_request(
            "stream_options should be set along with stream = true",
        ));
    }
    Ok(())
}
