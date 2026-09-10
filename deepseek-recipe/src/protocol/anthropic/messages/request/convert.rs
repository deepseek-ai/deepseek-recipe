use std::collections::{BTreeMap, HashSet};

use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::{IMAGE_SPECIAL_TOKEN, ImageSource};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use deepseek_recipe_core::util::json_formatter::stringify_python_style;
use itertools::Itertools;

use crate::protocol::image::{
    UNSUPPORTED_DOCUMENT_PLACEHOLDER, base64_image_source, external_image_source,
    file_id_unsupported, image_not_allowed,
};
use crate::protocol::validation::{
    image_special_token_not_allowed, validate_max_tokens, validate_sampling, validate_tool_name,
    validate_tool_parameters, validate_tool_text,
};
use crate::request::{
    ConversationRequest, ConversionError, ConversionOptions, InferenceOptions, ProtocolRequest,
    WebSearchBehavior,
};
use crate::stream::state_machine::{ParsingOptions, ReasoningStage};

use super::super::response::{MessagesChunkGenerator, MessagesResponse};
use super::schema::{
    MessagesContent, MessagesContentBlock, MessagesImageSource, MessagesMessage,
    MessagesReasoningEffort, MessagesRequest, MessagesTextBlock, MessagesTextOrTextBlocks,
    MessagesThinkingType, MessagesTool, MessagesToolChoice,
};

impl ProtocolRequest for MessagesRequest {
    type Response = MessagesResponse;

    /// Validate and convert Messages input into a conversation request.
    ///
    /// Explicit thinking overrides effort, which overrides the supplied default.
    /// Missing historical thinking is accepted, and supplied thinking is preserved.
    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError> {
        let (thinking_mode, reasoning_effort) = self.resolve_thinking(options);
        let inference_options = self.inference_options()?;
        let stop_sequences = self.stop_sequences.unwrap_or_default();
        let (tools, tool_choice) =
            convert_tools(self.tools, self.tool_choice, thinking_mode, &options)?;
        let parsing_options =
            Self::parsing_options(thinking_mode, &tools, tool_choice, stop_sequences);
        let messages = transform_messages(self.messages, self.system, &options)?;
        validate_tool_text(&tools, &messages)?;

        Ok(ConversationRequest {
            conversation: Conversation {
                messages,
                thinking_mode,
                tools,
                tool_choice,
                reasoning_effort,
                ..Conversation::default()
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
    ) -> MessagesChunkGenerator {
        MessagesChunkGenerator::new(id, model, request.conversation.thinking_mode)
    }
}

impl MessagesRequest {
    fn resolve_thinking(&self, options: ConversionOptions) -> (bool, Option<ReasoningEffort>) {
        let effort = self
            .output_config
            .as_ref()
            .and_then(|config| config.effort)
            .map(|effort| match effort {
                MessagesReasoningEffort::Low => ReasoningEffort::Low,
                MessagesReasoningEffort::Medium | MessagesReasoningEffort::High => {
                    ReasoningEffort::High
                }
                MessagesReasoningEffort::Xhigh => ReasoningEffort::Xhigh,
                MessagesReasoningEffort::Ultra | MessagesReasoningEffort::Max => {
                    ReasoningEffort::Max
                }
            });
        let thinking = match self.thinking.as_ref().map(|thinking| thinking.kind) {
            Some(MessagesThinkingType::Enabled) => true,
            Some(MessagesThinkingType::Disabled) => false,
            None => effort.is_some() || options.default_thinking_mode,
        };
        (
            thinking,
            thinking.then_some(effort.unwrap_or(ReasoningEffort::High)),
        )
    }

    fn parsing_options(
        thinking: bool,
        tools: &[ToolDefinition],
        tool_choice: ToolChoice,
        stop_sequences: Vec<String>,
    ) -> ParsingOptions {
        ParsingOptions {
            parse_tool_calls: !tools.is_empty(),
            reasoning_initial_stage: thinking.then_some(ReasoningStage::Start),
            stop_sequences,
            tool_call_initial_stage: tool_choice == ToolChoice::Required,
            ..ParsingOptions::default()
        }
    }

    fn inference_options(&self) -> Result<InferenceOptions, ConversionError> {
        validate_max_tokens(self.max_tokens, "max_tokens")?;
        validate_sampling(self.temperature, self.top_p)?;
        if let Some(stops) = &self.stop_sequences
            && (stops.len() > 16 || stops.iter().any(String::is_empty))
        {
            return Err(ConversionError::bad_request(
                "stop_sequences must contain at most 16 non-empty strings",
            ));
        }
        let disable_parallel_tool_use = match &self.tool_choice {
            Some(
                MessagesToolChoice::Auto {
                    disable_parallel_tool_use,
                }
                | MessagesToolChoice::Any {
                    disable_parallel_tool_use,
                }
                | MessagesToolChoice::Tool {
                    disable_parallel_tool_use,
                    ..
                },
            ) => *disable_parallel_tool_use,
            _ => None,
        };
        Ok(InferenceOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            thinking_budget_tokens: self
                .thinking
                .as_ref()
                .and_then(|thinking| thinking.budget_tokens),
            disable_parallel_tool_use,
        })
    }
}

fn convert_tools(
    tools: Option<Vec<MessagesTool>>,
    choice: Option<MessagesToolChoice>,
    thinking: bool,
    options: &ConversionOptions,
) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
    let mut names = HashSet::new();
    let mut definitions = Vec::new();
    let mut ignored_server_tool_names = HashSet::new();
    for (index, tool) in tools.unwrap_or_default().into_iter().enumerate() {
        if let Some(kind) = &tool.kind {
            if kind.starts_with("web_search") {
                match options.messages_web_search {
                    WebSearchBehavior::Ignore => {
                        ignored_server_tool_names.insert(tool.name.clone());
                        continue;
                    }
                    WebSearchBehavior::Reject => {
                        return Err(ConversionError::bad_request(format!(
                            "tools.{index}: server tools are not supported"
                        )));
                    }
                }
            }
            return Err(ConversionError::bad_request(format!(
                "tools.{index}: server tools are not supported"
            )));
        }
        validate_tool_name(&tool.name, &format!("tools.{index}.name"))?;
        if !names.insert(tool.name.clone()) {
            return Err(ConversionError::bad_request("Tool names must be unique"));
        }
        let parameters = tool.input_schema.ok_or_else(|| {
            ConversionError::bad_request(format!("tools.{index}: missing input_schema"))
        })?;
        validate_tool_parameters(&parameters, &format!("tools.{index}.input_schema"))?;
        definitions.push(ToolDefinition {
            name: tool.name,
            description: tool.description,
            parameters,
            strict: None,
        });
    }
    let choice = match choice {
        Some(MessagesToolChoice::None) => {
            definitions.clear();
            ToolChoice::None
        }
        Some(MessagesToolChoice::Tool { name, .. }) => {
            if ignored_server_tool_names.contains(&name) {
                // The named server tool was dropped; drop the choice too.
                ToolChoice::Auto
            } else if !names.contains(&name) {
                return Err(ConversionError::bad_request(format!(
                    "tool_choice: no tool named '{name}' was specified"
                )));
            } else if thinking {
                return Err(ConversionError::bad_request(
                    "Thinking mode does not support this tool_choice",
                ));
            } else {
                definitions.retain(|tool| tool.name == name);
                ToolChoice::Required
            }
        }
        _ => ToolChoice::Auto,
    };
    Ok((definitions, choice))
}

impl MessagesContentBlock {
    /// Convert text and tool references, substitute document placeholders, and
    /// collect images with one [`IMAGE_SPECIAL_TOKEN`] placeholder per image.
    fn into_content(self, path: &str) -> Result<(String, Vec<ImageSource>), ConversionError> {
        match self {
            MessagesContentBlock::Text { text } => Ok((text, Vec::new())),
            MessagesContentBlock::ToolReference { tool_name } => Ok((tool_name, Vec::new())),
            MessagesContentBlock::Image { source } => Ok((
                IMAGE_SPECIAL_TOKEN.to_string(),
                vec![image_source(source, path)?],
            )),
            MessagesContentBlock::Document => {
                Ok((UNSUPPORTED_DOCUMENT_PLACEHOLDER.to_string(), Vec::new()))
            }
            _ => Err(ConversionError::internal("unexpected content block type")),
        }
    }

    /// Convert textual content and document placeholders, rejecting images.
    fn into_text(self, path: &str, role: &str) -> Result<String, ConversionError> {
        let (content, image_sources) = self.into_content(path)?;
        if !image_sources.is_empty() {
            return Err(image_not_allowed(role));
        }
        Ok(content)
    }
}

impl MessagesTextBlock {
    /// Convert text and tool references, substitute document placeholders, and
    /// collect images with one [`IMAGE_SPECIAL_TOKEN`] placeholder per image.
    fn into_content(self, path: &str) -> Result<(String, Vec<ImageSource>), ConversionError> {
        match self {
            MessagesTextBlock::Text { text } => Ok((text, Vec::new())),
            MessagesTextBlock::ToolReference { tool_name } => Ok((tool_name, Vec::new())),
            MessagesTextBlock::Image { source } => Ok((
                IMAGE_SPECIAL_TOKEN.to_string(),
                vec![image_source(source, path)?],
            )),
            MessagesTextBlock::Document => {
                Ok((UNSUPPORTED_DOCUMENT_PLACEHOLDER.to_string(), Vec::new()))
            }
            MessagesTextBlock::Unsupported => Err(unsupported_block()),
        }
    }
}

impl MessagesTextOrTextBlocks {
    /// Join textual content and document placeholders, collecting image sources.
    fn into_content(self, path: &str) -> Result<(String, Vec<ImageSource>), ConversionError> {
        match self {
            MessagesTextOrTextBlocks::Raw(content) => Ok((content, Vec::new())),
            MessagesTextOrTextBlocks::Blocks(blocks) => {
                let mut texts = Vec::new();
                let mut image_sources = Vec::new();
                for (block_index, block) in blocks.into_iter().enumerate() {
                    let (text, images) = block.into_content(&format!("{path}[{block_index}]"))?;
                    texts.push(text);
                    image_sources.extend(images);
                }
                Ok((texts.join("\n\n"), image_sources))
            }
        }
    }

    /// Join textual content and document placeholders, rejecting images.
    fn into_text(self, path: &str, role: &str) -> Result<String, ConversionError> {
        let (content, image_sources) = self.into_content(path)?;
        if !image_sources.is_empty() {
            return Err(image_not_allowed(role));
        }
        Ok(content)
    }
}

/// Convert one Messages image reference.
fn image_source(source: MessagesImageSource, path: &str) -> Result<ImageSource, ConversionError> {
    match source {
        MessagesImageSource::Base64 { media_type, data } => {
            base64_image_source(&media_type, data, path)
        }
        MessagesImageSource::Url { url } => external_image_source(url, None, path),
        MessagesImageSource::File { file_id: _ } => Err(file_id_unsupported(path)),
    }
}

fn check_tool_use_resolved(
    message_idx: usize,
    tool_use_ids: &BTreeMap<String, usize>,
) -> Result<(), ConversionError> {
    if tool_use_ids.is_empty() {
        return Ok(());
    }
    Err(ConversionError::bad_request(format!(
        "messages.{message_idx}: `tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` block \
         must have a corresponding `tool_result` block in the next message.",
        tool_use_ids.keys().join(", ")
    )))
}

/// Return whether any message text or top-level system text carries
/// [`IMAGE_SPECIAL_TOKEN`].
///
/// Message text, tool references, thinking blocks, and tool result text are
/// checked. Tool call names and arguments are checked after conversion by
/// [`validate_tool_text`].
fn messages_contain_image_special_token(
    messages: &[MessagesMessage],
    system: Option<&MessagesTextOrTextBlocks>,
) -> bool {
    messages.iter().any(|message| match message {
        MessagesMessage::User(content) | MessagesMessage::Assistant(content) => {
            content_contains_image_special_token(content)
        }
        MessagesMessage::System(content) => {
            text_or_text_blocks_contain_image_special_token(content)
        }
    }) || system.is_some_and(text_or_text_blocks_contain_image_special_token)
}

/// Return whether one message content carries [`IMAGE_SPECIAL_TOKEN`].
fn content_contains_image_special_token(content: &MessagesContent) -> bool {
    match content {
        MessagesContent::Raw(content) => content.contains(IMAGE_SPECIAL_TOKEN),
        MessagesContent::Blocks(blocks) => blocks.iter().any(block_contains_image_special_token),
    }
}

/// Return whether one message content block carries [`IMAGE_SPECIAL_TOKEN`].
fn block_contains_image_special_token(block: &MessagesContentBlock) -> bool {
    match block {
        MessagesContentBlock::Text { text } => text.contains(IMAGE_SPECIAL_TOKEN),
        MessagesContentBlock::ToolReference { tool_name } => {
            tool_name.contains(IMAGE_SPECIAL_TOKEN)
        }
        MessagesContentBlock::Thinking { thinking, .. } => thinking.contains(IMAGE_SPECIAL_TOKEN),
        MessagesContentBlock::ToolResult { content, .. } => content
            .as_ref()
            .is_some_and(text_or_text_blocks_contain_image_special_token),
        _ => false,
    }
}

/// Return whether one text or text-block value carries [`IMAGE_SPECIAL_TOKEN`].
fn text_or_text_blocks_contain_image_special_token(content: &MessagesTextOrTextBlocks) -> bool {
    match content {
        MessagesTextOrTextBlocks::Raw(content) => content.contains(IMAGE_SPECIAL_TOKEN),
        MessagesTextOrTextBlocks::Blocks(blocks) => blocks.iter().any(|block| match block {
            MessagesTextBlock::Text { text } => text.contains(IMAGE_SPECIAL_TOKEN),
            MessagesTextBlock::ToolReference { tool_name } => {
                tool_name.contains(IMAGE_SPECIAL_TOKEN)
            }
            _ => false,
        }),
    }
}

fn transform_messages(
    messages: Vec<MessagesMessage>,
    system: Option<MessagesTextOrTextBlocks>,
    options: &ConversionOptions,
) -> Result<Vec<InputMessage>, ConversionError> {
    if messages.is_empty() {
        return Err(ConversionError::bad_request(
            "messages: at least one message is required",
        ));
    }
    if messages_contain_image_special_token(&messages, system.as_ref()) {
        return Err(image_special_token_not_allowed());
    }

    let mut transformed_messages = Vec::new();
    if let Some(system) = system {
        transformed_messages.push(InputMessage::System {
            content: system.into_text("system", "system")?,
        });
    }

    let mut tool_use_ids = BTreeMap::new();
    let messages_len = messages.len();

    for (message_idx, message) in messages.into_iter().enumerate() {
        match message {
            MessagesMessage::User(MessagesContent::Raw(content)) => {
                check_tool_use_resolved(message_idx, &tool_use_ids)?;
                transformed_messages.push(InputMessage::User {
                    content,
                    image_sources: Vec::new(),
                });
            }
            MessagesMessage::Assistant(MessagesContent::Raw(content)) => {
                check_tool_use_resolved(message_idx, &tool_use_ids)?;
                transformed_messages.push(InputMessage::Assistant {
                    content,
                    reasoning_content: None,
                    tool_calls: None,
                });
            }
            MessagesMessage::System(content) => {
                check_tool_use_resolved(message_idx, &tool_use_ids)?;
                let (content, image_sources) =
                    content.into_content(&format!("messages.{message_idx}"))?;
                transformed_messages.push(InputMessage::User {
                    content: format!("<system-reminder>\n{content}\n</system-reminder>"),
                    image_sources,
                });
            }
            MessagesMessage::User(MessagesContent::Blocks(blocks)) => {
                if blocks.is_empty() {
                    return Err(ConversionError::bad_request(format!(
                        "messages.{message_idx}: all messages must have non-empty content"
                    )));
                }

                let mut matched_tool_use_ids: HashSet<String> = HashSet::new();
                for (block_idx, block) in blocks.into_iter().enumerate() {
                    let path = format!("messages.{message_idx}.content[{block_idx}]");
                    match block {
                        MessagesContentBlock::Unsupported => return Err(unsupported_block()),
                        MessagesContentBlock::Text { .. }
                        | MessagesContentBlock::ToolReference { .. } => {
                            check_tool_use_resolved(message_idx, &tool_use_ids)?;
                            let content = block.into_text(&path, "user")?;
                            transformed_messages.push(InputMessage::User {
                                content,
                                image_sources: Vec::new(),
                            });
                        }
                        MessagesContentBlock::Image { source } => {
                            check_tool_use_resolved(message_idx, &tool_use_ids)?;
                            transformed_messages.push(InputMessage::User {
                                content: IMAGE_SPECIAL_TOKEN.to_string(),
                                image_sources: vec![image_source(source, &path)?],
                            });
                        }
                        MessagesContentBlock::Document => {
                            check_tool_use_resolved(message_idx, &tool_use_ids)?;
                            transformed_messages.push(InputMessage::User {
                                content: UNSUPPORTED_DOCUMENT_PLACEHOLDER.to_string(),
                                image_sources: Vec::new(),
                            });
                        }
                        MessagesContentBlock::ToolUse { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_idx}: `tool_use` blocks can only be in `assistant` messages"
                            )));
                        }
                        MessagesContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            if matched_tool_use_ids.contains(tool_use_id.as_str()) {
                                return Err(ConversionError::bad_request(format!(
                                    "messages.{message_idx}.content.{block_idx}: each tool_use must have a single result. Found \
                                         multiple `tool_result` blocks with id: {tool_use_id}"
                                )));
                            } else if !tool_use_ids.contains_key(tool_use_id.as_str()) {
                                return Err(ConversionError::bad_request(format!(
                                    "unexpected `messages.{message_idx}.content.{block_idx}: tool_use_id` found in `tool_result` \
                                         blocks: {tool_use_id}. Each `tool_result` block must have a corresponding `tool_use` block in \
                                         the previous message."
                                )));
                            } else {
                                tool_use_ids.remove(&tool_use_id);
                                matched_tool_use_ids.insert(tool_use_id.clone());
                            }
                            let (content, image_sources) = match content {
                                Some(content) => content.into_content(&path)?,
                                None => (String::new(), Vec::new()),
                            };
                            transformed_messages.push(InputMessage::Tool {
                                content,
                                image_sources,
                                tool_call_id: tool_use_id,
                            });
                        }
                        MessagesContentBlock::Thinking { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_idx}.content: thinking blocks may only be in `assistant` messages"
                            )));
                        }
                        MessagesContentBlock::ServerToolUse { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_idx}: `server_tool_use` blocks can only be in `assistant` messages"
                            )));
                        }
                        MessagesContentBlock::WebSearchToolResult { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_idx}: `web_search_tool_result` blocks can only be in `assistant` messages"
                            )));
                        }
                    }
                }

                if !tool_use_ids.is_empty() {
                    return Err(ConversionError::bad_request(format!(
                        "messages.{}:`tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` \
                             block must have a corresponding `tool_result` block in the next message.",
                        message_idx - 1,
                        tool_use_ids.keys().join(", ")
                    )));
                }
            }
            MessagesMessage::Assistant(MessagesContent::Blocks(blocks)) => {
                check_tool_use_resolved(message_idx, &tool_use_ids)?;
                if blocks.is_empty() {
                    return Err(ConversionError::bad_request(format!(
                        "messages.{message_idx}: all messages must have non-empty content"
                    )));
                }

                let mut content = String::new();
                let mut reasoning_content: Option<String> = None;
                let mut tool_calls: Option<Vec<ToolCall>> = None;

                for (block_idx, block) in blocks.into_iter().enumerate() {
                    let path = format!("messages.{message_idx}.content[{block_idx}]");
                    match block {
                        MessagesContentBlock::Unsupported => return Err(unsupported_block()),
                        MessagesContentBlock::Text { .. }
                        | MessagesContentBlock::ToolReference { .. }
                        | MessagesContentBlock::Image { .. }
                        | MessagesContentBlock::Document => {
                            if tool_calls.is_some() {
                                return Err(ConversionError::bad_request(format!(
                                    "messages.{message_idx}.{block_idx}: `tool_use` ids were found without `tool_result` blocks \
                                         immediately after: {}. Each `tool_use` block must have a corresponding `tool_result` block in \
                                         the next message.",
                                    tool_use_ids.keys().join(", ")
                                )));
                            }
                            if !content.is_empty() {
                                content.push_str("\n\n");
                            }
                            content.push_str(&block.into_text(&path, "assistant")?);
                        }
                        MessagesContentBlock::ToolUse { id, name, input } => {
                            let tool_use_id = id.clone();
                            let tool_call = ToolCall {
                                id,
                                name,
                                arguments: stringify_python_style(&input),
                            };
                            tool_calls.get_or_insert_with(Vec::new).push(tool_call);

                            if tool_use_ids.insert(tool_use_id, block_idx).is_some() {
                                return Err(ConversionError::bad_request(format!(
                                    "messages.{message_idx}.content.{block_idx}: `tool_use` ids must be unique"
                                )));
                            }
                        }
                        MessagesContentBlock::ToolResult { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_idx}: `tool_result` blocks can only be in `user` messages"
                            )));
                        }
                        MessagesContentBlock::Thinking { thinking, .. } => {
                            if let Some(reasoning_content) = &mut reasoning_content {
                                reasoning_content.push_str("\n\n");
                                reasoning_content.push_str(&thinking);
                            } else {
                                reasoning_content = Some(thinking);
                            }
                        }
                        MessagesContentBlock::ServerToolUse { .. }
                        | MessagesContentBlock::WebSearchToolResult { .. } => {
                            if options.messages_web_search == WebSearchBehavior::Reject {
                                return Err(ConversionError::bad_request(
                                    "Unsupported content block: server tools are not supported",
                                ));
                            }
                        }
                    }
                }
                transformed_messages.push(InputMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls,
                });
            }
        }
    }

    if !tool_use_ids.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "messages.{}:`tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` block must \
                 have a corresponding `tool_result` block in the next message.",
            messages_len - 1,
            tool_use_ids.keys().join(", ")
        )));
    }

    Ok(transformed_messages)
}

fn unsupported_block() -> ConversionError {
    ConversionError::bad_request("Unsupported content block")
}
