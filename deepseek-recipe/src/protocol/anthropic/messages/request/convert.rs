use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use deepseek_recipe_core::conversation::{Conversation, ResponseFormat};
use deepseek_recipe_core::json::stringify_spaced;
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::prompt;
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use itertools::Itertools;

use crate::protocol::client_tools::{
    ClientToolChoice, ClientToolDeclaration, convert_client_tools, normalize_message_tools,
};
use crate::protocol::image::{
    UNSUPPORTED_IMAGE_PLACEHOLDER, base64_image_source, external_image_source, file_id_unsupported,
    image_not_allowed,
};
use crate::protocol::output_constraint::resolve_output_constraint;
use crate::protocol::reasoning::resolve_thinking_mode;
use crate::protocol::reasoning_content::normalize_reasoning_content;
use crate::protocol::validation::{
    RawInferenceOptions, validate_stop_sequences, validate_tool_name,
};
use crate::request::{
    ConversationRequest, ConversionError, ConversionOptions, ModelVersion, ProtocolRequest,
    WebSearchBehavior,
};
use crate::stream::{ParsingOptions, ThinkingInitialStage};

use super::super::response::{MessagesChunkGenerator, MessagesResponse};
use super::schema::{
    MessagesContent, MessagesContentBlock, MessagesDocumentSource, MessagesImageSource,
    MessagesMessage, MessagesRequest, MessagesSystemBlock, MessagesSystemContent, MessagesTool,
    MessagesToolChoice, MessagesToolReference, MessagesToolResultBlock, MessagesToolResultContent,
};

/// The tools of a request after name validation.
struct MessagesToolSet {
    /// Client tools in declaration order.
    client_tools: Vec<ClientToolDeclaration>,
    /// Names of the dropped `web_search` server tools.
    dropped_server_tool_names: HashSet<String>,
    /// Whether the request declares at least one tool, including server tools.
    has_tools: bool,
}

impl ProtocolRequest for MessagesRequest {
    type Response = MessagesResponse;

    /// Validate and convert Messages input into a conversation request.
    ///
    /// An explicit `thinking` setting overrides `output_config.effort`, which
    /// overrides [`ConversionOptions::default_thinking_mode`]. Outside thinking
    /// mode, conversion clears the reasoning content of every assistant
    /// message. In thinking mode, the `thinking` blocks of an assistant
    /// message, joined with blank lines, become its reasoning content, and
    /// conversion handles missing reasoning content per
    /// [`ConversionOptions::reasoning_content_behavior`].
    /// Checks run in the order model name, tool names and deferred loading,
    /// messages, reasoning content, tool schemas and tool choice, stop sequences,
    /// and sampling parameters.
    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError> {
        let model_version = options.resolve_model_version(&self.model)?;
        let (thinking_mode, reasoning_effort) = resolve_thinking_mode(
            options.default_thinking_mode,
            self.thinking.as_ref().map(|thinking| thinking.r#type),
            self.output_config.as_ref().and_then(|config| config.effort),
        );
        let tool_set = partition_tools(self.tools, &options, model_version)?;
        let image_token = options.special_tokens.token(SpecialToken::Image);
        let mut messages = transform_messages(
            self.messages,
            self.system,
            image_token,
            options.messages_web_search,
            model_version,
            &tool_set.client_tools,
        )?;
        normalize_reasoning_content(
            &mut messages,
            thinking_mode,
            tool_set.has_tools,
            options.reasoning_content_behavior,
        )
        .map_err(|()| {
            ConversionError::bad_request(
                "The `content[].thinking` in the thinking mode must be passed back to the API.",
            )
        })?;
        let (tools, tool_choice) = convert_tools(tool_set, self.tool_choice, thinking_mode)?;
        normalize_message_tools(&mut messages, &tools);
        let output_constraint =
            resolve_output_constraint(&tools, tool_choice, &ResponseFormat::Text, false)?;
        let stop_sequences = self.stop_sequences.unwrap_or_default();
        validate_stop_sequences(&stop_sequences)?;
        let mut inference_options = RawInferenceOptions {
            thinking_mode,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            clamp_max_tokens: true,
            ..RawInferenceOptions::default()
        }
        .into_inference_options(options.resolve_max_tokens_upper_bound(thinking_mode))?;
        let parse_tool_calls = tool_choice != ToolChoice::None && !tools.is_empty();
        inference_options.top_p =
            Some(model_version.resolve_top_p(self.top_p, thinking_mode, parse_tool_calls));
        inference_options.output_constraint = output_constraint;
        let parsing_options = Self::parsing_options(
            thinking_mode,
            parse_tool_calls,
            tool_choice,
            stop_sequences,
            Arc::clone(&options.special_tokens),
        );

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
            model_version,
            stream: self.stream.unwrap_or(false),
            response_options: Default::default(),
        })
    }

    fn chunk_generator(
        request: &ConversationRequest,
        response_id: String,
        model: String,
    ) -> MessagesChunkGenerator {
        MessagesChunkGenerator::new(response_id, model, request.conversation.thinking_mode)
    }
}

impl MessagesRequest {
    fn parsing_options(
        thinking: bool,
        parse_tool_calls: bool,
        tool_choice: ToolChoice,
        stop_sequences: Vec<String>,
        special_tokens: Arc<dyn SpecialTokens>,
    ) -> ParsingOptions {
        ParsingOptions {
            parse_tool_calls,
            thinking_initial_stage: thinking.then_some(ThinkingInitialStage::ReasoningStart),
            stop_sequences,
            starts_in_tool_calls: tool_choice == ToolChoice::Required,
            special_tokens,
            ..ParsingOptions::default()
        }
    }
}

/// Validate client tool names and partition the declarations: client tools
/// are kept in order, `web_search` server tools are dropped or rejected per
/// [`ConversionOptions::messages_web_search`], and other server tools are
/// rejected.
fn partition_tools(
    tools: Option<Vec<MessagesTool>>,
    options: &ConversionOptions,
    model_version: ModelVersion,
) -> Result<MessagesToolSet, ConversionError> {
    let tools = tools.unwrap_or_default();
    let mut tool_set = MessagesToolSet {
        client_tools: Vec::new(),
        dropped_server_tool_names: HashSet::new(),
        has_tools: !tools.is_empty(),
    };
    for (index, tool) in tools.into_iter().enumerate() {
        match tool {
            MessagesTool::Server(tool) => {
                if tool.r#type.starts_with("web_search")
                    && options.messages_web_search == WebSearchBehavior::Drop
                {
                    tool_set
                        .dropped_server_tool_names
                        .insert(tool.name.unwrap_or_else(|| "web_search".to_owned()));
                    continue;
                }
                return Err(ConversionError::bad_request(format!(
                    "tools[{index}]: server tools are not supported"
                )));
            }
            MessagesTool::Client {
                name,
                description,
                input_schema,
                defer_loading,
            } => {
                validate_tool_name(&name, &format!("tools[{index}]"))?;
                tool_set.client_tools.push(ClientToolDeclaration {
                    name,
                    description,
                    parameters: Some(input_schema),
                    strict: None,
                    defer_loading: model_version.supports_mid_conversation_system()
                        && defer_loading == Some(true),
                });
            }
        }
    }
    if !tool_set.client_tools.is_empty()
        && tool_set.client_tools.iter().all(|tool| tool.defer_loading)
    {
        return Err(ConversionError::bad_request(
            "tools: at least one tool must have defer_loading=false. All tools cannot be deferred.",
        ));
    }
    Ok(tool_set)
}

/// Validate tool schemas and resolve the tool choice.
///
/// A `none` choice or a request without tools skips the schema checks and
/// returns no tool definitions. A choice that names a dropped `web_search`
/// server tool becomes `auto`; [`convert_client_tools`] resolves the other
/// choices.
fn convert_tools(
    tool_set: MessagesToolSet,
    choice: Option<MessagesToolChoice>,
    thinking_mode: bool,
) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
    if matches!(choice, Some(MessagesToolChoice::None)) {
        return Ok((Vec::new(), ToolChoice::None));
    }
    if !tool_set.has_tools {
        return Ok((Vec::new(), ToolChoice::Auto));
    }
    let choice = match choice {
        Some(MessagesToolChoice::Tool { name, .. }) => {
            if tool_set.dropped_server_tool_names.contains(&name) {
                ClientToolChoice::Auto
            } else {
                ClientToolChoice::Named(name)
            }
        }
        _ => ClientToolChoice::Auto,
    };
    convert_client_tools(tool_set.client_tools, choice, thinking_mode)
}

/// The content of one block: plain text and special tokens, plus the image
/// source of an image block.
struct MessagesBlockContent {
    content: Prompt,
    image_source: Option<ImageSource>,
}

/// Convert one image block to the `image_token` placeholder. A model without
/// image input receives the unsupported-image text and the source is not
/// validated. `path` prefixes the validation errors of the block, and
/// `image_path` is the location that the image source records: the block
/// itself for a user message, the message for a `system` role message, and
/// the `tool_result` block for a tool result image.
fn image_block(
    source: MessagesImageSource,
    path: &str,
    image_path: &str,
    image_token: &str,
    model_version: ModelVersion,
) -> Result<MessagesBlockContent, ConversionError> {
    if !model_version.supports_image() {
        return Ok(MessagesBlockContent {
            content: Prompt::plain(UNSUPPORTED_IMAGE_PLACEHOLDER),
            image_source: None,
        });
    }
    let image_source = match source {
        MessagesImageSource::Base64 { media_type, data } => {
            base64_image_source(media_type.into(), data, path, image_path)?
        }
        MessagesImageSource::Url { url } => external_image_source(url, path, image_path)?,
        MessagesImageSource::File { file_id: _ } => return Err(file_id_unsupported(path)),
    };
    Ok(MessagesBlockContent {
        content: Prompt::special(image_token),
        image_source: Some(image_source),
    })
}

fn document_block(source: MessagesDocumentSource) -> MessagesBlockContent {
    plain_block(format!(
        "[Unsupported document type: {}]",
        source.media_type.as_deref().unwrap_or("unknown")
    ))
}

fn plain_block(text: impl Into<String>) -> MessagesBlockContent {
    MessagesBlockContent {
        content: Prompt::plain(text),
        image_source: None,
    }
}

impl MessagesContentBlock {
    /// Convert a text, image, or document block. `path` prefixes the errors,
    /// and `image_path` is the location that an image source records; see
    /// [`image_block`].
    fn into_block_content(
        self,
        path: &str,
        image_path: &str,
        image_token: &str,
        model_version: ModelVersion,
    ) -> Result<MessagesBlockContent, ConversionError> {
        match self {
            MessagesContentBlock::Text { text } => Ok(plain_block(text)),
            MessagesContentBlock::Image { source } => {
                image_block(source, path, image_path, image_token, model_version)
            }
            MessagesContentBlock::Document { source } => Ok(document_block(source)),
            _ => Err(ConversionError::internal("unexpected content block type")),
        }
    }
}

impl MessagesToolResultContent {
    /// Join the result blocks with blank lines, collecting image sources. The
    /// image sources record `path`, the location of the `tool_result` block.
    fn into_content(
        self,
        path: &str,
        image_token: &str,
        model_version: ModelVersion,
    ) -> Result<(Prompt, Vec<ImageSource>), ConversionError> {
        match self {
            MessagesToolResultContent::Raw(content) => Ok((Prompt::plain(content), Vec::new())),
            MessagesToolResultContent::Blocks(blocks) => {
                let mut contents = Vec::new();
                let mut image_sources = Vec::new();
                for (block_index, block) in blocks.into_iter().enumerate() {
                    let block_path = format!("{path}.content[{block_index}]");
                    let block = match block {
                        MessagesToolResultBlock::Text { text } => plain_block(text),
                        MessagesToolResultBlock::Image { source } => {
                            image_block(source, &block_path, path, image_token, model_version)?
                        }
                        MessagesToolResultBlock::Document { source } => document_block(source),
                        MessagesToolResultBlock::ToolReference { tool_name } => {
                            plain_block(tool_name)
                        }
                    };
                    contents.push(block.content);
                    image_sources.extend(block.image_source);
                }
                Ok((Prompt::join(&contents, "\n\n"), image_sources))
            }
        }
    }
}

/// Convert the top-level `system` field. A first text block that starts with
/// `x-anthropic-billing-header`, compared case-insensitively, is dropped.
fn convert_system_field(system: MessagesSystemContent) -> Prompt {
    match system {
        MessagesSystemContent::Raw(content) => Prompt::plain(content),
        MessagesSystemContent::Blocks(blocks) => {
            let contents: Vec<Prompt> = blocks
                .into_iter()
                .enumerate()
                .filter_map(|(index, MessagesSystemBlock::Text { text })| {
                    let is_billing_header = index == 0
                        && text
                            .to_lowercase()
                            .starts_with("x-anthropic-billing-header");
                    (!is_billing_header).then(|| Prompt::plain(text))
                })
                .collect();
            Prompt::join(&contents, "\n\n")
        }
    }
}

/// Require every earlier `tool_use` block to have its `tool_result`.
fn ensure_no_pending_tool_uses(
    message_index: usize,
    pending_tool_use_ids: &BTreeMap<String, usize>,
) -> Result<(), ConversionError> {
    if pending_tool_use_ids.is_empty() {
        return Ok(());
    }
    Err(ConversionError::bad_request(format!(
        "messages.{message_index}: `tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` block \
         must have a corresponding `tool_result` block in the next message.",
        pending_tool_use_ids.keys().join(", ")
    )))
}

/// Convert the `system` field and the messages into conversation messages.
///
/// The `system` field becomes a system message. A `user` message becomes one
/// user message per text, image, or document block and one tool message per
/// `tool_result` block. A `system` role message becomes a system message in
/// V4.1. In V4, it becomes a user message wrapped in `<system-reminder>`
/// text, or nothing when its content is empty. An `assistant` message joins
/// its text blocks with blank lines, joins its `thinking` blocks into
/// reasoning content, converts its `tool_use` blocks into tool calls, and
/// drops its `server_tool_use` and `web_search_tool_result` blocks under
/// [`WebSearchBehavior::Drop`].
///
/// # Errors
///
/// Returns an error when `messages` is empty; when a `user` or `assistant`
/// message has an empty block list; when a `tool_use`, `server_tool_use`,
/// `thinking`, or `web_search_tool_result` block appears outside an
/// `assistant` message, or a `tool_result` block outside a `user` message;
/// when a `tool_use` ID repeats within one message, a `tool_result` answers no
/// `tool_use` of the preceding message or repeats an answered ID, a text,
/// image, or document block follows a `tool_use` block in an `assistant`
/// message, or a `tool_use` is not answered by the `tool_result` blocks at
/// the start of the next message; when, for a model with image input, an
/// image source is invalid or an `assistant` message holds an image block;
/// or when an `assistant` message holds a `server_tool_use` or
/// `web_search_tool_result` block under [`WebSearchBehavior::Reject`].
fn transform_messages(
    messages: Vec<MessagesMessage>,
    system: Option<MessagesSystemContent>,
    image_token: &str,
    web_search: WebSearchBehavior,
    model_version: ModelVersion,
    tools: &[ClientToolDeclaration],
) -> Result<Vec<InputMessage>, ConversionError> {
    if messages.is_empty() {
        return Err(ConversionError::bad_request(
            "messages: at least one message is required",
        ));
    }

    let mut transformed_messages = Vec::new();
    if let Some(system) = system {
        transformed_messages.push(InputMessage::System {
            content: convert_system_field(system),
            image_sources: Vec::new(),
            tools: None,
        });
    }

    let mut pending_tool_use_ids = BTreeMap::new();
    let message_count = messages.len();
    let tool_by_name: HashMap<_, _> = tools
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();

    for (message_index, message) in messages.into_iter().enumerate() {
        match message {
            MessagesMessage::User(MessagesContent::Raw(content)) => {
                ensure_no_pending_tool_uses(message_index, &pending_tool_use_ids)?;
                transformed_messages.push(InputMessage::User {
                    content: Prompt::plain(content),
                    image_sources: Vec::new(),
                });
            }
            MessagesMessage::Assistant(MessagesContent::Raw(content)) => {
                ensure_no_pending_tool_uses(message_index, &pending_tool_use_ids)?;
                transformed_messages.push(InputMessage::Assistant {
                    content,
                    reasoning_content: None,
                    tool_calls: None,
                    prefix: false,
                });
            }
            MessagesMessage::System(content) => {
                ensure_no_pending_tool_uses(message_index, &pending_tool_use_ids)?;
                let mut added_tools = Vec::new();
                let (content, image_sources) = match content {
                    MessagesContent::Raw(content) => (Prompt::plain(content), Vec::new()),
                    MessagesContent::Blocks(blocks) => convert_system_message_blocks(
                        blocks,
                        message_index,
                        image_token,
                        model_version,
                        &tool_by_name,
                        &mut added_tools,
                    )?,
                };
                if model_version.supports_mid_conversation_system() {
                    transformed_messages.push(InputMessage::System {
                        content,
                        image_sources,
                        tools: (!added_tools.is_empty()).then_some(added_tools),
                    });
                } else if !content.is_empty() {
                    transformed_messages.push(InputMessage::User {
                        content: prompt!("<system-reminder>\n{content}\n</system-reminder>"),
                        image_sources,
                    });
                }
            }
            MessagesMessage::User(MessagesContent::Blocks(blocks)) => {
                if blocks.is_empty() {
                    return Err(ConversionError::bad_request(format!(
                        "messages.{message_index}: all messages must have non-empty content"
                    )));
                }

                let mut matched_tool_use_ids: HashSet<String> = HashSet::new();
                let mut referenced_tools = Vec::new();
                for (block_index, block) in blocks.into_iter().enumerate() {
                    let path = format!("messages.{message_index}.content[{block_index}]");
                    match block {
                        MessagesContentBlock::Text { .. }
                        | MessagesContentBlock::Image { .. }
                        | MessagesContentBlock::Document { .. } => {
                            ensure_no_pending_tool_uses(message_index, &pending_tool_use_ids)?;
                            let block = block.into_block_content(
                                &path,
                                &path,
                                image_token,
                                model_version,
                            )?;
                            transformed_messages.push(InputMessage::User {
                                content: block.content,
                                image_sources: block.image_source.into_iter().collect(),
                            });
                        }
                        MessagesContentBlock::ToolUse { .. }
                        | MessagesContentBlock::ServerToolUse { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_index}: `tool_use` blocks can only be in `assistant` messages"
                            )));
                        }
                        MessagesContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            if matched_tool_use_ids.contains(tool_use_id.as_str()) {
                                return Err(ConversionError::bad_request(format!(
                                    "messages.{message_index}.content.{block_index}: each tool_use must have a single result. Found \
                                         multiple `tool_result` blocks with id: {tool_use_id}"
                                )));
                            } else if !pending_tool_use_ids.contains_key(tool_use_id.as_str()) {
                                return Err(ConversionError::bad_request(format!(
                                    "unexpected `messages.{message_index}.content.{block_index}: \
                                     tool_use_id` found in `tool_result` blocks: {tool_use_id}. \
                                     Each `tool_result` block must have a corresponding \
                                     `tool_use` block in the previous message."
                                )));
                            } else {
                                pending_tool_use_ids.remove(&tool_use_id);
                                matched_tool_use_ids.insert(tool_use_id.clone());
                            }
                            if let Some(MessagesToolResultContent::Blocks(blocks)) = &content {
                                for block in blocks {
                                    if let MessagesToolResultBlock::ToolReference { tool_name } =
                                        block
                                    {
                                        let tool = tool_by_name.get(tool_name.as_str()).ok_or_else(|| {
                                            ConversionError::bad_request(format!(
                                                "messages.{message_index}.content.{block_index}: tool reference \
                                                 '{tool_name}' not found in available tools"
                                            ))
                                        })?;
                                        referenced_tools.push(ToolDefinition::from(*tool));
                                    }
                                }
                            }
                            let (content, image_sources) = match content {
                                Some(content) => content.into_content(
                                    &format!(
                                        "messages.{message_index}.content.{block_index}.tool_result"
                                    ),
                                    image_token,
                                    model_version,
                                )?,
                                None => (Prompt::new(), Vec::new()),
                            };
                            transformed_messages.push(InputMessage::Tool {
                                content,
                                image_sources,
                                tool_call_id: Some(tool_use_id),
                            });
                        }
                        MessagesContentBlock::ToolAddition { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_index}: `tool_addition` blocks can only be in `system` messages"
                            )));
                        }
                        MessagesContentBlock::Thinking { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_index}.content: thinking blocks may only be in `assistant` messages"
                            )));
                        }
                        MessagesContentBlock::WebSearchToolResult { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_index}: `web_search_tool_result` blocks can only be in `assistant` messages"
                            )));
                        }
                    }
                }

                if !pending_tool_use_ids.is_empty() {
                    return Err(ConversionError::bad_request(format!(
                        "messages.{}:`tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` \
                             block must have a corresponding `tool_result` block in the next message.",
                        message_index - 1,
                        pending_tool_use_ids.keys().join(", ")
                    )));
                }
                if model_version.supports_mid_conversation_system() && !referenced_tools.is_empty()
                {
                    transformed_messages.push(InputMessage::System {
                        content: Prompt::new(),
                        image_sources: Vec::new(),
                        tools: Some(referenced_tools),
                    });
                }
            }
            MessagesMessage::Assistant(MessagesContent::Blocks(blocks)) => {
                ensure_no_pending_tool_uses(message_index, &pending_tool_use_ids)?;
                if blocks.is_empty() {
                    return Err(ConversionError::bad_request(format!(
                        "messages.{message_index}: all messages must have non-empty content"
                    )));
                }

                let mut content = String::new();
                let mut reasoning_content: Option<String> = None;
                let mut tool_calls: Option<Vec<ToolCall>> = None;

                for (block_index, block) in blocks.into_iter().enumerate() {
                    let path = format!("messages.{message_index}.content[{block_index}]");
                    match block {
                        MessagesContentBlock::Text { .. }
                        | MessagesContentBlock::Image { .. }
                        | MessagesContentBlock::Document { .. } => {
                            if tool_calls.is_some() {
                                return Err(ConversionError::bad_request(format!(
                                    "messages.{message_index}.{block_index}: `tool_use` ids were \
                                     found without `tool_result` blocks immediately after: {}. \
                                     Each `tool_use` block must have a corresponding \
                                     `tool_result` block in the next message.",
                                    pending_tool_use_ids.keys().join(", ")
                                )));
                            }
                            if !content.is_empty() {
                                content.push_str("\n\n");
                            }
                            let block = block.into_block_content(
                                &path,
                                &path,
                                image_token,
                                model_version,
                            )?;
                            if block.image_source.is_some() {
                                return Err(image_not_allowed("assistant"));
                            }
                            content.push_str(block.content.as_str());
                        }
                        MessagesContentBlock::ToolUse { id, name, input } => {
                            let tool_use_id = id.clone();
                            let tool_call = ToolCall {
                                id,
                                name,
                                arguments: stringify_spaced(&input),
                            };
                            tool_calls.get_or_insert_with(Vec::new).push(tool_call);

                            if pending_tool_use_ids
                                .insert(tool_use_id, block_index)
                                .is_some()
                            {
                                return Err(ConversionError::bad_request(format!(
                                    "messages.{message_index}.content.{block_index}: `tool_use` ids must be unique"
                                )));
                            }
                        }
                        MessagesContentBlock::ToolResult { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_index}: `tool_result` blocks can only be in `user` messages"
                            )));
                        }
                        MessagesContentBlock::ToolAddition { .. } => {
                            return Err(ConversionError::bad_request(format!(
                                "messages.{message_index}: `tool_addition` blocks can only be in `system` messages"
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
                            if web_search == WebSearchBehavior::Reject {
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
                    prefix: false,
                });
            }
        }
    }

    if !pending_tool_use_ids.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "messages.{}:`tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` block must \
                 have a corresponding `tool_result` block in the next message.",
            message_count - 1,
            pending_tool_use_ids.keys().join(", ")
        )));
    }

    Ok(transformed_messages)
}

/// Convert the blocks of a `system` role message, joined with blank lines.
/// Tool additions are collected separately from text, images, and documents.
/// Image sources record the message location, `messages.{message_index}`.
fn convert_system_message_blocks(
    blocks: Vec<MessagesContentBlock>,
    message_index: usize,
    image_token: &str,
    model_version: ModelVersion,
    tool_by_name: &HashMap<&str, &ClientToolDeclaration>,
    added_tools: &mut Vec<ToolDefinition>,
) -> Result<(Prompt, Vec<ImageSource>), ConversionError> {
    let message_path = format!("messages.{message_index}");
    let mut contents = Vec::new();
    let mut image_sources = Vec::new();
    for (block_index, block) in blocks.into_iter().enumerate() {
        match block {
            MessagesContentBlock::Text { .. }
            | MessagesContentBlock::Image { .. }
            | MessagesContentBlock::Document { .. } => {
                let path = format!("{message_path}.content[{block_index}]");
                let block =
                    block.into_block_content(&path, &message_path, image_token, model_version)?;
                contents.push(block.content);
                image_sources.extend(block.image_source);
            }
            MessagesContentBlock::ToolAddition { tool } => {
                let MessagesToolReference::ToolReference { name } = tool;
                let tool = tool_by_name.get(name.as_str()).ok_or_else(|| {
                    ConversionError::bad_request(format!(
                        "messages.{message_index}.content.{block_index}: `tool_addition` references unknown \
                         tool '{name}'"
                    ))
                })?;
                added_tools.push(ToolDefinition::from(*tool));
            }
            MessagesContentBlock::Thinking { .. } => {
                return Err(ConversionError::bad_request(format!(
                    "messages.{message_index}.content: thinking blocks may only be in `assistant` messages"
                )));
            }
            MessagesContentBlock::ToolUse { .. } | MessagesContentBlock::ServerToolUse { .. } => {
                return Err(ConversionError::bad_request(format!(
                    "messages.{message_index}: `tool_use` blocks can only be in `assistant` messages"
                )));
            }
            MessagesContentBlock::ToolResult { .. } => {
                return Err(ConversionError::bad_request(format!(
                    "messages.{message_index}: `tool_result` blocks can only be in `user` messages"
                )));
            }
            MessagesContentBlock::WebSearchToolResult { .. } => {
                return Err(ConversionError::bad_request(format!(
                    "messages.{message_index}: `web_search_tool_result` blocks can only be in `assistant` messages"
                )));
            }
        }
    }
    Ok((Prompt::join(&contents, "\n\n"), image_sources))
}
