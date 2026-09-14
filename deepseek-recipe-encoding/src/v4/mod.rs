use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use deepseek_recipe_core::util::json_formatter::stringify_python_style;

use crate::EncodingError;
use crate::PromptEncoding;
use crate::RenderedPrompt;
use crate::TokenizerEncoder;

pub mod dsv4;
pub mod dsv41;

/// Marks the start of the prompt.
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
/// Starts the reasoning content of an assistant turn.
pub const THINKING_START_TOKEN: &str = "<think>";
/// Ends the reasoning content of an assistant turn.
pub const THINKING_END_TOKEN: &str = "</think>";

/// Starts a system message in a V4.1 prompt.
pub const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
/// Starts a user message.
pub const USER_SP_TOKEN: &str = "<｜User｜>";
/// Starts an assistant message.
pub const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
/// Starts the latest reminder message.
pub const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
/// Terminates a message.
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
/// Tag name prefix of the markup that structures tool calls. The angle brackets
/// come from the surrounding template, as in `<｜DSML｜tool_calls>` for V4 and
/// `<｜DSML｜ calls>` for V4.1.
pub const DSML_SP_TOKEN: &str = "｜DSML｜";

fn parameter_template(
    dsml_token: &str,
    tool_parameter_tag_name: &str,
    key: &str,
    is_str: &str,
    value: &str,
) -> String {
    format!(
        "<{dsml_token}{tool_parameter_tag_name} name=\"{key}\" string=\"{is_str}\">{value}</{dsml_token}{tool_parameter_tag_name}>"
    )
}

fn render_tool_arguments(tool_call: &ToolCall, tool_parameter_tag_name: &str) -> String {
    let arguments =
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&tool_call.arguments)
            .unwrap_or_else(|err| {
                tracing::warn!(?err, "invalid tool call arguments");
                serde_json::Map::from_iter([(
                    "arguments".to_owned(),
                    tool_call.arguments.clone().into(),
                )])
            });
    arguments
        .iter()
        .map(|(key, value)| {
            let (is_str, kv_str) = match value.as_str() {
                Some(s) => ("true", s.to_owned()),
                None => ("false", stringify_python_style(value)),
            };
            parameter_template(DSML_SP_TOKEN, tool_parameter_tag_name, key, is_str, &kv_str)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) trait EncodingV4 {
    fn tokenizer(&self) -> Option<&dyn TokenizerEncoder>;

    fn supports_mid_conversation_system(&self) -> bool;

    fn system_token(&self) -> &'static str;

    fn tool_calls_block_name(&self) -> &'static str;

    fn tool_call_tag_name(&self) -> &'static str;

    fn tool_parameter_tag_name(&self) -> &'static str;

    fn render_reasoning_effort(
        &self,
        index: usize,
        thinking_mode: bool,
        effort: Option<ReasoningEffort>,
    ) -> String;
}

fn tool_call_template(encoding: &impl EncodingV4, name: &str, arguments: &str) -> String {
    let tool_call_tag_name = encoding.tool_call_tag_name();
    format!(
        "<{DSML_SP_TOKEN}{tool_call_tag_name} name=\"{name}\">\n{arguments}\n</{DSML_SP_TOKEN}{tool_call_tag_name}>"
    )
}

fn tool_calls_template(encoding: &impl EncodingV4, tool_calls: &str) -> String {
    let tc_block_name = encoding.tool_calls_block_name();
    format!("<{DSML_SP_TOKEN}{tc_block_name}>\n{tool_calls}\n</{DSML_SP_TOKEN}{tc_block_name}>")
}

fn render_tool_calls(encoding: &impl EncodingV4, tool_calls: &[ToolCall]) -> String {
    tool_calls
        .iter()
        .map(|tool_call| {
            tool_call_template(
                encoding,
                &tool_call.name,
                &render_tool_arguments(tool_call, encoding.tool_parameter_tag_name()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A rendered prompt fragment: `true` marks template control tokens (encoded
/// with added-token recognition), `false` marks message content (encoded as
/// plain BPE so user-supplied control-token literals are not smuggled in as
/// single control tokens).
type PromptSegment = (bool, String);

fn render_message_segments(
    encoding: &impl EncodingV4,
    messages: &[InputMessage],
    index: usize,
    thinking_mode: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Vec<PromptSegment> {
    let msg = &messages[index];
    let prev = messages[..index].last();
    let reasoning_effort_prompt =
        encoding.render_reasoning_effort(index, thinking_mode, reasoning_effort);
    let mut segments: Vec<PromptSegment> = Vec::new();
    let mut push = |is_template: bool, text: &str| {
        if !text.is_empty() {
            segments.push((is_template, text.to_string()));
        }
    };
    if index == 0
        && (!reasoning_effort_prompt.is_empty() || matches!(msg, InputMessage::System { .. }))
    {
        push(true, encoding.system_token());
    }
    push(false, &reasoning_effort_prompt);
    match msg {
        InputMessage::System { content } => {
            if index > 0 && encoding.supports_mid_conversation_system() {
                push(true, encoding.system_token());
            }
            push(false, content);
        }
        InputMessage::User { content, .. } => {
            if matches!(
                prev,
                Some(InputMessage::User { .. } | InputMessage::Tool { .. })
            ) {
                push(true, "\n\n");
            } else {
                push(true, USER_SP_TOKEN);
            }
            push(false, content);
        }
        InputMessage::LatestReminder { content } => {
            push(true, LATEST_REMINDER_SP_TOKEN);
            push(false, content);
        }
        InputMessage::Tool { content, .. } => {
            if matches!(
                prev,
                Some(InputMessage::User { .. } | InputMessage::Tool { .. })
            ) {
                push(true, "\n\n");
            } else {
                push(true, USER_SP_TOKEN);
            }
            push(false, &format!("<tool_result>{content}</tool_result>"));
        }
        InputMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => {
            let tool_calls_content = match tool_calls {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    format!(
                        "\n\n{}",
                        tool_calls_template(encoding, &render_tool_calls(encoding, tool_calls))
                    )
                }
                _ => String::new(),
            };
            push(true, ASSISTANT_SP_TOKEN);
            if thinking_mode && index > 0 {
                push(true, THINKING_START_TOKEN);
                if let Some(reasoning_content) = reasoning_content {
                    push(false, reasoning_content);
                }
                push(true, THINKING_END_TOKEN);
            } else {
                push(true, THINKING_END_TOKEN);
            }
            push(false, content);
            push(true, &tool_calls_content);
            push(true, EOS_TOKEN);
        }
    }
    segments
}

impl<T: EncodingV4> PromptEncoding for T {
    fn encode(&self, conversation: &Conversation) -> Result<Vec<u32>, EncodingError> {
        let tokenizer = self.tokenizer().ok_or(EncodingError::MissingTokenizer)?;
        let (segments, _) = render_parts(self, conversation);
        let mut ids = Vec::new();
        for (is_template, text) in segments {
            let seg_ids = if is_template {
                tokenizer.encode_ids(&text)
            } else {
                tokenizer.encode_content_ids(&text)
            }
            .map_err(EncodingError::Encode)?;
            ids.extend(seg_ids);
        }
        Ok(ids)
    }

    fn render_conversation(&self, conversation: &Conversation) -> RenderedPrompt {
        let (segments, image_sources) = render_parts(self, conversation);
        let prompt = segments
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<String>();
        RenderedPrompt {
            prompt,
            image_sources,
        }
    }
}

/// Renders the conversation into `(is_template, text)` segments plus the
/// ordered image sources. Template segments are encoded with added-token
/// recognition; content segments are encoded as plain BPE.
fn render_parts(
    encoding: &impl EncodingV4,
    conversation: &Conversation,
) -> (Vec<PromptSegment>, Vec<ImageSource>) {
    let mut messages = normalize_messages(encoding, &conversation.messages);
    let has_tools =
        conversation.tool_choice != ToolChoice::None && !conversation.tools.is_empty();
    let format_schema = match &conversation.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(stringify_python_style(&serde_json::json!({
            "type": "json_object"
        }))),
    };
    if has_tools || format_schema.is_some() {
        if !matches!(messages.first(), Some(InputMessage::System { .. })) {
            messages.insert(
                0,
                InputMessage::System {
                    content: String::new(),
                },
            );
        }
        if let Some(InputMessage::System { content }) = messages.first_mut() {
            if has_tools {
                content.push_str("\n\n");
                content.push_str(&render_tool_prompt(encoding, &conversation.tools));
            }
            if let Some(schema) = format_schema {
                content.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                content.push_str(&schema);
            }
        }
    }
    let mut segments: Vec<PromptSegment> = vec![(true, BOS_TOKEN.to_string())];
    for index in 0..messages.len() {
        segments.extend(render_message_segments(
            encoding,
            &messages,
            index,
            conversation.thinking_mode,
            conversation.reasoning_effort,
        ));
    }
    segments.push((true, ASSISTANT_SP_TOKEN.to_string()));
    segments.push((
        true,
        if conversation.thinking_mode {
            THINKING_START_TOKEN
        } else {
            THINKING_END_TOKEN
        }
        .to_string(),
    ));
    if conversation.tool_choice == ToolChoice::Required && !conversation.tools.is_empty() {
        segments.push((
            true,
            format!(
                "\n\n<{DSML_SP_TOKEN}{}>\n",
                encoding.tool_calls_block_name()
            ),
        ));
    }
    let image_sources = messages
        .iter()
        .filter_map(|message| match message {
            InputMessage::User { image_sources, .. }
            | InputMessage::Tool { image_sources, .. } => Some(image_sources.as_slice()),
            _ => None,
        })
        .flatten()
        .cloned()
        .collect();
    (segments, image_sources)
}

fn render_tool_prompt(encoding: &impl EncodingV4, tools: &[ToolDefinition]) -> String {
    let tool_schemas = tools
        .iter()
        .map(|tool| {
            stringify_python_style(&serde_json::json!({
              "name": tool.name,
              "description": tool.description.as_deref().unwrap_or_default(),
              "parameters": tool.parameters,
            }))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let dsml_token = DSML_SP_TOKEN;
    let tc_block_name = encoding.tool_calls_block_name();
    let tool_call_tag_name = encoding.tool_call_tag_name();
    let tool_parameter_tag_name = encoding.tool_parameter_tag_name();
    let thinking_start_token = THINKING_START_TOKEN;
    let thinking_end_token = THINKING_END_TOKEN;
    format!(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#
    )
}

fn normalize_messages(encoding: &impl EncodingV4, messages: &[InputMessage]) -> Vec<InputMessage> {
    let mut normalized = Vec::new();
    let mut has_non_system = false;
    for message in messages.iter().cloned() {
        match message {
            InputMessage::System { content } if !encoding.supports_mid_conversation_system() => {
                if !has_non_system {
                    if let Some(InputMessage::System { content: head }) = normalized.last_mut() {
                        if !head.is_empty() && !content.is_empty() {
                            head.push_str("\n\n");
                        }
                        head.push_str(&content);
                    } else {
                        normalized.push(InputMessage::System { content });
                    }
                } else if !content.is_empty() {
                    normalized.push(InputMessage::User {
                        content,
                        image_sources: Vec::new(),
                    });
                }
            }
            InputMessage::User {
                content,
                image_sources,
            } => {
                has_non_system = true;
                if let Some(InputMessage::User {
                    content: previous,
                    image_sources: previous_image_sources,
                }) = normalized.last_mut()
                {
                    previous.push_str("\n\n");
                    previous.push_str(&content);
                    previous_image_sources.extend(image_sources);
                } else {
                    normalized.push(InputMessage::User {
                        content,
                        image_sources,
                    });
                }
            }
            message => {
                has_non_system |= !matches!(message, InputMessage::System { .. });
                normalized.push(message);
            }
        }
    }
    sort_tool_results_by_call_order(&mut normalized);
    normalized
}

fn sort_tool_results_by_call_order(messages: &mut [InputMessage]) {
    let mut order: Vec<String> = Vec::new();
    let mut idx = 0;
    while idx < messages.len() {
        match &messages[idx] {
            InputMessage::Assistant {
                tool_calls: Some(tool_calls),
                ..
            } if !tool_calls.is_empty() => {
                order = tool_calls.iter().map(|tc| tc.id.clone()).collect();
                idx += 1;
            }
            InputMessage::User { .. } | InputMessage::Tool { .. } => {
                let start = idx;
                while idx < messages.len()
                    && matches!(
                        messages[idx],
                        InputMessage::User { .. } | InputMessage::Tool { .. }
                    )
                {
                    idx += 1;
                }
                let tool_idxs: Vec<usize> = (start..idx)
                    .filter(|&i| matches!(messages[i], InputMessage::Tool { .. }))
                    .collect();
                if tool_idxs.len() > 1 && !order.is_empty() {
                    let mut tools: Vec<InputMessage> = tool_idxs
                        .iter()
                        .map(|&i| {
                            std::mem::replace(
                                &mut messages[i],
                                InputMessage::LatestReminder {
                                    content: String::new(),
                                },
                            )
                        })
                        .collect();
                    tools.sort_by_key(|m| match m {
                        InputMessage::Tool { tool_call_id, .. } => {
                            order.iter().position(|id| id == tool_call_id).unwrap_or(0)
                        }
                        _ => 0,
                    });
                    for (&i, tool) in tool_idxs.iter().zip(tools) {
                        messages[i] = tool;
                    }
                }
            }
            _ => idx += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use deepseek_recipe_core::conversation::Conversation;
    use deepseek_recipe_core::messages::InputMessage;
    use tokenizers::Tokenizer;

    use super::dsv41::DeepseekV41Encoding;
    use crate::PromptEncoding;

    fn v41_encoding() -> DeepseekV41Encoding {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../static/tokenizers/v41/tokenizer.json"
        );
        DeepseekV41Encoding::new()
            .with_tokenizer(Tokenizer::from_file(path).expect("bundled v41 tokenizer"))
    }

    fn conversation_with(content: &str) -> Conversation {
        let mut conversation = Conversation::default();
        conversation.thinking_mode = false;
        conversation.messages.push(InputMessage::User {
            content: content.to_string(),
            image_sources: Vec::new(),
        });
        conversation
    }

    #[test]
    fn empty_user_message_keeps_template_baseline() {
        let ids = v41_encoding().encode(&conversation_with("")).unwrap();
        assert_eq!(ids.len(), 4, "baseline template is 4 ids, got {ids:?}");
    }

    #[test]
    fn user_supplied_control_token_literals_encode_as_plain_bpe() {
        let encoding = v41_encoding();
        let empty = encoding.encode(&conversation_with("")).unwrap();
        assert_eq!(empty.len(), 4);

        // <｜User｜>: the official API reports 9 prompt tokens (4 template + 5 content).
        let user = encoding
            .encode(&conversation_with("<｜User｜>"))
            .unwrap();
        assert_eq!(user.len(), 9, "expected 4 template + 5 content ids, got {user:?}");
        assert_eq!(&user[..2], &empty[..2], "template prefix must be unchanged");
        assert_eq!(
            &user[2..7],
            &[30u32, 28217, 6756, 28217, 32],
            "user content must be plain BPE (issue #5)"
        );
        assert_eq!(&user[7..], &empty[2..], "template suffix must be unchanged");
        assert!(
            !user[2..7].contains(&128803),
            "user content must not map to the <｜User｜> control token"
        );

        // <think>: the official API reports 7 prompt tokens (4 template + 3 content).
        let think = encoding.encode(&conversation_with("<think>")).unwrap();
        assert_eq!(think.len(), 7, "expected 4 template + 3 content ids, got {think:?}");
        assert_eq!(&think[..2], &empty[..2]);
        assert_eq!(&think[5..], &empty[2..]);
    }

    #[test]
    fn plain_text_content_is_unchanged() {
        let encoding = v41_encoding();
        let empty = encoding.encode(&conversation_with("")).unwrap();
        let x = encoding.encode(&conversation_with("x")).unwrap();
        assert_eq!(x.len(), 5, "expected 4 template + 1 content id, got {x:?}");
        assert_eq!(&x[..2], &empty[..2]);
        assert_eq!(&x[3..], &empty[2..]);
    }
}
