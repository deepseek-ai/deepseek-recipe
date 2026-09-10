use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN;
use deepseek_recipe_core::tools::ToolDefinition;
use deepseek_recipe_core::util::json_formatter::stringify_python_style;

use crate::request::ConversionError;

/// Return the error for input text carrying [`IMAGE_SPECIAL_TOKEN`].
///
/// An adapter inserts one placeholder per image source. A placeholder in input
/// text would reach the prompt without an image source, so the adapters reject
/// input text that spells it out.
pub(super) fn image_special_token_not_allowed() -> ConversionError {
    ConversionError::bad_request(format!(
        "The sub-string \"{IMAGE_SPECIAL_TOKEN}\" in your input is not allowed for this API. Please remove \
         \"{IMAGE_SPECIAL_TOKEN}\" from your input and try again."
    ))
}

/// Return the error for converted tool text carrying [`IMAGE_SPECIAL_TOKEN`].
///
/// A rendered tool prompt contains the name, description, and parameter schema
/// of every definition, and the tool calls of a historical assistant turn
/// contribute their names and arguments. Declared tool names are already
/// restricted by [`validate_tool_name`]; historical call names and arguments are
/// not.
pub(super) fn validate_tool_text(
    tools: &[ToolDefinition],
    messages: &[InputMessage],
) -> Result<(), ConversionError> {
    if tools
        .iter()
        .any(tool_definition_contains_image_special_token)
        || messages.iter().any(tool_calls_contain_image_special_token)
    {
        return Err(image_special_token_not_allowed());
    }
    Ok(())
}

/// Return whether a tool definition carries [`IMAGE_SPECIAL_TOKEN`].
fn tool_definition_contains_image_special_token(tool: &ToolDefinition) -> bool {
    tool.name.contains(IMAGE_SPECIAL_TOKEN)
        || tool
            .description
            .as_deref()
            .is_some_and(|description| description.contains(IMAGE_SPECIAL_TOKEN))
        || stringify_python_style(&tool.parameters).contains(IMAGE_SPECIAL_TOKEN)
}

/// Return whether the tool calls of one message carry [`IMAGE_SPECIAL_TOKEN`].
fn tool_calls_contain_image_special_token(message: &InputMessage) -> bool {
    let InputMessage::Assistant { tool_calls, .. } = message else {
        return false;
    };
    tool_calls.as_ref().is_some_and(|tool_calls| {
        tool_calls.iter().any(|tool_call| {
            tool_call.name.contains(IMAGE_SPECIAL_TOKEN)
                || tool_call.arguments.contains(IMAGE_SPECIAL_TOKEN)
        })
    })
}

pub(super) fn validate_max_tokens(value: Option<u32>, field: &str) -> Result<(), ConversionError> {
    if value == Some(0) {
        return Err(ConversionError::bad_request(format!(
            "{field} must be greater than zero"
        )));
    }
    Ok(())
}

pub(super) fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), ConversionError> {
    if temperature.is_some_and(|value| !(0.0..=2.0).contains(&value)) {
        return Err(ConversionError::bad_request(
            "temperature must be in [0, 2]",
        ));
    }
    if top_p.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0) {
        return Err(ConversionError::bad_request("top_p must be in (0, 1]"));
    }
    Ok(())
}

pub(super) fn validate_tool_name(name: &str, path: &str) -> Result<(), ConversionError> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err(ConversionError::bad_request(format!(
            "{path}: expected 1-128 ASCII letters, digits, underscores or hyphens"
        )));
    }
    Ok(())
}

pub(super) fn validate_tool_parameters(
    schema: &serde_json::Value,
    path: &str,
) -> Result<(), ConversionError> {
    if schema.get("type").and_then(serde_json::Value::as_str) != Some("object") {
        return Err(ConversionError::bad_request(format!(
            "{path} must be a JSON Schema of type object"
        )));
    }
    // Schema validation uses local data with HTTP and file resolution disabled.
    jsonschema::JSONSchema::compile(schema)
        .map(|_| ())
        .map_err(|error| ConversionError::bad_request(format!("{path}: {error}")))
}

pub(super) fn contains_json_instruction(messages: &[InputMessage]) -> bool {
    messages.iter().any(|message| {
        let content = match message {
            InputMessage::System { content }
            | InputMessage::User { content, .. }
            | InputMessage::Assistant { content, .. }
            | InputMessage::Tool { content, .. }
            | InputMessage::LatestReminder { content } => content,
        };
        content
            .as_bytes()
            .windows(4)
            .any(|part| part.eq_ignore_ascii_case(b"json"))
    })
}
