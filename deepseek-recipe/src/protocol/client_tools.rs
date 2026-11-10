//! Client tool conversion shared by the protocol adapters.

use std::collections::{HashMap, HashSet};

use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::protocol::validation::{
    duplicate_tool_names, normalize_tool_parameters, unknown_named_tool,
};
use crate::request::ConversionError;

/// A client tool declaration after name validation, before schema validation.
pub(super) struct ClientToolDeclaration {
    pub name: String,
    pub description: Option<String>,
    /// The parameter schema as declared. A missing schema becomes an empty
    /// object in the tool definition.
    pub parameters: Option<serde_json::Value>,
    pub strict: Option<bool>,
    pub defer_loading: bool,
}

impl From<&ClientToolDeclaration> for ToolDefinition {
    fn from(tool: &ClientToolDeclaration) -> Self {
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({})),
            strict: tool.strict,
            defer_loading: tool.defer_loading,
        }
    }
}

/// A tool choice that [`convert_client_tools`] resolves against the declared
/// tools. The adapter handles `none` and a request without tools before
/// conversion.
pub(super) enum ClientToolChoice {
    /// The model can answer or call a tool.
    Auto,
    /// The model must call a tool.
    Required,
    /// The model must call the named tool.
    Named(String),
}

/// Validate the parameter schemas of client tools and resolve the tool choice.
///
/// Each schema is normalized through [`normalize_tool_parameters`], and a
/// missing schema becomes an empty object. A named choice keeps only the
/// named tool definition. A named or `required` choice becomes
/// [`ToolChoice::Required`]; `auto` becomes [`ToolChoice::Auto`].
///
/// # Errors
///
/// Returns an error for duplicate tool names or an invalid parameter schema
/// in declaration order, then for a named choice that names no declared tool,
/// then for a named or `required` choice in thinking mode.
pub(super) fn convert_client_tools(
    tools: Vec<ClientToolDeclaration>,
    choice: ClientToolChoice,
    thinking_mode: bool,
) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
    let mut names = HashSet::new();
    let mut definitions = Vec::new();
    for tool in tools {
        if !names.insert(tool.name.clone()) {
            return Err(duplicate_tool_names());
        }
        let parameters = normalize_tool_parameters(&tool.name, tool.parameters, tool.strict)?;
        definitions.push(ToolDefinition {
            name: tool.name,
            description: tool.description,
            parameters: parameters.unwrap_or_else(|| serde_json::json!({})),
            strict: tool.strict,
            defer_loading: tool.defer_loading,
        });
    }
    let required = match choice {
        ClientToolChoice::Named(name) => {
            if !names.contains(&name) {
                return Err(unknown_named_tool(&name));
            }
            definitions.retain(|tool| tool.name == name);
            true
        }
        ClientToolChoice::Required => true,
        ClientToolChoice::Auto => false,
    };
    if required && thinking_mode {
        return Err(ConversionError::bad_request(
            "Thinking mode does not support this tool_choice",
        ));
    }
    let choice = if required {
        ToolChoice::Required
    } else {
        ToolChoice::Auto
    };
    Ok((definitions, choice))
}

/// Retain only selected tools in historical declarations and use their
/// normalized definitions. Drop system messages with no content or tools.
pub(super) fn normalize_message_tools(messages: &mut Vec<InputMessage>, tools: &[ToolDefinition]) {
    let tools_by_name: HashMap<_, _> = tools
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();
    messages.retain_mut(|message| {
        if let InputMessage::System {
            content,
            tools: Some(tools),
            ..
        } = message
        {
            *tools = tools
                .iter()
                .filter_map(|tool| {
                    tools_by_name
                        .get(tool.name.as_str())
                        .map(|tool| (*tool).clone())
                })
                .collect();
            return !content.is_empty() || !tools.is_empty();
        }
        true
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaration(name: &str) -> ClientToolDeclaration {
        ClientToolDeclaration {
            name: name.to_owned(),
            description: None,
            parameters: None,
            strict: None,
            defer_loading: false,
        }
    }

    #[test]
    fn a_named_choice_keeps_the_named_tool() {
        let (definitions, choice) = convert_client_tools(
            vec![declaration("a"), declaration("b")],
            ClientToolChoice::Named("b".to_owned()),
            false,
        )
        .expect("the tools convert");
        assert_eq!(choice, ToolChoice::Required);
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "b");
        assert_eq!(definitions[0].parameters, serde_json::json!({}));
    }

    #[test]
    fn errors_follow_the_documented_order() {
        let error = |tools, choice, thinking_mode| {
            convert_client_tools(tools, choice, thinking_mode)
                .expect_err("the tools are rejected")
                .to_string()
        };
        assert_eq!(
            error(
                vec![declaration("a"), declaration("a")],
                ClientToolChoice::Named("x".to_owned()),
                true,
            ),
            "Tool names must be unique."
        );
        assert_eq!(
            error(
                vec![declaration("a")],
                ClientToolChoice::Named("x".to_owned()),
                true,
            ),
            "Invalid value for 'tool_call': no function named 'x' was specified in the 'tools' \
             parameter."
        );
        assert_eq!(
            error(vec![declaration("a")], ClientToolChoice::Required, true),
            "Thinking mode does not support this tool_choice"
        );
    }
}
