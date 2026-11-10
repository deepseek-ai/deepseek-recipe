//! Normalization of historical `reasoning_content`.
//!
//! In thinking mode, rendering expects every assistant message to carry
//! reasoning content. [`normalize_reasoning_content`] fills missing reasoning
//! content with an empty string or, per [`ReasoningContentBehavior`], rejects
//! it. A tool call ID with the empty-thinking marker records that its turn
//! produced no reasoning content.

use deepseek_recipe_core::messages::InputMessage;

use crate::protocol::tool_call_id::is_empty_thinking_tool_call_id;
use crate::request::ReasoningContentBehavior;

/// Normalize the reasoning content of every assistant message in `messages`.
///
/// Outside thinking mode, the reasoning content of every assistant message
/// becomes `None`. In thinking mode, supplied reasoning content is kept and
/// missing reasoning content becomes an empty string. Under
/// [`ReasoningContentBehavior::Reject`], missing reasoning content is rejected
/// instead when all of the following hold:
///
/// - The message follows the last user message, or `messages` contains no
///   user message.
/// - The message has tool calls, or `has_tools` is `true`.
/// - No tool call ID of the message carries the empty-thinking marker.
///
/// # Errors
///
/// Returns `Err(())` at the first rejected message. The messages before it
/// have been normalized.
pub(crate) fn normalize_reasoning_content(
    messages: &mut [InputMessage],
    thinking_mode: bool,
    has_tools: bool,
    behavior: ReasoningContentBehavior,
) -> Result<(), ()> {
    let last_user_index = messages
        .iter()
        .rposition(|message| matches!(message, InputMessage::User { .. }));

    for (index, message) in messages.iter_mut().enumerate() {
        let InputMessage::Assistant {
            reasoning_content,
            tool_calls,
            ..
        } = message
        else {
            continue;
        };

        if !thinking_mode {
            *reasoning_content = None;
            continue;
        }

        if reasoning_content.is_some() {
            continue;
        }

        // Missing reasoning content of an assistant message before the last
        // user message is never rejected.
        let required = last_user_index.is_none_or(|last_user_index| index > last_user_index);

        let is_empty_thinking = tool_calls.as_ref().is_some_and(|tool_calls| {
            tool_calls
                .iter()
                .any(|tool_call| is_empty_thinking_tool_call_id(&tool_call.id))
        });

        // Reasoning content is required for a message after the last user
        // message that has tool calls or belongs to a request with tools,
        // unless one of its tool call IDs carries the empty-thinking marker.
        let must_be_passed_back =
            required && !is_empty_thinking && (tool_calls.is_some() || has_tools);
        if must_be_passed_back && behavior == ReasoningContentBehavior::Reject {
            return Err(());
        }
        *reasoning_content = Some(String::new());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use deepseek_recipe_core::messages::ToolCall;

    use super::*;

    fn tool_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "f".to_string(),
            arguments: "{}".to_string(),
        }
    }

    fn assistant(
        reasoning_content: Option<&str>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> InputMessage {
        InputMessage::Assistant {
            content: String::new(),
            reasoning_content: reasoning_content.map(str::to_string),
            tool_calls,
            prefix: false,
        }
    }

    fn user() -> InputMessage {
        InputMessage::User {
            content: deepseek_recipe_core::prompt::Prompt::plain("hi"),
            image_sources: Vec::new(),
        }
    }

    #[test]
    fn non_thinking_clears_reasoning_content() {
        let mut messages = vec![assistant(Some("kept"), None)];
        normalize_reasoning_content(
            &mut messages,
            false,
            false,
            ReasoningContentBehavior::FillEmpty,
        )
        .unwrap();
        let InputMessage::Assistant {
            reasoning_content, ..
        } = &messages[0]
        else {
            unreachable!()
        };
        assert_eq!(*reasoning_content, None);
    }

    #[test]
    fn ignore_fills_missing_final_round_reasoning() {
        let mut messages = vec![
            user(),
            assistant(
                None,
                Some(vec![tool_call("call_00_abcdefghijklmnopqrst0000")]),
            ),
        ];
        normalize_reasoning_content(
            &mut messages,
            true,
            true,
            ReasoningContentBehavior::FillEmpty,
        )
        .unwrap();
        let InputMessage::Assistant {
            reasoning_content, ..
        } = &messages[1]
        else {
            unreachable!()
        };
        assert_eq!(*reasoning_content, Some(String::new()));
    }

    #[test]
    fn reject_rejects_missing_final_round_reasoning() {
        let mut messages = vec![
            user(),
            assistant(
                None,
                Some(vec![tool_call("call_00_abcdefghijklmnopqrst0000")]),
            ),
        ];
        let result = normalize_reasoning_content(
            &mut messages,
            true,
            true,
            ReasoningContentBehavior::Reject,
        );
        assert!(result.is_err());
    }

    #[test]
    fn empty_thinking_marker_is_never_rejected() {
        let id = crate::protocol::tool_call_id::new_tool_call_id(0, true);
        let mut messages = vec![user(), assistant(None, Some(vec![tool_call(&id)]))];
        normalize_reasoning_content(&mut messages, true, true, ReasoningContentBehavior::Reject)
            .unwrap();
        let InputMessage::Assistant {
            reasoning_content, ..
        } = &messages[1]
        else {
            unreachable!()
        };
        assert_eq!(*reasoning_content, Some(String::new()));
    }

    #[test]
    fn existing_reasoning_is_preserved() {
        let mut messages = vec![assistant(Some("kept"), None)];
        normalize_reasoning_content(&mut messages, true, false, ReasoningContentBehavior::Reject)
            .unwrap();
        let InputMessage::Assistant {
            reasoning_content, ..
        } = &messages[0]
        else {
            unreachable!()
        };
        assert_eq!(reasoning_content.as_deref(), Some("kept"));
    }
}
