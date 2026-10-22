use crate::request::ConversionError;

use super::schema::{MessagesContent, MessagesContentBlock, MessagesMessage};

/// Anthropic combines consecutive user or assistant messages into one turn.
/// Validate tool-call boundaries only after those turns have been assembled.
pub(super) fn coalesce_messages(
    messages: Vec<MessagesMessage>,
) -> Result<Vec<(usize, MessagesMessage)>, ConversionError> {
    let mut turns = Vec::with_capacity(messages.len());
    for (index, message) in messages.into_iter().enumerate() {
        if matches!(
            &message,
            MessagesMessage::User(MessagesContent::Blocks(blocks))
                | MessagesMessage::Assistant(MessagesContent::Blocks(blocks))
                if blocks.is_empty()
        ) {
            return Err(ConversionError::bad_request(format!(
                "messages.{index}: all messages must have non-empty content"
            )));
        }
        match (turns.last_mut(), message) {
            (Some((_, MessagesMessage::User(left))), MessagesMessage::User(right))
            | (Some((_, MessagesMessage::Assistant(left))), MessagesMessage::Assistant(right)) => {
                append_content(left, right);
            }
            (_, message) => turns.push((index, message)),
        }
    }
    Ok(turns)
}

fn append_content(left: &mut MessagesContent, right: MessagesContent) {
    let previous = std::mem::replace(left, MessagesContent::Blocks(Vec::new()));
    let mut blocks = into_blocks(previous);
    blocks.extend(into_blocks(right));
    *left = MessagesContent::Blocks(blocks);
}

fn into_blocks(content: MessagesContent) -> Vec<MessagesContentBlock> {
    match content {
        MessagesContent::Raw(text) => vec![MessagesContentBlock::Text { text }],
        MessagesContent::Blocks(blocks) => blocks,
    }
}
