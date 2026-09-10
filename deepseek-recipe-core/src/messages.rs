use crate::multimodal::ImageSource;

/// A message in the conversation supplied for prompt rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMessage {
    System {
        content: String,
    },
    User {
        content: String,
        image_sources: Vec<ImageSource>,
    },
    Assistant {
        content: String,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        content: String,
        image_sources: Vec<ImageSource>,
        tool_call_id: String,
    },
    LatestReminder {
        content: String,
    },
}

/// A historical tool call and its arguments serialized as JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
