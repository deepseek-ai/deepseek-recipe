use serde::{Deserialize, Serialize};

use crate::messages::InputMessage;
use crate::tools::{ToolChoice, ToolDefinition};

/// A conversation and its prompt configuration.
///
/// Sampling parameters and transport options belong to the caller. Thinking is
/// enabled by default.
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Messages in conversation order.
    pub messages: Vec<InputMessage>,
    /// Enable thinking mode for the next assistant turn.
    pub thinking_mode: bool,
    /// Available client tools, in declaration order.
    pub tools: Vec<ToolDefinition>,
    /// Tool selection after protocol-specific normalization.
    pub tool_choice: ToolChoice,
    /// Reasoning effort used when thinking mode is enabled.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Requested answer format used during prompt rendering.
    pub response_format: ResponseFormat,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            thinking_mode: true,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            reasoning_effort: None,
            response_format: ResponseFormat::Text,
        }
    }
}

/// Output format described in the inference prompt.
#[derive(Debug, Clone, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
}

/// Protocol-independent reasoning effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    High,
    Xhigh,
    Max,
}
