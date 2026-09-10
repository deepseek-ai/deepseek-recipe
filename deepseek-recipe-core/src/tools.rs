//! Client tool definitions used when constructing a conversation.

/// A function the caller may execute after receiving a tool call.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Function name.
    pub name: String,
    /// Human-readable purpose of the function.
    pub description: Option<String>,
    /// JSON Schema describing the input object.
    pub parameters: serde_json::Value,
    /// Requested strict argument validation. Enforcement belongs to the caller.
    pub strict: Option<bool>,
}

/// Normalized tool selection for a rendered conversation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model may answer or call a tool.
    #[default]
    Auto,
    /// Render the prompt with tool definitions omitted.
    None,
    /// Start the assistant output inside a tool-call block.
    Required,
}
