/// How an adapter handles supported `web_search` declarations and tool choices.
/// Messages also applies this setting to assistant server-tool content blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WebSearchBehavior {
    /// Drop matching declarations and choices, plus Messages server-tool blocks.
    #[default]
    Ignore,
    /// Reject matching declarations and choices, plus Messages server-tool blocks.
    Reject,
}

/// Defaults for conversion from protocol requests.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ConversionOptions {
    /// Default thinking mode for requests with an unspecified thinking mode.
    ///
    /// Defaults to `true`.
    pub default_thinking_mode: bool,
    /// Behavior for Responses `web_search` tool declarations and tool choices.
    /// Historical `web_search_call` input items are ignored under either setting.
    ///
    /// Defaults to [`WebSearchBehavior::Ignore`].
    pub responses_web_search: WebSearchBehavior,
    /// Behavior for the Messages `web_search` server tool, its named tool
    /// choice, and its `server_tool_use` / `web_search_tool_result` blocks.
    ///
    /// Defaults to [`WebSearchBehavior::Reject`].
    pub messages_web_search: WebSearchBehavior,
}

impl ConversionOptions {
    /// Construct options that enable thinking for requests that do not
    /// specify a thinking mode.
    pub const fn new() -> Self {
        Self {
            default_thinking_mode: true,
            responses_web_search: WebSearchBehavior::Ignore,
            messages_web_search: WebSearchBehavior::Reject,
        }
    }

    /// Set the default thinking mode, consuming and returning `self`.
    #[must_use]
    pub fn with_default_thinking_mode(mut self, default_thinking_mode: bool) -> Self {
        self.default_thinking_mode = default_thinking_mode;
        self
    }

    /// Set the Responses `web_search` behavior, consuming and returning `self`.
    #[must_use]
    pub fn with_responses_web_search(mut self, responses_web_search: WebSearchBehavior) -> Self {
        self.responses_web_search = responses_web_search;
        self
    }

    /// Set the Messages `web_search` behavior, consuming and returning `self`.
    #[must_use]
    pub fn with_messages_web_search(mut self, messages_web_search: WebSearchBehavior) -> Self {
        self.messages_web_search = messages_web_search;
        self
    }
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self::new()
    }
}
