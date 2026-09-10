/// Prompt-side token counts supplied by the inference backend. Unspecified
/// counts default to zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptUsage {
    /// Total input tokens, including tokens read from the prompt cache.
    pub prompt_tokens: usize,
    /// Input tokens read from the prompt cache.
    pub prompt_cache_hit_tokens: usize,
}

/// Completion-side token counts accumulated by the stream processor from
/// `Text.content_tokens` and token IDs that decode into text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompletionUsage {
    /// Tokens accounted for by processed chunks, including discarded markup
    /// and the entire chunk containing a local stop sequence. Undecoded trailing
    /// token IDs and chunks after the stop are excluded.
    pub completion_tokens: usize,
}

/// Text and metadata received from an inference backend.
#[derive(Debug)]
pub enum InferenceChunk {
    /// Initial metadata reported before the model produces output.
    Ready {
        system_fingerprint: Option<String>,
        prompt_usage: PromptUsage,
    },
    /// Text produced by the model, and the tokens that text accounts for.
    Text {
        content: String,
        content_tokens: usize,
    },
    /// One generated token id, decoded by the processor's tokenizer.
    Token { token_id: u32 },
    /// Terminal metadata reported after the model stops.
    Finish {
        finish_reason: InferenceFinishReason,
    },
}

/// The finish reason reported by the inference backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceFinishReason {
    Stop,
    Length,
    ContentFilter,
}

/// Final result after applying the stream parser's tool and stop-sequence
/// rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    StopSequence,
    ContentFilter,
    /// The source ended without an explicit finish reason.
    EndOfStream,
}
