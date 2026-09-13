//! Shared request data and the conversion contract for protocol adapters.

use deepseek_recipe_core::conversation::Conversation;

use crate::response::ProtocolResponse;
use crate::stream::state_machine::ParsingOptions;

pub use options::{ConversionOptions, WebSearchBehavior};

mod options;

/// Inference parameters extracted from a protocol request.
///
/// Unspecified parameters remain `None`; model-specific defaults belong to the
/// caller.
#[derive(Debug, Default)]
pub struct InferenceOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Passed through for applications that support a thinking budget.
    pub thinking_budget_tokens: Option<u64>,
    /// Passed through for the caller to control parallel tool execution.
    pub disable_parallel_tool_use: Option<bool>,
}

/// A conversation with inference parameters and output parsing options.
///
/// Protocol adapters produce this representation. The caller renders the
/// conversation, tokenizes the prompt, and constructs the inference request.
/// Protocol-specific metadata is available on the protocol request before
/// conversion.
#[derive(Debug)]
pub struct ConversationRequest {
    pub conversation: Conversation,
    pub inference_options: InferenceOptions,
    pub parsing_options: ParsingOptions,
    /// Original model name, when exposed by the adapter. The caller resolves it.
    pub model: Option<String>,
    pub stream: bool,
    /// Number of parallel completions the caller should generate for this
    /// conversation.
    ///
    /// Always `1` for protocols that don't expose a multi-completion parameter.
    /// Chat Completions surfaces this from its `n` request field. The caller
    /// runs inference once per completion and tags each with a distinct
    /// `choice_index` via [`ProtocolRequest::chunk_generator_for_choice`].
    pub completion_count: u32,
}

impl ConversationRequest {
    /// Construct a request with default inference, parsing, and transport options.
    pub fn new(conversation: Conversation) -> Self {
        Self {
            conversation,
            inference_options: InferenceOptions::default(),
            parsing_options: ParsingOptions::default(),
            model: None,
            stream: false,
            completion_count: 1,
        }
    }
}

/// Convert protocol input into a shared conversation request.
pub trait ProtocolRequest: 'static + Send + Sync {
    type Response: ProtocolResponse;

    /// Validate and convert the protocol's supported request fields.
    ///
    /// Extract any protocol-specific metadata needed by the application first.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::BadRequest`] for invalid or unsupported input, or
    /// [`ConversionError::Internal`] if an internal conversion step fails.
    fn convert(self, options: ConversionOptions) -> Result<ConversationRequest, ConversionError>;

    /// Construct protocol events using the converted request's resolved settings.
    ///
    /// Equivalent to [`Self::chunk_generator_for_choice`] with `choice_index`
    /// `0`. Protocols that never produce more than one completion can keep
    /// implementing only this method.
    fn chunk_generator(
        request: &ConversationRequest,
        id: String,
        model: String,
    ) -> <Self::Response as ProtocolResponse>::ChunkGenerator;

    /// Construct protocol events for one completion of a `completion_count > 1`
    /// request.
    ///
    /// `choice_index` identifies which of the caller's parallel completions
    /// this generator's output belongs to; the generator must tag every event
    /// it produces with that index. The caller runs one inference per
    /// completion, drives each through its own [`crate::stream::StreamProcessor`]
    /// built from the generator returned here, and merges (streaming) or
    /// accumulates (complete response) the results.
    ///
    /// The default implementation ignores `choice_index` and delegates to
    /// [`Self::chunk_generator`], which is correct for every protocol except
    /// Chat Completions today.
    fn chunk_generator_for_choice(
        request: &ConversationRequest,
        id: String,
        model: String,
        choice_index: u32,
    ) -> <Self::Response as ProtocolResponse>::ChunkGenerator {
        let _ = choice_index;
        Self::chunk_generator(request, id, model)
    }
}

/// A failure while converting protocol input into a conversation request.
#[derive(Debug)]
pub enum ConversionError {
    /// The input is invalid or uses an unsupported feature.
    BadRequest { detail: String },
    /// An internal conversion operation failed.
    Internal { detail: String },
}

impl ConversionError {
    pub fn bad_request(detail: impl Into<String>) -> Self {
        ConversionError::BadRequest {
            detail: detail.into(),
        }
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        ConversionError::Internal {
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest { detail } | Self::Internal { detail } => f.write_str(detail),
        }
    }
}

impl std::error::Error for ConversionError {}
