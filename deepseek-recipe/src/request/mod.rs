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
    fn chunk_generator(
        request: &ConversationRequest,
        id: String,
        model: String,
    ) -> <Self::Response as ProtocolResponse>::ChunkGenerator;
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
