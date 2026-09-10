//! Parse model output and turn it into protocol-specific response events.

pub use decoder::TokenizerDecoder;
pub use inference::{
    CompletionUsage, FinishReason, InferenceChunk, InferenceFinishReason, PromptUsage,
};
pub use processor::StreamProcessor;

mod decoder;
mod inference;
mod processor;
pub mod state_machine;

/// An error that ends stream processing before normal completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// A token-id chunk arrived without an attached tokenizer.
    MissingTokenizer,
    /// The tokenizer failed to decode buffered token ids.
    Decode {
        /// The decoder's failure message.
        detail: String,
    },
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTokenizer => {
                formatter.write_str("token chunk received without a tokenizer")
            }
            Self::Decode { detail } => write!(formatter, "failed to decode token ids: {detail}"),
        }
    }
}

impl std::error::Error for StreamError {}

/// A parsed piece of model output, before protocol-specific event conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputChunk {
    Start {
        system_fingerprint: Option<String>,
        usage: PromptUsage,
    },
    Raw {
        content: String,
    },
    Reasoning {
        content: String,
    },
    ToolCallBegin,
    ToolCall {
        tool_name: String,
        arguments: String,
    },
    ToolArgumentsDelta {
        content: String,
    },
    Finish {
        reason: FinishReason,
        stop_sequence: Option<String>,
        usage: CompletionUsage,
    },
}

/// Convert parsed output into a protocol's response events.
pub trait ChunkGenerator: Send + 'static {
    /// Event type emitted by this protocol.
    type Chunk;

    /// Produce zero or more events for one parsed output chunk.
    fn generate(&mut self, chunk: OutputChunk) -> impl Future<Output = Vec<Self::Chunk>> + Send;
}
