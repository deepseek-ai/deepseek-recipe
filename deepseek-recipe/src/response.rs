//! Protocol response construction and event accumulation.

use serde::Serialize;

use crate::stream::ChunkGenerator;
use crate::util::append_delta::AppendDelta;

/// A protocol response that can accumulate events from its chunk generator.
pub trait ProtocolResponse:
    'static + Send + Serialize + AppendDelta<<Self::ChunkGenerator as ChunkGenerator>::Chunk>
{
    /// Generator for this protocol's response events.
    type ChunkGenerator: ChunkGenerator;

    /// Construct a response with caller-supplied identity and input token usage.
    fn new(
        id: String,
        model: String,
        created: u64,
        prompt_tokens: usize,
        prompt_cache_hit_tokens: usize,
    ) -> Self;

    /// Return the SSE event name, or `None` for an unnamed data event.
    fn chunk_event_type(
        _chunk: &<Self::ChunkGenerator as ChunkGenerator>::Chunk,
    ) -> Option<&'static str> {
        None
    }

    /// Return a final transport sentinel if the protocol requires one.
    fn done_message() -> Option<String> {
        None
    }
}
