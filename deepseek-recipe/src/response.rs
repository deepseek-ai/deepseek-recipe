//! Protocol response construction and chunk accumulation.

use serde::Serialize;

use crate::stream::ChunkGenerator;

/// A protocol response that can accumulate the chunks of its chunk generator.
pub trait ProtocolResponse:
    'static + Send + Serialize + Accumulate<<Self::ChunkGenerator as ChunkGenerator>::Chunk>
{
    /// Generator for this protocol's response chunks.
    type ChunkGenerator: ChunkGenerator;

    /// Construct a response with caller-supplied identity and input token usage.
    /// `created` is a Unix timestamp in seconds.
    fn new(
        id: String,
        model: String,
        created: u64,
        prompt_tokens: usize,
        prompt_cache_hit_tokens: usize,
    ) -> Self;

    /// Return the SSE event name of `chunk`, or `None` for an unnamed data
    /// event.
    fn sse_event_name(
        _chunk: &<Self::ChunkGenerator as ChunkGenerator>::Chunk,
    ) -> Option<&'static str> {
        None
    }

    /// Return the data of the final SSE message, such as `[DONE]`, when the
    /// protocol ends its stream with one.
    fn sse_done_data() -> Option<String> {
        None
    }
}

/// Accumulate an incremental update into an existing value.
///
/// The caller must apply updates in arrival order.
pub trait Accumulate<D> {
    /// Apply one update.
    fn accumulate(&mut self, delta: D);
}

/// `None` leaves the value unchanged; `Some` accumulates into an existing value
/// or sets an absent one.
impl<T> Accumulate<Option<T>> for Option<T>
where
    T: Accumulate<T>,
{
    fn accumulate(&mut self, delta: Option<T>) {
        let Some(delta) = delta else {
            return;
        };
        if let Some(inner) = self {
            inner.accumulate(delta);
        } else {
            *self = Some(delta);
        }
    }
}

impl<T> Accumulate<Vec<T>> for Vec<T> {
    fn accumulate(&mut self, delta: Vec<T>) {
        self.extend(delta);
    }
}

impl Accumulate<String> for String {
    fn accumulate(&mut self, delta: String) {
        self.push_str(&delta);
    }
}

impl Accumulate<&str> for String {
    fn accumulate(&mut self, delta: &str) {
        self.push_str(delta);
    }
}
