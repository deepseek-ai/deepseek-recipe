//! Decode token ids into text for the stream processor.

use std::sync::Arc;

use tokenizers::Tokenizer;

use super::StreamError;

/// Replacement character emitted while a multi-token character is incomplete.
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

/// Decode token ids into text.
///
/// Implemented for [`Tokenizer`]; other backends can implement it to feed a
/// stream processor.
pub trait TokenizerDecoder: Send {
    /// Decode `ids` into text, skipping special tokens when requested.
    fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String>;
}

impl TokenizerDecoder for Tokenizer {
    fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.decode(ids, skip_special_tokens)
            .map_err(|error| error.to_string())
    }
}

impl<T> TokenizerDecoder for Arc<T>
where
    T: TokenizerDecoder + Sync,
{
    fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        (**self).decode_ids(ids, skip_special_tokens)
    }
}

/// Buffer token ids until they decode into text.
pub struct StreamDecoder {
    decoder: Box<dyn TokenizerDecoder>,
    ids: Vec<u32>,
    pending: usize,
}

impl StreamDecoder {
    pub fn new(decoder: Box<dyn TokenizerDecoder>) -> Self {
        Self {
            decoder,
            ids: Vec::new(),
            pending: 0,
        }
    }

    /// Return the text contributed by the buffered token ids and the number of
    /// ids it accounts for, or `None` while no text is available yet.
    ///
    /// Special tokens drive the parser's state machine, so they stay in the
    /// decoded text. A decoder failure is returned as a stream error.
    pub fn decode(&mut self, token_id: u32) -> Result<Option<(String, usize)>, StreamError> {
        self.pending += 1;
        self.ids.push(token_id);
        let content = self
            .decoder
            .decode_ids(&self.ids, false)
            .map_err(|detail| StreamError::Decode { detail })?;
        if content.is_empty() || content.ends_with(REPLACEMENT_CHARACTER) {
            return Ok(None);
        }
        self.ids.clear();
        Ok(Some((content, std::mem::take(&mut self.pending))))
    }
}
