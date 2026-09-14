//! Encode rendered prompts into token ids.

use std::sync::Arc;

use tokenizers::{AddedVocabulary, Tokenizer};

/// Encode text into model token ids.
///
/// Implemented for [`Tokenizer`]; other tokenizers can implement it to back a
/// prompt encoding.
pub trait TokenizerEncoder: Send + Sync {
    /// Encode `text` into token ids without added special tokens.
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String>;

    /// Encode `text` as plain BPE: added-token literals in the text are not
    /// recognized. Used for user-supplied content so that control-token
    /// literals (e.g. `<｜User｜>`) cannot be smuggled into the prompt as
    /// single control tokens.
    fn encode_content_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_ids(text)
    }
}

impl TokenizerEncoder for Tokenizer {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }

    fn encode_content_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut plain = self.clone();
        plain.with_added_vocabulary(AddedVocabulary::new());
        plain.encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }
}

impl<T: TokenizerEncoder> TokenizerEncoder for Arc<T> {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        (**self).encode_ids(text)
    }

    fn encode_content_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        (**self).encode_content_ids(text)
    }
}
