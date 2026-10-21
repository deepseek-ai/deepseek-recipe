//! Encode rendered prompts into token IDs.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use tokenizers::{AddedVocabulary, Tokenizer};

/// An encoder that maps prompt text to model token IDs.
///
/// Implemented for [`PromptTokenizer`]; other tokenizers can implement it to
/// provide token encoding for a prompt encoding.
pub trait TokenizerEncoder: Send + Sync {
    /// Encode `text` into token IDs.
    ///
    /// `special_ranges` are the byte ranges of `text` that hold explicit
    /// special tokens. The caller must pass ranges in ascending order that do
    /// not overlap, lie within `text`, and start and end on `char` boundaries;
    /// [`RenderedPrompt::special_ranges`] meets these conditions.
    ///
    /// The implementation must encode each range as the token ID of its
    /// spelling and the plain text between the ranges without the added
    /// vocabulary, so a special-token spelling in plain text encodes as the
    /// token IDs of its characters. The implementation must add no tokens,
    /// such as a BOS token, beyond those that the text encodes to.
    ///
    /// # Errors
    ///
    /// Returns an error message when the implementation cannot encode `text`.
    ///
    /// [`RenderedPrompt::special_ranges`]: crate::RenderedPrompt::special_ranges
    fn encode_ids(&self, text: &str, special_ranges: &[Range<usize>]) -> Result<Vec<u32>, String>;
}

/// A tokenizer prepared for prompt encoding.
///
/// It holds a copy of the tokenizer without its added vocabulary, which
/// encodes plain text, and the added vocabulary, which supplies special token
/// IDs. Padding and truncation are disabled in the copy: plain regions are
/// fragments of one prompt, not independent sequences. The caller's tokenizer
/// is unchanged.
#[derive(Clone)]
pub struct PromptTokenizer {
    plain_tokenizer: Tokenizer,
    added_vocabulary: HashMap<String, u32>,
}

impl PromptTokenizer {
    /// Prepare `tokenizer` for prompt encoding. This clones the tokenizer once.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let added_vocabulary = tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .iter()
            .map(|(token, &id)| (token.clone(), id))
            .collect();
        let mut plain_tokenizer = tokenizer.clone();
        plain_tokenizer.with_added_vocabulary(AddedVocabulary::new());
        plain_tokenizer.with_padding(None);
        plain_tokenizer
            .with_truncation(None)
            .expect("disabling truncation is valid");
        Self {
            plain_tokenizer,
            added_vocabulary,
        }
    }

    fn special_token_id(&self, token: &str) -> Result<u32, String> {
        self.added_vocabulary
            .get(token)
            .copied()
            .ok_or_else(|| format!("special token {token:?} is not in the added vocabulary"))
    }
}

impl TokenizerEncoder for PromptTokenizer {
    /// Encode the plain text regions as one batch with the tokenizer without
    /// its added vocabulary, and each special range as the added-vocabulary
    /// token ID of its spelling. The batch runs in parallel unless the
    /// `TOKENIZERS_PARALLELISM` environment variable disables it. The
    /// tokenizer adds no tokens, such as a BOS token, beyond those that the
    /// text encodes to, and does not truncate the plain regions.
    ///
    /// # Errors
    ///
    /// Returns an error message when the tokenizer fails to encode a plain
    /// text region or the spelling of a special range is not in the added
    /// vocabulary.
    ///
    /// # Panics
    ///
    /// Panics if a range starts before the end of the previous range, ends
    /// before it starts, extends past the end of `text`, or does not start and
    /// end on `char` boundaries.
    fn encode_ids(&self, text: &str, special_ranges: &[Range<usize>]) -> Result<Vec<u32>, String> {
        // The plain region before each special range, then the one after the
        // last range; a region can be empty.
        let mut regions = Vec::with_capacity(special_ranges.len() + 1);
        let mut start = 0;
        for range in special_ranges {
            regions.push(&text[start..range.start]);
            start = range.end;
        }
        regions.push(&text[start..]);
        let encodings = self
            .plain_tokenizer
            .encode_batch_fast(regions, false)
            .map_err(|error| error.to_string())?;

        let mut ids = Vec::new();
        let mut encodings = encodings.iter();
        for range in special_ranges {
            if let Some(encoding) = encodings.next() {
                ids.extend_from_slice(encoding.get_ids());
            }
            ids.push(self.special_token_id(&text[range.clone()])?);
        }
        if let Some(encoding) = encodings.next() {
            ids.extend_from_slice(encoding.get_ids());
        }
        Ok(ids)
    }
}

impl<T: TokenizerEncoder + ?Sized> TokenizerEncoder for Arc<T> {
    fn encode_ids(&self, text: &str, special_ranges: &[Range<usize>]) -> Result<Vec<u32>, String> {
        (**self).encode_ids(text, special_ranges)
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

    use super::*;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../static/tokenizers/v41/tokenizer.json"
        ))
        .expect("the bundled V4.1 tokenizer loads")
    }

    fn assert_prompt_regions(padding: Option<PaddingStrategy>, truncation: bool) {
        let original = tokenizer();
        let plain = "one two three four five";
        let bos = "<｜begin▁of▁sentence｜>";
        let eos = "<｜end▁of▁sentence｜>";
        let text = format!("{bos}{plain}{eos}");
        let ranges = [0..bos.len(), bos.len() + plain.len()..text.len()];
        let mut expected = vec![original.token_to_id(bos).unwrap()];
        let plain_ids = original.encode(plain, false).unwrap().get_ids().to_vec();
        assert!(plain_ids.len() > 2, "the fixture must exercise truncation");
        expected.extend_from_slice(&plain_ids);
        expected.push(original.token_to_id(eos).unwrap());

        let mut configured = original.clone();
        configured.with_padding(padding.clone().map(|strategy| PaddingParams {
            strategy,
            pad_id: 0,
            ..PaddingParams::default()
        }));
        configured
            .with_truncation(truncation.then_some(TruncationParams {
                max_length: 2,
                ..TruncationParams::default()
            }))
            .unwrap();
        let before = configured.to_string(false).unwrap();
        let encoder = PromptTokenizer::new(&configured);
        assert_eq!(
            encoder.encode_ids(&text, &ranges).unwrap(),
            expected,
            "padding={padding:?}, truncation={truncation}"
        );
        assert_eq!(configured.to_string(false).unwrap(), before);
        assert_eq!(encoder.encode_ids(plain, &[]).unwrap(), plain_ids);
        assert!(encoder.encode_ids("", &[]).unwrap().is_empty());
    }

    #[test]
    fn prompt_regions_are_not_padded() {
        assert_prompt_regions(Some(PaddingStrategy::Fixed(32)), false);
        assert_prompt_regions(Some(PaddingStrategy::BatchLongest), false);
    }

    #[test]
    fn prompt_regions_are_not_truncated() {
        assert_prompt_regions(None, true);
    }

    #[test]
    fn configured_tokenizers_preserve_both_model_prompts() {
        use deepseek_recipe_core::conversation::Conversation;
        use deepseek_recipe_core::messages::InputMessage;
        use deepseek_recipe_core::prompt::Prompt;

        use crate::PromptEncoding;
        use crate::v4::{dsv4::DeepseekV4Encoding, dsv41::DeepseekV41Encoding};

        let original = tokenizer();
        let mut configured = original.clone();
        configured.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_to_multiple_of: Some(8),
            ..PaddingParams::default()
        }));
        configured
            .with_truncation(Some(TruncationParams {
                max_length: 2,
                ..TruncationParams::default()
            }))
            .unwrap();
        // Saved tokenizer.json files retain these settings, including when
        // loaded through the Python bindings.
        let configured =
            Tokenizer::from_bytes(configured.to_string(false).unwrap().as_bytes()).unwrap();
        let conversation = Conversation {
            messages: vec![InputMessage::User {
                content: Prompt::plain("Please preserve every word in this message."),
                image_sources: Vec::new(),
            }],
            thinking_mode: false,
            ..Conversation::default()
        };
        let reference = PromptTokenizer::new(&original);
        let actual = PromptTokenizer::new(&configured);
        assert_eq!(
            DeepseekV4Encoding::new()
                .with_tokenizer(actual.clone())
                .encode(&conversation)
                .unwrap(),
            DeepseekV4Encoding::new()
                .with_tokenizer(reference.clone())
                .encode(&conversation)
                .unwrap()
        );
        assert_eq!(
            DeepseekV41Encoding::new()
                .with_tokenizer(actual)
                .encode(&conversation)
                .unwrap(),
            DeepseekV41Encoding::new()
                .with_tokenizer(reference)
                .encode(&conversation)
                .unwrap()
        );
    }
}
