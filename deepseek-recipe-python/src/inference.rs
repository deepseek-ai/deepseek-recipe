//! Python representations of backend inference chunks and usage snapshots.

use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use pyo3::prelude::*;

/// Prompt-side token counts supplied by the inference backend.
///
/// Unspecified counts default to zero. The binding does not infer token counts
/// from generated text.
#[pyclass(name = "PromptUsage", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyPromptUsage {
    inner: PromptUsage,
}

#[pymethods]
impl PyPromptUsage {
    #[new]
    #[pyo3(signature = (*, prompt_tokens = 0, prompt_cache_hit_tokens = 0))]
    fn new(prompt_tokens: usize, prompt_cache_hit_tokens: usize) -> Self {
        Self {
            inner: PromptUsage {
                prompt_tokens,
                prompt_cache_hit_tokens,
            },
        }
    }

    /// Total input tokens, including tokens read from the prompt cache.
    #[getter]
    fn prompt_tokens(&self) -> usize {
        self.inner.prompt_tokens
    }

    /// Input tokens read from the prompt cache.
    #[getter]
    fn prompt_cache_hit_tokens(&self) -> usize {
        self.inner.prompt_cache_hit_tokens
    }
}

/// The finish reason reported by the inference backend.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(
    name = "InferenceFinishReason",
    module = "deepseek_recipe._native",
    eq,
    frozen
)]
pub(crate) enum PyInferenceFinishReason {
    Stop,
    Length,
    ContentFilter,
}

impl PyInferenceFinishReason {
    fn to_core(self) -> InferenceFinishReason {
        match self {
            Self::Stop => InferenceFinishReason::Stop,
            Self::Length => InferenceFinishReason::Length,
            Self::ContentFilter => InferenceFinishReason::ContentFilter,
        }
    }

    fn from_core(reason: InferenceFinishReason) -> Self {
        match reason {
            InferenceFinishReason::Stop => Self::Stop,
            InferenceFinishReason::Length => Self::Length,
            InferenceFinishReason::ContentFilter => Self::ContentFilter,
        }
    }
}

/// One backend inference chunk, constructed with `ready`, `text`, `token`, or
/// `finish`.
///
/// Ready chunks carry the prompt usage snapshot; text chunks carry the tokens
/// their content accounts for. Omitted counts default to zero.
#[pyclass(name = "InferenceChunk", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyInferenceChunk {
    inner: InferenceChunk,
}

impl PyInferenceChunk {
    pub(crate) fn to_core(&self) -> InferenceChunk {
        match &self.inner {
            InferenceChunk::Ready {
                system_fingerprint,
                prompt_usage,
            } => InferenceChunk::Ready {
                system_fingerprint: system_fingerprint.clone(),
                prompt_usage: *prompt_usage,
            },
            InferenceChunk::Text {
                content,
                content_tokens,
            } => InferenceChunk::Text {
                content: content.clone(),
                content_tokens: *content_tokens,
            },
            InferenceChunk::Token { token_id } => InferenceChunk::Token {
                token_id: *token_id,
            },
            InferenceChunk::Finish { finish_reason } => InferenceChunk::Finish {
                finish_reason: *finish_reason,
            },
        }
    }
}

#[pymethods]
impl PyInferenceChunk {
    /// Construct the backend's initial metadata chunk.
    ///
    /// Omitted usage defaults to zero. The system fingerprint is carried into
    /// protocol responses that expose it, such as OpenAI Chat Completions.
    #[staticmethod]
    #[pyo3(signature = (*, prompt_usage = None, system_fingerprint = None))]
    fn ready(
        prompt_usage: Option<PyRef<'_, PyPromptUsage>>,
        system_fingerprint: Option<String>,
    ) -> Self {
        Self {
            inner: InferenceChunk::Ready {
                system_fingerprint,
                prompt_usage: prompt_usage.map(|usage| usage.inner).unwrap_or_default(),
            },
        }
    }

    /// Construct one text chunk from the backend.
    ///
    /// `content_tokens` is the number of tokens this chunk's content accounts
    /// for; the stream processor accumulates it across chunks.
    #[staticmethod]
    #[pyo3(signature = (text, content_tokens = 0))]
    fn text(text: String, content_tokens: usize) -> Self {
        Self {
            inner: InferenceChunk::Text {
                content: text,
                content_tokens,
            },
        }
    }

    /// Construct one token-id chunk, decoded by the processor's tokenizer.
    #[staticmethod]
    fn token(token_id: u32) -> Self {
        Self {
            inner: InferenceChunk::Token { token_id },
        }
    }

    /// Construct the backend's terminal chunk.
    #[staticmethod]
    fn finish(finish_reason: PyRef<'_, PyInferenceFinishReason>) -> Self {
        Self {
            inner: InferenceChunk::Finish {
                finish_reason: (*finish_reason).to_core(),
            },
        }
    }

    /// The chunk kind: `ready`, `text`, `token`, or `finish`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            InferenceChunk::Ready { .. } => "ready",
            InferenceChunk::Text { .. } => "text",
            InferenceChunk::Token { .. } => "token",
            InferenceChunk::Finish { .. } => "finish",
        }
    }

    /// The backend content, present only on text chunks.
    #[getter]
    fn content(&self) -> Option<&str> {
        match &self.inner {
            InferenceChunk::Text { content, .. } => Some(content),
            InferenceChunk::Ready { .. }
            | InferenceChunk::Token { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }

    /// The backend token id, present only on token chunks.
    #[getter]
    fn token_id(&self) -> Option<u32> {
        match &self.inner {
            InferenceChunk::Token { token_id } => Some(*token_id),
            InferenceChunk::Ready { .. }
            | InferenceChunk::Text { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }

    /// The backend finish reason, present only on finish chunks.
    #[getter]
    fn finish_reason(&self) -> Option<PyInferenceFinishReason> {
        match &self.inner {
            InferenceChunk::Finish { finish_reason, .. } => {
                Some(PyInferenceFinishReason::from_core(*finish_reason))
            }
            InferenceChunk::Ready { .. }
            | InferenceChunk::Text { .. }
            | InferenceChunk::Token { .. } => None,
        }
    }

    /// The prompt usage snapshot, present only on ready chunks.
    #[getter]
    fn prompt_usage(&self) -> Option<PyPromptUsage> {
        match &self.inner {
            InferenceChunk::Ready { prompt_usage, .. } => Some(PyPromptUsage {
                inner: *prompt_usage,
            }),
            InferenceChunk::Text { .. }
            | InferenceChunk::Token { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }

    /// The tokens this chunk's content accounts for, present only on text
    /// chunks.
    #[getter]
    fn content_tokens(&self) -> Option<usize> {
        match &self.inner {
            InferenceChunk::Text { content_tokens, .. } => Some(*content_tokens),
            InferenceChunk::Ready { .. }
            | InferenceChunk::Token { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }

    /// The backend fingerprint, present only on ready chunks.
    #[getter]
    fn system_fingerprint(&self) -> Option<&str> {
        match &self.inner {
            InferenceChunk::Ready {
                system_fingerprint, ..
            } => system_fingerprint.as_deref(),
            InferenceChunk::Text { .. }
            | InferenceChunk::Token { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }
}
