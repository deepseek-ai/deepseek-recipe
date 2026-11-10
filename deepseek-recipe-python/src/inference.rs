//! Python representations of backend inference chunks and usage snapshots.

use deepseek_recipe::stream::{
    InferenceChunk, InferenceFinishReason, InferenceTokenLogprob, PromptUsage,
};
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

/// The log probability of one token and its `(token_id, logprob)` alternatives.
type InferenceTokenLogprobTuple = (f32, Vec<(u32, f32)>);

/// One backend inference chunk, constructed with `ready`, `text`, `tokens`,
/// or `finish`.
///
/// Ready chunks carry the prompt usage snapshot; text chunks carry the number
/// of tokens their content accounts for. Omitted counts default to zero.
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
                token_count,
            } => InferenceChunk::Text {
                content: content.clone(),
                token_count: *token_count,
            },
            InferenceChunk::Tokens {
                token_ids,
                logprobs,
            } => InferenceChunk::Tokens {
                token_ids: token_ids.clone(),
                logprobs: logprobs.clone(),
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
    /// protocol responses that expose it, such as Chat Completions.
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
    /// `token_count` is the number of tokens this chunk's content accounts
    /// for; the stream processor accumulates it across chunks.
    #[staticmethod]
    #[pyo3(signature = (content, token_count = 0))]
    fn text(content: String, token_count: usize) -> Self {
        Self {
            inner: InferenceChunk::Text {
                content,
                token_count,
            },
        }
    }

    /// Construct a chunk with the token IDs of one backend output, decoded
    /// together by the processor's tokenizer.
    ///
    /// `logprobs`, when given, holds one `(logprob, top_logprobs)` pair per
    /// token ID, where `top_logprobs` lists `(token_id, logprob)` alternatives.
    #[staticmethod]
    #[pyo3(signature = (token_ids, logprobs = None))]
    fn tokens(token_ids: Vec<u32>, logprobs: Option<Vec<InferenceTokenLogprobTuple>>) -> Self {
        Self {
            inner: InferenceChunk::Tokens {
                token_ids,
                logprobs: logprobs.map(|logprobs| {
                    logprobs
                        .into_iter()
                        .map(|(logprob, top_logprobs)| InferenceTokenLogprob {
                            logprob,
                            top_logprobs,
                        })
                        .collect()
                }),
            },
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

    /// The chunk kind: `ready`, `text`, `tokens`, or `finish`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            InferenceChunk::Ready { .. } => "ready",
            InferenceChunk::Text { .. } => "text",
            InferenceChunk::Tokens { .. } => "tokens",
            InferenceChunk::Finish { .. } => "finish",
        }
    }

    /// The backend content, present only on text chunks.
    #[getter]
    fn content(&self) -> Option<&str> {
        match &self.inner {
            InferenceChunk::Text { content, .. } => Some(content),
            InferenceChunk::Ready { .. }
            | InferenceChunk::Tokens { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }

    /// The token IDs of a `tokens` chunk.
    #[getter]
    fn token_ids(&self) -> Option<Vec<u32>> {
        match &self.inner {
            InferenceChunk::Tokens { token_ids, .. } => Some(token_ids.clone()),
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
            | InferenceChunk::Tokens { .. } => None,
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
            | InferenceChunk::Tokens { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }

    /// The number of tokens this chunk's content accounts for, present only
    /// on text chunks.
    #[getter]
    fn token_count(&self) -> Option<usize> {
        match &self.inner {
            InferenceChunk::Text { token_count, .. } => Some(*token_count),
            InferenceChunk::Ready { .. }
            | InferenceChunk::Tokens { .. }
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
            | InferenceChunk::Tokens { .. }
            | InferenceChunk::Finish { .. } => None,
        }
    }
}
