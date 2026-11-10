//! Python bindings for Rust response generators, accumulation, and streams.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use deepseek_recipe::anthropic::messages::response::{
    MessagesChunkGenerator, MessagesResponse, MessagesStreamEvent,
};
use deepseek_recipe::openai::chat_completion::response::{
    ChatCompletionChunk, ChatCompletionChunkGenerator, ChatCompletionResponse,
};
use deepseek_recipe::openai::responses::response::{
    ResponsesChunkGenerator, ResponsesEcho, ResponsesResponse, ResponsesStreamEvent,
};
use deepseek_recipe::response::Accumulate;
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::ParsingOptions;
use deepseek_recipe::stream::{ChunkGenerator, InferenceChunk, StreamError, StreamProcessor};
use futures::Stream;
use futures::channel::mpsc;
use futures::executor::block_on;
use futures::future::poll_fn;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::inference::PyInferenceChunk;
use crate::parsing::PyParsingOptions;
use crate::tokenizer::PyTokenizer;

/// A Rust Chat Completions generator consumed by a stream processor.
#[pyclass(
    name = "ChatCompletionChunkGenerator",
    module = "deepseek_recipe._native"
)]
pub(crate) struct PyChatCompletionChunkGenerator {
    inner: Option<ChatCompletionChunkGenerator>,
}

impl PyChatCompletionChunkGenerator {
    pub(crate) fn from_core(inner: ChatCompletionChunkGenerator) -> Self {
        Self { inner: Some(inner) }
    }
}

/// Request settings that Responses response objects report, as the request
/// specified them.
///
/// `ResponsesRequest.echo` reads them from a request; the constructor builds
/// the settings of an unspecified request. The getters expose the scalar
/// settings; the reasoning, text, tool choice, and tool settings are reported
/// only through the response.
#[pyclass(name = "ResponsesEcho", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyResponsesEcho {
    pub(crate) inner: ResponsesEcho,
}

#[pymethods]
impl PyResponsesEcho {
    /// Construct the settings of an unspecified request: no instructions,
    /// output token limit, or user, temperature and `top_p` of 1.0, and zero
    /// `top_logprobs`.
    #[new]
    fn new() -> Self {
        Self {
            inner: ResponsesEcho::default(),
        }
    }

    /// The requested `instructions`.
    #[getter]
    fn instructions(&self) -> Option<&str> {
        self.inner.instructions.as_deref()
    }

    /// The requested `max_output_tokens`.
    #[getter]
    fn max_output_tokens(&self) -> Option<u32> {
        self.inner.max_output_tokens
    }

    /// The requested temperature or 1.0, not the sampling temperature.
    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature
    }

    /// The requested `top_p` or 1.0, not the sampling `top_p`.
    #[getter]
    fn top_p(&self) -> f32 {
        self.inner.top_p
    }

    /// The requested `top_logprobs`, or 0.
    #[getter]
    fn top_logprobs(&self) -> u8 {
        self.inner.top_logprobs
    }

    /// The requested `user`.
    #[getter]
    fn user(&self) -> Option<&str> {
        self.inner.user.as_deref()
    }
}

/// A Rust Responses generator consumed by a stream processor.
#[pyclass(name = "ResponsesChunkGenerator", module = "deepseek_recipe._native")]
pub(crate) struct PyResponsesChunkGenerator {
    inner: Option<ResponsesChunkGenerator>,
}

impl PyResponsesChunkGenerator {
    pub(crate) fn from_core(inner: ResponsesChunkGenerator) -> Self {
        Self { inner: Some(inner) }
    }
}

#[pymethods]
impl PyResponsesChunkGenerator {
    /// Consume this generator and configure the namespace tool names.
    fn with_namespace_tool_map(
        &mut self,
        namespace_tool_map: HashMap<String, (String, String)>,
    ) -> PyResult<Self> {
        Ok(Self::from_core(
            take_generator(&mut self.inner)?.with_namespace_tool_map(namespace_tool_map),
        ))
    }

    /// Consume this generator and configure the reported request settings.
    fn with_echo(&mut self, echo: &PyResponsesEcho) -> PyResult<Self> {
        Ok(Self::from_core(
            take_generator(&mut self.inner)?.with_echo(echo.inner.clone()),
        ))
    }
}

/// A Rust Messages generator consumed by a stream processor.
#[pyclass(name = "MessagesChunkGenerator", module = "deepseek_recipe._native")]
pub(crate) struct PyMessagesChunkGenerator {
    inner: Option<MessagesChunkGenerator>,
}

impl PyMessagesChunkGenerator {
    pub(crate) fn from_core(inner: MessagesChunkGenerator) -> Self {
        Self { inner: Some(inner) }
    }
}

#[pymethods]
impl PyMessagesChunkGenerator {
    #[new]
    fn new(id: String, model: String, thinking_mode: bool) -> Self {
        Self::from_core(MessagesChunkGenerator::new(id, model, thinking_mode))
    }

    /// Consume this generator and return one that emits `signature` at the end
    /// of each thinking block. The default signature is the response ID.
    fn with_signature(&mut self, signature: String) -> PyResult<Self> {
        Ok(Self::from_core(
            take_generator(&mut self.inner)?.with_signature(signature),
        ))
    }
}

fn take_generator<G>(generator: &mut Option<G>) -> PyResult<G> {
    generator
        .take()
        .ok_or_else(|| PyRuntimeError::new_err("chunk generator has been consumed"))
}

fn to_json<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value)
        .map_err(|error| PyRuntimeError::new_err(format!("failed to serialize response: {error}")))
}

// Each wrapper delegates serialization and accumulation to its Rust type.
macro_rules! response_bindings {
    (
        $py_chunk:ident, $chunk:ident, $chunk_name:literal,
        $py_response:ident, $response:ident, $response_name:literal
    ) => {
        /// A protocol chunk consumed by response accumulation.
        #[pyclass(name = $chunk_name, module = "deepseek_recipe._native")]
        pub(crate) struct $py_chunk {
            inner: Option<$chunk>,
        }

        #[pymethods]
        impl $py_chunk {
            /// Serialize this chunk with the Rust protocol representation.
            fn to_json(&self) -> PyResult<String> {
                to_json(self.as_core()?)
            }
        }

        impl $py_chunk {
            fn as_core(&self) -> PyResult<&$chunk> {
                self.inner
                    .as_ref()
                    .ok_or_else(|| PyRuntimeError::new_err("protocol chunk has been consumed"))
            }
        }

        /// A Rust protocol response with explicit accumulation.
        #[pyclass(name = $response_name, module = "deepseek_recipe._native")]
        pub(crate) struct $py_response {
            inner: $response,
        }

        #[pymethods]
        impl $py_response {
            #[new]
            fn new(
                id: String,
                model: String,
                created: u64,
                prompt_tokens: usize,
                prompt_cache_hit_tokens: usize,
            ) -> Self {
                Self {
                    inner: $response::new(
                        id,
                        model,
                        created,
                        prompt_tokens,
                        prompt_cache_hit_tokens,
                    ),
                }
            }

            /// Apply `chunk` to this response. `chunk` cannot be used afterward;
            /// a consumed chunk raises `RuntimeError`.
            fn accumulate(&mut self, chunk: &mut $py_chunk) -> PyResult<()> {
                let chunk = chunk
                    .inner
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("protocol chunk has been consumed"))?;
                self.inner.accumulate(chunk);
                Ok(())
            }

            /// Serialize the current Rust response, including partial state.
            fn to_json(&self) -> PyResult<String> {
                to_json(&self.inner)
            }

            /// Return the SSE event name of `chunk`, or `None` for an unnamed
            /// data event.
            #[staticmethod]
            fn sse_event_name(chunk: &$py_chunk) -> PyResult<Option<&'static str>> {
                Ok($response::sse_event_name(chunk.as_core()?))
            }

            /// Return the data of the final SSE message, such as `[DONE]`, when
            /// the protocol ends its stream with one.
            #[staticmethod]
            fn sse_done_data() -> Option<String> {
                $response::sse_done_data()
            }
        }
    };
}

response_bindings!(
    PyChatCompletionChunk,
    ChatCompletionChunk,
    "ChatCompletionChunk",
    PyChatCompletionResponse,
    ChatCompletionResponse,
    "ChatCompletionResponse"
);
response_bindings!(
    PyResponsesStreamEvent,
    ResponsesStreamEvent,
    "ResponsesStreamEvent",
    PyResponsesResponse,
    ResponsesResponse,
    "ResponsesResponse"
);
response_bindings!(
    PyMessagesStreamEvent,
    MessagesStreamEvent,
    "MessagesStreamEvent",
    PyMessagesResponse,
    MessagesResponse,
    "MessagesResponse"
);

enum ProtocolChunk {
    ChatCompletion(ChatCompletionChunk),
    /// Boxed because response events carry the complete response object.
    Responses(Box<ResponsesStreamEvent>),
    Messages(MessagesStreamEvent),
}

impl ProtocolChunk {
    fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::ChatCompletion(chunk) => {
                Ok(Py::new(py, PyChatCompletionChunk { inner: Some(chunk) })?.into_any())
            }
            Self::Responses(chunk) => Ok(Py::new(
                py,
                PyResponsesStreamEvent {
                    inner: Some(*chunk),
                },
            )?
            .into_any()),
            Self::Messages(chunk) => {
                Ok(Py::new(py, PyMessagesStreamEvent { inner: Some(chunk) })?.into_any())
            }
        }
    }
}

type ChunkStream = Pin<Box<dyn Stream<Item = Result<ProtocolChunk, StreamError>> + Send>>;

struct Processor {
    input: Option<mpsc::Sender<InferenceChunk>>,
    events: Option<ChunkStream>,
    input_pending: Arc<AtomicBool>,
    finished: bool,
    closed: bool,
}

/// Adapt Python input to the Rust processor's inference stream.
///
/// Construction consumes the generator. Each `push` produces the protocol
/// chunks available before the processor needs another input. `finish` closes
/// the input normally; `close` releases state without generating successful
/// completion.
#[pyclass(name = "StreamProcessor", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyStreamProcessor {
    state: Mutex<Processor>,
}

impl PyStreamProcessor {
    fn from_generator<G>(
        generator: G,
        options: ParsingOptions,
        tokenizer: Option<Arc<Tokenizer>>,
        wrap: fn(G::Chunk) -> ProtocolChunk,
    ) -> Self
    where
        G: ChunkGenerator,
        G::Chunk: Send,
    {
        let (sender, mut receiver) = mpsc::channel(1);
        let input_pending = Arc::new(AtomicBool::new(false));
        let receiver_pending = Arc::clone(&input_pending);
        let input = futures::stream::poll_fn(move |cx| {
            let result = Pin::new(&mut receiver).poll_next(cx);
            if result.is_pending() {
                receiver_pending.store(true, Ordering::Relaxed);
            }
            result
        });
        let processor = StreamProcessor::new(generator, options);
        let processor = match tokenizer {
            Some(tokenizer) => processor.with_tokenizer(tokenizer),
            None => processor,
        };
        let events = futures::StreamExt::map(processor.into_protocol_stream(input), move |event| {
            event.map(wrap)
        });
        Self {
            state: Mutex::new(Processor {
                input: Some(sender),
                events: Some(Box::pin(events)),
                input_pending,
                finished: false,
                closed: false,
            }),
        }
    }

    fn with_state<T: Send>(
        &self,
        py: Python<'_>,
        operation: impl FnOnce(&mut Processor) -> PyResult<T> + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let mut state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("stream processor lock poisoned"))?;
            operation(&mut state)
        })
    }
}

#[pymethods]
impl PyStreamProcessor {
    #[new]
    #[pyo3(signature = (generator, options, tokenizer = None))]
    fn new(
        generator: &Bound<'_, PyAny>,
        options: &PyParsingOptions,
        tokenizer: Option<PyRef<'_, PyTokenizer>>,
    ) -> PyResult<Self> {
        let tokenizer = tokenizer.map(|tokenizer| tokenizer.inner.clone());
        if let Ok(mut generator) =
            generator.extract::<PyRefMut<'_, PyChatCompletionChunkGenerator>>()
        {
            Ok(Self::from_generator(
                take_generator(&mut generator.inner)?,
                options.inner.clone(),
                tokenizer,
                ProtocolChunk::ChatCompletion,
            ))
        } else if let Ok(mut generator) =
            generator.extract::<PyRefMut<'_, PyResponsesChunkGenerator>>()
        {
            Ok(Self::from_generator(
                take_generator(&mut generator.inner)?,
                options.inner.clone(),
                tokenizer,
                |chunk| ProtocolChunk::Responses(Box::new(chunk)),
            ))
        } else if let Ok(mut generator) =
            generator.extract::<PyRefMut<'_, PyMessagesChunkGenerator>>()
        {
            Ok(Self::from_generator(
                take_generator(&mut generator.inner)?,
                options.inner.clone(),
                tokenizer,
                ProtocolChunk::Messages,
            ))
        } else {
            Err(PyTypeError::new_err("expected a protocol chunk generator"))
        }
    }

    /// Process one inference chunk until Rust needs another input.
    /// Raises `RuntimeError` after completion or close, or on a stream error: a
    /// `tokens` chunk without a tokenizer, a decoding failure, or a log
    /// probability count that differs from the token ID count. A stream error
    /// closes the processor.
    fn push(&self, py: Python<'_>, chunk: &PyInferenceChunk) -> PyResult<Vec<Py<PyAny>>> {
        let chunk = chunk.to_core();
        self.with_state(py, |state| state.push(chunk))?
            .into_iter()
            .map(|chunk| chunk.into_python(py))
            .collect()
    }

    /// Close the input normally and process its buffered output.
    /// Repeated calls after completion produce no chunks. Calls after close or
    /// a stream error raise `RuntimeError`.
    fn finish(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.with_state(py, Processor::finish)?
            .into_iter()
            .map(|chunk| chunk.into_python(py))
            .collect()
    }

    /// Whether the Rust stream reached normal completion and has not been closed.
    #[getter]
    fn finished(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_state(py, |state| Ok(state.finished && !state.closed))
    }

    /// Release state without producing normal completion chunks.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.with_state(py, |state| {
            state.close();
            Ok(())
        })
    }
}

impl Processor {
    fn drain(&mut self) -> PyResult<Vec<ProtocolChunk>> {
        let mut chunks = Vec::new();
        block_on(poll_fn(|cx| {
            loop {
                // Only a pending input read ends this batch. Other pending
                // futures retain their wakeups and continue on this executor.
                self.input_pending.store(false, Ordering::Relaxed);
                let event = self.events.as_mut().unwrap().as_mut().poll_next(cx);
                match event {
                    Poll::Ready(Some(Ok(chunk))) => chunks.push(chunk),
                    Poll::Ready(Some(Err(error))) => {
                        self.close();
                        return Poll::Ready(Err(PyRuntimeError::new_err(error.to_string())));
                    }
                    Poll::Ready(None) => {
                        self.finished = true;
                        self.input = None;
                        self.events = None;
                        return Poll::Ready(Ok(std::mem::take(&mut chunks)));
                    }
                    Poll::Pending if self.input_pending.load(Ordering::Relaxed) => {
                        return Poll::Ready(Ok(std::mem::take(&mut chunks)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }))
    }

    fn push(&mut self, chunk: InferenceChunk) -> PyResult<Vec<ProtocolChunk>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("stream processor is closed"));
        }
        if self.finished {
            return Err(PyRuntimeError::new_err("stream processor has finished"));
        }
        if let Err(error) = self.input.as_mut().unwrap().try_send(chunk) {
            self.close();
            return Err(PyRuntimeError::new_err(format!(
                "failed to submit inference chunk: {error}"
            )));
        }
        self.drain()
    }

    fn finish(&mut self) -> PyResult<Vec<ProtocolChunk>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("stream processor is closed"));
        }
        if self.finished {
            return Ok(Vec::new());
        }
        self.input = None;
        self.drain()
    }

    fn close(&mut self) {
        self.closed = true;
        self.input = None;
        self.events = None;
    }
}
