//! Request handling and mock inference for the Rust HTTP server example.
//!
//! Requests use DeepSeek V4.1 prompt rendering, resolve images with the default
//! image resolver, and use the server's mock inference backend. The HTTP server
//! supports SSE and complete JSON responses for Chat Completions, Responses,
//! and Messages. The Python example composes its own handlers and mock backend.

use std::future::Future;
use std::pin::{Pin, pin};
use std::time::{SystemTime, UNIX_EPOCH};

use deepseek_recipe::anthropic::MessagesRequest;
pub use deepseek_recipe::error_response::ErrorResponse;
use deepseek_recipe::openai::responses::response::ResponsesResponse;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
pub use deepseek_recipe::request::ConversionError;
use deepseek_recipe::request::{ConversationRequest, ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor,
};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_core::multimodal::MultiModalData;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_image::{
    ImageError, ImageQuota, ImageResolver, OpenCvImagePreprocessor, ReqwestImageFetcher,
};
use serde::{Serialize, de::DeserializeOwned};
use tokio_stream::{Stream, StreamExt};

const MOCK_ID: &str = "mock-id";
const MOCK_MODEL: &str = "deepseek-flash";

type PreparedProcessor<R> = (
    String,
    MultiModalData,
    StreamProcessor<<R as ProtocolResponse>::ChunkGenerator>,
);

/// A lazy response stream whose successful items are complete SSE frames.
///
/// A serialization or processing error terminates the transport's response with
/// an error; consumers must not replace failed events with empty frames.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<String, serde_json::Error>> + Send>>;

/// A validated request's prompt, resolved images, and protocol response stream.
pub struct PreparedResponse {
    /// DeepSeek V4.1 prompt, currently unused by the mock inference backend.
    pub prompt: String,
    /// Images resolved from the request and preprocessed for the backend.
    pub multi_modal_data: MultiModalData,
    /// Protocol events, including the terminal sentinel when required.
    pub events: ResponseStream,
}

/// Deserialize a protocol request and prepare the server's response.
///
/// Supports `chat_completions`, `responses`, and `messages`.
/// All protocols currently produce SSE regardless of the request's `stream`
/// setting and use the mock backend defined in this crate.
///
/// # Errors
///
/// Returns a conversion error for unknown protocols, malformed JSON, invalid
/// requests, or unusable images. The error has status code 400 for invalid input
/// and 500 for internal conversion failures. Validation finishes before the
/// response stream is returned.
pub async fn handle_request(
    protocol: &str,
    body: &[u8],
) -> Result<PreparedResponse, ConversionError> {
    match protocol {
        "chat_completions" => parse_request(body, prepare_chat_completion_response).await,
        "responses" => parse_request(body, prepare_responses_response).await,
        "messages" => parse_request(body, prepare_response::<MessagesRequest>).await,
        _ => Err(ConversionError::bad_request(format!(
            "unknown protocol: {protocol}"
        ))),
    }
}

async fn parse_request<T, F, Fut>(
    body: &[u8],
    prepare: F,
) -> Result<PreparedResponse, ConversionError>
where
    T: DeserializeOwned,
    F: FnOnce(T) -> Fut,
    Fut: Future<Output = Result<PreparedResponse, ConversionError>>,
{
    let request = serde_json::from_slice::<T>(body)
        .map_err(|error| ConversionError::bad_request(format!("invalid JSON request: {error}")))?;
    prepare(request).await
}

/// Convert an already deserialized request and prepare the server's response.
///
/// Uses [`ConversionOptions::default()`], including enabled thinking unless the
/// protocol explicitly overrides it. Images are resolved with the default
/// resolver. The resulting stream shares the request's resolved parsing options,
/// including stop sequences and tool parsing.
///
/// # Errors
///
/// Returns the protocol conversion error as an HTTP 400 or 500 error before
/// producing any response events.
pub async fn prepare_response<T>(req: T) -> Result<PreparedResponse, ConversionError>
where
    T: ProtocolRequest,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Serialize + Send,
{
    let request = req.convert(ConversionOptions::default())?;
    prepare_converted_response::<T>(request).await
}

/// Prepare SSE from a validated request without repeating protocol conversion.
///
/// The request must come from `T::convert`. The caller selects SSE transport;
/// the request's original stream setting is preserved for the chunk generator.
///
/// # Errors
///
/// Returns a conversion error with status code 400 or 500 when an image cannot
/// be resolved or preprocessed.
pub async fn prepare_converted_response<T>(
    request: ConversationRequest,
) -> Result<PreparedResponse, ConversionError>
where
    T: ProtocolRequest,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Serialize + Send,
{
    prepare_converted_response_with::<T>(request, |generator| generator).await
}

/// Prepare Chat Completions SSE using the request's usage inclusion setting.
///
/// # Errors
///
/// Returns the protocol conversion error before producing response events, or a
/// conversion error with status code 400 or 500 when an image cannot be resolved
/// or preprocessed.
pub async fn prepare_chat_completion_response(
    req: ChatCompletionRequest,
) -> Result<PreparedResponse, ConversionError> {
    let include_usage = req.include_usage();
    let request = req.convert(ConversionOptions::default())?;
    prepare_converted_response_with::<ChatCompletionRequest>(request, |generator| {
        generator.with_include_usage(include_usage)
    })
    .await
}

/// Prepare Responses SSE using the request's custom tool declarations.
///
/// # Errors
///
/// Returns a conversion error for invalid requests or unusable images.
pub async fn prepare_responses_response(
    req: ResponsesRequest,
) -> Result<PreparedResponse, ConversionError> {
    let custom_tool_names = req.custom_tool_names();
    let request = req.convert(ConversionOptions::default())?;
    prepare_converted_response_with::<ResponsesRequest>(request, move |generator| {
        generator.with_custom_tool_names(custom_tool_names)
    })
    .await
}

async fn prepare_converted_response_with<T>(
    request: ConversationRequest,
    configure_generator: impl FnOnce(
        <T::Response as ProtocolResponse>::ChunkGenerator,
    ) -> <T::Response as ProtocolResponse>::ChunkGenerator,
) -> Result<PreparedResponse, ConversionError>
where
    T: ProtocolRequest,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Serialize + Send,
{
    let (prompt, multi_modal_data, processor) =
        prepare_processor::<T>(request, configure_generator).await?;
    let events = processor.process(mock_inference()).map(|chunk| {
        let chunk = chunk.map_err(|error| serde_json::Error::io(std::io::Error::other(error)))?;
        let data = serde_json::to_string(&chunk)?;
        Ok(
            match <T::Response as ProtocolResponse>::chunk_event_type(&chunk) {
                Some(event_type) => format!("event: {event_type}\ndata: {data}\n\n"),
                None => format!("data: {data}\n\n"),
            },
        )
    });
    let done = <T::Response as ProtocolResponse>::done_message()
        .map(|message| Ok(format!("data: {message}\n\n")));

    Ok(PreparedResponse {
        prompt,
        multi_modal_data,
        events: Box::pin(events.chain(tokio_stream::iter(done))),
    })
}

/// Aggregate the mock inference output into a complete protocol response.
///
/// The request must come from `T::convert`. Uses the same parsing and identity
/// as [`prepare_converted_response`] without repeating protocol conversion.
/// The protocol response implements event accumulation. Events are consumed
/// incrementally.
///
/// # Errors
///
/// Returns a conversion error with status code 400 or 500 when an image cannot
/// be resolved or preprocessed.
pub async fn complete_response<T>(
    request: ConversationRequest,
) -> Result<T::Response, ConversionError>
where
    T: ProtocolRequest,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Send,
{
    complete_response_with::<T>(request, |generator| generator).await
}

/// Aggregate Responses output using the request's custom tool declarations.
///
/// # Errors
///
/// Returns a conversion error for invalid requests, unusable images, or failed
/// output processing.
pub async fn complete_responses_response(
    req: ResponsesRequest,
) -> Result<ResponsesResponse, ConversionError> {
    let custom_tool_names = req.custom_tool_names();
    let request = req.convert(ConversionOptions::default())?;
    complete_response_with::<ResponsesRequest>(request, move |generator| {
        generator.with_custom_tool_names(custom_tool_names)
    })
    .await
}

async fn complete_response_with<T>(
    request: ConversationRequest,
    configure_generator: impl FnOnce(
        <T::Response as ProtocolResponse>::ChunkGenerator,
    ) -> <T::Response as ProtocolResponse>::ChunkGenerator,
) -> Result<T::Response, ConversionError>
where
    T: ProtocolRequest,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Send,
{
    let (_, _, processor) = prepare_processor::<T>(request, configure_generator).await?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let mut response = T::Response::new(MOCK_ID.into(), MOCK_MODEL.into(), created, 0, 0);
    let mut chunks = pin!(processor.process(mock_inference()));
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|error| ConversionError::internal(error.to_string()))?;
        response.append(chunk);
    }
    Ok(response)
}

async fn prepare_processor<T>(
    request: ConversationRequest,
    configure_generator: impl FnOnce(
        <T::Response as ProtocolResponse>::ChunkGenerator,
    ) -> <T::Response as ProtocolResponse>::ChunkGenerator,
) -> Result<PreparedProcessor<T::Response>, ConversionError>
where
    T: ProtocolRequest,
{
    let rendered = DeepseekV41Encoding::new().render_conversation(&request.conversation);
    let resolver = default_resolver()?;
    let mut quota = ImageQuota::new();
    let multi_modal_data = resolver
        .resolve(&rendered.image_sources, &mut quota)
        .await
        .map_err(image_error)?;
    tracing::debug!(
        prompt = rendered.prompt,
        image_count = multi_modal_data.images.len(),
        "rendered prompt"
    );
    let generator = configure_generator(T::chunk_generator(
        &request,
        MOCK_ID.into(),
        MOCK_MODEL.into(),
    ));
    Ok((
        rendered.prompt,
        multi_modal_data,
        StreamProcessor::new(generator, request.parsing_options),
    ))
}

/// Build the resolver used by the example server.
fn default_resolver()
-> Result<ImageResolver<ReqwestImageFetcher, OpenCvImagePreprocessor>, ConversionError> {
    let fetcher =
        ReqwestImageFetcher::new().map_err(|error| ConversionError::internal(error.to_string()))?;
    Ok(ImageResolver::new(fetcher, OpenCvImagePreprocessor))
}

/// Map an image failure to a conversion error. Invalid input is a client error.
fn image_error(error: ImageError) -> ConversionError {
    match error {
        ImageError::Client(_) | ImageError::TokenBudget(_) => {
            ConversionError::internal(error.to_string())
        }
        _ => ConversionError::bad_request(error.to_string()),
    }
}

fn mock_inference() -> impl Stream<Item = InferenceChunk> + Send {
    tokio_stream::iter(vec![
        InferenceChunk::Ready {
            system_fingerprint: None,
            prompt_usage: PromptUsage::default(),
        },
        InferenceChunk::Text {
            content: "Hello ".to_string(),
            content_tokens: 1,
        },
        InferenceChunk::Text {
            content: "world!".to_string(),
            content_tokens: 1,
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ])
}
