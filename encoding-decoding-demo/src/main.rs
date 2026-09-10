use std::net::SocketAddr;
use std::pin::pin;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::error_response::ErrorResponse;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{
    ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest,
};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::state_machine::ReasoningStage;
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, StreamProcessor,
};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::v4::{
    ASSISTANT_SP_TOKEN, BOS_TOKEN, DSML_SP_TOKEN, EOS_TOKEN, LATEST_REMINDER_SP_TOKEN,
    SYSTEM_SP_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN, USER_SP_TOKEN,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio_stream::StreamExt;

const DEFAULT_ADDR: &str = "127.0.0.1:7778";

const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_JS: &str = include_str!("../static/app.js");
const STYLE_CSS: &str = include_str!("../static/style.css");
// DeepSeek icon: https://www.deepseek.com/
const FAVICON_PNG: &[u8] = include_bytes!("../static/favicon-32x32.png");

struct Highlight {
    name: &'static str,
    literals: &'static [&'static str],
    description: &'static str,
}

impl Highlight {
    const fn new(
        name: &'static str,
        literals: &'static [&'static str],
        description: &'static str,
    ) -> Self {
        Self {
            name,
            literals,
            description,
        }
    }

    fn find_in(&self, rest: &str) -> Option<usize> {
        self.literals
            .iter()
            .filter_map(|literal| Some(tag_start(rest, rest.find(literal)?, literal)))
            .min()
    }
}

const HIGHLIGHTS: &[Highlight] = &[
    Highlight::new(
        "bos",
        &[BOS_TOKEN],
        "Begin of sentence. Marks the start of the prompt.",
    ),
    Highlight::new(
        "system",
        &[SYSTEM_SP_TOKEN],
        "System turn marker. Starts a system message.",
    ),
    Highlight::new(
        "user",
        &[USER_SP_TOKEN],
        "User turn marker. Starts a user message.",
    ),
    Highlight::new(
        "assistant",
        &[ASSISTANT_SP_TOKEN],
        "Assistant turn marker. Starts an assistant message.",
    ),
    Highlight::new(
        "thinking_start",
        &[THINKING_START_TOKEN],
        "Starts the reasoning content.",
    ),
    Highlight::new(
        "thinking_end",
        &[THINKING_END_TOKEN],
        "Ends the reasoning content.",
    ),
    Highlight::new(
        "latest_reminder",
        &[LATEST_REMINDER_SP_TOKEN],
        "Starts the latest reminder message.",
    ),
    Highlight::new(
        "eos",
        &[EOS_TOKEN],
        "End of sentence. Terminates the message.",
    ),
    Highlight::new(
        "dsml",
        &[DSML_SP_TOKEN],
        "DeepSeek markup tag. Structures tool calls in the prompt.",
    ),
    Highlight::new(
        "tool_result",
        &["<tool_result>", "</tool_result>"],
        "Contains the result of a tool call.",
    ),
    Highlight::new(
        "image",
        &[IMAGE_SPECIAL_TOKEN],
        "Marks one image in the prompt.",
    ),
    Highlight::new(
        "system_reminder",
        &["<system-reminder>", "</system-reminder>"],
        "Contains a system message within the conversation.",
    ),
];

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Segment {
    Text {
        text: String,
    },
    Token {
        name: String,
        token: String,
        description: String,
    },
}

impl Segment {
    fn text(text: &str) -> Self {
        Self::Text {
            text: text.to_owned(),
        }
    }

    fn token(highlight: &Highlight, token: &str) -> Self {
        Self::Token {
            name: highlight.name.to_owned(),
            token: token.to_owned(),
            description: highlight.description.to_owned(),
        }
    }
}

fn segment(prompt: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut rest = prompt;

    while let Some((start, highlight)) = find_earliest(rest) {
        let (before, tag) = rest.split_at(start);
        if !before.is_empty() {
            segments.push(Segment::text(before));
        }
        let end = tag.find('>').map_or(tag.len(), |index| index + 1);
        let (token, tail) = tag.split_at(end);
        segments.push(Segment::token(highlight, token));
        rest = tail;
    }
    if !rest.is_empty() {
        segments.push(Segment::text(rest));
    }
    segments
}

fn find_earliest(rest: &str) -> Option<(usize, &'static Highlight)> {
    HIGHLIGHTS
        .iter()
        .filter_map(|highlight| Some((highlight.find_in(rest)?, highlight)))
        .min_by_key(|(start, _)| *start)
}

fn tag_start(rest: &str, index: usize, literal: &str) -> usize {
    if literal.starts_with('<') {
        return index;
    }
    let mut start = index;
    if rest[..start].ends_with('/') {
        start -= 1;
    }
    if rest[..start].ends_with('<') {
        start -= 1;
    }
    start
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ApiFormat {
    ChatCompletions,
    Responses,
    Messages,
}

#[derive(Debug, Deserialize)]
struct RenderRequest {
    format: ApiFormat,
    body: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct RenderResponse {
    prompt: String,
    segments: Vec<Segment>,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DecodeFinishReason {
    #[default]
    Stop,
    Length,
    ContentFilter,
}

impl From<DecodeFinishReason> for InferenceFinishReason {
    fn from(reason: DecodeFinishReason) -> Self {
        match reason {
            DecodeFinishReason::Stop => Self::Stop,
            DecodeFinishReason::Length => Self::Length,
            DecodeFinishReason::ContentFilter => Self::ContentFilter,
        }
    }
}

#[derive(Debug, Deserialize)]
struct DecodeRequest {
    format: ApiFormat,
    body: serde_json::Value,
    output: String,
    #[serde(default)]
    finish_reason: DecodeFinishReason,
}

#[derive(Debug, Serialize)]
struct DecodeResponse {
    response: serde_json::Value,
    segments: Vec<Segment>,
}

type DemoError = (StatusCode, Json<ErrorResponse>);

fn bad_request(detail: String) -> DemoError {
    conversion_failed(ConversionError::bad_request(detail))
}

fn conversion_failed(error: ConversionError) -> DemoError {
    let status =
        StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(error.into_response()))
}

fn convert_request<T>(mut body: serde_json::Value) -> Result<ConversationRequest, DemoError>
where
    T: ProtocolRequest + DeserializeOwned,
{
    // Requests use V4.1 prompt rendering and a default model name.
    if let Some(body) = body.as_object_mut() {
        body.entry("model")
            .or_insert_with(|| "encoding-decoding-demo".into());
    }
    let request: T = serde_json::from_value(body)
        .map_err(|err| bad_request(format!("invalid request body: {err}")))?;
    request
        .convert(ConversionOptions::default())
        .map_err(conversion_failed)
}

fn render_prompt<T>(body: serde_json::Value) -> Result<String, DemoError>
where
    T: ProtocolRequest + DeserializeOwned,
{
    let request = convert_request::<T>(body)?;
    Ok(DeepseekV41Encoding::new()
        .render_conversation(&request.conversation)
        .prompt)
}

async fn render_handler(
    Json(request): Json<RenderRequest>,
) -> Result<Json<RenderResponse>, DemoError> {
    let prompt = match request.format {
        ApiFormat::ChatCompletions => render_prompt::<ChatCompletionRequest>(request.body)?,
        ApiFormat::Responses => render_prompt::<ResponsesRequest>(request.body)?,
        ApiFormat::Messages => render_prompt::<MessagesRequest>(request.body)?,
    };
    let segments = segment(&prompt);
    Ok(Json(RenderResponse { prompt, segments }))
}

async fn decode_output<T>(request: DecodeRequest) -> Result<serde_json::Value, DemoError>
where
    T: ProtocolRequest + DeserializeOwned,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Send,
{
    let mut converted = convert_request::<T>(request.body)?;
    // This endpoint always accumulates a complete response, including usage,
    // even when the input request originally selected streaming transport.
    converted.stream = false;
    let (output, finish_reason) = match request.output.split_once(EOS_TOKEN) {
        Some((output, _)) => (output, InferenceFinishReason::Stop),
        None => (request.output.as_str(), request.finish_reason.into()),
    };
    // The demo accepts complete assistant output. Consume its leading frame
    // markers here; the stream parser normally starts after a prompt prefill.
    let output = output.strip_prefix(ASSISTANT_SP_TOKEN).unwrap_or(output);
    let (thinking, output) = match output.strip_prefix(THINKING_START_TOKEN) {
        Some(output) => (true, output),
        None => (
            false,
            output.strip_prefix(THINKING_END_TOKEN).unwrap_or(output),
        ),
    };
    converted.conversation.thinking_mode = thinking;
    converted.parsing_options.reasoning_initial_stage = thinking.then_some(ReasoningStage::Start);
    converted.parsing_options.tool_call_initial_stage = false;
    let model = converted
        .model
        .clone()
        .unwrap_or_else(|| "encoding-decoding-demo".into());
    let id = "demo-id".to_owned();
    let generator = T::chunk_generator(&converted, id.clone(), model.clone());
    let processor = StreamProcessor::new(generator, converted.parsing_options);
    let inference = tokio_stream::iter([
        InferenceChunk::Text {
            content: output.to_owned(),
            // Raw text alone does not supply tokenizer or backend usage data.
            content_tokens: 0,
        },
        InferenceChunk::Finish { finish_reason },
    ]);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let mut response = T::Response::new(id, model, created, 0, 0);
    let mut chunks = pin!(processor.process(inference));
    while let Some(chunk) = chunks.next().await {
        response
            .append(chunk.map_err(|error| {
                conversion_failed(ConversionError::internal(error.to_string()))
            })?);
    }
    serde_json::to_value(response)
        .map_err(|error| conversion_failed(ConversionError::internal(error.to_string())))
}

async fn decode_handler(
    Json(request): Json<DecodeRequest>,
) -> Result<Json<DecodeResponse>, DemoError> {
    let segments = segment(&request.output);
    let response = match request.format {
        ApiFormat::ChatCompletions => decode_output::<ChatCompletionRequest>(request).await?,
        ApiFormat::Responses => decode_output::<ResponsesRequest>(request).await?,
        ApiFormat::Messages => decode_output::<MessagesRequest>(request).await?,
    };
    Ok(Json(DecodeResponse { response, segments }))
}

fn asset(content_type: &'static str, body: &'static str) -> impl IntoResponse {
    binary_asset(content_type, body.as_bytes())
}

fn binary_asset(content_type: &'static str, body: &'static [u8]) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
}

#[tokio::main]
async fn main() {
    let addr: SocketAddr = std::env::var("DEMO_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_owned())
        .parse()
        .expect("invalid DEMO_ADDR");

    let app = Router::new()
        .route(
            "/",
            get(|| async { asset("text/html; charset=utf-8", INDEX_HTML) }),
        )
        .route(
            "/app.js",
            get(|| async { asset("text/javascript; charset=utf-8", APP_JS) }),
        )
        .route(
            "/style.css",
            get(|| async { asset("text/css; charset=utf-8", STYLE_CSS) }),
        )
        .route(
            "/favicon-32x32.png",
            get(|| async { binary_asset("image/png", FAVICON_PNG) }),
        )
        .route("/api/render", post(render_handler))
        .route("/api/decode", post(decode_handler));

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("deepseek-recipe demo page: http://{addr}");
    axum::serve(listener, app).await.unwrap();
}
