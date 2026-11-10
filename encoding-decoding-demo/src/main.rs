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
    ConversationRequest, ConversionError, ConversionOptions, ModelVersion, ProtocolRequest,
};
use deepseek_recipe::response::{Accumulate, ProtocolResponse};
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, StreamProcessor, ThinkingInitialStage,
};
use deepseek_recipe_core::special_tokens::SpecialToken;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
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
        &[SpecialToken::Bos.default_str()],
        "Beginning-of-sentence token (BOS). Marks the start of the prompt.",
    ),
    Highlight::new(
        "system",
        &[SpecialToken::System.default_str()],
        "System turn marker. Starts a system message.",
    ),
    Highlight::new(
        "user",
        &[SpecialToken::User.default_str()],
        "User turn marker. Starts a user message.",
    ),
    Highlight::new(
        "assistant",
        &[SpecialToken::Assistant.default_str()],
        "Assistant turn marker. Starts an assistant message.",
    ),
    Highlight::new(
        "thinking_start",
        &[SpecialToken::ThinkingStart.default_str()],
        "Starts the reasoning content.",
    ),
    Highlight::new(
        "thinking_end",
        &[SpecialToken::ThinkingEnd.default_str()],
        "Ends the reasoning content.",
    ),
    Highlight::new(
        "latest_reminder",
        &[SpecialToken::LatestReminder.default_str()],
        "Starts the latest reminder message.",
    ),
    Highlight::new(
        "eos",
        &[SpecialToken::Eos.default_str()],
        "End-of-sentence token (EOS). Ends an assistant message.",
    ),
    Highlight::new(
        "dsml",
        &[SpecialToken::Dsml.default_str()],
        "DeepSeek markup tag. Structures tool calls in the prompt.",
    ),
    Highlight::new(
        "tool_result",
        &["<tool_result>", "</tool_result>"],
        "Contains the result of a tool call.",
    ),
    Highlight::new(
        "image",
        &[SpecialToken::Image.default_str()],
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
enum ParseFinishReason {
    #[default]
    Stop,
    Length,
    ContentFilter,
}

impl From<ParseFinishReason> for InferenceFinishReason {
    fn from(reason: ParseFinishReason) -> Self {
        match reason {
            ParseFinishReason::Stop => Self::Stop,
            ParseFinishReason::Length => Self::Length,
            ParseFinishReason::ContentFilter => Self::ContentFilter,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ParseRequest {
    format: ApiFormat,
    body: serde_json::Value,
    output: String,
    #[serde(default)]
    finish_reason: ParseFinishReason,
}

#[derive(Debug, Serialize)]
struct ParseResponse {
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
    (status, Json(error.into_error_response()))
}

/// Convert a protocol request, returning it alongside the requested model name
/// echoed into responses.
fn convert_request<T>(
    mut body: serde_json::Value,
) -> Result<(ConversationRequest, String), DemoError>
where
    T: ProtocolRequest + DeserializeOwned,
{
    // A request without a model name uses the default name, which targets V4.1.
    const DEFAULT_MODEL: &str = "encoding-decoding-demo";
    if let Some(body) = body.as_object_mut() {
        body.entry("model").or_insert_with(|| DEFAULT_MODEL.into());
    }
    let model = body["model"].as_str().unwrap_or(DEFAULT_MODEL).to_owned();
    let request: T = serde_json::from_value(body)
        .map_err(|err| bad_request(format!("invalid request body: {err}")))?;
    // The demo also accepts the DeepSeek API model names.
    let converted = request
        .convert(
            ConversionOptions::default()
                .with_model_alias(DEFAULT_MODEL, ModelVersion::DeepseekV41Flash)
                .with_model_alias("deepseek-v4-pro", ModelVersion::DeepseekV4Pro)
                .with_model_alias("deepseek-flash", ModelVersion::DeepseekV41Flash),
        )
        .map_err(conversion_failed)?;
    Ok((converted, model))
}

fn render_prompt<T>(body: serde_json::Value) -> Result<String, DemoError>
where
    T: ProtocolRequest + DeserializeOwned,
{
    let (request, _) = convert_request::<T>(body)?;
    // The model version of the request selects the prompt encoding.
    let rendered = match request.model_version {
        ModelVersion::DeepseekV4Pro => {
            DeepseekV4Encoding::new().render_conversation(&request.conversation)
        }
        _ => DeepseekV41Encoding::new().render_conversation(&request.conversation),
    };
    Ok(rendered.prompt)
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

async fn parse_output<T>(request: ParseRequest) -> Result<serde_json::Value, DemoError>
where
    T: ProtocolRequest + DeserializeOwned,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Send,
{
    let (mut converted, model) = convert_request::<T>(request.body)?;
    // This endpoint always accumulates a complete response, including usage,
    // even when the input request originally selected streaming transport.
    converted.stream = false;
    let (output, finish_reason) = match request.output.split_once(SpecialToken::Eos.default_str()) {
        Some((output, _)) => (output, InferenceFinishReason::Stop),
        None => (request.output.as_str(), request.finish_reason.into()),
    };
    // The stream parser expects output that starts after the prompt's
    // assistant-turn start (`<｜Assistant｜><think>` or `</think>`). The demo
    // accepts complete assistant output, so it strips that start here.
    let output = output
        .strip_prefix(SpecialToken::Assistant.default_str())
        .unwrap_or(output);
    let (thinking, output) = match output.strip_prefix(SpecialToken::ThinkingStart.default_str()) {
        Some(output) => (true, output),
        None => (
            false,
            output
                .strip_prefix(SpecialToken::ThinkingEnd.default_str())
                .unwrap_or(output),
        ),
    };
    converted.conversation.thinking_mode = thinking;
    converted.parsing_options.thinking_initial_stage =
        thinking.then_some(ThinkingInitialStage::ReasoningStart);
    converted.parsing_options.starts_in_tool_calls = false;
    let id = "demo-id".to_owned();
    let generator = T::chunk_generator(&converted, id.clone(), model.clone());
    let processor = StreamProcessor::new(generator, converted.parsing_options);
    let inference = tokio_stream::iter([
        InferenceChunk::Text {
            content: output.to_owned(),
            // Raw text alone does not supply tokenizer or backend usage data.
            token_count: 0,
        },
        InferenceChunk::Finish { finish_reason },
    ]);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let mut response = T::Response::new(id, model, created, 0, 0);
    let mut chunks = pin!(processor.into_protocol_stream(inference));
    while let Some(chunk) = chunks.next().await {
        response
            .accumulate(chunk.map_err(|error| {
                conversion_failed(ConversionError::internal(error.to_string()))
            })?);
    }
    serde_json::to_value(response)
        .map_err(|error| conversion_failed(ConversionError::internal(error.to_string())))
}

async fn parse_handler(
    Json(request): Json<ParseRequest>,
) -> Result<Json<ParseResponse>, DemoError> {
    let segments = segment(&request.output);
    let response = match request.format {
        ApiFormat::ChatCompletions => parse_output::<ChatCompletionRequest>(request).await?,
        ApiFormat::Responses => parse_output::<ResponsesRequest>(request).await?,
        ApiFormat::Messages => parse_output::<MessagesRequest>(request).await?,
    };
    Ok(Json(ParseResponse { response, segments }))
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
        .route("/api/parse", post(parse_handler));

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("deepseek-recipe demo page: http://{addr}");
    axum::serve(listener, app).await.unwrap();
}
