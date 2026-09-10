use std::net::SocketAddr;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::ChunkGenerator;
use serde::{Serialize, de::DeserializeOwned};
use server_rs::{
    ConversionError, ErrorResponse, PreparedResponse, complete_response,
    complete_responses_response, prepare_chat_completion_response, prepare_converted_response,
    prepare_responses_response,
};

type ApiError = (StatusCode, Json<ErrorResponse>);

async fn api_handler<T>(Json(req): Json<T>) -> Result<Response, ApiError>
where
    T: ProtocolRequest + DeserializeOwned,
    <<T::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk: Serialize + Send,
{
    let request = req
        .convert(ConversionOptions::default())
        .map_err(http_error)?;
    if !request.stream {
        let response = complete_response::<T>(request).await.map_err(http_error)?;
        return Ok(Json(response).into_response());
    }
    prepare_converted_response::<T>(request)
        .await
        .map(sse_response)
        .map_err(http_error)
}

async fn chat_completion_handler(
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if request.stream == Some(true) {
        return prepare_chat_completion_response(request)
            .await
            .map(sse_response)
            .map_err(http_error);
    }
    api_handler(Json(request)).await
}

async fn responses_handler(Json(request): Json<ResponsesRequest>) -> Result<Response, ApiError> {
    if request.stream == Some(true) {
        return prepare_responses_response(request)
            .await
            .map(sse_response)
            .map_err(http_error);
    }
    complete_responses_response(request)
        .await
        .map(|response| Json(response).into_response())
        .map_err(http_error)
}

fn http_error(error: ConversionError) -> ApiError {
    let status =
        StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(error.into_response()))
}

fn sse_response(prepared: PreparedResponse) -> Response {
    let mut response = Response::new(Body::from_stream(prepared.events));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response
}

#[tokio::main]
async fn main() {
    let addr: SocketAddr = std::env::var("SERVER_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:7777".to_string())
        .parse()
        .expect("invalid SERVER_ADDR");
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completion_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/messages", post(api_handler::<MessagesRequest>));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
