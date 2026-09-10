"""FastAPI example with an application-supplied inference backend."""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from time import time
from uuid import uuid4

from anyio import CancelScope
from deepseek_recipe import (
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    CalcResizeError,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversationRequest,
    ConversionError,
    ConversionOptions,
    DeepseekV4Encoding,
    DeepseekV41Encoding,
    ImageError,
    ImageInfo,
    ImageQuota,
    ImageResolver,
    InferenceChunk,
    InferenceFinishReason,
    InferenceOptions,
    MessagesRequest,
    MessagesResponse,
    OpenCvImagePreprocessor,
    ReqwestImageFetcher,
    ResponsesRequest,
    ResponsesResponse,
    StreamProcessor,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

PROTOCOL_TYPES = {
    "chat_completions": (ChatCompletionRequest, ChatCompletionResponse),
    "responses": (ResponsesRequest, ResponsesResponse),
    "messages": (MessagesRequest, MessagesResponse),
}


@dataclass(frozen=True)
class PreparedRequest:
    """The example server's rendered input and protocol response settings."""

    protocol: str
    conversation_request: ConversationRequest
    prompt: str
    images: list[ImageInfo]
    image_token_adjustment: int
    include_usage: bool = False
    custom_tool_names: frozenset[str] = frozenset()

    @property
    def model(self) -> str | None:
        return self.conversation_request.model

    @property
    def stream(self) -> bool:
        return self.conversation_request.stream

    @property
    def inference_options(self) -> InferenceOptions:
        return self.conversation_request.inference_options


Inference = Callable[[PreparedRequest], AsyncIterator[InferenceChunk]]


class RequestError(Exception):
    """A request preparation failure with an HTTP status chosen by the server."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def prepare_request(
    protocol: str,
    body: bytes,
    *,
    model_version: str = "v4.1",
    options: ConversionOptions | None = None,
) -> PreparedRequest:
    """Convert a protocol request, render its prompt, and resolve its images.

    `options` configures Rust conversion defaults, including the web-search
    behavior of the Responses and Messages protocols. It defaults to
    `ConversionOptions()`, which ignores Responses `web_search` declarations and
    rejects Messages `web_search` declarations.
    """
    request_type, _ = PROTOCOL_TYPES[protocol]
    try:
        request = request_type(body)
    except ValueError as error:
        raise RequestError(str(error)) from error
    include_usage = request.include_usage() if protocol == "chat_completions" else False
    custom_tool_names = (
        frozenset(request.custom_tool_names()) if protocol == "responses" else frozenset()
    )
    converted = request.convert(options if options is not None else ConversionOptions())
    encoding_type = {"v4": DeepseekV4Encoding, "v4.1": DeepseekV41Encoding}[model_version]
    rendered = encoding_type().render_conversation(converted.conversation)
    images = []
    image_token_adjustment = 0
    if rendered.image_sources:
        try:
            resolver = ImageResolver(ReqwestImageFetcher(), OpenCvImagePreprocessor())
            multimodal = resolver.resolve(rendered.image_sources, ImageQuota())
        except ImageError as error:
            status_code = 500 if error.kind in {"Client", "TokenBudget"} else 400
            raise RequestError(str(error), status_code) from error
        images = multimodal.images
        try:
            image_token_adjustment = multimodal.image_token_adjustment()
        except CalcResizeError as error:
            raise RequestError(str(error), 500) from error
    return PreparedRequest(
        protocol,
        converted,
        rendered.prompt,
        images,
        image_token_adjustment,
        include_usage,
        custom_tool_names,
    )


class InferenceStreamingResponse(StreamingResponse):
    """Close the response iterator after a disconnect or a failed send."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


async def mock_inference(request: PreparedRequest) -> AsyncIterator[InferenceChunk]:
    """Produce a fixed answer for the example server."""
    yield InferenceChunk.ready()
    if request.prompt.endswith(THINKING_START_TOKEN):
        yield InferenceChunk.text(THINKING_END_TOKEN, content_tokens=1)
    yield InferenceChunk.text("Hello ", content_tokens=1)
    yield InferenceChunk.text("world!", content_tokens=1)
    yield InferenceChunk.finish(finish_reason=InferenceFinishReason.Stop)


async def response_body(
    request: PreparedRequest, infer: Inference
) -> AsyncIterator[str]:
    request_type, response_type = PROTOCOL_TYPES[request.protocol]
    response_id = str(uuid4())
    model = request.model or "deepseek-flash"
    generator = request_type.chunk_generator(request.conversation_request, response_id, model)
    if request.protocol == "chat_completions":
        generator = generator.with_include_usage(request.include_usage)
    elif request.protocol == "responses":
        generator = generator.with_custom_tool_names(request.custom_tool_names)
    processor = StreamProcessor(generator, request.conversation_request.parsing_options)
    response = None if request.stream else response_type(response_id, model, int(time()), 0, 0)
    inference = None
    try:
        inference = aiter(infer(request))
        while True:
            try:
                chunk = await anext(inference)
            except StopAsyncIteration:
                chunks = processor.finish()
            else:
                chunks = processor.push(chunk)
            for output in chunks:
                if request.stream:
                    yield sse_frame(response_type.chunk_event_type(output), output.to_json())
                else:
                    response.append(output)
            if processor.finished:
                break
        if request.stream:
            done = response_type.done_message()
            if done is not None:
                yield sse_frame(None, done)
        else:
            yield response.to_json()
    finally:
        try:
            close = getattr(inference, "aclose", None)
            if close is not None:
                with CancelScope(shield=True):
                    await close()
        finally:
            processor.close()


def sse_frame(event: str | None, data: str) -> str:
    """Format one protocol chunk or completion marker for HTTP streaming."""
    prefix = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {data}\n\n"


def create_app(
    infer: Inference = mock_inference, *, options: ConversionOptions | None = None
) -> FastAPI:
    """Create a server using an asynchronous inference iterator.

    `options` is passed to every request conversion. It defaults to
    `ConversionOptions()`, which keeps the Rust conversion defaults.

    The backend owns cancellation protection for asynchronous cleanup inside
    its own iterator. The server protects the explicit aclose() call.
    """
    app = FastAPI(title="deepseek-recipe", version="0.1.0")

    @app.exception_handler(ConversionError)
    async def conversion_error_handler(_request: Request, exc: ConversionError) -> Response:
        return Response(
            content=exc.body, status_code=exc.status_code, media_type="application/json"
        )

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Request, exc: RequestError) -> Response:
        error_type = "internal_error" if exc.status_code >= 500 else "invalid_request_error"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {
                "message": str(exc), "type": error_type, "param": None, "code": error_type,
            }},
        )

    def api_handler(protocol: str) -> Callable[[Request], Awaitable[Response]]:
        async def handler(request: Request) -> Response:
            body = await request.body()
            prepared = await run_in_threadpool(prepare_request, protocol, body, options=options)
            output = response_body(prepared, infer)
            if prepared.stream:
                return InferenceStreamingResponse(output, media_type="text/event-stream")
            content = "".join([part async for part in output])
            return Response(content, media_type="application/json")

        return handler

    for path, protocol in (
        ("/v1/chat/completions", "chat_completions"),
        ("/v1/responses", "responses"),
        ("/v1/messages", "messages"),
    ):
        app.add_api_route(path, api_handler(protocol), methods=["POST"], name=protocol)
    return app


app = create_app()
