from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

__version__: str

BOS_TOKEN: str
EOS_TOKEN: str
SYSTEM_SP_TOKEN: str
USER_SP_TOKEN: str
ASSISTANT_SP_TOKEN: str
LATEST_REMINDER_SP_TOKEN: str
THINKING_START_TOKEN: str
THINKING_END_TOKEN: str
DSML_SP_TOKEN: str
IMAGE_SPECIAL_TOKEN: str
IMAGE_SPECIAL_TOKEN_ID: int

ToolChoice: TypeAlias = Literal["auto", "none", "required"]
ReasoningEffort: TypeAlias = Literal["low", "high", "xhigh", "max"]
ResponseFormat: TypeAlias = Literal["text", "json_object"]
ImageDetail: TypeAlias = Literal["low", "high", "original", "auto"]
Role: TypeAlias = Literal["system", "user", "assistant", "tool", "latest_reminder"]
ImageSourceKind: TypeAlias = Literal["data_url", "url", "bytes"]
RequestBody: TypeAlias = bytes | str | dict[str, Any]

class ConversionError(Exception):
    """Raised when a protocol request is invalid or cannot be converted."""

    status_code: int
    body: str

class ImageError(Exception):
    """A Rust image error with its original variant and retry classification."""

    kind: str
    is_retryable: bool

class CalcResizeError(ValueError):
    """The Rust image token calculation did not converge."""

class ToolCall:
    """A historical tool call and its arguments."""

    def __init__(self, id: str, name: str, arguments: str) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> str:
        """Caller-supplied arguments, expected to encode a JSON object; not validated."""

class ToolDefinition:
    """A client tool definition."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        parameters: Any = None,
        strict: bool | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str | None: ...
    @property
    def parameters(self) -> Any:
        """Input JSON Schema of the tool."""

    @property
    def strict(self) -> bool | None: ...

class ImageSource:
    """Base class of the images supplied with a conversation."""

    @property
    def kind(self) -> ImageSourceKind: ...
    @property
    def detail(self) -> ImageDetail: ...
    @property
    def data_url(self) -> str | None: ...
    @property
    def url(self) -> str | None: ...
    @property
    def data(self) -> bytes | None: ...

class DataUrlImageSource(ImageSource):
    """An image referenced by a data URL with a base64 or percent-encoded body."""

    def __init__(self, data_url: str, detail: ImageDetail = "high") -> None: ...

class UrlImageSource(ImageSource):
    """An image referenced by an external URL."""

    def __init__(self, url: str, detail: ImageDetail = "high") -> None: ...

class BytesImageSource(ImageSource):
    """An image supplied as encoded bytes."""

    def __init__(self, data: bytes, detail: ImageDetail = "high") -> None: ...

class Message:
    """Base class of the messages in a conversation."""

    @property
    def role(self) -> Role: ...
    @property
    def content(self) -> str: ...
    @property
    def reasoning_content(self) -> str | None: ...
    @property
    def tool_calls(self) -> list[ToolCall] | None: ...
    @property
    def tool_call_id(self) -> str | None: ...
    @property
    def image_sources(self) -> list[ImageSource]: ...

class SystemMessage(Message):
    """A system message."""

    def __init__(self, content: str) -> None: ...

class UserMessage(Message):
    """A user message."""

    def __init__(
        self, content: str, image_sources: Sequence[ImageSource] | None = None
    ) -> None: ...

class AssistantMessage(Message):
    """An assistant message."""

    def __init__(
        self,
        content: str,
        reasoning_content: str | None = None,
        tool_calls: Sequence[ToolCall] | None = None,
    ) -> None: ...

class ToolMessage(Message):
    """A tool result message."""

    def __init__(
        self,
        content: str,
        tool_call_id: str,
        image_sources: Sequence[ImageSource] | None = None,
    ) -> None: ...

class LatestReminderMessage(Message):
    """A reminder message that carries additional instructions."""

    def __init__(self, content: str) -> None: ...

class Conversation:
    """A conversation and its prompt configuration."""

    def __init__(
        self,
        messages: Sequence[Message] | None = None,
        *,
        thinking_mode: bool = True,
        tools: Sequence[ToolDefinition] | None = None,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: ReasoningEffort | None = None,
        response_format: ResponseFormat = "text",
    ) -> None: ...
    @property
    def messages(self) -> list[Message]: ...
    @property
    def thinking_mode(self) -> bool: ...
    @property
    def tools(self) -> list[ToolDefinition]: ...
    @property
    def tool_choice(self) -> ToolChoice: ...
    @property
    def reasoning_effort(self) -> ReasoningEffort | None: ...
    @property
    def response_format(self) -> ResponseFormat: ...

class RenderedPrompt:
    """A rendered prompt and the images its placeholders refer to."""

    @property
    def prompt(self) -> str: ...
    @property
    def image_sources(self) -> list[ImageSource]: ...

class DeepseekV4Encoding:
    """The Rust V4 prompt renderer. Token encoding requires an attached tokenizer."""

    def __init__(self) -> None: ...
    def with_tokenizer(self, tokenizer: Tokenizer) -> DeepseekV4Encoding: ...
    def render_conversation(self, conversation: Conversation) -> RenderedPrompt: ...
    def encode(self, conversation: Conversation) -> list[int]: ...

class DeepseekV41Encoding:
    """The Rust V4.1 prompt renderer. Token encoding requires an attached tokenizer."""

    def __init__(self) -> None: ...
    def with_tokenizer(self, tokenizer: Tokenizer) -> DeepseekV41Encoding: ...
    def render_conversation(self, conversation: Conversation) -> RenderedPrompt: ...
    def encode(self, conversation: Conversation) -> list[int]: ...

class WebSearchBehavior:
    """Handling of supported web-search declarations, choices, and Messages blocks."""

    Ignore: WebSearchBehavior
    Reject: WebSearchBehavior

class ConversionOptions:
    """Conversion defaults: Responses web search is ignored; Messages rejects it."""

    def __init__(
        self,
        *,
        default_thinking_mode: bool = True,
        responses_web_search: WebSearchBehavior = WebSearchBehavior.Ignore,
        messages_web_search: WebSearchBehavior = WebSearchBehavior.Reject,
    ) -> None: ...
    @property
    def default_thinking_mode(self) -> bool: ...
    @property
    def responses_web_search(self) -> WebSearchBehavior: ...
    @property
    def messages_web_search(self) -> WebSearchBehavior: ...
    def with_default_thinking_mode(self, default_thinking_mode: bool) -> ConversionOptions: ...
    def with_responses_web_search(
        self, responses_web_search: WebSearchBehavior
    ) -> ConversionOptions: ...
    def with_messages_web_search(
        self, messages_web_search: WebSearchBehavior
    ) -> ConversionOptions: ...

class ConversationRequest:
    """A Rust conversation request with inference and parsing options."""

    def __init__(self, conversation: Conversation) -> None: ...
    @property
    def conversation(self) -> Conversation: ...
    @property
    def inference_options(self) -> InferenceOptions: ...
    @property
    def parsing_options(self) -> ParsingOptions: ...
    @property
    def model(self) -> str | None: ...
    @property
    def stream(self) -> bool: ...

class ChatCompletionRequest:
    """A Rust Chat Completions request; even failed conversion consumes it."""

    def __init__(self, body: RequestBody) -> None: ...
    def include_usage(self) -> bool:
        """Read the stream usage setting before consuming this request."""

    def convert(self, options: ConversionOptions) -> ConversationRequest: ...
    @staticmethod
    def chunk_generator(
        request: ConversationRequest, id: str, model: str
    ) -> ChatCompletionChunkGenerator: ...

class ResponsesRequest:
    """A Rust Responses request; even failed conversion consumes it."""

    def __init__(self, body: RequestBody) -> None: ...
    def custom_tool_names(self) -> set[str]:
        """Read the declared custom tool names before consuming this request."""

    def convert(self, options: ConversionOptions) -> ConversationRequest: ...
    @staticmethod
    def chunk_generator(
        request: ConversationRequest, id: str, model: str
    ) -> ResponsesChunkGenerator: ...

class MessagesRequest:
    """A Rust Messages request; even failed conversion consumes it."""

    def __init__(self, body: RequestBody) -> None: ...
    def convert(self, options: ConversionOptions) -> ConversationRequest: ...
    @staticmethod
    def chunk_generator(
        request: ConversationRequest, id: str, model: str
    ) -> MessagesChunkGenerator: ...

class InferenceOptions:
    """Converted inference parameters; the backend applies unspecified defaults."""

    @property
    def max_tokens(self) -> int | None: ...
    @property
    def temperature(self) -> float | None: ...
    @property
    def top_p(self) -> float | None: ...
    @property
    def thinking_budget_tokens(self) -> int | None: ...
    @property
    def disable_parallel_tool_use(self) -> bool | None: ...

class ImageInfo:
    """Encoded image bytes and their dimensions in pixels."""

    def __init__(self, data: bytes, width: int, height: int) -> None: ...
    @property
    def data(self) -> bytes: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class MultiModalData:
    """Images in the order their placeholders appear in a prompt."""

    def __init__(self, images: Sequence[ImageInfo] | None = None) -> None: ...
    @property
    def images(self) -> list[ImageInfo]: ...
    def is_empty(self) -> bool: ...
    def image_token_adjustment(self) -> int:
        """Image input tokens beyond one token per prompt placeholder."""

class ImageQuota:
    """Source count and encoded bytes recorded across resolve calls.

    A call records its sources after it preprocesses every one of them, so a
    failed call adds nothing.
    """

    def __init__(self) -> None: ...
    def image_count(self) -> int: ...
    def byte_size(self) -> int: ...

class PreprocessOptions:
    """Options passed to one Rust image preprocessing operation."""

    def __init__(
        self, detail: ImageDetail, max_dimension_px: int, low_detail_max_dimension_px: int
    ) -> None: ...
    @property
    def detail(self) -> ImageDetail: ...
    @property
    def max_dimension_px(self) -> int: ...
    @property
    def low_detail_max_dimension_px(self) -> int: ...

class ReqwestImageFetcher:
    """Fetches external image URLs through the Rust HTTP client."""

    def __init__(self) -> None: ...
    @staticmethod
    def with_max_bytes(max_bytes: int) -> ReqwestImageFetcher: ...
    def fetch(self, url: str) -> bytes: ...

class OpenCvImagePreprocessor:
    """Preprocesses encoded images through the Rust OpenCV implementation."""

    def __init__(self) -> None: ...
    def preprocess(self, data: bytes, options: PreprocessOptions) -> ImageInfo: ...

class ImageResolver:
    """Resolves sources using explicit Rust image components and request quota."""

    def __init__(
        self, fetcher: ReqwestImageFetcher, preprocessor: OpenCvImagePreprocessor
    ) -> None: ...
    def resolve(self, sources: Sequence[ImageSource], quota: ImageQuota) -> MultiModalData:
        """Resolve images and update the supplied quota; a failed call leaves it unchanged."""

class PromptUsage:
    """Prompt-side token counts from the backend."""

    def __init__(
        self, *, prompt_tokens: int = 0, prompt_cache_hit_tokens: int = 0,
    ) -> None: ...
    @property
    def prompt_tokens(self) -> int: ...
    @property
    def prompt_cache_hit_tokens(self) -> int: ...

class InferenceFinishReason:
    """Backend finish reasons; output parsing determines the protocol finish reason."""

    Stop: InferenceFinishReason
    Length: InferenceFinishReason
    ContentFilter: InferenceFinishReason

class InferenceChunk:
    """One backend update, constructed with ready(), text(), token(), or finish()."""

    @staticmethod
    def ready(
        *, prompt_usage: PromptUsage | None = None, system_fingerprint: str | None = None,
    ) -> InferenceChunk: ...
    @staticmethod
    def text(text: str, content_tokens: int = 0) -> InferenceChunk: ...
    @staticmethod
    def token(token_id: int) -> InferenceChunk: ...
    @staticmethod
    def finish(finish_reason: InferenceFinishReason) -> InferenceChunk: ...
    @property
    def kind(self) -> Literal["ready", "text", "token", "finish"]: ...
    @property
    def content(self) -> str | None: ...
    @property
    def token_id(self) -> int | None: ...
    @property
    def finish_reason(self) -> InferenceFinishReason | None: ...
    @property
    def prompt_usage(self) -> PromptUsage | None: ...
    @property
    def content_tokens(self) -> int | None:
        """The tokens this chunk's content accounts for; text chunks only."""
    @property
    def system_fingerprint(self) -> str | None: ...

class ReasoningStage:
    """Position within reasoning content when parsing begins."""

    Start: ReasoningStage
    Reasoning: ReasoningStage
    Content: ReasoningStage

    def start_from_reasoning(self) -> bool: ...

class ParsingOptions:
    """Settings passed directly to the Rust stream parser."""

    def __init__(
        self,
        *,
        parse_tool_calls: bool = True,
        tool_call_initial_stage: bool = False,
        parse_json_output: bool = False,
        reasoning_initial_stage: ReasoningStage | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> None: ...
    parse_tool_calls: bool
    tool_call_initial_stage: bool
    parse_json_output: bool
    reasoning_initial_stage: ReasoningStage | None
    @property
    def stop_sequences(self) -> list[str]: ...
    @stop_sequences.setter
    def stop_sequences(self, value: Sequence[str]) -> None: ...

class ChatCompletionChunkGenerator:
    """A Rust Chat Completions generator consumed by a stream processor."""

    def __init__(
        self, id: str, model: str, include_usage: bool, thinking_mode: bool
    ) -> None: ...
    def with_include_usage(self, include_usage: bool) -> ChatCompletionChunkGenerator:
        """Consume this generator and construct one with the supplied usage setting."""

class ResponsesChunkGenerator:
    """A Rust Responses generator consumed by a stream processor."""

    def __init__(self, id: str, model: str) -> None: ...
    def with_custom_tool_names(
        self, custom_tool_names: set[str] | frozenset[str]
    ) -> ResponsesChunkGenerator:
        """Consume this generator and construct one with the supplied custom tool names."""

class MessagesChunkGenerator:
    """A Rust Messages generator consumed by a stream processor."""

    def __init__(self, id: str, model: str, thinking_mode: bool) -> None: ...
    def with_signature(self, signature: str) -> MessagesChunkGenerator:
        """Consume this generator and construct one with the supplied signature."""

class ChatCompletionChunk:
    """A Rust Chat Completions chunk consumed by response accumulation."""

    def to_json(self) -> str: ...

class ResponsesStreamEvent:
    """A Rust Responses event consumed by response accumulation."""

    def to_json(self) -> str: ...

class MessagesStreamEvent:
    """A Rust Messages event consumed by response accumulation."""

    def to_json(self) -> str: ...

class ChatCompletionResponse:
    """A Rust Chat Completions response with explicit accumulation."""

    def __init__(
        self, id: str, model: str, created: int, prompt_tokens: int, prompt_cache_hit_tokens: int
    ) -> None: ...
    def append(self, chunk: ChatCompletionChunk) -> None:
        """Consume this chunk and apply its delta to the response."""

    def to_json(self) -> str: ...
    @staticmethod
    def chunk_event_type(chunk: ChatCompletionChunk) -> str | None: ...
    @staticmethod
    def done_message() -> str | None: ...

class ResponsesResponse:
    """A Rust Responses response with explicit accumulation."""

    def __init__(
        self, id: str, model: str, created: int, prompt_tokens: int, prompt_cache_hit_tokens: int
    ) -> None: ...
    def append(self, chunk: ResponsesStreamEvent) -> None:
        """Consume this event and apply its delta to the response."""

    def to_json(self) -> str: ...
    @staticmethod
    def chunk_event_type(chunk: ResponsesStreamEvent) -> str | None: ...
    @staticmethod
    def done_message() -> str | None: ...

class MessagesResponse:
    """A Rust Messages response with explicit accumulation."""

    def __init__(
        self, id: str, model: str, created: int, prompt_tokens: int, prompt_cache_hit_tokens: int
    ) -> None: ...
    def append(self, chunk: MessagesStreamEvent) -> None:
        """Consume this event and apply its delta to the response."""

    def to_json(self) -> str: ...
    @staticmethod
    def chunk_event_type(chunk: MessagesStreamEvent) -> str | None: ...
    @staticmethod
    def done_message() -> str | None: ...

class Tokenizer:
    """A HuggingFace tokenizer loaded from a `tokenizer.json` file."""

    @staticmethod
    def from_file(path: str) -> Tokenizer:
        """Load a tokenizer from a `tokenizer.json` file."""

    @staticmethod
    def from_str(text: str) -> Tokenizer:
        """Load a tokenizer from the contents of a `tokenizer.json` file."""

    @staticmethod
    def from_pretrained(
        identifier: str, revision: str | None = None, token: str | None = None
    ) -> Tokenizer:
        """Download and cache `tokenizer.json` from the HuggingFace Hub."""

    def encode(self, text: str) -> list[int]: ...

class StreamProcessor:
    """Adapt Python inference chunks to the Rust processor's input stream."""

    def __init__(
        self,
        generator: ChatCompletionChunkGenerator | ResponsesChunkGenerator | MessagesChunkGenerator,
        options: ParsingOptions,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Consume the generator and construct a Rust stream processor.

        The optional tokenizer decodes token-id chunks. Without one, a token
        chunk raises RuntimeError and closes the processor.
        """

    @property
    def finished(self) -> bool:
        """Whether processing completed normally and the processor remains unclosed."""

    def push(
        self, chunk: InferenceChunk
    ) -> list[ChatCompletionChunk | ResponsesStreamEvent | MessagesStreamEvent]:
        """Process one chunk; raises RuntimeError after completion, close, or decode failure."""

    def finish(self) -> list[ChatCompletionChunk | ResponsesStreamEvent | MessagesStreamEvent]:
        """Process EOF; repeated calls after completion yield no chunks.

        Calls after close or a decoding failure raise RuntimeError.
        """

    def close(self) -> None:
        """Release processing state without producing successful completion."""
