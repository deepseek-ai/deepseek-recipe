from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

__version__: str

IMAGE_SPECIAL_TOKEN_ID: int

ToolChoice: TypeAlias = Literal["auto", "none", "required"]
ReasoningEffort: TypeAlias = Literal["low", "high", "xhigh", "max"]
ResponseFormat: TypeAlias = Literal["text", "json_object", "json_schema", "regex"]
ImageDetail: TypeAlias = Literal["low", "high", "original", "auto"]
Role: TypeAlias = Literal["system", "user", "assistant", "tool", "latest_reminder"]
ImageSourceKind: TypeAlias = Literal["data_url", "url", "bytes"]
RequestBody: TypeAlias = bytes | str | dict[str, Any]

class SpecialToken:
    """A kind of special token."""

    Bos: SpecialToken
    Eos: SpecialToken
    ThinkingStart: SpecialToken
    ThinkingEnd: SpecialToken
    System: SpecialToken
    User: SpecialToken
    Assistant: SpecialToken
    LatestReminder: SpecialToken
    Dsml: SpecialToken
    Image: SpecialToken
    Ref: SpecialToken
    RefEnd: SpecialToken
    Box: SpecialToken
    BoxEnd: SpecialToken
    Point: SpecialToken
    PointEnd: SpecialToken
    Polygon: SpecialToken
    PolygonEnd: SpecialToken

    def default_str(self) -> str:
        """Return the default literal spelling of this token."""

class SpecialTokens:
    """A mapping from each `SpecialToken` to its spelling.

    Every keyword defaults to the default spelling of its token, so only
    overridden tokens need to be supplied.
    """

    def __init__(
        self,
        *,
        bos: str | None = None,
        eos: str | None = None,
        thinking_start: str | None = None,
        thinking_end: str | None = None,
        system: str | None = None,
        user: str | None = None,
        assistant: str | None = None,
        latest_reminder: str | None = None,
        dsml: str | None = None,
        image: str | None = None,
        ref: str | None = None,
        ref_end: str | None = None,
        box: str | None = None,
        box_end: str | None = None,
        point: str | None = None,
        point_end: str | None = None,
        polygon: str | None = None,
        polygon_end: str | None = None,
    ) -> None: ...
    def token(self, token: SpecialToken) -> str:
        """Return the spelling of `token` in this mapping."""

class ConversionError(Exception):
    """Raised when a protocol request is invalid or cannot be converted."""

    status_code: int
    body: str

class ImageError(Exception):
    """A failure to resolve, fetch, or preprocess images, including request limits.

    `kind` names the Rust `ImageError` variant, and `is_retryable` holds its
    retry classification.
    """

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
    """A client tool definition.

    `strict` is passed through. Direct construction does not normalize
    `parameters`, unlike conversion, which normalizes the parameters of a tool
    with `strict: true`.
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        parameters: Any = None,
        strict: bool | None = None,
        defer_loading: bool = False,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str | None: ...
    @property
    def parameters(self) -> Any:
        """Input JSON Schema of the tool."""

    @property
    def strict(self) -> bool | None:
        """The `strict` setting, passed through without normalizing `parameters`."""
    @property
    def defer_loading(self) -> bool: ...

class ImageSource:
    """Base class of the image sources supplied with a conversation.

    Every source records a `path`: the request location of the image as error
    messages report it, such as `messages.0.content[1]`. Conversion fills it; a
    source constructed directly has an empty path unless the `path` keyword
    supplies one.
    """

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
    @property
    def path(self) -> str:
        """Request location of the image; empty when the source has none."""

class DataUrlImageSource(ImageSource):
    """An image referenced by a data URL with a base64 or percent-encoded body."""

    def __init__(self, data_url: str, detail: ImageDetail = "high", *, path: str = "") -> None: ...

class UrlImageSource(ImageSource):
    """An image referenced by an external URL."""

    def __init__(self, url: str, detail: ImageDetail = "high", *, path: str = "") -> None: ...

class BytesImageSource(ImageSource):
    """An image supplied as encoded bytes."""

    def __init__(self, data: bytes, detail: ImageDetail = "high", *, path: str = "") -> None: ...

class Message:
    """Base class of the messages in a conversation.

    Every message has a role and content. On a message of another role, the
    role-specific properties have fixed values: `reasoning_content`,
    `tool_calls`, and `tool_call_id` are `None`, `prefix` is `False`, and
    `image_sources` is an empty list.
    """

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
    @property
    def prefix(self) -> bool:
        """Whether an assistant message is continued instead of starting a new turn."""

class SystemMessage(Message):
    """A system message.

    See `UserMessage` for image placeholder rules.
    """

    def __init__(
        self,
        content: str,
        image_sources: Sequence[ImageSource] | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        *,
        image_token: str | None = None,
    ) -> None: ...
    @property
    def tools(self) -> list[ToolDefinition] | None: ...

class UserMessage(Message):
    """A user message.

    The first occurrences of `image_token` in `content`, one per image source,
    become the placeholders of the image sources; later occurrences are plain
    text. `image_token` defaults to the default image token spelling; a caller
    that renders with a `SpecialTokens` mapping whose image token differs
    passes that mapping's spelling. When `content` holds fewer occurrences than
    there are image sources, the message keeps every image source, so a
    rendered prompt holds fewer placeholders than image sources; construction
    does not report the mismatch. An empty `image_token` raises `ValueError`.
    """

    def __init__(
        self,
        content: str,
        image_sources: Sequence[ImageSource] | None = None,
        *,
        image_token: str | None = None,
    ) -> None: ...

class AssistantMessage(Message):
    """An assistant message."""

    def __init__(
        self,
        content: str,
        reasoning_content: str | None = None,
        tool_calls: Sequence[ToolCall] | None = None,
        prefix: bool = False,
    ) -> None: ...

class ToolMessage(Message):
    """A tool result message.

    The first occurrences of `image_token` in `content`, one per image source,
    become the placeholders of the image sources; later occurrences are plain
    text. `image_token` defaults to the default image token spelling; a caller
    that renders with a `SpecialTokens` mapping whose image token differs
    passes that mapping's spelling. When `content` holds fewer occurrences than
    there are image sources, the message keeps every image source, so a
    rendered prompt holds fewer placeholders than image sources; construction
    does not report the mismatch. An empty `image_token` raises `ValueError`.
    """

    def __init__(
        self,
        content: str,
        tool_call_id: str | None = None,
        image_sources: Sequence[ImageSource] | None = None,
        *,
        image_token: str | None = None,
    ) -> None: ...

class LatestReminderMessage(Message):
    """A reminder message that carries additional instructions."""

    def __init__(self, content: str) -> None: ...

class Conversation:
    """A conversation and its prompt configuration.

    Thinking mode is enabled by default, tool selection defaults to `auto`, and
    the response format defaults to `text`. `response_format` accepts `text` or
    `json_object`; `response_schema` selects the `json_schema` format with that
    JSON Schema, and `response_regex` selects the `regex` format with that
    regular expression. Supplying both, or either with `json_object`, raises
    `ValueError`.
    """

    def __init__(
        self,
        messages: Sequence[Message] | None = None,
        *,
        thinking_mode: bool = True,
        tools: Sequence[ToolDefinition] | None = None,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: ReasoningEffort | None = None,
        response_format: Literal["text", "json_object"] = "text",
        response_schema: Any | None = None,
        response_regex: str | None = None,
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
    def response_format(self) -> ResponseFormat:
        """Requested answer format; the caller enforces it during inference."""
    @property
    def response_schema(self) -> Any | None:
        """The JSON Schema of a `json_schema` response format, or `None`."""
    @property
    def response_regex(self) -> str | None:
        """The regular expression of a `regex` response format, or `None`."""

class RenderedPrompt:
    """A rendered prompt and the image sources its placeholders refer to."""

    @property
    def prompt(self) -> str: ...
    @property
    def special_ranges(self) -> list[tuple[int, int]]:
        """Ascending, disjoint ranges of `prompt` that hold explicit special tokens.

        Each range is a half-open `(start, end)` pair of code point offsets into
        `prompt` (Python `str` indices), so `prompt[start:end]` is the spelling of
        one token. The Rust `RenderedPrompt::special_ranges` field holds byte
        offsets instead.

        The text between the ranges is plain text, which prompt encoding encodes
        without the tokenizer's added vocabulary. `Tokenizer.encode` uses the full
        vocabulary, so it encodes a special-token spelling in plain text as that
        token's ID. `DeepseekV4Encoding.encode` and `DeepseekV41Encoding.encode`
        perform prompt encoding.
        """
    @property
    def image_sources(self) -> list[ImageSource]: ...

class DeepseekV4Encoding:
    """The V4 prompt encoding.

    It renders a conversation and, with an attached tokenizer, encodes it into
    token IDs.
    """

    def __init__(self) -> None: ...
    def with_tokenizer(self, tokenizer: Tokenizer) -> DeepseekV4Encoding:
        """Return a new prompt encoding that uses `tokenizer` for `encode`."""
    def with_special_tokens(self, special_tokens: SpecialTokens) -> DeepseekV4Encoding:
        """Return a new prompt encoding that renders prompts with `special_tokens`."""
    def render_conversation(self, conversation: Conversation) -> RenderedPrompt: ...
    def encode(self, conversation: Conversation) -> list[int]:
        """Encode a conversation into token IDs using the attached tokenizer.

        Raises `RuntimeError` when no tokenizer is attached or encoding fails.
        """

class DeepseekV41Encoding:
    """The V4.1 prompt encoding.

    It renders a conversation and, with an attached tokenizer, encodes it into
    token IDs.
    """

    def __init__(self) -> None: ...
    def with_tokenizer(self, tokenizer: Tokenizer) -> DeepseekV41Encoding:
        """Return a new prompt encoding that uses `tokenizer` for `encode`."""
    def with_special_tokens(self, special_tokens: SpecialTokens) -> DeepseekV41Encoding:
        """Return a new prompt encoding that renders prompts with `special_tokens`."""
    def render_conversation(self, conversation: Conversation) -> RenderedPrompt: ...
    def encode(self, conversation: Conversation) -> list[int]:
        """Encode a conversation into token IDs using the attached tokenizer.

        Raises `RuntimeError` when no tokenizer is attached or encoding fails.
        """

class WebSearchBehavior:
    """Handling of supported web-search declarations, choices, and Messages blocks."""

    Drop: WebSearchBehavior
    """Drop matching declarations and choices, plus Messages server-tool blocks."""
    Reject: WebSearchBehavior
    """Reject matching declarations and choices, plus Messages server-tool blocks."""

class ReasoningContentBehavior:
    """Handling of assistant history missing reasoning content in thinking mode."""

    FillEmpty: ReasoningContentBehavior
    """Fill missing reasoning content with an empty string."""
    Reject: ReasoningContentBehavior
    """Reject the request when reasoning content that must be passed back is missing."""

class ModelVersion:
    """The model version a converted conversation targets.

    Conversion resolves it from the request's model name through
    `ConversionOptions.model_versions_by_name`, and it selects the prompt
    encoding: `DeepseekV4Encoding` or `DeepseekV41Encoding`.
    """

    DeepseekV4Pro: ModelVersion
    """DeepSeek V4 Pro, rendered by `DeepseekV4Encoding`."""
    DeepseekV41Flash: ModelVersion
    """DeepSeek V4.1 Flash, rendered by `DeepseekV41Encoding`."""

class ConversionOptions:
    """Options for protocol request conversion.

    By default, thinking mode is enabled, missing reasoning content is filled
    with an empty string, Responses `web_search` is dropped, Messages
    `web_search` is rejected, the model version is resolved from the request's
    model name (`DeepSeek-V4-Pro` or `DeepSeek-V4.1-Flash`), and the
    `max_tokens` upper bound is 393216 in thinking mode and outside it.
    """

    def __init__(
        self,
        *,
        default_thinking_mode: bool = True,
        reasoning_content_behavior: ReasoningContentBehavior = ReasoningContentBehavior.FillEmpty,
        responses_web_search: WebSearchBehavior = WebSearchBehavior.Drop,
        messages_web_search: WebSearchBehavior = WebSearchBehavior.Reject,
        thinking_max_tokens_upper_bound: int = 393216,
        non_thinking_max_tokens_upper_bound: int = 393216,
    ) -> None: ...
    @property
    def default_thinking_mode(self) -> bool:
        """Whether requests that do not specify a thinking mode use thinking mode.

        Defaults to `True`.
        """
    @property
    def model_versions_by_name(self) -> dict[str, ModelVersion]:
        """The target model version of each accepted request model name.

        Defaults to `DeepSeek-V4-Pro` and `DeepSeek-V4.1-Flash`; conversion
        rejects a request whose model is not a key.
        """
    @property
    def thinking_max_tokens_upper_bound(self) -> int:
        """The largest accepted `max_tokens` (`max_output_tokens` in Responses)
        of a request in thinking mode.

        Chat Completions and Responses reject a larger value; Messages lowers it
        to this bound.
        """
    @property
    def non_thinking_max_tokens_upper_bound(self) -> int:
        """The largest accepted `max_tokens` (`max_output_tokens` in Responses)
        of a request outside thinking mode.

        Chat Completions and Responses reject a larger value; Messages lowers it
        to this bound.
        """
    @property
    def reasoning_content_behavior(self) -> ReasoningContentBehavior:
        """Missing-reasoning-content behavior, defaulting to `FillEmpty`."""
    @property
    def responses_web_search(self) -> WebSearchBehavior:
        """Responses web-search behavior, defaulting to `Drop`."""
    @property
    def messages_web_search(self) -> WebSearchBehavior:
        """Messages web-search behavior, defaulting to `Reject`."""
    @property
    def special_tokens(self) -> SpecialTokens:
        """Special token mapping, defaulting to the built-in mapping."""
    def with_default_thinking_mode(self, default_thinking_mode: bool) -> ConversionOptions:
        """Construct options with the supplied default thinking mode."""
    def with_reasoning_content_behavior(
        self, reasoning_content_behavior: ReasoningContentBehavior
    ) -> ConversionOptions:
        """Construct options with the supplied missing-reasoning-content behavior."""
    def with_responses_web_search(
        self, responses_web_search: WebSearchBehavior
    ) -> ConversionOptions:
        """Construct options with the supplied Responses web-search behavior."""
    def with_messages_web_search(
        self, messages_web_search: WebSearchBehavior
    ) -> ConversionOptions:
        """Construct options with the supplied Messages web-search behavior."""
    def with_special_tokens(self, special_tokens: SpecialTokens) -> ConversionOptions:
        """Construct options with the supplied special token mapping."""
    def with_model_alias(self, model: str, model_version: ModelVersion) -> ConversionOptions:
        """Accept `model` as a name of `model_version`.

        Adds the name, or replaces the version of an existing one.
        """
    def with_model_versions_by_name(
        self, model_versions_by_name: dict[str, ModelVersion]
    ) -> ConversionOptions:
        """Replace the whole table of accepted model names.

        Drops the default names unless the mapping includes them.
        """
    def with_thinking_max_tokens_upper_bound(
        self, thinking_max_tokens_upper_bound: int
    ) -> ConversionOptions: ...
    def with_non_thinking_max_tokens_upper_bound(self, non_thinking_max_tokens_upper_bound: int) -> ConversionOptions: ...

class ConversationRequest:
    """A Rust conversation request with inference and parsing options."""

    def __init__(self, conversation: Conversation, model_version: ModelVersion) -> None:
        """Construct a request with default inference, parsing, and transport options."""
    @property
    def conversation(self) -> Conversation: ...
    @property
    def inference_options(self) -> InferenceOptions: ...
    @property
    def parsing_options(self) -> ParsingOptions: ...
    @property
    def model_version(self) -> ModelVersion:
        """The target model version.

        Conversion resolves it from the request's model name; the constructor
        takes it as an argument.
        """
    @property
    def stream(self) -> bool:
        """Whether the protocol request asks for a streamed response."""
    def resolve_max_tokens(self, prompt_tokens: int, context_limit: int) -> int:
        """Resolve the `max_tokens` to send to the backend.

        `prompt_tokens` is the number of tokens in the encoded prompt, with each
        image counted at its image token count instead of as one placeholder,
        and `context_limit` is the number of tokens the model's context holds.
        A requested `max_tokens` is kept; an unspecified one becomes 8192
        outside thinking mode, 131072 in thinking mode with `max` effort, and
        65536 in thinking mode otherwise. The result is lowered to the context
        left after the prompt and raised to at least 1. Raises `ConversionError`
        when `prompt_tokens` plus the requested `max_tokens` exceeds
        `context_limit`, or, without a requested `max_tokens`, when
        `prompt_tokens` alone exceeds it.
        """

class ChatCompletionRequest:
    """A Rust Chat Completions request; even failed conversion consumes it."""

    def __init__(self, body: RequestBody) -> None: ...
    @property
    def model(self) -> str:
        """The requested model name; read it before `convert` consumes the request."""
    def convert(self, options: ConversionOptions) -> ConversationRequest: ...
    @staticmethod
    def chunk_generator(
        request: ConversationRequest, id: str, model: str
    ) -> ChatCompletionChunkGenerator: ...

class ResponsesRequest:
    """A Rust Responses request; even failed conversion consumes it."""

    def __init__(self, body: RequestBody) -> None: ...
    @property
    def model(self) -> str:
        """The requested model name; read it before `convert` consumes the request."""
    def namespace_tool_map(self, options: ConversionOptions) -> dict[str, tuple[str, str]]:
        """Return the function names of the declared namespace tools.

        Each function name maps to `(namespace, local name)`, as conversion with
        `options` assigns them; read them before `convert` consumes this request.
        Raises `ConversionError` for an invalid tool declaration.
        """
    def echo(self) -> ResponsesEcho:
        """Return the request settings that responses report.

        Read them before `convert` consumes this request.
        """

    def convert(self, options: ConversionOptions) -> ConversationRequest: ...
    @staticmethod
    def chunk_generator(
        request: ConversationRequest, id: str, model: str
    ) -> ResponsesChunkGenerator: ...

class MessagesRequest:
    """A Rust Messages request; even failed conversion consumes it."""

    def __init__(self, body: RequestBody) -> None: ...
    @property
    def model(self) -> str:
        """The requested model name; read it before `convert` consumes the request."""
    def convert(self, options: ConversionOptions) -> ConversationRequest: ...
    @staticmethod
    def chunk_generator(
        request: ConversationRequest, id: str, model: str
    ) -> MessagesChunkGenerator: ...

class InferenceOptions:
    """Inference parameters extracted by protocol conversion.

    Conversion always sets `temperature` (1.0 in thinking mode) and `top_p` to
    their resolved values. Other unspecified parameters are `None`;
    `ConversationRequest.resolve_max_tokens` supplies the default `max_tokens`,
    and the caller applies the other model-specific defaults.
    """

    @property
    def max_tokens(self) -> int | None:
        """Maximum number of output tokens, or `None` when the request sets none.

        `ConversationRequest.resolve_max_tokens` supplies the default after the
        prompt is encoded.
        """
    @property
    def temperature(self) -> float | None:
        """Sampling temperature: 1.0 in thinking mode, otherwise the requested value or 1.0."""
    @property
    def top_p(self) -> float | None:
        """Final `top_p` for the target model version."""
    @property
    def logprobs(self) -> bool:
        """Whether the request asks for the log probability of each output token."""
    @property
    def top_logprobs(self) -> int | None:
        """Number of most likely alternatives reported per output token, in `[0, 20]`."""
    @property
    def seed(self) -> int | None:
        """Sampling seed in `[0, 2^63)`; the caller chooses an unspecified seed."""
    @property
    def output_constraint(self) -> OutputConstraint | None:
        """The constraint that constrained decoding would enforce on the output,
        or `None`; whether the backend applies it is the caller's decision.
        """

class OutputConstraint:
    """A constraint on the model output that the backend can enforce by
    constrained decoding, as a kind name and a constraint string.
    """

    @property
    def kind(self) -> str:
        """`json`, `regex`, `tool_call`, or `tool_call_required`."""
    @property
    def schema(self) -> str:
        """The schema as compact JSON, the regular expression, the tool list as
        a compact JSON array of Chat Completions function tools with
        `defer_loading: true` on deferred tools, or an empty string for any JSON
        value.
        """

class ImageInfo:
    """A preprocessed image for the inference backend.

    It holds the encoded image bytes and the image dimensions in pixels after
    preprocessing.
    """

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
        """Return the number of inference input tokens beyond one per placeholder.

        The V4.1 image token specification gives the image token count of each
        image, which occupies one placeholder in the prompt; the adjustment is the
        sum of the differences. Raises `CalcResizeError` when fitting an image
        does not converge.
        """

class ImageQuota:
    """The image count and encoded bytes recorded by the completed resolve calls of one request.

    To apply the image count and total byte limits across the resolve calls of a
    request, the caller must pass the same quota to every call. A call records
    its sources after it preprocesses every one of them, so a failed call adds
    nothing.
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
    """An image fetcher that downloads external image URLs through the Rust `reqwest` client.

    Without a destination filter, this fetcher connects to any address. When
    untrusted clients can supply image URLs, the caller should install a filter
    with `with_destination_filter` to protect against SSRF.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def with_max_bytes(max_bytes: int) -> ReqwestImageFetcher: ...
    def with_destination_filter(
        self, destination_filter: Callable[[str, str], bool]
    ) -> ReqwestImageFetcher:
        """Return a fetcher with this size limit that connects only to allowed addresses.

        `destination_filter(host, address)` receives the host of the image URL or
        of a redirect target and one address as a string, such as `"10.0.0.1"`,
        and returns whether the fetcher can connect to it. For a host name, the
        fetcher connects only to the resolved addresses the filter allows. A host
        with no allowed address raises `ImageError` with kind
        `"DestinationRejected"`. A callable that raises or returns a non-boolean
        value rejects the address. The returned fetcher uses no proxy. The
        library does not include a filter implementation; the caller writes the
        rule. Raises `TypeError` when `destination_filter` is not callable.
        """
    def fetch(self, url: str) -> bytes: ...

class OpenCvImagePreprocessor:
    """An image preprocessor that uses the Rust OpenCV implementation."""

    def __init__(self) -> None: ...
    def preprocess(self, data: bytes, options: PreprocessOptions) -> ImageInfo: ...

class ImageResolver:
    """An image resolver built from explicit fetching and preprocessing components."""

    def __init__(
        self, fetcher: ReqwestImageFetcher, preprocessor: OpenCvImagePreprocessor
    ) -> None: ...
    def resolve(self, sources: Sequence[ImageSource], quota: ImageQuota) -> MultiModalData:
        """Resolve image sources and update the supplied quota.

        A failed call leaves the quota unchanged.
        """

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
    """One backend update, constructed with `ready`, `text`, `tokens`, or `finish`."""

    @staticmethod
    def ready(
        *, prompt_usage: PromptUsage | None = None, system_fingerprint: str | None = None,
    ) -> InferenceChunk: ...
    @staticmethod
    def text(content: str, token_count: int = 0) -> InferenceChunk:
        """Construct a text chunk from `content` and the number of tokens it accounts for."""
    @staticmethod
    def tokens(
        token_ids: Sequence[int],
        logprobs: Sequence[tuple[float, Sequence[tuple[int, float]]]] | None = None,
    ) -> InferenceChunk:
        """Construct a chunk from the token IDs of one backend output.

        `logprobs` holds one `(logprob, top_logprobs)` pair per token ID.
        """
    @staticmethod
    def finish(finish_reason: InferenceFinishReason) -> InferenceChunk: ...
    @property
    def kind(self) -> Literal["ready", "text", "tokens", "finish"]: ...
    @property
    def content(self) -> str | None: ...
    @property
    def token_ids(self) -> list[int] | None:
        """The token IDs of a `tokens` chunk."""
    @property
    def finish_reason(self) -> InferenceFinishReason | None: ...
    @property
    def prompt_usage(self) -> PromptUsage | None: ...
    @property
    def token_count(self) -> int | None:
        """The number of tokens this chunk's content accounts for; text chunks only."""
    @property
    def system_fingerprint(self) -> str | None: ...

class ThinkingInitialStage:
    """The stage where parsing begins in thinking mode."""

    ReasoningStart: ThinkingInitialStage
    """Begin the reasoning content of a new assistant turn, discarding leading newlines."""
    ReasoningContinuation: ThinkingInitialStage
    """Continue the reasoning content of a prefix continuation, preserving leading newlines."""
    ContentContinuation: ThinkingInitialStage
    """Continue the answer content of a prefix continuation whose reasoning content has ended.

    Leading newlines are preserved.
    """

    def starts_in_reasoning(self) -> bool:
        """Return whether parsing begins in reasoning content."""

class ParsingOptions:
    """Settings passed directly to the Rust stream parser."""

    def __init__(
        self,
        *,
        parse_tool_calls: bool = True,
        starts_in_tool_calls: bool = False,
        parse_json_output: bool = False,
        thinking_initial_stage: ThinkingInitialStage | None = None,
        stop_sequences: Sequence[str] | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None: ...
    parse_tool_calls: bool
    """Whether to parse model-specific tool-call markup."""
    starts_in_tool_calls: bool
    """Whether parsing begins inside the tool-call block that the prompt opened.

    Used for a required tool choice; takes precedence over `thinking_initial_stage`.
    """
    parse_json_output: bool
    """Whether to recognize fenced or bare JSON output and replace the text around it with spaces."""
    thinking_initial_stage: ThinkingInitialStage | None
    """The stage where parsing begins in thinking mode; `None` means thinking mode is off.

    A stage also makes the processor report reasoning tokens.
    """
    @property
    def stop_sequences(self) -> list[str]: ...
    @stop_sequences.setter
    def stop_sequences(self, value: Sequence[str]) -> None: ...
    @property
    def special_tokens(self) -> SpecialTokens: ...
    @special_tokens.setter
    def special_tokens(self, value: SpecialTokens) -> None: ...

class ChatCompletionChunkGenerator:
    """A Rust Chat Completions generator consumed by a stream processor."""

class ResponsesChunkGenerator:
    """A Rust Responses generator consumed by a stream processor."""

    def with_namespace_tool_map(
        self, namespace_tool_map: dict[str, tuple[str, str]]
    ) -> ResponsesChunkGenerator:
        """Consume this generator and construct one that splits namespace tool names."""
    def with_echo(self, echo: ResponsesEcho) -> ResponsesChunkGenerator:
        """Consume this generator and construct one whose responses report the request settings."""

class ResponsesEcho:
    """Request settings that Responses response objects report, as the request specified them.

    `ResponsesRequest.echo` reads them from a request; the constructor builds
    the settings of an unspecified request. The properties expose the scalar
    settings; the reasoning, text, tool choice, and tool settings are reported
    only through the response.
    """

    def __init__(self) -> None:
        """Construct the settings of an unspecified request.

        No instructions, output token limit, or user; temperature and `top_p` of
        1.0; zero `top_logprobs`.
        """
    @property
    def instructions(self) -> str | None: ...
    @property
    def max_output_tokens(self) -> int | None: ...
    @property
    def temperature(self) -> float:
        """The requested temperature or 1.0, not the sampling temperature."""
    @property
    def top_p(self) -> float:
        """The requested `top_p` or 1.0, not the sampling `top_p`."""
    @property
    def top_logprobs(self) -> int:
        """The requested `top_logprobs`, or 0."""
    @property
    def user(self) -> str | None: ...

class MessagesChunkGenerator:
    """A Rust Messages generator consumed by a stream processor."""

    def __init__(self, id: str, model: str, thinking_mode: bool) -> None: ...
    def with_signature(self, signature: str) -> MessagesChunkGenerator:
        """Consume this generator and construct one with the supplied signature.

        The new generator emits `signature` at the end of each thinking block. The
        default signature is the response ID.
        """

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
    def accumulate(self, chunk: ChatCompletionChunk) -> None:
        """Consume this chunk and apply its delta to the response."""

    def to_json(self) -> str: ...
    @staticmethod
    def sse_event_name(chunk: ChatCompletionChunk) -> str | None:
        """Return the SSE event name of `chunk`, or `None` for an unnamed data event."""
    @staticmethod
    def sse_done_data() -> str | None:
        """Return the data of the final SSE message, `[DONE]`."""

class ResponsesResponse:
    """A Rust Responses response with explicit accumulation."""

    def __init__(
        self, id: str, model: str, created: int, prompt_tokens: int, prompt_cache_hit_tokens: int
    ) -> None: ...
    def accumulate(self, chunk: ResponsesStreamEvent) -> None:
        """Consume this event and apply its delta to the response."""

    def to_json(self) -> str: ...
    @staticmethod
    def sse_event_name(chunk: ResponsesStreamEvent) -> str | None:
        """Return the SSE event name of `chunk`."""
    @staticmethod
    def sse_done_data() -> str | None:
        """Return `None`: a Responses stream has no final data message."""

class MessagesResponse:
    """A Rust Messages response with explicit accumulation."""

    def __init__(
        self, id: str, model: str, created: int, prompt_tokens: int, prompt_cache_hit_tokens: int
    ) -> None: ...
    def accumulate(self, chunk: MessagesStreamEvent) -> None:
        """Consume this event and apply its delta to the response."""

    def to_json(self) -> str: ...
    @staticmethod
    def sse_event_name(chunk: MessagesStreamEvent) -> str | None:
        """Return the SSE event name of `chunk`."""
    @staticmethod
    def sse_done_data() -> str | None:
        """Return `None`: a Messages stream has no final data message."""

class Tokenizer:
    """A Hugging Face tokenizer loaded from a `tokenizer.json` file.

    A prompt encoding uses it, after `with_tokenizer`, to encode conversations into
    token IDs. A stream processor uses it to decode the token IDs of `tokens`
    inference chunks.
    """

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
        """Download and cache `tokenizer.json` from the Hugging Face Hub."""

    def encode(self, text: str) -> list[int]:
        """Encode `text` into token IDs with the full vocabulary, including the added vocabulary.

        A special-token spelling in `text` encodes as that token's ID. The
        post-processor adds no special tokens, such as a BOS token. This is not
        prompt encoding, which encodes plain text without the added vocabulary:
        `DeepseekV4Encoding.encode` and `DeepseekV41Encoding.encode` perform
        prompt encoding.
        """

class StreamProcessor:
    """Adapt Python inference chunks to the Rust processor's input stream."""

    def __init__(
        self,
        generator: ChatCompletionChunkGenerator | ResponsesChunkGenerator | MessagesChunkGenerator,
        options: ParsingOptions,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Consume the generator and construct a Rust stream processor.

        The optional tokenizer decodes `tokens` chunks. Without one, such a chunk
        raises `RuntimeError` and closes the processor.
        """

    @property
    def finished(self) -> bool:
        """Whether processing completed normally and the processor remains unclosed."""

    def push(
        self, chunk: InferenceChunk
    ) -> list[ChatCompletionChunk | ResponsesStreamEvent | MessagesStreamEvent]:
        """Process one chunk.

        Raises `RuntimeError` after completion or close, or on a stream error: a
        `tokens` chunk without a tokenizer, a decoding failure, or a log
        probability count that differs from the token ID count. A stream error
        closes the processor.
        """

    def finish(self) -> list[ChatCompletionChunk | ResponsesStreamEvent | MessagesStreamEvent]:
        """Close the input and process buffered output; repeated calls after completion yield no chunks.

        Calls after close or a stream error raise `RuntimeError`.
        """

    def close(self) -> None:
        """Release processing state without producing successful completion."""
