"""Smoke tests for the Rust conversion, rendering, response, and image bindings."""

import http.server
import json
import sys
import threading
from pathlib import Path

import pytest

from deepseek_recipe import (
    IMAGE_SPECIAL_TOKEN_ID,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Conversation,
    ConversionError,
    ConversionOptions,
    DeepseekV4Encoding,
    DeepseekV41Encoding,
    ImageError,
    ImageQuota,
    ImageResolver,
    InferenceChunk,
    InferenceFinishReason,
    MessagesRequest,
    MessagesResponse,
    ModelVersion,
    OpenCvImagePreprocessor,
    PromptUsage,
    ReasoningContentBehavior,
    ReqwestImageFetcher,
    ResponsesRequest,
    ResponsesResponse,
    SpecialToken,
    SpecialTokens,
    StreamProcessor,
    SystemMessage,
    ThinkingInitialStage,
    Tokenizer,
    ToolDefinition,
    ToolMessage,
    UrlImageSource,
    UserMessage,
    WebSearchBehavior,
)

REQUEST = {
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "hi"}],
}
RESPONSES_REQUEST = {"model": "DeepSeek-V4.1-Flash", "input": "hi"}

TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "tokenizers" / "v41" / "tokenizer.json"
)


def run_processor(generator, parsing_options, chunks, tokenizer=None):
    """Push `chunks` through a stream processor and return every protocol chunk."""
    processor = StreamProcessor(generator, parsing_options, tokenizer)
    outputs = []
    try:
        for chunk in chunks:
            outputs.extend(processor.push(chunk))
        outputs.extend(processor.finish())
    finally:
        processor.close()
    return outputs


def answer_chunks(content, finish_reason=InferenceFinishReason.Stop):
    """Return the inference chunks of one answer that closes an empty reasoning block."""
    return [
        InferenceChunk.ready(),
        InferenceChunk.text(SpecialToken.ThinkingEnd.default_str() + content, token_count=1),
        InferenceChunk.finish(finish_reason),
    ]


def test_rendering():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    encoding = DeepseekV41Encoding()
    rendered = encoding.render_conversation(Conversation(messages=[UserMessage("hi")]))

    assert converted.conversation.messages[0].content == "hi"
    assert "hi" in rendered.prompt
    assert rendered.image_sources == []
    assert encoding.render_conversation(converted.conversation).prompt == rendered.prompt


def test_system_message_images():
    source = UrlImageSource("https://example.com/a.png")
    message = SystemMessage("<｜image｜>", [source])
    conversation = Conversation(messages=[message])
    rendered = DeepseekV41Encoding().render_conversation(conversation)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ids = DeepseekV41Encoding().with_tokenizer(tokenizer).encode(conversation)

    assert rendered.image_sources[0].url == source.url
    assert ids.count(IMAGE_SPECIAL_TOKEN_ID) == 1


def test_special_tokens_mapping():
    assert SpecialTokens().token(SpecialToken.User) == SpecialToken.User.default_str()

    custom = SpecialTokens(user="<u>", image="IMG")
    configured = ConversionOptions().with_special_tokens(custom)
    assert configured.special_tokens.token(SpecialToken.User) == "<u>"

    rendered = (
        DeepseekV41Encoding()
        .with_special_tokens(custom)
        .render_conversation(Conversation(messages=[UserMessage("hi")]))
    )
    assert "<u>" in rendered.prompt

    converted = ChatCompletionRequest(
        {
            "model": "DeepSeek-V4.1-Flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        },
                    ],
                }
            ],
        }
    ).convert(configured)
    assert converted.conversation.messages[0].content == "hi\n\nIMG"


def test_encoding_with_tokenizer():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    with pytest.raises(RuntimeError, match="no tokenizer"):
        DeepseekV41Encoding().encode(converted.conversation)

    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    prompt_ids = encoding.encode(converted.conversation)

    # The prompt holds no special-token spelling in plain text, so encoding
    # with the full vocabulary gives the same IDs.
    assert prompt_ids == tokenizer.encode(encoding.render_conversation(converted.conversation).prompt)


def test_prompt_encoding_differs_from_tokenizer_on_special_token_spellings():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    conversation = Conversation(messages=[UserMessage("a<｜User｜>")], thinking_mode=False)
    rendered = encoding.render_conversation(conversation)
    (user_token_id,) = tokenizer.encode(SpecialToken.User.default_str())

    prompt_ids = encoding.encode(conversation)
    full_vocabulary_ids = tokenizer.encode(rendered.prompt)

    # Prompt encoding encodes the spelling in the message as plain text; only
    # the template's token is the special token ID. `Tokenizer.encode` encodes
    # both occurrences as that ID.
    assert prompt_ids.count(user_token_id) == 1
    assert full_vocabulary_ids.count(user_token_id) == 2
    assert prompt_ids != full_vocabulary_ids

def test_conversation_constructor_structured_output_formats():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with_schema = Conversation(messages=[UserMessage("hi")], response_schema=schema)
    assert with_schema.response_format == "json_schema"
    assert with_schema.response_schema == schema
    assert with_schema.response_regex is None
    # Rendering writes the schema into the prompt.
    assert '"properties"' in DeepseekV41Encoding().render_conversation(with_schema).prompt

    with_regex = Conversation(response_regex="[0-9]+")
    assert with_regex.response_format == "regex"
    assert with_regex.response_regex == "[0-9]+"
    assert with_regex.response_schema is None

    assert Conversation(response_format="json_object").response_format == "json_object"
    with pytest.raises(ValueError, match="cannot be combined"):
        Conversation(response_schema=schema, response_regex="[0-9]+")
    with pytest.raises(ValueError, match="cannot be combined"):
        Conversation(response_format="json_object", response_schema=schema)
    with pytest.raises(ValueError, match="cannot be combined"):
        Conversation(response_format="json_object", response_regex="[0-9]+")


@pytest.mark.parametrize(
    "model,deferred",
    [
        ("DeepSeek-V4-Pro", False),
        ("DeepSeek-V4.1-Flash", True),
    ],
)
def test_deferred_loading_follows_model_version(model, deferred):
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "system",
                "content": [
                    {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "lookup"}},
                ],
            },
        ],
        "tools": [
            {"name": "search", "input_schema": {"type": "object"}},
            {"name": "lookup", "input_schema": {"type": "object"}, "defer_loading": True},
        ],
    }
    converted = MessagesRequest(body).convert(ConversionOptions())
    encoding = DeepseekV41Encoding() if deferred else DeepseekV4Encoding()
    prompt = encoding.render_conversation(converted.conversation).prompt

    assert converted.conversation.tools[1].defer_loading is deferred
    assert ("### Updated Tool Schemas" in prompt) is deferred
    assert '"name": "lookup"' in prompt


def test_web_search_conversion_options():
    defaults = ConversionOptions()
    constructed = ConversionOptions(
        default_thinking_mode=False,
        responses_web_search=WebSearchBehavior.Reject,
        messages_web_search=WebSearchBehavior.Drop,
    )
    configured = (
        defaults.with_default_thinking_mode(False)
        .with_responses_web_search(WebSearchBehavior.Reject)
        .with_messages_web_search(WebSearchBehavior.Drop)
    )
    assert defaults.default_thinking_mode
    assert defaults.responses_web_search == WebSearchBehavior.Drop
    assert defaults.messages_web_search == WebSearchBehavior.Reject
    for options in (constructed, configured):
        assert not options.default_thinking_mode
        assert options.responses_web_search == WebSearchBehavior.Reject
        assert options.messages_web_search == WebSearchBehavior.Drop

    responses = {
        "model": "DeepSeek-V4.1-Flash",
        "input": "hi",
        "tools": [{"type": "web_search"}],
    }
    messages = {
        **REQUEST,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    assert ResponsesRequest(responses).convert(defaults).conversation.tools == []
    with pytest.raises(ConversionError, match="server tools are not supported"):
        MessagesRequest(messages).convert(defaults)

    converted = MessagesRequest(messages).convert(configured)
    assert converted.conversation.tools == []
    assert not converted.conversation.thinking_mode
    with pytest.raises(ConversionError, match="Server tools are not supported"):
        ResponsesRequest(responses).convert(configured)


def test_reasoning_content_conversion_options():
    defaults = ConversionOptions()
    constructed = ConversionOptions(
        reasoning_content_behavior=ReasoningContentBehavior.Reject
    )
    configured = defaults.with_reasoning_content_behavior(
        ReasoningContentBehavior.Reject
    )
    assert defaults.reasoning_content_behavior == ReasoningContentBehavior.FillEmpty
    for options in (constructed, configured):
        assert options.reasoning_content_behavior == ReasoningContentBehavior.Reject

    request = {
        "model": "DeepSeek-V4.1-Flash",
        "thinking": {"type": "enabled"},
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_00_abcdefghijklmnopqrst0000",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_00_abcdefghijklmnopqrst0000",
                "content": "ok",
            },
        ],
    }
    with pytest.raises(ConversionError, match="must be passed back"):
        ChatCompletionRequest(request).convert(constructed)

    converted = ChatCompletionRequest(request).convert(defaults)
    assistant = converted.conversation.messages[1]
    assert assistant.reasoning_content == ""


def test_model_version_mapping():
    # The model name selects the version; the table defaults to the model names.
    defaults = ConversionOptions()
    assert defaults.model_versions_by_name == {
        "DeepSeek-V4-Pro": ModelVersion.DeepseekV4Pro,
        "DeepSeek-V4.1-Flash": ModelVersion.DeepseekV41Flash,
    }

    overridden = defaults.with_model_alias("custom-model", ModelVersion.DeepseekV4Pro)
    assert overridden.model_versions_by_name == {
        **defaults.model_versions_by_name,
        "custom-model": ModelVersion.DeepseekV4Pro,
    }

    bulk = defaults.with_model_versions_by_name(
        {"custom-model": ModelVersion.DeepseekV4Pro, "another-model": ModelVersion.DeepseekV41Flash}
    )
    assert bulk.model_versions_by_name == {
        "custom-model": ModelVersion.DeepseekV4Pro,
        "another-model": ModelVersion.DeepseekV41Flash,
    }
    with pytest.raises(TypeError):
        defaults.with_model_alias("custom-model", "DeepSeek-V4-Pro")

    body = {"model": "custom-model", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(ConversionError, match="supported API model names"):
        ChatCompletionRequest(body).convert(defaults)
    assert ChatCompletionRequest(body).convert(overridden).model_version == ModelVersion.DeepseekV4Pro
    ChatCompletionRequest(body).convert(bulk)
    converted = ChatCompletionRequest({**body, "model": "DeepSeek-V4.1-Flash"}).convert(defaults)
    assert converted.model_version == ModelVersion.DeepseekV41Flash
    with pytest.raises(ConversionError, match="supported API model names"):
        ChatCompletionRequest({**body, "model": "DeepSeek-V4.1-Flash"}).convert(bulk)

    # A model version is hashable, so it can key a table of prompt encodings.
    encodings = {ModelVersion.DeepseekV4Pro: "v4", ModelVersion.DeepseekV41Flash: "v41"}
    assert encodings[converted.model_version] == "v41"

def test_protocol_requests_expose_model_until_converted():
    for request_type, body in (
        (ChatCompletionRequest, REQUEST),
        (ResponsesRequest, RESPONSES_REQUEST),
        (MessagesRequest, REQUEST),
    ):
        request = request_type(body)
        assert request.model == "DeepSeek-V4.1-Flash"
        request.convert(ConversionOptions())
        with pytest.raises(RuntimeError, match="consumed"):
            request.model
        with pytest.raises(RuntimeError, match="consumed"):
            request.convert(ConversionOptions())


def test_output_constraints_reach_inference_options():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    strict_tools = ChatCompletionRequest({
        **REQUEST,
        "thinking": {"type": "disabled"},
        "tools": [{"type": "function", "function": {"name": "f", "strict": True, "parameters": schema}}],
    }).convert(ConversionOptions())
    constraint = strict_tools.inference_options.output_constraint
    assert constraint.kind == "tool_call"
    assert json.loads(constraint.schema)[0]["function"]["parameters"] == {
        **schema, "additionalProperties": False, "required": ["a"],
    }
    unconstrained = ResponsesRequest(RESPONSES_REQUEST).convert(ConversionOptions())
    assert unconstrained.inference_options.output_constraint is None


def test_resolve_max_tokens_reports_the_context_length_error():
    thinking = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    assert thinking.resolve_max_tokens(10, 131072) == 65536
    with pytest.raises(ConversionError, match="maximum context length is 131072 tokens") as rejected:
        thinking.resolve_max_tokens(131073, 131072)
    assert rejected.value.status_code == 400


def test_max_tokens_upper_bound_rejects_or_clamps():
    options = ConversionOptions(thinking_max_tokens_upper_bound=100, non_thinking_max_tokens_upper_bound=50)

    # A thinking request uses the thinking bound; `thinking: disabled` and
    # `reasoning.effort: none` select the other bound.
    with pytest.raises(ConversionError, match=r"valid range of max_tokens is \[1, 100\]"):
        ChatCompletionRequest({**REQUEST, "max_tokens": 200}).convert(options)
    with pytest.raises(ConversionError, match=r"valid range of max_tokens is \[1, 50\]"):
        ChatCompletionRequest(
            {**REQUEST, "max_tokens": 100, "thinking": {"type": "disabled"}}
        ).convert(options)
    with pytest.raises(ConversionError, match=r"valid range of max_tokens is \[1, 100\]"):
        ResponsesRequest({**RESPONSES_REQUEST, "max_output_tokens": 200}).convert(options)
    with pytest.raises(ConversionError, match=r"valid range of max_tokens is \[1, 50\]"):
        ResponsesRequest(
            {**RESPONSES_REQUEST, "max_output_tokens": 100, "reasoning": {"effort": "none"}}
        ).convert(options)
    clamped = MessagesRequest({**REQUEST, "max_tokens": 200}).convert(options)
    assert clamped.inference_options.max_tokens == 100
    clamped = MessagesRequest(
        {**REQUEST, "max_tokens": 200, "thinking": {"type": "disabled"}}
    ).convert(options)
    assert clamped.inference_options.max_tokens == 50

def test_custom_inference():
    converted = ChatCompletionRequest({**REQUEST, "temperature": 0.25}).convert(
        ConversionOptions()
    )
    generator = ChatCompletionRequest.chunk_generator(converted, "smoke-response", "smoke-model")
    processor = StreamProcessor(generator, converted.parsing_options)
    response = ChatCompletionResponse("smoke-response", "smoke-model", 0, 0, 0)
    try:
        # Thinking mode samples at temperature 1.0.
        assert converted.inference_options.temperature == 1.0
        non_thinking = ChatCompletionRequest(
            {**REQUEST, "temperature": 0.25, "thinking": {"type": "disabled"}}
        ).convert(ConversionOptions())
        assert non_thinking.inference_options.temperature == 0.25
        for chunk in (
            InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=4)),
            InferenceChunk.text(
                SpecialToken.ThinkingEnd.default_str() + "Custom answer", token_count=2
            ),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
        body = json.loads(response.to_json())

        assert processor.finished
        assert body["id"] == "smoke-response"
        assert body["choices"][0]["message"]["content"] == "Custom answer"
        assert body["usage"]["prompt_tokens"] == 4
        assert body["usage"]["completion_tokens"] == 2
    finally:
        processor.close()


def test_responses_echo_request_settings():
    request = ResponsesRequest(
        {
            "model": "DeepSeek-V4.1-Flash",
            "input": "hi",
            "instructions": "be brief",
            "temperature": 0.5,
            "tools": [{"type": "function", "name": "f"}],
            "user": "u",
        }
    )
    converted = request.convert(ConversionOptions())
    generator = ResponsesRequest.chunk_generator(
        converted, "smoke-response", "smoke-model"
    )
    processor = StreamProcessor(generator, converted.parsing_options)
    response = ResponsesResponse("smoke-response", "smoke-model", 0, 0, 0)
    try:
        for chunk in (
            InferenceChunk.ready(),
            InferenceChunk.text("hello", token_count=1),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
        body = json.loads(response.to_json())

        assert body["instructions"] == "be brief"
        # The response reports the requested temperature, not the sampling one.
        assert converted.inference_options.temperature == 1.0
        assert body["temperature"] == 0.5
        assert body["tools"] == [
            {"type": "function", "name": "f", "description": None, "parameters": None, "strict": None}
        ]
        assert body["user"] == "u"
        assert body["top_p"] == 1.0
    finally:
        processor.close()


def test_token_chunks():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "token-response", "smoke-model")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    token_ids = tokenizer.encode(SpecialToken.ThinkingEnd.default_str() + "Custom answer")
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = ChatCompletionResponse("token-response", "smoke-model", 0, 0, 0)
    try:
        for chunk in (
            InferenceChunk.ready(),
            *(InferenceChunk.tokens([token_id]) for token_id in token_ids),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
        body = json.loads(response.to_json())

        assert body["choices"][0]["message"]["content"] == "Custom answer"
        assert body["usage"]["completion_tokens"] == len(token_ids)
    finally:
        processor.close()


def test_split_character_tokens():
    # A character outside the vocabulary decodes from several byte tokens; the
    # ids buffered while it was incomplete still count as completion tokens.
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "token-response", "smoke-model")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    token_ids = tokenizer.encode(SpecialToken.ThinkingEnd.default_str() + "🦀")
    assert len(token_ids) > 2
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = ChatCompletionResponse("token-response", "smoke-model", 0, 0, 0)
    try:
        for chunk in (
            InferenceChunk.ready(),
            *(InferenceChunk.tokens([token_id]) for token_id in token_ids),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
        body = json.loads(response.to_json())

        assert body["choices"][0]["message"]["content"] == "🦀"
        assert body["usage"]["completion_tokens"] == len(token_ids)
    finally:
        processor.close()


def test_undecodable_token_ignored():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "token-response", "smoke-model")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = ChatCompletionResponse("token-response", "smoke-model", 0, 0, 0)
    try:
        for chunk in (
            InferenceChunk.ready(),
            InferenceChunk.tokens([999_999_999]),
            InferenceChunk.text(SpecialToken.ThinkingEnd.default_str() + "Hi", token_count=1),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
        body = json.loads(response.to_json())

        assert body["choices"][0]["message"]["content"] == "Hi"
        assert body["usage"]["completion_tokens"] == 1
    finally:
        processor.close()


def test_token_chunk_without_tokenizer_errors():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "token-response", "smoke-model")
    processor = StreamProcessor(generator, converted.parsing_options)
    try:
        with pytest.raises(RuntimeError, match="without a tokenizer"):
            processor.push(InferenceChunk.tokens([1]))
    finally:
        processor.close()


def test_tokenizer_from_str():
    # Loading from the file's contents gives the same ids as loading the file.
    text = TOKENIZER_PATH.read_text()
    assert Tokenizer.from_str(text).encode("Custom answer") == Tokenizer.from_file(
        str(TOKENIZER_PATH)
    ).encode("Custom answer")


def test_tokenizer_from_str_errors():
    with pytest.raises(RuntimeError):
        Tokenizer.from_str("not json")


def test_foreign_tokenizer_errors():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "py-response", "smoke-model")
    with pytest.raises(TypeError):
        StreamProcessor(generator, converted.parsing_options, object())


def test_image_resolution():
    # One 50-by-40 RGB PNG exercises the native image decoder and WebP encoder.
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAADIAAAAoCAIAAAAzED4bAAAAPElEQVR4nO3OUQkAIBBA"
        "sQvht/2jGMsYPmSwAJu1T9A8H2hpaWlpaTVoaWkVaGlpFWhpaRVoaWkVaH3QuqTYo"
        "YpppISkAAAAAElFTkSuQmCC"
    )
    converted = ChatCompletionRequest({
        **REQUEST,
        "messages": [{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + png},
        }]}],
    }).convert(ConversionOptions())
    rendered = DeepseekV41Encoding().render_conversation(converted.conversation)
    resolver = ImageResolver(ReqwestImageFetcher(), OpenCvImagePreprocessor())
    multimodal = resolver.resolve(rendered.image_sources, ImageQuota())

    assert SpecialToken.Image.default_str() in rendered.prompt
    (image,) = multimodal.images
    assert image.width > 0 and image.height > 0
    assert image.data[:4] == b"RIFF" and image.data[8:12] == b"WEBP"
    assert multimodal.image_token_adjustment() > 0


UNSUPPORTED_IMAGE = (
    "You have uploaded an unsupported image. Please make sure your image is valid and has one "
    "of the following formats: webp, png, jpeg, and gif."
)
# Valid base64 with a PNG media type that does not decode as an image.
UNDECODABLE_PNG_DATA_URL = "data:image/png;base64,AAAAAAAA"
# A 1-by-1 PNG.
TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.mark.parametrize(
    ("request_type", "body", "expected"),
    [
        (
            ChatCompletionRequest,
            {
                **REQUEST,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + TINY_PNG}},
                    {"type": "image_url", "image_url": {"url": UNDECODABLE_PNG_DATA_URL}},
                ]}],
            },
            f".messages[0].image[1]: {UNSUPPORTED_IMAGE}",
        ),
        (
            MessagesRequest,
            {
                "model": "DeepSeek-V4.1-Flash",
                "max_tokens": 16,
                "messages": [
                    {"role": "system", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": TINY_PNG}},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAAAAAA"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": TINY_PNG}},
                    ]},
                ],
            },
            f"messages.0.image[1]: {UNSUPPORTED_IMAGE}",
        ),
        (
            ResponsesRequest,
            {
                "model": "DeepSeek-V4.1-Flash",
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64," + TINY_PNG},
                    {"type": "input_image", "image_url": UNDECODABLE_PNG_DATA_URL},
                ]}],
            },
            f"input[0].image[1]: {UNSUPPORTED_IMAGE}",
        ),
    ],
    ids=["chat", "messages", "responses"],
)
def test_image_errors_report_the_request_location(request_type, body, expected):
    converted = request_type(json.dumps(body)).convert(ConversionOptions())
    rendered = DeepseekV41Encoding().render_conversation(converted.conversation)
    assert all(source.path for source in rendered.image_sources)
    resolver = ImageResolver(ReqwestImageFetcher(), OpenCvImagePreprocessor())
    with pytest.raises(ImageError) as rejected:
        resolver.resolve(rendered.image_sources, ImageQuota())
    assert str(rejected.value) == expected
    assert rejected.value.kind == "UnsupportedMediaType"

def test_destination_filter_rejects_addresses():
    checked = []

    def reject_all(host, address):
        checked.append((host, address))
        return False

    fetcher = ReqwestImageFetcher().with_destination_filter(reject_all)
    with pytest.raises(ImageError) as rejected:
        fetcher.fetch("http://127.0.0.1:9/image.png")
    assert rejected.value.kind == "DestinationRejected"
    assert not rejected.value.is_retryable
    assert checked == [("127.0.0.1", "127.0.0.1")]

    with pytest.raises(ImageError) as rejected:
        fetcher.fetch("http://localhost:9/image.png")
    assert rejected.value.kind == "DestinationRejected"
    # The resolved address depends on the host's name resolution.
    assert any(host == "localhost" for host, _ in checked)


class ImageHandler(http.server.BaseHTTPRequestHandler):
    """Serve three bytes at any path, and redirect `/redirect` to `localhost`."""

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/image.png")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "3")
        self.end_headers()
        self.wfile.write(b"abc")

    def log_message(self, *args):
        pass


@pytest.fixture
def image_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_destination_filter_allows_addresses(image_server):
    url = f"http://127.0.0.1:{image_server.server_port}/image.png"
    fetcher = ReqwestImageFetcher().with_destination_filter(lambda host, address: True)
    assert fetcher.fetch(url) == b"abc"


def test_destination_filter_rejects_redirect_target(image_server):
    checked = []

    def loopback_only(host, address):
        checked.append(host)
        return host == "127.0.0.1"

    fetcher = ReqwestImageFetcher().with_destination_filter(loopback_only)
    assert fetcher.fetch(f"http://127.0.0.1:{image_server.server_port}/image.png") == b"abc"
    with pytest.raises(ImageError) as rejected:
        fetcher.fetch(f"http://127.0.0.1:{image_server.server_port}/redirect")
    assert rejected.value.kind == "DestinationRejected"
    # A fetch outside a resolve call has no request location, so the text
    # starts with the image index alone.
    assert str(rejected.value) == (
        f"image[0]: Failed to download image from http://127.0.0.1:{image_server.server_port}/redirect"
    )
    assert checked[-1] == "localhost"

def test_destination_filter_returning_non_bool_rejects(monkeypatch):
    reported = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)

    fetcher = ReqwestImageFetcher().with_destination_filter(lambda host, address: "yes")
    with pytest.raises(ImageError) as rejected:
        fetcher.fetch("http://127.0.0.1:9/image.png")
    assert rejected.value.kind == "DestinationRejected"
    assert [type(report.exc_value) for report in reported] == [TypeError]


def test_destination_filter_that_raises_rejects(monkeypatch):
    reported = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)

    def broken(host, address):
        raise RuntimeError("filter failure")

    fetcher = ReqwestImageFetcher().with_destination_filter(broken)
    with pytest.raises(ImageError) as rejected:
        fetcher.fetch("http://127.0.0.1:9/image.png")
    assert rejected.value.kind == "DestinationRejected"
    assert [type(report.exc_value) for report in reported] == [RuntimeError]


def test_circular_dict_rejected():
    # A self-referencing container must raise instead of exhausting the stack.
    parameters = {}
    parameters["self"] = parameters
    with pytest.raises(ValueError, match="circular reference"):
        ToolDefinition(name="f", parameters=parameters)


def test_circular_list_rejected():
    parameters = []
    parameters.append(parameters)
    with pytest.raises(ValueError, match="circular reference"):
        ToolDefinition(name="f", parameters=parameters)


def test_indirect_cycle_rejected():
    outer = {}
    inner = {"outer": outer}
    outer["inner"] = inner
    with pytest.raises(ValueError, match="circular reference"):
        ToolDefinition(name="f", parameters=outer)


def test_cycle_through_tuple_rejected():
    parameters = {"items": []}
    parameters["items"].append((parameters,))
    with pytest.raises(ValueError, match="circular reference"):
        ToolDefinition(name="f", parameters=parameters)


def test_circular_request_body_rejected():
    body = dict(REQUEST)
    body["self"] = body
    with pytest.raises(ValueError, match="circular reference"):
        ChatCompletionRequest(body)


def test_deep_nesting_rejected():
    parameters = {}
    node = parameters
    for _ in range(200):
        child = {}
        node["next"] = child
        node = child
    with pytest.raises(RecursionError):
        ToolDefinition(name="f", parameters=parameters)


def test_nesting_within_limit_accepted():
    parameters = {}
    node = parameters
    for _ in range(127):
        child = {}
        node["next"] = child
        node = child
    assert ToolDefinition(name="f", parameters=parameters).parameters == parameters


def test_shared_subtree_accepted():
    # A reference used twice is a shared subtree, not a circular reference.
    shared = {"type": "string"}
    parameters = {"type": "object", "properties": {"a": shared, "b": shared}}
    assert ToolDefinition(name="f", parameters=parameters).parameters == parameters


def test_token_chunks_carry_log_probabilities():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    converted = ChatCompletionRequest(
        {**REQUEST, "logprobs": True, "top_logprobs": 1, "thinking": {"type": "disabled"}}
    ).convert(ConversionOptions())
    assert converted.inference_options.logprobs
    assert converted.inference_options.top_logprobs == 1
    generator = ChatCompletionRequest.chunk_generator(converted, "logprobs", "smoke-model")
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = ChatCompletionResponse("logprobs", "smoke-model", 0, 0, 0)
    token_ids = tokenizer.encode("Hi there")
    try:
        for chunk in (
            InferenceChunk.ready(),
            InferenceChunk.tokens(
                token_ids, [(-0.5, [(token_ids[0], -0.5)]) for _ in token_ids]
            ),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
        body = json.loads(response.to_json())
        logprobs = body["choices"][0]["logprobs"]["content"]
        assert "".join(entry["token"] for entry in logprobs) == "Hi there"
        assert all(entry["logprob"] == -0.5 for entry in logprobs)
        assert logprobs[0]["top_logprobs"][0]["bytes"] == list(logprobs[0]["token"].encode())
    finally:
        processor.close()

def test_image_placeholders_split_content():
    source = UrlImageSource("https://example.com/a.png")
    image_token = SpecialToken.Image.default_str()
    encoding = DeepseekV41Encoding()

    def placeholders(rendered):
        return [
            (start, end)
            for start, end in rendered.special_ranges
            if rendered.prompt[start:end] == image_token
        ]

    # One spelling per source becomes a placeholder; the extra spelling stays plain text.
    message = UserMessage(f"{image_token}a{image_token}b{image_token}", [source, source])
    rendered = encoding.render_conversation(Conversation(messages=[message], thinking_mode=False))
    assert len(placeholders(rendered)) == 2
    assert rendered.prompt.count(image_token) == 3
    assert [image.url for image in rendered.image_sources] == [source.url, source.url]
    assert message.content == f"{image_token}a{image_token}b{image_token}"

    # Fewer spellings than sources: every source is kept without a placeholder.
    message = ToolMessage("no placeholder", "call_1", [source])
    rendered = encoding.render_conversation(Conversation(messages=[message], thinking_mode=False))
    assert placeholders(rendered) == []
    assert image_token not in rendered.prompt
    assert len(rendered.image_sources) == 1
    assert len(message.image_sources) == 1


def test_messages_accept_an_explicit_image_token():
    custom = SpecialTokens(image="IMG")
    encoding = DeepseekV41Encoding().with_special_tokens(custom)
    source = UrlImageSource("https://example.com/a.png")

    for message in (
        SystemMessage("see IMG", [source], image_token="IMG"),
        UserMessage("see IMG", [source], image_token="IMG"),
        ToolMessage("see IMG", "call_1", [source], image_token="IMG"),
    ):
        rendered = encoding.render_conversation(Conversation(messages=[message], thinking_mode=False))
        assert [rendered.prompt[start:end] for start, end in rendered.special_ranges if rendered.prompt[start:end] == "IMG"] == ["IMG"]

    # Without the keyword, the custom spelling is plain text and no placeholder is written.
    rendered = encoding.render_conversation(
        Conversation(messages=[UserMessage("see IMG", [source])], thinking_mode=False)
    )
    assert all(rendered.prompt[start:end] != "IMG" for start, end in rendered.special_ranges)
    with pytest.raises(ValueError, match="image_token must not be empty"):
        SystemMessage("x", image_token="")
    with pytest.raises(ValueError, match="image_token must not be empty"):
        UserMessage("x", image_token="")
    with pytest.raises(ValueError, match="image_token must not be empty"):
        ToolMessage("x", image_token="")

def test_responses_namespace_tool_map():
    namespace = {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "f"}]}
    request = ResponsesRequest({**RESPONSES_REQUEST, "tools": [namespace]})
    namespace_tool_map = request.namespace_tool_map(ConversionOptions())
    assert list(namespace_tool_map.values()) == [("ns", "f")]
    (function_name,) = namespace_tool_map
    assert "ns" in function_name and "f" in function_name

    converted = request.convert(ConversionOptions())
    assert [tool.name for tool in converted.conversation.tools] == [function_name]
    with pytest.raises(RuntimeError, match="consumed"):
        request.namespace_tool_map(ConversionOptions())
    generator = (
        ResponsesRequest.chunk_generator(converted, "ns", "m")
        .with_namespace_tool_map(namespace_tool_map)
    )
    outputs = run_processor(generator, converted.parsing_options, answer_chunks("hello"))
    assert ResponsesResponse.sse_event_name(outputs[-1]) == "response.completed"

    invalid = {**namespace, "tools": [{"type": "custom", "name": "c"}]}
    with pytest.raises(ConversionError, match="custom tools are not allowed inside a namespace"):
        ResponsesRequest({**RESPONSES_REQUEST, "tools": [invalid]}).namespace_tool_map(
            ConversionOptions()
        )

def test_logprob_count_mismatch_closes_the_processor():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    converted = ChatCompletionRequest({**REQUEST, "logprobs": True}).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "logprobs", "m")
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    try:
        processor.push(InferenceChunk.ready())
        with pytest.raises(RuntimeError, match="1 log probabilities for 2 token IDs"):
            processor.push(InferenceChunk.tokens([1, 2], [(0.0, [])]))
        assert not processor.finished
        with pytest.raises(RuntimeError, match="closed"):
            processor.push(InferenceChunk.text("x"))
        with pytest.raises(RuntimeError, match="closed"):
            processor.finish()
    finally:
        processor.close()


def test_the_thinking_mode_selects_the_thinking_initial_stage():
    thinking = ChatCompletionRequest(REQUEST).convert(ConversionOptions()).parsing_options
    assert thinking.thinking_initial_stage == ThinkingInitialStage.ReasoningStart
    non_thinking = ChatCompletionRequest(
        {**REQUEST, "thinking": {"type": "disabled"}}
    ).convert(ConversionOptions())
    assert non_thinking.parsing_options.thinking_initial_stage is None


def test_sse_framing_of_the_three_responses():
    assert ChatCompletionResponse.sse_done_data() == "[DONE]"
    assert ResponsesResponse.sse_done_data() is None
    assert MessagesResponse.sse_done_data() is None

    for request_type, response_type, body, first_event, last_event in (
        (ChatCompletionRequest, ChatCompletionResponse, REQUEST, None, None),
        (
            ResponsesRequest,
            ResponsesResponse,
            RESPONSES_REQUEST,
            "response.created",
            "response.completed",
        ),
        (MessagesRequest, MessagesResponse, REQUEST, "message_start", "message_stop"),
    ):
        converted = request_type(body).convert(ConversionOptions())
        generator = request_type.chunk_generator(converted, "sse", "m")
        outputs = run_processor(generator, converted.parsing_options, answer_chunks("x"))
        assert outputs
        assert response_type.sse_event_name(outputs[0]) == first_event
        assert response_type.sse_event_name(outputs[-1]) == last_event
        response = response_type("sse", "m", 0, 0, 0)
        for output in outputs:
            response.accumulate(output)
        assert json.loads(response.to_json())["id"] == "sse"
        with pytest.raises(RuntimeError, match="consumed"):
            response_type.sse_event_name(outputs[0])
