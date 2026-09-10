"""Smoke tests for the Rust conversion, rendering, response, and image bindings."""

import json
from pathlib import Path

import pytest

from deepseek_recipe import (
    IMAGE_SPECIAL_TOKEN,
    THINKING_END_TOKEN,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Conversation,
    ConversionError,
    ConversionOptions,
    DeepseekV41Encoding,
    ImageQuota,
    ImageResolver,
    InferenceChunk,
    InferenceFinishReason,
    MessagesRequest,
    OpenCvImagePreprocessor,
    PromptUsage,
    ReqwestImageFetcher,
    ResponsesRequest,
    StreamProcessor,
    Tokenizer,
    ToolDefinition,
    UserMessage,
    WebSearchBehavior,
)

REQUEST = {
    "model": "smoke-model",
    "messages": [{"role": "user", "content": "hi"}],
}

TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "tokenizers" / "v41" / "tokenizer.json"
)


def test_rendering():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    encoding = DeepseekV41Encoding()
    rendered = encoding.render_conversation(Conversation(messages=[UserMessage("hi")]))

    assert converted.conversation.messages[0].content == "hi"
    assert "hi" in rendered.prompt
    assert rendered.image_sources == []
    assert encoding.render_conversation(converted.conversation).prompt == rendered.prompt


def test_encoding_with_tokenizer():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    with pytest.raises(RuntimeError, match="no tokenizer"):
        DeepseekV41Encoding().encode(converted.conversation)

    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    prompt_ids = encoding.encode(converted.conversation)

    assert prompt_ids == tokenizer.encode(encoding.render_conversation(converted.conversation).prompt)


def test_web_search_conversion_options():
    defaults = ConversionOptions()
    constructed = ConversionOptions(
        default_thinking_mode=False,
        responses_web_search=WebSearchBehavior.Reject,
        messages_web_search=WebSearchBehavior.Ignore,
    )
    configured = (
        defaults.with_default_thinking_mode(False)
        .with_responses_web_search(WebSearchBehavior.Reject)
        .with_messages_web_search(WebSearchBehavior.Ignore)
    )
    assert defaults.default_thinking_mode
    assert defaults.responses_web_search == WebSearchBehavior.Ignore
    assert defaults.messages_web_search == WebSearchBehavior.Reject
    for options in (constructed, configured):
        assert not options.default_thinking_mode
        assert options.responses_web_search == WebSearchBehavior.Reject
        assert options.messages_web_search == WebSearchBehavior.Ignore

    responses = {
        "model": "smoke-model",
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


def test_custom_inference():
    converted = ChatCompletionRequest({**REQUEST, "temperature": 0.25}).convert(
        ConversionOptions()
    )
    generator = ChatCompletionRequest.chunk_generator(converted, "smoke-response", "smoke-model")
    processor = StreamProcessor(generator, converted.parsing_options)
    response = ChatCompletionResponse("smoke-response", "smoke-model", 0, 0, 0)
    try:
        assert converted.inference_options.temperature == 0.25
        for chunk in (
            InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=4)),
            InferenceChunk.text(
                THINKING_END_TOKEN + "Custom answer", content_tokens=2
            ),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.append(output)
        for output in processor.finish():
            response.append(output)
        body = json.loads(response.to_json())

        assert processor.finished
        assert body["id"] == "smoke-response"
        assert body["choices"][0]["message"]["content"] == "Custom answer"
        assert body["usage"]["prompt_tokens"] == 4
        assert body["usage"]["completion_tokens"] == 2
    finally:
        processor.close()


def test_token_chunks():
    converted = ChatCompletionRequest(REQUEST).convert(ConversionOptions())
    generator = ChatCompletionRequest.chunk_generator(converted, "token-response", "smoke-model")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    token_ids = tokenizer.encode(THINKING_END_TOKEN + "Custom answer")
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = ChatCompletionResponse("token-response", "smoke-model", 0, 0, 0)
    try:
        for chunk in (
            InferenceChunk.ready(),
            *(InferenceChunk.token(token_id) for token_id in token_ids),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.append(output)
        for output in processor.finish():
            response.append(output)
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
    token_ids = tokenizer.encode(THINKING_END_TOKEN + "🦀")
    assert len(token_ids) > 2
    processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = ChatCompletionResponse("token-response", "smoke-model", 0, 0, 0)
    try:
        for chunk in (
            InferenceChunk.ready(),
            *(InferenceChunk.token(token_id) for token_id in token_ids),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.append(output)
        for output in processor.finish():
            response.append(output)
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
            InferenceChunk.token(999_999_999),
            InferenceChunk.text(THINKING_END_TOKEN + "Hi", content_tokens=1),
            InferenceChunk.finish(InferenceFinishReason.Stop),
        ):
            for output in processor.push(chunk):
                response.append(output)
        for output in processor.finish():
            response.append(output)
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
            processor.push(InferenceChunk.token(1))
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

    assert IMAGE_SPECIAL_TOKEN in rendered.prompt
    (image,) = multimodal.images
    assert image.width > 0 and image.height > 0
    assert image.data[:4] == b"RIFF" and image.data[8:12] == b"WEBP"
    assert multimodal.image_token_adjustment() > 0


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

