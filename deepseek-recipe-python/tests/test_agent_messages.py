"""Codex agent input must reach the rendered prompt through the shared converter."""

from pathlib import Path

import pytest

from deepseek_recipe import (
    ConversionError,
    ConversionOptions,
    DeepseekV41Encoding,
    ResponsesRequest,
    Tokenizer,
    UserMessage,
)


def agent_message(content):
    return {
        "type": "agent_message",
        "author": "/root/worker",
        "recipient": "/root",
        "content": content,
    }


def test_agent_message_reaches_prompt_and_token_ids():
    request = {
        "model": "test-model",
        "input": [agent_message([
            {"type": "input_text", "text": "Message Type: FINAL_ANSWER\nPayload:"},
            {"type": "input_text", "text": "CHILD_RESULT_92714: 世界 🦀"},
        ])],
    }
    converted = ResponsesRequest(request).convert(ConversionOptions())
    (message,) = converted.conversation.messages
    assert isinstance(message, UserMessage)
    assert "from /root/worker to /root" in message.content
    assert "FINAL_ANSWER\nPayload:\nCHILD_RESULT_92714: 世界 🦀" in message.content

    tokenizer_path = Path(__file__).resolve().parents[2] / "static/tokenizers/v41/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    prompt = encoding.render_conversation(converted.conversation).prompt
    token_ids = encoding.encode(converted.conversation)
    assert "CHILD_RESULT_92714: 世界 🦀" in prompt
    assert token_ids
    assert token_ids == tokenizer.encode(prompt)


@pytest.mark.parametrize("content", [
    [],
    [{"type": "input_text", "text": "  "}],
    [{"type": "encrypted_content", "encrypted_content": "secret-ciphertext"}],
    [
        {"type": "input_text", "text": "VISIBLE_PREFIX"},
        {"type": "encrypted_content", "encrypted_content": "secret-ciphertext"},
    ],
])
def test_unreadable_agent_input_is_rejected(content):
    request = ResponsesRequest({"model": "test-model", "input": [agent_message(content)]})
    with pytest.raises(ConversionError) as error:
        request.convert(ConversionOptions())
    assert "secret-ciphertext" not in str(error.value)
    assert "VISIBLE_PREFIX" not in str(error.value)
