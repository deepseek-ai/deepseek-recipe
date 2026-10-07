"""Smoke tests for the FastAPI example and its inference callback."""

import importlib.util
from pathlib import Path

import pytest
from deepseek_recipe import (
    THINKING_END_TOKEN,
    ConversionOptions,
    InferenceChunk,
    WebSearchBehavior,
)
from fastapi.testclient import TestClient

SPEC = importlib.util.spec_from_file_location(
    "recipe_server_example", Path(__file__).resolve().parents[1] / "main.py"
)
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)

PROTOCOL_REQUESTS = [
    (
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
        "data: [DONE]",
    ),
    ("/v1/responses", {"input": "hi"}, "event: response.completed"),
    (
        "/v1/messages",
        {"messages": [{"role": "user", "content": "hi"}]},
        "event: message_stop",
    ),
]


def response_text(path, body):
    if path == "/v1/chat/completions":
        return body["choices"][0]["message"]["content"]
    if path == "/v1/responses":
        return "".join(
            content["text"]
            for item in body["output"]
            if item["type"] == "message"
            for content in item["content"]
            if content["type"] == "output_text"
        )
    return "".join(item["text"] for item in body["content"] if item["type"] == "text")


@pytest.mark.parametrize("path,payload,terminal", PROTOCOL_REQUESTS)
@pytest.mark.parametrize("stream", [False, True])
def test_example_server(path, payload, terminal, stream):
    with TestClient(SERVER.app) as client:
        response = client.post(
            path, json={"model": "deepseek-flash", "stream": stream, **payload}
        )
    assert response.status_code == 200, response.text
    assert "cache-control" not in response.headers
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "Hello " in response.text and "world!" in response.text
        assert terminal in response.text
    else:
        assert response.headers["content-type"] == "application/json"
        assert response_text(path, response.json()) == "Hello world!"


def test_custom_inference():
    async def infer(request):
        assert "hi" in request.prompt
        assert request.inference_options.temperature == 0.25
        assert request.images == []
        assert request.image_token_adjustment == 0
        yield InferenceChunk.text(THINKING_END_TOKEN + "Custom answer")

    with TestClient(SERVER.create_app(infer)) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "custom-model", "input": "hi", "temperature": 0.25},
        )
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "custom-model"
    assert response_text("/v1/responses", response.json()) == "Custom answer"


@pytest.mark.parametrize("stream", [False, True])
def test_agent_input_reaches_inference_and_unreadable_content_does_not(stream):
    prompts = []

    async def infer(request):
        prompts.append(request.prompt)
        assert "from /root/worker to /root" in request.prompt
        assert "CHILD_RESULT_92714" in request.prompt
        yield InferenceChunk.text(THINKING_END_TOKEN + "Agent result received")

    item = {
        "type": "agent_message",
        "author": "/root/worker",
        "recipient": "/root",
        "content": [{"type": "input_text", "text": "CHILD_RESULT_92714"}],
    }
    with TestClient(SERVER.create_app(infer)) as client:
        accepted = client.post("/v1/responses", json={
            "model": "test-model", "input": [item], "stream": stream,
        })
        assert accepted.status_code == 200, accepted.text
        assert "Agent result received" in accepted.text
        encrypted = {"type": "encrypted_content", "encrypted_content": "secret-ciphertext"}
        for content in [[encrypted], [*item["content"], encrypted]]:
            rejected = client.post("/v1/responses", json={
                "model": "test-model", "input": [{**item, "content": content}], "stream": stream,
            })
            assert rejected.status_code == 400, rejected.text
            assert "secret-ciphertext" not in rejected.text
    assert len(prompts) == 1


def test_invalid_request():
    with TestClient(SERVER.app) as client:
        response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert "cache-control" not in response.headers
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_web_search_conversion_options():
    messages = {
        "model": "deepseek-flash",
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    responses = {
        "model": "deepseek-flash",
        "stream": False,
        "input": "hi",
        "tools": [{"type": "web_search"}],
    }

    # The default options reject Messages web search and ignore Responses web search.
    with TestClient(SERVER.app) as client:
        assert client.post("/v1/messages", json=messages).status_code == 400
        assert client.post("/v1/responses", json=responses).status_code == 200

    options = ConversionOptions(
        responses_web_search=WebSearchBehavior.Reject,
        messages_web_search=WebSearchBehavior.Ignore,
    )
    with TestClient(SERVER.create_app(options=options)) as client:
        accepted = client.post("/v1/messages", json=messages)
        rejected = client.post("/v1/responses", json=responses)
    assert accepted.status_code == 200, accepted.text
    assert response_text("/v1/messages", accepted.json()) == "Hello world!"
    assert rejected.status_code == 400
