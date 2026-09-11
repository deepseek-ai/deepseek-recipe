"""Smoke tests for the FastAPI example and its inference callback."""

import importlib.util
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
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


class _StubImageHost(BaseHTTPRequestHandler):
    """A local origin that answers every GET with a fixed status code."""

    status = 503

    def do_GET(self):  # noqa: N802
        self.send_response(self.status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@contextmanager
def image_host(status):
    """Serve one image URL whose origin always answers `status`."""
    _StubImageHost.status = status
    server = HTTPServer(("127.0.0.1", 0), _StubImageHost)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/image.png"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "upstream,expected_status,expected_type",
    [(503, 500, "internal_error"), (404, 400, "invalid_request_error")],
)
def test_image_fetch_failure_follows_retryability(
    upstream, expected_status, expected_type
):
    """An upstream image failure is the server's to own only when it is retryable.

    `ImageError.is_retryable` is the library's classification: `Fetch` is always
    retryable and `FetchStatus` is retryable at 5xx. Answering a retryable upstream
    failure with 400 tells the client its request was malformed and should not be
    retried, which inverts the classification the library hands over.
    """
    with image_host(upstream) as url:
        with TestClient(SERVER.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-flash",
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe this"},
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        }
                    ],
                },
            )
    assert response.status_code == expected_status, response.text
    assert response.json()["error"]["type"] == expected_type
