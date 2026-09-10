# server-py

A FastAPI example with Python mock inference. The server combines the Rust
components exposed by `deepseek_recipe` for request conversion, prompt rendering,
image resolution, and response processing. Request preparation and HTTP transport
are implemented in [main.py](main.py).

## Usage

Use Python 3.10+ in a virtual environment and install the
[Rust and OpenCV prerequisites](../docs/development.md#native-dependencies). From the repository
root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install ./deepseek-recipe-python -r server-py/requirements.txt
python3 -m uvicorn main:app --app-dir server-py --host 127.0.0.1 --port 7777
```

Use `--host` and `--port` to change the listening address.
On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## API

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`

All three protocols support JSON for `stream=false` and SSE for `stream=true`.

For example, in a second terminal:

```sh
curl http://127.0.0.1:7777/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

The response contains the mock answer `Hello world!`. Set `"stream":true` and
add `-N` to curl to inspect SSE output. No API key or model weights are required.

## Custom inference

Use `create_app(infer=...)` in [main.py](main.py) to supply a backend. The callback
receives the server's `PreparedRequest`, including the prompt, inference options,
and processed images, and produces an asynchronous iterator of `InferenceChunk`
values.

Use `create_app(options=...)` to change the Rust conversion defaults. The default
`ConversionOptions()` ignores Responses `web_search` declarations and rejects
Messages `web_search` declarations; switching to the other value reverses that
behavior (`Reject` for Responses and `Ignore` for Messages).

## Deployment scope

This example has no authentication and uses an image fetcher that does not
filter private network addresses. Keep it on loopback for local use; see the
[deployment considerations](../docs/development.md#example-servers) before
exposing it to other users. The Rust example uses the same default port.
