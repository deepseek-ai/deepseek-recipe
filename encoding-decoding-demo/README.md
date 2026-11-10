# Encoding & Decoding Demo

A web interface for rendering DeepSeek V4 and V4.1 prompts, inspecting special
tokens, and parsing complete model output into Chat Completions, Responses, and
Messages responses. The demo renders prompts and parses model output; it does
not encode or decode token IDs.

## Usage

Use the Rust toolchain pinned in the repository and a C/C++ compiler. This
example does not require OpenCV, model weights, or an API key. From the
repository root:

```sh
cargo run -p encoding-decoding-demo --locked
```

Open [http://127.0.0.1:7778](http://127.0.0.1:7778). Set `DEMO_ADDR` to change
the listening address.

Use the Render / Parse buttons to switch between modes; only the selected
mode's editor and result are shown. In Render mode, select an API format, edit
its request JSON, then render it and inspect the prompt. Render mode shows
prompt text; it does not encode the prompt into token IDs. In Parse mode, edit
the complete assistant output and parse it into the selected format's protocol
response. Model output is displayed with the same special-token highlights as
the prompt. Switching modes or formats keeps each request draft and reuses the
same model output.

The request's `model` name selects the prompt encoding. `DeepSeek-V4-Pro` and
`deepseek-v4-pro` render V4 prompts; `DeepSeek-V4.1-Flash`, `deepseek-flash`,
and `encoding-decoding-demo` render V4.1 prompts. A request without a `model`
uses `encoding-decoding-demo`. Any other name is rejected with HTTP 400.

The example output includes `<think>…</think>` reasoning, answer text, two DSML
weather tool calls (Beijing and Shanghai, each with string and numeric
arguments), and the end-of-sentence token (EOS) `<｜end▁of▁sentence｜>`. Each
protocol's sample request declares the weather tool and includes the preceding
tool result and follow-up question.

The protocol response view shows only answer content, reasoning content, and
tool calls. It omits finish status and server-generated metadata such as
response and tool-call IDs, model names, timestamps, token usage, and thinking
signatures. Tool inputs and arguments remain intact, including payload fields
with those names. The HTTP endpoint responds with the complete protocol
response.

Parse mode accepts complete assistant output, including the opening `<think>`
when reasoning is present, and an optional leading `<｜Assistant｜>`. It strips
these leading tokens before passing the content to the library's stream parser.
Output without an opening `<think>` is answer content; a leading `</think>`,
which the prompt ends with outside thinking mode, is also accepted.
Parsing starts outside tool-call markup because the complete output includes
its DSML opening block. Tool declarations, JSON output mode, and stop sequences
come from the selected protocol request.

Edit the model output, then use the adjacent Parse button or Cmd/Ctrl+Enter.
The UI uses the default `stop` finish reason without displaying protocol status.
If the pasted output contains the EOS, the demo treats its first occurrence as
the end of the output and discards that EOS and everything after it.

## HTTP endpoints

`POST /api/render` accepts `{ "format": "…", "body": { … } }` and responds with
the rendered prompt and its highlighted segments.

`POST /api/parse` accepts the same request plus `output` and an optional
`finish_reason` (default: `stop`). The formats are `chat_completions`,
`responses`, and `messages`. For example:

```sh
curl http://127.0.0.1:7778/api/parse \
  -H 'content-type: application/json' \
  -d '{
    "format": "chat_completions",
    "body": {"messages": [{"role": "user", "content": "Hello"}]},
    "output": "<think>Respond in English.</think>Hello!<｜end▁of▁sentence｜>",
    "finish_reason": "stop"
  }'
```

The result is `{ "response": { … }, "segments": [ … ] }`. `segments` preserves
the complete supplied model output for highlighting, including the leading
`<｜Assistant｜>` and `<think>`. `response` is the complete protocol response,
even when the input request sets `stream: true`. The demo uses the supplied
model name or `encoding-decoding-demo`, a demonstration response ID, and zero
token usage because no tokenizer or backend usage data is supplied.
The endpoint parses the output with the library's `StreamProcessor` and
accumulates the protocol chunks into the complete response. API callers can
supply `length` or `content_filter` as the backend finish reason. Output that
contains the EOS uses `stop` instead, and matched request stop sequences take
precedence.

This example has no authentication. Keep the default loopback address for local
use. It renders prompts and parses model output; it does not encode
conversations into token IDs, decode token IDs, execute inference, run tools, or
fetch image URLs.
