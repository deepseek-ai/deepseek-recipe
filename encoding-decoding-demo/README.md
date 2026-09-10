# Encoding & Decoding Demo

A web interface for rendering DeepSeek V4.1 prompts, inspecting special tokens,
and decoding complete model output into Chat Completions, Responses, and Messages
API formats.

## Usage

Use the Rust toolchain pinned in the repository and a C/C++ compiler. This
example does not require OpenCV, model weights, or an API key. From the
repository root:

```sh
cargo run -p encoding-decoding-demo --locked
```

Open [http://127.0.0.1:7778](http://127.0.0.1:7778). Set `DEMO_ADDR` to change the
listening address.

Use the Encode / Decode buttons to switch between modes; only the selected
mode's editor and result are shown. In Encode mode, select an API format, edit
its request JSON, and render the request to inspect the model prompt. In Decode
mode, edit the complete assistant output and decode it. Model output is
rendered with the same special-token highlights as the prompt. Switching modes or
formats keeps each request draft and reuses the same model output.

The example output includes `<think>…</think>` reasoning, answer text, two DSML
weather tool calls (Beijing and Shanghai, each with string and numeric
arguments), and the end-of-turn marker `<｜end▁of▁sentence｜>`. Each protocol's
sample request declares the weather tool and includes the preceding tool
result and follow-up question.

The decoded JSON view shows only response content, reasoning, and tool calls.
It omits finish status and server-generated metadata such as response
and tool-call IDs, model names, timestamps, token usage, and thinking
signatures. Tool inputs and arguments remain intact, including payload fields
with those names. The HTTP endpoint still returns the complete protocol
response.

The decoder accepts complete assistant content, including the opening
`<think>` when reasoning is present, and an optional leading `<｜Assistant｜>`.
It consumes these frame markers before passing the content to the library's
stream parser. Output without an opening `<think>` is answer content; a
leading `</think>` from a non-thinking assistant prefix is also accepted.
Parsing starts outside tool-call markup because the complete output includes
its DSML opening block. Tool declarations, JSON output mode, and stop sequences
come from the selected protocol request.

Edit the model output, then use the adjacent decode button or Cmd/Ctrl+Enter. The UI
uses the default `stop` finish reason without displaying protocol status. If
the pasted output contains `<｜end▁of▁sentence｜>`, the demo treats its first
occurrence as EOS and discards that marker and everything after it.

## HTTP endpoints

`POST /api/render` accepts `{ "format": "…", "body": { … } }` and returns the
prompt with highlighted segments.

`POST /api/decode` accepts the same request plus `output` and an optional
`finish_reason` (default: `stop`). The formats are `chat_completions`,
`responses`, and `messages`. For example:

```sh
curl http://127.0.0.1:7778/api/decode \
  -H 'content-type: application/json' \
  -d '{
    "format": "chat_completions",
    "body": {"messages": [{"role": "user", "content": "Hello"}]},
    "output": "<think>Respond in English.</think>Hello!<｜end▁of▁sentence｜>",
    "finish_reason": "stop"
  }'
```

The result is `{ "response": { … }, "segments": [ … ] }`. `segments` preserves
the complete supplied model output for highlighting, including its frame
markers. `response` is the complete protocol response, even when the input
request sets `stream: true`. The demo
uses the supplied model name or `encoding-decoding-demo`, a demonstration response ID,
and zero token usage because no tokenizer or backend usage data is supplied.
Decoding uses the library's stream parser and protocol response accumulators.
API callers can supply `length` or `content_filter` as the backend finish
reason. EOS uses `stop` instead, and matched request stop sequences take
precedence.

This example has no authentication. Keep the default loopback address for local
use. It renders prompts and decodes supplied text; it does not execute
inference, run tools, or fetch image URLs.
