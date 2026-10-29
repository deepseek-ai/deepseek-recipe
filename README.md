<div align="center">
  <img src="static/deepseek-whale.svg" alt="DeepSeek" width="240">
</div>

# deepseek-recipe

**English** | [中文](README.zh.md)

deepseek-recipe is a collection of Rust libraries and Python bindings that
uniformly convert API requests in different formats into the Conversation
format, render conversations into prompts for DeepSeek models, encode
conversations into token IDs with an attached tokenizer, and convert model
output into responses in the corresponding format. Use these components to
connect an inference backend to API services that support multiple formats. The
caller must provide model inference, tool execution, and HTTP transport.

[Getting started](#using-deepseek-recipe) ·
[Streaming response](docs/streaming.md) ·
[Use with tokenizer](docs/tokenizer.md) · [License](#license)

## Supported scope

- **Request/response formats:** Conversion of Messages, Chat Completions, and
  Responses requests, [Streaming response](docs/streaming.md), and complete
  responses. Supports text, images, thinking mode, client tool calls, and Chat
  Completions `prefix` continuation.
- **Prompts:** Rendering of DeepSeek V4 and V4.1 conversations into prompts, and
  encoding of conversations into token IDs with an attached tokenizer.
- **Generation settings:** Thinking mode, reasoning effort, `temperature`,
  `top_p`, and output token limits. After the caller encodes the prompt,
  `ConversationRequest::resolve_max_tokens` checks the prompt and `max_tokens`
  against the context limit and supplies the `max_tokens` default when the
  request sets none: 8192 outside thinking mode, 65536 in thinking mode, and
  131072 in thinking mode with `max` effort.
- **Output parsing:** Reasoning content, tool calls, JSON object output, and
  stop sequences. Usage reports `reasoning_tokens` when the stream processor has an
  attached tokenizer: the processor encodes the reasoning content with it to
  count the tokens.
- **Structured output:** Chat Completions `response_format` of type
  `json_schema` or `regex`, and Responses `text.format` of type `json_schema`.
  Conversion validates a JSON Schema, and rendering writes it into the prompt; a
  regex is passed through as `ResponseFormat::Regex`. Conversion also selects
  the `OutputConstraint` of `InferenceOptions`: the tool call block of a request
  with tools and a tool choice other than `none`, otherwise the JSON, JSON
  Schema, or regex of its response format. The caller decides whether its
  backend enforces it by constrained decoding.
- **Images:** Provided as base64 or external URLs, and preprocessed for
  DeepSeek V4.1 with OpenCV by the image component. JPEG EXIF rotation and
  mirroring are applied before resizing and WebP encoding. DeepSeek V4 does not
  accept images; conversion replaces each image in a V4 request with the text
  `[Unsupported Image]`. The component's `ReqwestImageFetcher`
  connects to any address by default, so a service that fetches URLs from
  untrusted requests should install a `DestinationFilter`.
- **Tool definitions:** Function tools, and for Responses also tool namespaces
  and the `apply_patch` custom tool. V4.1 supports deferred tool loading in
  Messages and client-executed tool search in Responses.
- **Log probabilities:** `logprobs` and `top_logprobs` (Chat Completions) and
  `top_logprobs` (Responses) are converted into `InferenceOptions`. The caller
  supplies the backend's values in `InferenceChunk::Tokens`; with an attached
  tokenizer, the Chat Completions and Responses adapters report them. The
  library does not compute probabilities.

## Using deepseek-recipe

The examples below convert a Chat Completions request into a DeepSeek V4.1
prompt. The library resolves the model version from the request's `model` name
and accepts `DeepSeek-V4-Pro` and `DeepSeek-V4.1-Flash` by default; register
other names, such as `deepseek-flash`, with
`ConversionOptions::with_model_alias`.

### Python

#### Installation

Python 3.10+:

```sh
python3 -m pip install deepseek-recipe
```

#### Example

```python
from deepseek_recipe import ChatCompletionRequest, ConversionOptions, DeepseekV41Encoding

request = ChatCompletionRequest({
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Hello"}],
})
converted = request.convert(ConversionOptions())
rendered = DeepseekV41Encoding().render_conversation(converted.conversation)
print(rendered.prompt)
```

### Rust

#### Installation

```sh
cargo add deepseek-recipe@0.2 deepseek-recipe-encoding@0.2 serde_json@1
```

See the [development guide](docs/development.md) for source builds and image
dependencies.

#### Example

```rust
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }))?;
    let converted = request.convert(ConversionOptions::default())?;
    let rendered = DeepseekV41Encoding::new().render_conversation(&converted.conversation);
    println!("{}", rendered.prompt);
    Ok(())
}
```

### Encoding & Decoding Demo

Run the demo from the repository root to render prompts from requests and parse
model output into protocol responses:

```sh
cargo run -p encoding-decoding-demo --locked
```

Open [http://127.0.0.1:7778](http://127.0.0.1:7778).

### More examples

Rust and Python both support converting model output into streaming responses.
See the [Streaming response](docs/streaming.md). To encode conversations into
token IDs or decode backend token IDs, see
[use with tokenizer](docs/tokenizer.md).

## Packages and example projects

| Package | Purpose |
| --- | --- |
| [`deepseek-recipe`](https://docs.rs/deepseek-recipe) | Protocol conversion and model output parsing. |
| [`deepseek-recipe-core`](https://docs.rs/deepseek-recipe-core) | Shared conversation, message, image, and tool types. |
| [`deepseek-recipe-encoding`](https://docs.rs/deepseek-recipe-encoding) | DeepSeek V4 and V4.1 prompt rendering and token encoding. |
| [`deepseek-recipe-image`](https://docs.rs/deepseek-recipe-image) | Image fetching and preprocessing. |
| [`deepseek-recipe-macros`](https://docs.rs/deepseek-recipe-macros) | The `prompt!` macro used by prompt rendering. |
| [`deepseek-recipe-python`](deepseek-recipe-python/README.md) | Python bindings, imported as `deepseek_recipe`. |
| [encoding-decoding-demo](encoding-decoding-demo/README.md) | A web interface that renders prompts, highlights special tokens, and parses complete model output into Chat Completions, Responses, or Messages responses. |
| [server-rs](server-rs/README.md) | An Axum API example with mock inference. |
| [server-py](server-py/README.md) | A FastAPI example with mock inference. |

## License

Project code and public documentation are licensed under the
[MIT License](LICENSE). Bundled tokenizer notices are in
[static/tokenizers/README.md](static/tokenizers/README.md).
