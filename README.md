<div align="center">
  <img src="static/deepseek-whale.svg" alt="DeepSeek" width="240">
</div>

# deepseek-recipe

**English** | [中文](README.zh.md)

deepseek-recipe is a collection of Rust libraries and Python bindings that
uniformly convert API requests in different formats into the Conversation
format, encode them into prompts for DeepSeek models, and convert model output
into responses in the corresponding format. Use these components to connect an
inference backend to API services that support multiple formats. Model
inference, tool execution, and HTTP transport must be provided externally.

[Getting started](#using-deepseek-recipe) · [Streaming response](docs/streaming.md) ·
[Use with tokenizer](docs/tokenizer.md) · [License](#license)

## Supported scope

- **Request/response formats:** Conversion of Messages, Chat Completions, and
  Responses requests, [Streaming response](docs/streaming.md), and complete
  responses. Supports text, images, thinking, and client tool calls.
- **Prompts:** Encoding of DeepSeek V4 and V4.1 conversations into prompts or token IDs.
- **Generation settings:** Thinking mode, reasoning effort, `temperature`, `top_p`,
  and output token limits.
- **Output parsing:** Thinking, tool calls, JSON object output, and stop sequences.
- **Images:** Provided as base64 or external URLs. The image component provides
  DeepSeek V4.1 preprocessing with OpenCV.
- **Tool definitions:** Function tools; the Responses API also supports tool
  namespaces and the `apply_patch` custom tool.

## Not yet supported

- Token probabilities (`logprobs` and `top_logprobs`).
- Document content, audio/video input, and file retrieval by `file_id`.
- Server tool execution, such as `web_search`.
- JSON Schema and regex output constraints, or enforcement of tool `strict` settings.
- Multiple completions per Chat Completions request (`n > 1`).
- Responses custom tool definitions other than `apply_patch`.
- Responses conversation storage and context retrieval through `previous_response_id`.
- Responses encrypted thinking content (`encrypted_content`).

## Using deepseek-recipe

To convert a Chat Completions request into a DeepSeek V4.1 prompt:

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
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
})
converted = request.convert(ConversionOptions())
rendered = DeepseekV41Encoding().render_conversation(converted.conversation)
print(rendered.prompt)
```

### Rust

#### Installation

```sh
cargo add deepseek-recipe@0.1 deepseek-recipe-encoding@0.1
```

See the [development guide](docs/development.md) for source builds and image dependencies.

#### Example

```rust
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }))?;
    let converted = request.convert(ConversionOptions::default())?;
    let rendered = DeepseekV41Encoding::new().render_conversation(&converted.conversation);
    println!("{}", rendered.prompt);
    Ok(())
}
```

### Encoding & Decoding Demo

Run the encoding and decoding demo from the repository root:

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
| [`deepseek-recipe-python`](deepseek-recipe-python/README.md) | Python bindings, imported as `deepseek_recipe`. |
| [encoding-decoding-demo](encoding-decoding-demo/README.md) | A web interface for encoding prompts, inspecting special tokens, and decoding complete model output into Chat Completions, Responses, or Messages. |
| [server-rs](server-rs/README.md) | An Axum API example with mock inference. |
| [server-py](server-py/README.md) | A FastAPI example with mock inference. |

## License

Project code and public documentation are licensed under the [MIT License](LICENSE).
Bundled tokenizer notices are in [static/tokenizers/README.md](static/tokenizers/README.md).
