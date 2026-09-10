# Use with tokenizer

**English** | [中文](tokenizer.zh.md)

With a tokenizer attached, a `PromptEncoding` encodes the prompt into model
token IDs, and a `StreamProcessor` accepts token IDs. Neither loads a tokenizer
on its own: pass one with `with_tokenizer`, then call the corresponding method.
Using `encode` or passing token IDs without an attached tokenizer returns an
error.

Use the tokenizer that matches your model. This repository bundles copies under
[`static/tokenizers/`](../static/tokenizers/README.md) for the V4 and V4.1
prompt templates; point the loader at `tokenizer.json` in the matching
directory. A tokenizer from the Hugging Face Hub works too, but its special
token spelling must match the encoding's constants, and its image token must
resolve to the same ID.

## Rust

Add the dependencies to your application:

```sh
cargo add deepseek-recipe@0.1 deepseek-recipe-encoding@0.1 serde_json@1 tokenizers@0.23
```

Save this as `src/main.rs` and run `cargo run`. The tokenizer path below points
at this repository's bundled file; use your own `tokenizer.json` otherwise:

```rust
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde_json::json;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }))?;
    let converted = request.convert(ConversionOptions::default())?;

    // Attach the tokenizer, then encode the conversation's tokens.
    let tokenizer = Tokenizer::from_file("static/tokenizers/v41/tokenizer.json")?;
    let encoding = DeepseekV41Encoding::new().with_tokenizer(tokenizer);
    let token_ids = encoding.encode(&converted.conversation)?;

    println!("{token_ids:?}");
    Ok(())
}
```

`Tokenizer` implements the `TokenizerEncoder` trait used by `with_tokenizer`.
Any other type implementing that trait can be attached instead. Encoding is a
two-step operation: `render_conversation` returns the prompt text and its image
placeholders, and `encode` tokenizes that prompt with added special tokens
disabled, because the prompt already contains the special token text.

### Decoding

A backend can return token IDs instead of text. Attach the same tokenizer to the
stream processor to decode those chunks. The example also needs `tokio-stream`,
plus `tokio` with the `macros` and `rt-multi-thread` features, as listed in
[response streaming](streaming.md):

```rust
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, StreamProcessor};
use serde_json::json;
use tokenizers::Tokenizer;
use tokio_stream::{StreamExt, iter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "disabled"},
    }))?;
    let converted = request.convert(ConversionOptions::default())?;
    let generator =
        ChatCompletionRequest::chunk_generator(&converted, "id-1".into(), "deepseek-flash".into());

    // The backend answers with token IDs; the attached tokenizer decodes them.
    let tokenizer = Tokenizer::from_file("static/tokenizers/v41/tokenizer.json")?;
    let token_ids = tokenizer.encode("Hello there!", false)?.get_ids().to_vec();
    let processor =
        StreamProcessor::new(generator, converted.parsing_options).with_tokenizer(tokenizer);

    let inference = iter(
        token_ids
            .into_iter()
            .map(|token_id| InferenceChunk::Token { token_id })
            .chain([InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop,
            }]),
    );

    let mut chunks = std::pin::pin!(processor.process(inference));
    while let Some(chunk) = chunks.next().await {
        println!("{}", serde_json::to_string(&chunk?)?);
    }
    Ok(())
}
```

The processor decodes ids incrementally and buffers them while a multi-token
character is incomplete; ids that produced text count as completion tokens.
Special tokens stay in the decoded text because they drive output parsing. A
`Token` chunk without an attached tokenizer fails the stream with
`StreamError::MissingTokenizer`. `Tokenizer` implements the `TokenizerDecoder`
trait used by `with_tokenizer`.

## Python

Install the package with `python3 -m pip install deepseek-recipe`:

```python
from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    DeepseekV41Encoding,
    Tokenizer,
)

request = ChatCompletionRequest({
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
})
converted = request.convert(ConversionOptions())

tokenizer = Tokenizer.from_file("static/tokenizers/v41/tokenizer.json")
encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
print(encoding.encode(converted.conversation))
```

`Tokenizer.from_pretrained` downloads and caches a `tokenizer.json` from the
Hugging Face Hub, and `Tokenizer.from_str` loads the file contents directly.
`render_conversation` and `encode` return the same results as in Rust.

### Decoding

Pass the tokenizer as the processor's third argument. Token-id chunks are
decoded with the same behavior as in Rust; without a tokenizer, a token chunk
raises `RuntimeError` and closes the processor.

```python
from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    InferenceChunk,
    InferenceFinishReason,
    StreamProcessor,
    Tokenizer,
)

converted = ChatCompletionRequest({
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "thinking": {"type": "disabled"},
}).convert(ConversionOptions())
generator = ChatCompletionRequest.chunk_generator(converted, "id-1", "deepseek-flash")

# The backend answers with token IDs; the tokenizer passed to the processor
# decodes them.
tokenizer = Tokenizer.from_file("static/tokenizers/v41/tokenizer.json")
token_ids = tokenizer.encode("Hello there!")
processor = StreamProcessor(generator, converted.parsing_options, tokenizer)

chunks = []
for chunk in (
    *(InferenceChunk.token(token_id) for token_id in token_ids),
    InferenceChunk.finish(InferenceFinishReason.Stop),
):
    chunks.extend(processor.push(chunk))
chunks.extend(processor.finish())

for chunk in chunks:
    print(chunk.to_json())
processor.close()
```

See the [project README](../README.md) for supported scope and the
[response streaming](streaming.md) guide for the full chunk sequence.
