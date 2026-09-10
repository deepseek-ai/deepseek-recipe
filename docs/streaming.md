# Streaming response

**English** | [中文](streaming.zh.md)

`StreamProcessor` converts backend output into Messages, Chat Completions, or
Responses events. The examples below use mock inference and print Chat
Completions chunks. The application supplies the backend and HTTP transport.

## Rust

Add the dependencies to your application:

```sh
cargo add deepseek-recipe@0.1 serde_json@1 tokio-stream@0.1
cargo add tokio@1 --features macros,rt-multi-thread
```

Save this as `src/main.rs` and run `cargo run`:

```rust
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe::stream::{
    InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor,
};
use serde_json::json;
use tokio_stream::{StreamExt, iter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let body = json!({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "disabled"},
    });

    // The chunk generator uses the converted request's resolved settings.
    let request: ChatCompletionRequest = serde_json::from_value(body)?;
    let converted = request.convert(ConversionOptions::default())?;
    let generator =
        ChatCompletionRequest::chunk_generator(&converted, "id-1".into(), "deepseek-flash".into());
    let processor = StreamProcessor::new(generator, converted.parsing_options);

    // Mock the backend's inference chunks.
    let inference = iter(vec![
        InferenceChunk::Ready {
            system_fingerprint: Some("fp-1".into()),
            prompt_usage: PromptUsage {
                prompt_tokens: 4,
                prompt_cache_hit_tokens: 0,
            },
        },
        InferenceChunk::Text {
            content: "Hello!".into(),
            content_tokens: 2,
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ]);

    // Convert them into protocol chunks.
    let mut chunks = std::pin::pin!(processor.process(inference));
    while let Some(chunk) = chunks.next().await {
        println!("{}", serde_json::to_string(&chunk?)?);
    }
    Ok(())
}
```

## Python

Install the package with `python3 -m pip install deepseek-recipe`.
`push` processes one backend chunk; `finish` completes response processing.

```python
from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    InferenceChunk,
    InferenceFinishReason,
    PromptUsage,
    StreamProcessor,
)

request = ChatCompletionRequest({
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "thinking": {"type": "disabled"},
})
converted = request.convert(ConversionOptions())
generator = ChatCompletionRequest.chunk_generator(converted, "id-1", "deepseek-flash")
processor = StreamProcessor(generator, converted.parsing_options)

# Mock the backend's inference chunks.
chunks = []
for chunk in [
    InferenceChunk.ready(system_fingerprint="fp-1", prompt_usage=PromptUsage(prompt_tokens=4)),
    InferenceChunk.text("Hello!", content_tokens=2),
    InferenceChunk.finish(InferenceFinishReason.Stop),
]:
    chunks.extend(processor.push(chunk))
chunks.extend(processor.finish())

for chunk in chunks:
    print(chunk.to_json())
processor.close()
```

See the [project README](../README.md) for supported scope and the
[Rust API documentation](https://docs.rs/deepseek-recipe) for API contracts.
