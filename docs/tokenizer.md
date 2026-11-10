# Use with tokenizer

**English** | [中文](tokenizer.zh.md)

With a tokenizer attached, a `PromptEncoding` encodes conversations into model
token IDs, and a `StreamProcessor` accepts token IDs. Neither loads a tokenizer
on its own: pass one with `with_tokenizer`, then call the corresponding method.
Without an attached tokenizer, `encode` returns
`EncodingError::MissingTokenizer`, and a `Tokens` chunk fails the stream with
`StreamError::MissingTokenizer`.

Use the tokenizer that matches your model. This repository bundles copies under
[`static/tokenizers/`](../static/tokenizers/README.md) for the V4 and V4.1
prompt encodings; load `tokenizer.json` from the matching directory. You can
also load a tokenizer from the Hugging Face Hub. Its added vocabulary must
contain the spelling of every special token that rendering writes; otherwise
`encode` returns `EncodingError::Encode`. `IMAGE_SPECIAL_TOKEN_ID` is the token
ID of the image token in the bundled V4.1 tokenizer; a caller that uses this
constant needs a tokenizer that assigns the image token the same token ID.

## Rust

Add the dependencies to your application:

```sh
cargo add deepseek-recipe@0.2 deepseek-recipe-encoding@0.2 serde_json@1 tokenizers@0.23
```

Save this as `src/main.rs`. The tokenizer path is relative to the working
directory, so run the application from this repository's root with
`cargo run --manifest-path path/to/your-app/Cargo.toml`; elsewhere, use an
absolute path or your own `tokenizer.json`:

```rust
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer};
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde_json::json;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }))?;
    let converted = request.convert(ConversionOptions::default())?;

    // Attach the tokenizer, then encode the conversation into token IDs.
    let tokenizer = Tokenizer::from_file("static/tokenizers/v41/tokenizer.json")?;
    let encoding = DeepseekV41Encoding::new().with_tokenizer(PromptTokenizer::new(&tokenizer));
    let token_ids = encoding.encode(&converted.conversation)?;

    println!("{token_ids:?}");
    Ok(())
}
```

Pass `with_tokenizer` a `PromptTokenizer`, which prepares a `Tokenizer` by
keeping a copy without the added vocabulary, or attach any other
`TokenizerEncoder` implementation. `encode` renders the conversation and then
tokenizes the prompt.

`render_conversation` performs only the rendering step: it returns the prompt
text, the byte ranges of its explicit special tokens (`special_ranges`), and its
image sources. Rendering inserts the template's control tokens, the placeholders
of a message's image sources, and the grounding tokens in assistant content and
reasoning content as explicit special tokens. During tokenization, each special
range encodes as the token ID of its spelling, and the text between special
ranges is plain text, encoded without the added vocabulary. `PromptTokenizer`
encodes the plain regions as one batch, which the `tokenizers` crate can encode
in parallel, depending on its `TOKENIZERS_PARALLELISM` setting.

A special-token spelling typed into a user message is plain text, so it cannot
act as a control token. A caller that tokenizes `RenderedPrompt::prompt` itself
must apply `special_ranges` the same way.

After encoding, the caller resolves the `max_tokens` to send with
`ConversationRequest::resolve_max_tokens(prompt_tokens, context_limit)`.
`prompt_tokens` is the number of encoded token IDs plus the image token counts
beyond one per placeholder (`MultiModalData::image_token_adjustment`), and
`context_limit` is the number of tokens the model's context holds. A requested
`max_tokens` is kept. An unspecified one becomes 8192 outside thinking mode,
131072 in thinking mode with `max` reasoning effort, and 65536 in thinking mode
otherwise, lowered to the context left after the prompt. The method returns a
`ConversionError` when the prompt and the requested `max_tokens` together
exceed `context_limit`.

### Decoding

A backend can produce token IDs instead of text. Attach the same tokenizer to
the stream processor to decode those chunks. The example also needs
`tokio-stream`, plus `tokio` with the `macros` and `rt-multi-thread` features,
as listed in [response streaming](streaming.md). The tokenizer path has the same
requirement as in the encoding example:

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
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "disabled"},
    }))?;
    let converted = request.convert(ConversionOptions::default())?;
    let generator = ChatCompletionRequest::chunk_generator(
        &converted,
        "id-1".into(),
        "DeepSeek-V4.1-Flash".into(),
    );

    // The backend answers with token IDs; the attached tokenizer decodes them.
    let tokenizer = Tokenizer::from_file("static/tokenizers/v41/tokenizer.json")?;
    let token_ids = tokenizer.encode("Hello there!", false)?.get_ids().to_vec();
    let processor =
        StreamProcessor::new(generator, converted.parsing_options).with_tokenizer(tokenizer);

    let inference = iter([
        InferenceChunk::Tokens { token_ids, logprobs: None },
        InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop },
    ]);

    let mut chunks = std::pin::pin!(processor.into_protocol_stream(inference));
    while let Some(chunk) = chunks.next().await {
        println!("{}", serde_json::to_string(&chunk?)?);
    }
    Ok(())
}
```

The processor decodes token IDs incrementally and buffers them while a
multi-token character is incomplete; token IDs that produced text count as
completion tokens. Each special token is decoded as its own piece and marked, so
the output parser recognizes `</think>` and the DSML tags only as tokens; the
same text spelled with plain tokens is content. Text chunks carry no token
boundaries, so their markers match by spelling. A `Tokens` chunk without an
attached tokenizer fails the stream with `StreamError::MissingTokenizer`.
`Tokenizer` implements the `StreamTokenizer` trait used by `with_tokenizer`.

`InferenceChunk::Tokens` carries the token IDs of one backend output, which the
processor decodes together. When the converted request sets
`InferenceOptions::logprobs`, the caller supplies one `InferenceTokenLogprob`
per token ID: the decoded text then carries the log probabilities of its
tokens, and Chat Completions and Responses report them. A `Tokens` chunk whose log
probability count differs from its token ID count fails the stream with
`StreamError::LogprobCountMismatch`. In thinking mode, the processor also uses
the attached tokenizer to count `reasoning_tokens` for usage by encoding the
emitted reasoning content, for text chunks as well as token chunks. Without a
tokenizer, Chat Completions usage omits `completion_tokens_details`, and
Responses usage reports `reasoning_tokens` as `0`.

## Python

Install the package with `python3 -m pip install deepseek-recipe`. The tokenizer
path has the same requirement as in the Rust example:

```python
from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    DeepseekV41Encoding,
    Tokenizer,
)

request = ChatCompletionRequest({
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Hello"}],
})
converted = request.convert(ConversionOptions())

tokenizer = Tokenizer.from_file("static/tokenizers/v41/tokenizer.json")
encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
print(encoding.encode(converted.conversation))
```

`Tokenizer.from_pretrained` downloads and caches a `tokenizer.json` from the
Hugging Face Hub, and `Tokenizer.from_str` loads the file contents directly.
`render_conversation` and `encode` return the same results as in Rust; `encode`
raises `RuntimeError` when no tokenizer is attached or encoding fails. After
encoding, `ConversationRequest.resolve_max_tokens` resolves `max_tokens` as in
Rust and raises `ConversionError` where Rust returns one.

### Decoding

Pass the tokenizer as the processor's third argument. A `tokens` chunk is
decoded with the same behavior as in Rust; without a tokenizer, such a chunk
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
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "thinking": {"type": "disabled"},
}).convert(ConversionOptions())
generator = ChatCompletionRequest.chunk_generator(converted, "id-1", "DeepSeek-V4.1-Flash")

# The backend answers with token IDs; the tokenizer passed to the processor
# decodes them.
tokenizer = Tokenizer.from_file("static/tokenizers/v41/tokenizer.json")
token_ids = tokenizer.encode("Hello there!")
processor = StreamProcessor(generator, converted.parsing_options, tokenizer)

chunks = []
for chunk in (
    InferenceChunk.tokens(token_ids),
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
