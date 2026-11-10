# 使用 tokenizer

[English](tokenizer.md) | **中文**

`PromptEncoding` 附加 tokenizer 后可以把对话编码为模型的 token IDs，`StreamProcessor` 附加 tokenizer 后可以直接传入 token IDs。`PromptEncoding` 和 `StreamProcessor` 不会自动加载 tokenizer，需要先用 `with_tokenizer` 传入再调用相应方法。未附加 tokenizer 时，`encode` 返回 `EncodingError::MissingTokenizer`，`Tokens` chunk 会以 `StreamError::MissingTokenizer` 结束流。

请使用与模型匹配的 tokenizer。本仓库在 [`static/tokenizers/`](../static/tokenizers/README.md) 下内置了 V4 与 V4.1 prompt encoding 对应的副本，从对应目录加载 `tokenizer.json` 即可。也可以从 Hugging Face Hub 加载 tokenizer：其 added vocabulary 必须包含渲染写入的每个 special token 的拼写，否则 `encode` 返回 `EncodingError::Encode`。`IMAGE_SPECIAL_TOKEN_ID` 是内置 V4.1 tokenizer 中图片 token 的 token ID；使用该常量的调用方需要图片 token ID 与之相同的 tokenizer。

## Rust

为你的应用添加依赖：

```sh
cargo add deepseek-recipe@0.2 deepseek-recipe-encoding@0.2 serde_json@1 tokenizers@0.23
```

将以下代码保存为 `src/main.rs`。tokenizer 路径相对于工作目录，因此请在本仓库根目录用 `cargo run --manifest-path path/to/your-app/Cargo.toml` 运行应用；在其他目录运行时，请改用绝对路径或你自己的 `tokenizer.json`：

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

    // 附加 tokenizer，然后把对话编码为 token IDs。
    let tokenizer = Tokenizer::from_file("static/tokenizers/v41/tokenizer.json")?;
    let encoding = DeepseekV41Encoding::new().with_tokenizer(PromptTokenizer::new(&tokenizer));
    let token_ids = encoding.encode(&converted.conversation)?;

    println!("{token_ids:?}");
    Ok(())
}
```

给 `with_tokenizer` 传入 `PromptTokenizer`（它为 `Tokenizer` 另存一份不含 added vocabulary 的副本），或传入任何其他 `TokenizerEncoder` 实现。`encode` 先渲染对话，再对 prompt 做分词。

`render_conversation` 只执行渲染这一步：它返回 prompt 文本、其中显式 special token 的字节区间（`special_ranges`）以及图片来源。渲染把模板的控制 token、消息图片来源对应的占位符，以及 assistant 内容和 reasoning 内容中的 grounding token 作为显式 special token 插入。分词时，每个 special 区间编码为其拼写对应的 token ID，区间之间的文本是 plain text，不使用 added vocabulary 编码；`PromptTokenizer` 把各 plain 区域作为一个 batch 编码；`tokenizers` crate 可以并行编码该 batch，取决于其 `TOKENIZERS_PARALLELISM` 设置。

因此，用户消息里写出的 special token 拼写是 plain text，不会成为控制 token。调用方自行对 `RenderedPrompt::prompt` 分词时必须同样处理 `special_ranges`。

编码之后，调用方用 `ConversationRequest::resolve_max_tokens(prompt_tokens, context_limit)` 解析要下发的 `max_tokens`。`prompt_tokens` 是编码得到的 token ID 数加上图片超出占位符的 token 数（`MultiModalData::image_token_adjustment`），`context_limit` 是模型上下文可容纳的 token 数。请求指定的 `max_tokens` 保持不变；未指定时，非思考模式取 8192，思考模式且 reasoning effort 为 `max` 时取 131072，其余思考模式取 65536，并降到 prompt 之后剩余的上下文以内。prompt 与请求的 `max_tokens` 之和超过 `context_limit` 时，该方法返回 `ConversionError`。

### 解码

后端可以产生 token IDs 而不是文本。把同一个 tokenizer 附加到 stream processor 即可解码这些 chunk。该示例还需要 `tokio-stream`，以及带 `macros` 和 `rt-multi-thread` feature 的 `tokio`，见[流式响应](streaming.zh.md)。tokenizer 路径的要求与编码示例相同：

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

    // 后端返回 token IDs，附加的 tokenizer 负责解码。
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

processor 会增量解码 token IDs，并在多 token 字符尚未完整时先缓冲；产生文本的 token IDs 计入 completion token。每个 special token 单独解码成一段并带标记，因此输出解析器只把 token 形式的 `</think>` 与 DSML 标签当作标记；用普通 token 拼出的同样文本是内容。文本 chunk 没有 token 边界，其标记按拼写匹配。未附加 tokenizer 时，`Tokens` chunk 会以 `StreamError::MissingTokenizer` 结束流。`Tokenizer` 实现了 `with_tokenizer` 所需的 `StreamTokenizer` trait。

`InferenceChunk::Tokens` 携带后端一次输出的全部 token IDs，processor 会一起解码。转换后的请求设置了 `InferenceOptions::logprobs` 时，调用方为每个 token ID 提供一个 `InferenceTokenLogprob`：解码出的文本会带上其 token 的 log probability，由 Chat Completions 与 Responses 输出。log probability 数量与 token ID 数量不一致的 `Tokens` chunk 会以 `StreamError::LogprobCountMismatch` 结束流。思考模式下，processor 还会用附加的 tokenizer 编码已输出的 reasoning 内容，统计 usage 中的 `reasoning_tokens`，文本 chunk 与 token chunk 均如此。未附加 tokenizer 时，Chat Completions 的 usage 不含 `completion_tokens_details`，Responses 的 usage 中 `reasoning_tokens` 为 `0`。

## Python

安装包：`python3 -m pip install deepseek-recipe`。tokenizer 路径的要求与 Rust 示例相同：

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

`Tokenizer.from_pretrained` 从 Hugging Face Hub 下载并缓存 `tokenizer.json`，`Tokenizer.from_str` 直接加载文件内容。`render_conversation` 与 `encode` 的结果与 Rust 一致；未附加 tokenizer 或编码失败时，`encode` 抛出 `RuntimeError`。编码之后，`ConversationRequest.resolve_max_tokens` 按与 Rust 相同的规则解析 `max_tokens`，在 Rust 返回错误的情况下抛出 `ConversionError`。

### 解码

把 tokenizer 作为 processor 的第三个参数传入。`tokens` chunk 的解码行为与 Rust 一致；未传入 tokenizer 时，这类 chunk 会抛出 `RuntimeError` 并关闭 processor。

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

# 后端返回 token IDs，传给 processor 的 tokenizer 负责解码。
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

支持范围见[项目 README](../README.zh.md)，完整 chunk 序列见[流式响应](streaming.zh.md)。
