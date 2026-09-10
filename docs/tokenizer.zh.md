# 使用 tokenizer

[English](tokenizer.md) | **中文**

`PromptEncoding` 附加 tokenizer 后可以直接把 prompt 编码为模型的 token IDs，`StreamProcessor` 附加 tokenizer 后可以直接传入 token IDs。`PromptEncoding` 和 `StreamProcessor` 不会自动加载 tokenizer，需要先用 `with_tokenizer` 传入再调用相应方法。未附加 tokenizer 直接使用 `encode` / 传入 token IDs 会返回错误。

请使用与模型匹配的 tokenizer。本仓库在 [`static/tokenizers/`](../static/tokenizers/README.md) 下内置了 V4 与 V4.1 prompt 模板对应的副本，把加载路径指向对应目录中的 `tokenizer.json` 即可。使用 Hugging Face Hub 上的 tokenizer 也可以，但其 special token 拼写必须与 encoding 的常量一致，且 image token 必须解析到相同的 ID。

## Rust

为你的应用添加依赖：

```sh
cargo add deepseek-recipe@0.1 deepseek-recipe-encoding@0.1 serde_json@1 tokenizers@0.23
```

将以下代码保存为 `src/main.rs` 并运行 `cargo run`。下面的 tokenizer 路径指向本仓库内置的文件，否则请换成你自己的 `tokenizer.json`：

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

    // 附加 tokenizer，然后编码对话的 token。
    let tokenizer = Tokenizer::from_file("static/tokenizers/v41/tokenizer.json")?;
    let encoding = DeepseekV41Encoding::new().with_tokenizer(tokenizer);
    let token_ids = encoding.encode(&converted.conversation)?;

    println!("{token_ids:?}");
    Ok(())
}
```

`Tokenizer` 实现了 `with_tokenizer` 所需的 `TokenizerEncoder` trait；任何实现了该 trait 的类型都可以传入。编码分两步：`render_conversation` 返回 prompt 文本及其 image 占位符，`encode` 再对该 prompt 做分词；分词时不额外添加 special token，因为 prompt 文本中已经包含 special token。

### 解码

后端可以返回 token IDs 而不是文本。把同一个 tokenizer 附加到 stream processor 即可解码这些 chunk。该示例还需要 `tokio-stream`，以及带 `macros` 和 `rt-multi-thread` feature 的 `tokio`，见[流式响应](streaming.zh.md)：

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

    // 后端返回 token IDs，附加的 tokenizer 负责解码。
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

processor 会增量解码 id，并在多 token 字符尚未完整时先缓冲；产生文本的 id 计入 completion token。special token 保留在解码结果中，因为它驱动输出解析。未附加 tokenizer 时，`Token` chunk 会以 `StreamError::MissingTokenizer` 结束流。`Tokenizer` 实现了 `with_tokenizer` 所需的 `TokenizerDecoder` trait。

## Python

安装包：`python3 -m pip install deepseek-recipe`。

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

`Tokenizer.from_pretrained` 从 Hugging Face Hub 下载并缓存 `tokenizer.json`，`Tokenizer.from_str` 直接加载文件内容。`render_conversation` 与 `encode` 的结果与 Rust 一致。

### 解码

把 tokenizer 作为 processor 的第三个参数传入。token-id chunk 的解码行为与 Rust 一致；未传入 tokenizer 时，token chunk 会抛出 `RuntimeError` 并关闭 processor。

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

# 后端返回 token IDs，传给 processor 的 tokenizer 负责解码。
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

支持范围见[项目 README](../README.zh.md)，完整 chunk 序列见[流式响应](streaming.zh.md)。
