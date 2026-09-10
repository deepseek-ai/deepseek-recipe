# 流式响应

[English](streaming.md) | **中文**

`StreamProcessor` 把后端输出转换为 Messages、Chat Completions 或 Responses 事件。以下示例使用 mock 推理并打印 Chat Completions chunk。后端与 HTTP 传输由应用提供。

## Rust

为你的应用添加依赖：

```sh
cargo add deepseek-recipe@0.1 serde_json@1 tokio-stream@0.1
cargo add tokio@1 --features macros,rt-multi-thread
```

将以下代码保存为 `src/main.rs`，然后运行 `cargo run`：

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

    // chunk 生成器使用转换后请求的最终设置。
    let request: ChatCompletionRequest = serde_json::from_value(body)?;
    let converted = request.convert(ConversionOptions::default())?;
    let generator =
        ChatCompletionRequest::chunk_generator(&converted, "id-1".into(), "deepseek-flash".into());
    let processor = StreamProcessor::new(generator, converted.parsing_options);

    // 模拟后端的推理 chunk。
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

    // 将其转换为协议 chunk。
    let mut chunks = std::pin::pin!(processor.process(inference));
    while let Some(chunk) = chunks.next().await {
        println!("{}", serde_json::to_string(&chunk?)?);
    }
    Ok(())
}
```

## Python

安装包：`python3 -m pip install deepseek-recipe`。
`push` 处理一个后端 chunk；`finish` 完成响应处理。

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

# 模拟后端的推理 chunk。
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

支持范围见[项目 README](../README.zh.md)，API 约定见 [Rust API 文档](https://docs.rs/deepseek-recipe)。
