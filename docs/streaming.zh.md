# 流式响应

[English](streaming.md) | **中文**

`StreamProcessor` 把后端输出转换为 Messages、Chat Completions 或 Responses 事件。以下示例使用 mock 推理并打印 Chat Completions chunk。后端与 HTTP 传输由应用提供。思考模式下，只有为 processor 附加 tokenizer 后，它才会统计 usage 中的 `reasoning_tokens`，见[使用 tokenizer](tokenizer.zh.md#解码)。

## 首 token 时延（TTFT）的计时口径

`InferenceChunk::Ready`（Python 侧 `InferenceChunk.ready(...)`）会立刻产出一个**起始 chunk**：
在 Chat Completions 格式下它带 `delta: {"role": "assistant"}`、`content` 为**空**，真正的第一个
token 在之后的 chunk 里。

因此，若以"第一个 SSE 帧"计 TTFT，量到的其实是**请求被接受 + 分词完成**的时间，此时 prefill
尚未开始，这段时间随后会以"token 间隔变大"的形式体现出来。

应以第一个 `delta.content` 非空的 chunk 计时（思考模式下同时考虑 `delta.reasoning_content`）：

```python
first_frame_ts = None
first_token_ts = None
for chunk in stream:
    now = time.perf_counter()
    first_frame_ts = first_frame_ts or now
    delta = chunk["choices"][0].get("delta", {})
    if delta.get("content") or delta.get("reasoning_content"):
        first_token_ts = first_token_ts or now
```

我们在 32K 输入上实测，两种口径可相差一个数量级以上（0.2 s 与 14.6 s），报告中务必写清用的是哪一种。

## Rust

为你的应用添加依赖：

```sh
cargo add deepseek-recipe@0.2 serde_json@1 tokio-stream@0.1
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
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "disabled"},
    });

    // chunk 生成器使用转换后请求的最终设置。
    let request: ChatCompletionRequest = serde_json::from_value(body)?;
    let converted = request.convert(ConversionOptions::default())?;
    let generator = ChatCompletionRequest::chunk_generator(
        &converted,
        "id-1".into(),
        "DeepSeek-V4.1-Flash".into(),
    );
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
            token_count: 2,
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ]);

    // 将其转换为协议 chunk。
    let mut chunks = std::pin::pin!(processor.into_protocol_stream(inference));
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
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "thinking": {"type": "disabled"},
})
converted = request.convert(ConversionOptions())
generator = ChatCompletionRequest.chunk_generator(converted, "id-1", "DeepSeek-V4.1-Flash")
processor = StreamProcessor(generator, converted.parsing_options)

# 模拟后端的推理 chunk。
chunks = []
for chunk in [
    InferenceChunk.ready(system_fingerprint="fp-1", prompt_usage=PromptUsage(prompt_tokens=4)),
    InferenceChunk.text("Hello!", token_count=2),
    InferenceChunk.finish(InferenceFinishReason.Stop),
]:
    chunks.extend(processor.push(chunk))
chunks.extend(processor.finish())

for chunk in chunks:
    print(chunk.to_json())
processor.close()
```

支持范围见[项目 README](../README.zh.md)，API 约定见 [Rust API 文档](https://docs.rs/deepseek-recipe)。
