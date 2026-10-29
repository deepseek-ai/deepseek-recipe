<div align="center">
  <img src="static/deepseek-whale.svg" alt="DeepSeek" width="240">
</div>

# deepseek-recipe

[English](README.md) | **中文**

deepseek-recipe 包含一套 Rust 库以及对应的 Python bindings：支持将不同格式的 API 请求统一转换为 Conversation 格式，将对话渲染为适用于 DeepSeek 模型的 prompt，通过附加的 tokenizer 将对话编码为 token IDs，并支持将模型输出转换为对应格式的响应。
可用于把推理后端接入兼容多种格式的 API 服务；调用方必须提供模型推理、工具执行和 HTTP 传输。

[快速上手](#使用-deepseek-recipe) · [流式响应](docs/streaming.zh.md) · [使用 tokenizer](docs/tokenizer.zh.md) · [许可证](#许可证)

## 支持范围

- **请求/响应格式：** Messages、Chat Completions 与 Responses 的请求转换、[流式响应](docs/streaming.zh.md)和完整响应，支持文本、图片、思考模式、客户端工具调用和 Chat Completions 的 `prefix` 续写。
- **Prompt：** 支持将 DeepSeek V4 与 V4.1 的对话渲染为 prompt，并通过附加的 tokenizer 将对话编码为 token IDs。
- **生成参数：** 思考模式、reasoning effort、`temperature`、`top_p` 和输出 token 上限。调用方编码 prompt 之后，`ConversationRequest::resolve_max_tokens` 按上下文长度检查 prompt 与 `max_tokens`，并给未指定的 `max_tokens` 取默认值：非思考模式 8192，思考模式 65536，思考模式且 `max` effort 时 131072。
- **输出解析：** reasoning 内容、工具调用、JSON object 输出和 stop sequence。流式处理器附加 tokenizer 后，usage 会给出 `reasoning_tokens`：processor 用该 tokenizer 编码 reasoning 内容来计数。
- **结构化输出：** Chat Completions 的 `response_format`（`json_schema` 与 `regex` 类型）和 Responses 的 `text.format`（`json_schema` 类型）。转换会校验 JSON Schema，渲染时把它写入 prompt；正则表达式以 `ResponseFormat::Regex` 透传。转换还会选出 `InferenceOptions` 的 `OutputConstraint`：带工具且 tool choice 不为 `none` 的请求是工具调用块，否则是 response format 对应的 JSON、JSON Schema 或正则。是否由后端以约束解码执行，由调用方决定。
- **图片：** 支持以 base64 或外部 URL 传入，由图片组件基于 OpenCV 完成 DeepSeek V4.1 的预处理。JPEG 的 EXIF 旋转和镜像方向会在缩放及 WebP 编码前应用到像素。DeepSeek V4 不接受图片，转换把 V4 请求中的每张图片替换为文本 `[Unsupported Image]`。图片组件的 `ReqwestImageFetcher` 默认不限制连接的地址，因此如果服务会拉取不受信任请求中的 URL，应为它安装 `DestinationFilter`。
- **工具定义：** 函数工具；Responses 还支持工具命名空间和 `apply_patch` 自定义工具。V4.1 支持 Messages 的工具延迟加载和 Responses 的客户端工具搜索。
- **Log probability：** Chat Completions 的 `logprobs`、`top_logprobs` 以及 Responses 的 `top_logprobs` 会转换到 `InferenceOptions`。调用方在 `InferenceChunk::Tokens` 中提供后端的值；附加 tokenizer 后，Chat Completions 与 Responses 的 adapter 会输出这些值。本库不计算概率。

## 使用 deepseek-recipe

以下示例将 Chat Completions 请求转换为 DeepSeek V4.1 prompt。本库根据请求的 `model` 名解析模型版本，默认接受 `DeepSeek-V4-Pro` 与 `DeepSeek-V4.1-Flash`；其他名称（例如 `deepseek-flash`）用 `ConversionOptions::with_model_alias` 注册。

### Python

#### 安装

Python 3.10+：

```sh
python3 -m pip install deepseek-recipe
```

#### 示例

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

#### 安装

```sh
cargo add deepseek-recipe@0.2 deepseek-recipe-encoding@0.2 serde_json@1
```

源码构建与图片依赖见[开发指南](docs/development.md)。

#### 示例

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

在仓库根目录运行 demo，把请求渲染为 prompt，并将模型输出解析为协议响应：

```sh
cargo run -p encoding-decoding-demo --locked
```

打开 [http://127.0.0.1:7778](http://127.0.0.1:7778)。

### 更多示例

Rust 和 Python 均支持将模型输出转换为流式响应，见[流式响应示例](docs/streaming.zh.md)；将对话编码为 token IDs、以及直接解码后端产生的 token IDs 的示例见[使用 tokenizer](docs/tokenizer.zh.md)。

## 包与示例

| 包 | 用途 |
| --- | --- |
| [`deepseek-recipe`](https://docs.rs/deepseek-recipe) | 协议转换与模型输出解析。 |
| [`deepseek-recipe-core`](https://docs.rs/deepseek-recipe-core) | 共享的对话、消息、图片与工具类型。 |
| [`deepseek-recipe-encoding`](https://docs.rs/deepseek-recipe-encoding) | DeepSeek V4 与 V4.1 的 prompt 渲染与 token 编码。 |
| [`deepseek-recipe-image`](https://docs.rs/deepseek-recipe-image) | 图片拉取与预处理。 |
| [`deepseek-recipe-macros`](https://docs.rs/deepseek-recipe-macros) | prompt 渲染使用的 `prompt!` 宏。 |
| [`deepseek-recipe-python`](deepseek-recipe-python/README.md) | Python bindings，以 `deepseek_recipe` 导入。 |
| [encoding-decoding-demo](encoding-decoding-demo/README.md) | 渲染 prompt、高亮 special token，并将完整模型输出解析为 Chat Completions、Responses 或 Messages 响应的 Web 界面。 |
| [server-rs](server-rs/README.md) | 使用 mock 推理的 Axum API 示例。 |
| [server-py](server-py/README.md) | 使用 mock 推理的 FastAPI 示例。 |

## 许可证

项目代码和公开文档采用 [MIT License](LICENSE)。随仓库分发的 tokenizer 声明见 [static/tokenizers/README.md](static/tokenizers/README.md)。
