<div align="center">
  <img src="static/deepseek-whale.svg" alt="DeepSeek" width="240">
</div>

# deepseek-recipe

[English](README.md) | **中文**

deepseek-recipe 包含一套 Rust 库以及对应的 Python bindings：支持将不同格式的 API 请求统一转换为 Conversation 格式，编码成适用于 DeepSeek 模型的 prompt，并支持将模型输出转换为对应格式的响应。
可用于把推理后端接入兼容多种格式的 API 服务；模型推理、工具执行和 HTTP 传输需由外部提供。

[快速上手](#使用-deepseek-recipe) · [流式响应](docs/streaming.zh.md) · [使用 tokenizer](docs/tokenizer.zh.md) · [许可证](#许可证)

## 支持范围

- **请求/响应格式：** Messages、Chat Completions 与 Responses 的请求转换、[流式响应](docs/streaming.zh.md)和完整响应，支持文本、图片、思考和客户端工具调用。
- **Prompt：** 支持将 DeepSeek V4 与 V4.1 的对话编码为 prompt 或 token IDs。
- **生成参数：** 思考模式、reasoning effort、`temperature`、`top_p` 和输出 token 上限。
- **输出解析：** 思考、工具调用、JSON object 输出和 stop sequence。
- **图片：** 支持以 base64 或外部 URL 传入。提供基于 OpenCV 的 DeepSeek V4.1 图片预处理。
- **工具定义：** 支持函数工具；Responses API 还支持工具命名空间和 `apply_patch` 自定义工具。

## 尚未支持

- Token 概率（`logprobs` 和 `top_logprobs`）。
- 文档内容、音视频输入和通过 `file_id` 获取文件。
- 服务端工具执行，例如 `web_search`。
- JSON Schema、正则表达式输出约束，以及工具 `strict` 设置的强制执行。
- Chat Completions 单次请求生成多个候选回复（`n > 1`）。
- Responses 中除 `apply_patch` 外的自定义工具定义。
- Responses 的会话存储，以及通过 `previous_response_id` 获取上下文。
- Responses 的加密思考内容（`encrypted_content`）。

## 使用 deepseek-recipe

将 Chat Completions 请求转换为 DeepSeek V4.1 prompt：

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
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
})
converted = request.convert(ConversionOptions())
rendered = DeepseekV41Encoding().render_conversation(converted.conversation)
print(rendered.prompt)
```

### Rust

#### 安装

```sh
cargo add deepseek-recipe@0.1 deepseek-recipe-encoding@0.1
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

在仓库根目录运行编解码 demo：

```sh
cargo run -p encoding-decoding-demo --locked
```

打开 [http://127.0.0.1:7778](http://127.0.0.1:7778)。

### 更多示例

Rust 和 Python 均支持将模型输出转换为流式响应，见[流式响应示例](docs/streaming.zh.md)；将对话编码为 token IDs、以及直接解码后端返回的 token IDs 的示例见[使用 tokenizer](docs/tokenizer.zh.md)。

## 包与示例

| 包 | 用途 |
| --- | --- |
| [`deepseek-recipe`](https://docs.rs/deepseek-recipe) | 协议转换与模型输出解析。 |
| [`deepseek-recipe-core`](https://docs.rs/deepseek-recipe-core) | 共享的对话、消息、图片与工具类型。 |
| [`deepseek-recipe-encoding`](https://docs.rs/deepseek-recipe-encoding) | DeepSeek V4 与 V4.1 的 prompt 渲染与 token 编码。 |
| [`deepseek-recipe-image`](https://docs.rs/deepseek-recipe-image) | 图片拉取与预处理。 |
| [`deepseek-recipe-python`](deepseek-recipe-python/README.md) | Python bindings，以 `deepseek_recipe` 导入。 |
| [encoding-decoding-demo](encoding-decoding-demo/README.md) | 编码 prompt、查看 special token，并将完整模型输出解码为 Chat Completions、Responses 或 Messages 的 Web 界面。 |
| [server-rs](server-rs/README.md) | 使用 mock 推理的 Axum API 示例。 |
| [server-py](server-py/README.md) | 支持 mock 推理的 FastAPI 示例。 |

## 许可证

项目代码和公开文档采用 [MIT License](LICENSE)。随仓库分发的 tokenizer 声明见 [static/tokenizers/README.md](static/tokenizers/README.md)。
