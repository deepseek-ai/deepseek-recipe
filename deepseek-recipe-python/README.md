# deepseek-recipe-python

Python bindings for deepseek-recipe: conversion of Messages, Chat Completions,
and Responses requests, rendering of DeepSeek V4/V4.1 conversations into
prompts, encoding of conversations into token IDs with an attached tokenizer,
image preprocessing, and response streaming. The caller must provide model
inference, tool execution, and HTTP transport.

## Installation

Python 3.10 or later:

```sh
python3 -m pip install deepseek-recipe
```

Import as `deepseek_recipe`. Source builds require Rust, OpenCV 4.x, and
Clang/libclang.

## Example

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

## License

MIT. The distribution includes the license text.
