# deepseek-recipe-python

Python bindings for deepseek-recipe: conversion of Messages, Chat Completions,
and Responses requests, DeepSeek V4/V4.1 prompt and token encoding, image
preprocessing, and response streaming. Model inference, tool execution, and HTTP
transport are provided by the application.

## Installation

Python 3.10 or later:

```sh
python3 -m pip install deepseek-recipe
```

Import as `deepseek_recipe`. Source builds require Rust, OpenCV 4.x, and Clang/libclang.

## Example

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

## License

MIT. The distribution includes the license text.
