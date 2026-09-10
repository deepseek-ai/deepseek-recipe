# Bundled tokenizers

These files are modified copies of tokenizers published by DeepSeek. The local
changes align the image special token with the prompt encodings in this
repository. Use the matching tokenizer when converting rendered prompts to
token IDs; the encodings do not load these files automatically.

| Directory | Upstream model | Local changes and verification |
| --- | --- | --- |
| `v4/` | [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | [V4 tokenizer notes](v4/README.md) |
| `v41/` | [DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | [V4.1 tokenizer notes](v41/README.md) |

Both upstream tokenizers are provided under the MIT License, with
`Copyright (c) 2023 DeepSeek`. The complete upstream notice is preserved in
[LICENSE](LICENSE) and applies to the bundled tokenizer copies. Retain this
notice when redistributing them.

Each tokenizer's README records the upstream revision used for comparison,
the local SHA-256 checksum, and the semantic changes. When updating a tokenizer,
verify its provenance and license, compare the JSON contents, and update these
records together.
