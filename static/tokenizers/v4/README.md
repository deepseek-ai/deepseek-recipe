# V4 tokenizer

[`tokenizer.json`](tokenizer.json) is a modified copy of the Hugging Face file
[`deepseek-ai/DeepSeek-V4-Pro/tokenizer.json`][upstream] with the image tokens
reassigned.

## Local changes

| id | this file | upstream |
| --- | --- | --- |
| 129264 | `<｜image｜>` special=true | `<｜image2｜>` special=false |
| 129279 | `<｜image2｜>` special=false | `<｜image｜>` special=true |

All other JSON entries match the comparison revision below.

## Provenance and license

- Comparison revision: `b5968e9190ef611bbf34a7229255be88a0e937c1`, verified on 2026-09-10.
- Local SHA-256: `97d2f31b020d18b5aee5c9b3d5b4efb10ea210f3fe3f7dffe3f1cd90542d6b19`.
- Upstream license: [MIT][upstream-license], `Copyright (c) 2023 DeepSeek`.
- Preserved license notice: [../LICENSE](../LICENSE).

The original import revision was not recorded; the comparison revision identifies
the upstream JSON used to verify the changes above.

[upstream]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json
[upstream-license]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/LICENSE
