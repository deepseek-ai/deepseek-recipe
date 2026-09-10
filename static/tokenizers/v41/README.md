# V4.1 tokenizer

[`tokenizer.json`](tokenizer.json) is a modified copy of the Hugging Face file
[`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tokenizer.json`][upstream] with the image
token reassigned.

## Local changes

| id | this file | upstream |
| --- | --- | --- |
| 129264 | `<｜image｜>` special=true | `<｜deepseek_image｜>` special=false |

All other JSON entries match the comparison revision below.

## Provenance and license

- Comparison revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`, verified on 2026-09-10.
- Local SHA-256: `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`.
- Upstream license: [MIT][upstream-license], `Copyright (c) 2023 DeepSeek`.
- Preserved license notice: [../LICENSE](../LICENSE).

The original import revision was not recorded; the comparison revision identifies
the upstream JSON used to verify the changes above.

[upstream]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/tokenizer.json
[upstream-license]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/LICENSE
