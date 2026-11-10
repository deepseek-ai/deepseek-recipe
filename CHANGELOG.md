# Changelog

This file lists the changes to the Rust crates and to the `deepseek-recipe`
Python distribution. All crates and the distribution share one version. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/): before 1.0, a new
minor version can contain incompatible changes.

## Unreleased

## 0.2.0 - 2026-10-20

This release changes the public API of every crate and of the Python package.
[Migrating from 0.1](#migrating-from-01) lists the renamed and removed items.

### Added

- **Model versions.** Conversion resolves a `ModelVersion` (V4 or V4.1) from
  the request's `model` name through
  `ConversionOptions::model_versions_by_name`. The defaults are
  `DeepSeek-V4-Pro` and `DeepSeek-V4.1-Flash`; `with_model_alias` and
  `with_model_versions_by_name` register other names.
  `ConversationRequest::model_version` reports the result. V4 does not accept
  images (`ModelVersion::supports_image`): conversion replaces each image of a
  V4 request with the text `[Unsupported Image]`. Python:
  `ConversionOptions.model_versions_by_name`, `with_model_alias`, and
  `with_model_versions_by_name`.
- **Sampling values by model version.** Conversion resolves `temperature` and
  `top_p` instead of passing them through. `temperature` is 1.0 in thinking
  mode and otherwise the requested value or 1.0. `top_p` follows
  `ModelVersion::resolve_top_p`: in thinking mode, V4.1 raises it to at least
  0.95, and V4 uses 0.95 with tools and 1.0 without; outside thinking mode,
  V4.1 uses 1.0 and V4 uses the requested value.
- **Log probabilities.** Conversion maps Chat Completions `logprobs` and
  `top_logprobs`, and Responses `top_logprobs`, into `InferenceOptions`. The
  caller supplies backend values in `InferenceChunk::Tokens`, and the Chat
  Completions and Responses adapters report them. `OutputChunk::Content` and
  `OutputChunk::Reasoning` carry them as `logprobs`.
- **Reasoning token counts.** `CompletionUsage::reasoning_tokens`: in thinking
  mode, a stream processor with an attached tokenizer counts the tokens of the
  reasoning content, and Chat Completions and Responses report them.
- **Structured output.** `ResponseFormat::JsonSchema` and
  `ResponseFormat::Regex`. Conversion validates a Chat Completions
  `response_format` or Responses `text.format` of type `json_schema`, which 0.1
  accepted and ignored, and rendering writes the schema into the prompt. A Chat
  Completions `regex` response format passes through as
  `ResponseFormat::Regex`.
- **Explicit special tokens.** The new `deepseek-recipe-macros` crate provides
  the `prompt!` macro, and `deepseek-recipe-core` provides `Prompt`.
  `RenderedPrompt::special_ranges` records the byte ranges of the explicit
  special tokens, and `PromptTokenizer` encodes only those ranges as special
  tokens. Apart from the grounding tokens in assistant content and reasoning
  content, a special-token spelling in message text encodes as plain text.
- **Prefix continuation.** A final Chat Completions assistant message with
  `prefix: true` continues that message instead of starting a new assistant
  turn. `InputMessage::Assistant` gains `prefix`. Python: the `prefix` argument
  of `AssistantMessage`.
- **Deferred tool loading and tool search.** V4.1 supports deferred tool
  loading in Messages and client-executed `tool_search` in Responses.
  `ToolDefinition` gains `defer_loading`, and `InputMessage::System` gains
  `tools`. Python: `ToolDefinition.defer_loading` and `SystemMessage.tools`.
- **System images.** V4.1 accepts images in system messages.
  `InputMessage::System` gains `image_sources`. Python: the `image_sources`
  and `image_token` arguments of `SystemMessage`.
- **Image destination filtering.** `DestinationFilter` and
  `ReqwestImageFetcher::with_destination_filter` check the URL host, every
  resolved address, and every redirect target. A rejected destination fails
  with `ImageError::DestinationRejected`. Python:
  `ReqwestImageFetcher.with_destination_filter`.
- **Request settings in responses.** Conversion preserves usage inclusion,
  tool output types, namespace mappings, and echoed request settings in
  `ConversationRequest::response_options`. `ProtocolRequest::chunk_generator`
  applies them when constructing a generator. Python: the request types'
  `chunk_generator` methods.
- **Max tokens upper bounds.**
  `ConversionOptions::thinking_max_tokens_upper_bound` and
  `ConversionOptions::non_thinking_max_tokens_upper_bound`, set by the matching
  `with_*` methods: the largest accepted `max_tokens` in thinking mode and
  outside it, both defaulting to `DEFAULT_MAX_TOKENS_UPPER_BOUND` (393216).
  `resolve_max_tokens_upper_bound` selects the bound of a request's thinking
  mode; Messages lowers a larger value to it, Chat Completions and Responses
  reject it. Python: the matching keywords, properties, and `with_*` methods
  of `ConversionOptions`.
- **Output constraints.** `InferenceOptions::output_constraint` holds the
  `OutputConstraint` that constrained decoding would enforce. A request with
  tools and a tool choice other than `none` gets `ToolCalls`, which lists the
  tools and whether a call is required. Otherwise a JSON object or JSON Schema
  response format gets `Json`, and a regex response format gets `Regex`.
  `OutputConstraint::kind` returns `json`, `regex`, `tool_call`, or
  `tool_call_required`, and `OutputConstraint::schema` returns the constraint
  text. Whether the backend applies the constraint is the caller's decision.
  Python: `OutputConstraint` and `InferenceOptions.output_constraint`.
- `ConversationRequest::resolve_max_tokens`: after the caller encodes the
  prompt, it resolves the `max_tokens` to send from the prompt's token count
  and a caller-supplied context limit. A request without `max_tokens` gets 8192
  outside thinking mode, 65536 in thinking mode, and 131072 in thinking mode
  with `max` effort (`DEFAULT_NON_THINKING_MAX_TOKENS`,
  `DEFAULT_THINKING_MAX_TOKENS`, `DEFAULT_MAX_EFFORT_MAX_TOKENS`), lowered to
  the context left after the prompt. When the prompt and the requested
  `max_tokens` together exceed the context limit, it returns a `400 Bad
  Request` error whose message matches the DeepSeek API: `This model's maximum
  context length is {context_limit} tokens. ...`. Python:
  `ConversationRequest.resolve_max_tokens`.
- `SpecialToken` names the grounding tokens (`Ref`, `RefEnd`, `Box`, `BoxEnd`,
  `Point`, `PointEnd`, `Polygon`, `PolygonEnd`), so a `SpecialTokens` mapping
  can respell them; `SpecialTokensMap` has a field for each. The encoding crate
  lists them as the assistant special tokens that rendering inserts from
  assistant content and reasoning content. Python: the matching `SpecialToken`
  members and `SpecialTokens` keywords.
- `deepseek_recipe_core::multimodal::UNSUPPORTED_IMAGE_ERROR_MESSAGE`: the text
  of the unsupported-image error.
- Shared request types `RequestReasoningEffort` and `ThinkingType`.
- Request schema types: `MessagesMetadata` validates `metadata`, `MessagesTool`
  distinguishes client tools from server tools (`MessagesServerTool`), and
  `ResponsesWebSearchTool` deserializes and validates the `search_context_size`
  and `user_location` of a `web_search` or `web_search_2025_08_26` tool.
- Python: the `ModelVersion` enum, a `ResponsesEcho` constructor with
  read-only getters, the `response_schema` and `response_regex` arguments of
  `Conversation`, the `image_token` argument of `UserMessage` and
  `ToolMessage`, an optional `tool_call_id` of `ToolMessage`, and the `model`
  property of the protocol requests, readable before `convert`.

### Changed

- Conversion rejects a request whose `model` is not a key of
  `ConversionOptions::model_versions_by_name`. Register any additional name,
  such as `deepseek-flash`, with `with_model_alias`.
- Validation error messages and the order of the checks match the DeepSeek API.
- A Responses reasoning item without text adds nothing to the conversation;
  `encrypted_content` is accepted and ignored.
- V4.1 preserves Messages `system` and Responses `developer` messages as
  system messages.
- Messages document blocks require a `source` object. Their unsupported-content
  text includes the media type.
- A complete Chat Completions response numbers its `tool_calls` with `index`
  (`ChatCompletionIndexedToolCall`), as the streamed deltas do.
- `StreamProcessor::with_tokenizer` takes a `StreamTokenizer`. Besides
  `decode_ids`, the trait has the optional methods `id_to_token`, for log
  probabilities, `count_tokens`, for reasoning token counts, and
  `is_special_token`, for marker matching.
- `PromptTokenizer` encodes the plain text regions of a prompt as one batch
  through `Tokenizer::encode_batch_fast` instead of one region at a time. The
  `tokenizers` crate encodes the batch in parallel unless its
  `TOKENIZERS_PARALLELISM` environment variable disables parallelism.
- Conversion rejects a request that mixes strict and non-strict tools, and a
  request that sets a response format together with strict tools, unless the
  tool choice is `none`.
- With token input, the stream processor decodes each special token as its own
  piece. Output parsing matches the thinking end token (`</think>` by default)
  and the DSML token within DSML tags only against decoded special tokens; the
  same text spelled with plain tokens is content. The log probabilities of
  these special tokens are reported with the marker rather than with the
  surrounding content. Text chunks still match markers by spelling.
- The Python distribution takes its version from the Cargo package, so
  `deepseek_recipe.__version__` matches the installed distribution.

### Removed

- `InferenceChunk::Token`. Send one `InferenceChunk::Tokens` chunk instead.
- The `arguments` field of `OutputChunk::ToolCall`, now `ToolCallNamed`.
  Arguments arrive in `ToolCallArgumentsDelta` chunks.
- The `util` modules of `deepseek-recipe` and `deepseek-recipe-core`, and the
  public `state_machine` module and `StreamDecoder` type of `deepseek-recipe`.
  Output parsing and token decoding are internal to `StreamProcessor`;
  `ParsingOptions` and `ThinkingInitialStage` remain public in `stream`.
- The per-protocol reasoning effort and thinking type enums, replaced by
  `RequestReasoningEffort` and `ThinkingType`.
- `MessagesContentBlock::ToolReference`. A Messages request can contain a tool
  reference only in `tool_result.content` or `tool_addition.tool`.
- Public constructors of `ChatCompletionChunkGenerator` and
  `ResponsesChunkGenerator`. Use `ProtocolRequest::chunk_generator` with a
  converted request. Python: use the request type's `chunk_generator` method.
- `ChatCompletionRequest::include_usage`,
  `ChatCompletionChunkGenerator::with_include_usage`,
  `ResponsesRequest::custom_tool_names`, and
  `ResponsesChunkGenerator::with_custom_tool_names`, including their Python
  equivalents.
- `InferenceOptions::thinking_budget_tokens` and
  `InferenceOptions::disable_parallel_tool_use`. Conversion accepts and ignores
  Messages `thinking.budget_tokens` and `disable_parallel_tool_use` and Chat
  Completions `parallel_tool_calls`, as the DeepSeek API does. Python: the
  `thinking_budget_tokens` and `disable_parallel_tool_use` properties of
  `InferenceOptions`.
- Python: `InferenceChunk.token()` and `InferenceChunk.token_id`.

### Migrating from 0.1

#### Rust

| 0.1 | 0.2 |
| --- | --- |
| `ConversationRequest::new(conversation)` | `ConversationRequest::new(conversation, model_version)` |
| `ConversationRequest::model` | `ConversationRequest::model_version`; the model name stays on the protocol request |
| `ChatCompletionChunkGenerator::new(...)`, `ResponsesChunkGenerator::new(...)` | `ChatCompletionRequest::chunk_generator(&converted, id, model)`, `ResponsesRequest::chunk_generator(&converted, id, model)` |
| `ConversionError::into_response` | `ConversionError::into_error_response` |
| `StreamProcessor::process` | `StreamProcessor::into_protocol_stream` |
| `TokenizerDecoder` | `StreamTokenizer` |
| `InferenceChunk::Text { content, content_tokens }` | `InferenceChunk::Text { content, token_count }` |
| `InferenceChunk::Token { token_id }` | `InferenceChunk::Tokens { token_ids: vec![token_id], logprobs: None }` |
| `OutputChunk::Start { usage }` | `OutputChunk::Start { prompt_usage }` |
| `OutputChunk::Raw { content }` | `OutputChunk::Content { content, logprobs }` |
| `OutputChunk::Reasoning { content }` | `OutputChunk::Reasoning { content, logprobs }` |
| `OutputChunk::ToolCall { tool_name, arguments }` | `OutputChunk::ToolCallNamed { tool_name }` |
| `OutputChunk::ToolArgumentsDelta { content }` | `OutputChunk::ToolCallArgumentsDelta { arguments }` |
| `OutputChunk::Finish { reason, usage, .. }` | `OutputChunk::Finish { finish_reason, completion_usage, .. }` |
| `ParsingOptions::reasoning_initial_stage: Option<ReasoningStage>` | `ParsingOptions::thinking_initial_stage: Option<ThinkingInitialStage>` |
| `ParsingOptions::tool_call_initial_stage` | `ParsingOptions::starts_in_tool_calls` |
| `ReasoningStage::start_from_reasoning` | `ThinkingInitialStage::starts_in_reasoning` |
| `ReasoningContentBehavior::Ignore` | `ReasoningContentBehavior::FillEmpty` |
| `WebSearchBehavior::Ignore` | `WebSearchBehavior::Drop` |
| `util::append_delta::AppendDelta` | `response::Accumulate` |
| `util::json_formatter` | `deepseek_recipe_core::json` |
| `ImageTokenSpec::calc_token_len` | `ImageTokenSpec::calc_token_count` |
| `ImageTokenSpec::max_n_token` | `ImageTokenSpec::max_token_count` |
| `MessagesReasoningEffort`, `ChatCompletionReasoningEffort`, `ResponsesReasoningEffort` | `RequestReasoningEffort` |
| `MessagesThinkingType`, `ChatCompletionThinkingType` | `ThinkingType` |
| `MessagesTextOrTextBlocks`, `MessagesTextBlock` | `MessagesSystemContent`, `MessagesSystemBlock` |
| `MessagesTool { name, description, input_schema, kind }` | `MessagesTool::Client { name, description, input_schema, defer_loading }`, `MessagesTool::Server(MessagesServerTool)` |
| `MessagesResponseContent`, `MessagesDelta`, `MessagesStopInfo` | `MessagesResponseContentBlock`, `MessagesContentBlockDelta`, `MessagesMessageDelta` |
| `ChatCompletionToolChoiceMode`, `ChatCompletionNamedToolChoice` | variants of `ChatCompletionToolChoiceOption` |
| `ResponsesToolChoiceMode` | variants of `ResponsesToolChoice` |

#### Python

| 0.1 | 0.2 |
| --- | --- |
| `ConversationRequest(conversation)` | `ConversationRequest(conversation, model_version)` |
| `ConversationRequest.model` | `ConversationRequest.model_version`; read the name from the protocol request's `model` before `convert` |
| `ChatCompletionChunkGenerator(...)`, `ResponsesChunkGenerator(...)` | `ChatCompletionRequest.chunk_generator(converted, id, model)`, `ResponsesRequest.chunk_generator(converted, id, model)` |
| `*Response.append(chunk)` | `*Response.accumulate(chunk)` |
| `*Response.chunk_event_type(chunk)` | `*Response.sse_event_name(chunk)` |
| `*Response.done_message()` | `*Response.sse_done_data()` |
| `InferenceChunk.text(text, content_tokens)` | `InferenceChunk.text(content, token_count)` |
| `InferenceChunk.content_tokens` | `InferenceChunk.token_count` |
| `InferenceChunk.token(token_id)` | `InferenceChunk.tokens([token_id])` |
| `ReasoningStage` | `ThinkingInitialStage` |
| `ReasoningStage.start_from_reasoning` | `ThinkingInitialStage.starts_in_reasoning` |
| model version strings such as `"DeepSeek-V4-Pro"` | `ModelVersion.DeepseekV4Pro`, `ModelVersion.DeepseekV41Flash` |

## 0.1.0 - 2026-09-10

Initial release: conversion of Messages, Chat Completions, and Responses
requests into a shared conversation, DeepSeek V4 and V4.1 prompt rendering,
image fetching and preprocessing, streaming output parsing into protocol
responses, and the `deepseek-recipe` Python distribution.
