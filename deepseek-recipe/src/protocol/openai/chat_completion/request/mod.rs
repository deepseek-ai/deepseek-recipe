//! Chat Completions request deserialization and conversion.
//!
//! [`ChatCompletionRequest`] accepts messages, sampling parameters, client
//! tools, and JSON object output. An explicit `thinking` setting overrides
//! `reasoning_effort`, which overrides the caller's thinking default. Sampling
//! values are validated and passed through to the caller.
//!
//! Text, images, and historical tool calls are supported. A tool result must
//! follow the assistant message that requested it, and tool names must be
//! unique. Named or required tool choices require thinking to be disabled when
//! tools are present. JSON Schema output, `logprobs`, and `top_logprobs` are
//! accepted and ignored. Function-tool `strict` flags are passed through for the
//! caller to enforce. Model resolution and tool execution belong to the caller.
//!
//! Input text must not spell out the image placeholder. Every placeholder in a
//! prompt corresponds to one image source, and the adapter inserts them for
//! image blocks only, so message content, assistant reasoning content, tool
//! definitions, and historical tool calls that carry the placeholder spelling
//! are rejected.

pub use schema::*;

mod convert;
mod schema;
