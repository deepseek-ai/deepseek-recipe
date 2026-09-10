//! Responses request deserialization and conversion.
//!
//! [`ResponsesRequest`] supports text, images, plaintext reasoning history,
//! instructions, sampling parameters, reasoning effort, and JSON object output.
//! Client tools include functions, namespaces, and the `apply_patch` custom tool.
//!
//! Document content, encrypted reasoning, conversation storage, and hosted tool
//! execution are outside the supported scope. Web-search declarations and
//! choices are ignored by default; [`crate::request::WebSearchBehavior::Reject`]
//! rejects them instead.
//!
//! Input text must not spell out the image placeholder. Every placeholder in a
//! prompt corresponds to one image source, and the adapter inserts them for
//! image blocks only, so text input, instructions, message text, tool output
//! text, reasoning text, tool definitions, and historical tool calls that carry
//! the placeholder spelling are rejected.
//!
//! JSON Schema output, `logprobs`, and `top_logprobs` are accepted and ignored.
//! Function-tool `strict` settings are passed through for the caller to enforce.
//! The caller supplies model inference, tool execution, and HTTP transport.

pub use schema::*;

mod convert;
mod schema;
mod tools;
