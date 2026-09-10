//! Anthropic Messages request conversion and response events.
//!
//! Supports text, images, client tools, and thinking blocks. Document content
//! and hosted tool execution are outside the supported scope. Server tools are
//! rejected by default; web-search handling is configurable through
//! [`crate::request::ConversionOptions::messages_web_search`].
//!
//! Input text must not spell out the image placeholder. Every placeholder in a
//! prompt corresponds to one image source, and the adapter inserts them for
//! image blocks only, so message text, tool references, thinking blocks, tool
//! result text, the top-level `system` field, tool definitions, and historical
//! tool calls are rejected when they carry the placeholder spelling.
//!
//! Use [`crate::request::ProtocolRequest::convert`] to produce a shared
//! [`crate::request::ConversationRequest`], and
//! [`crate::request::ProtocolRequest::chunk_generator`] to construct response
//! events. [`crate::stream::StreamProcessor`] parses backend output into typed
//! [`response::MessagesStreamEvent`] values, which can also be accumulated into
//! a complete [`response::MessagesResponse`].
//!
//! The caller supplies model inference, tool execution, token usage, and HTTP
//! transport.

pub mod request;
pub mod response;
