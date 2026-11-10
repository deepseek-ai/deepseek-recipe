//! Anthropic Messages request conversion and response events.
//!
//! Supports text, images, client tools, and thinking blocks. Document content
//! and hosted tool execution are outside the supported scope. Server tools are
//! rejected by default; web-search handling is configurable through
//! [`crate::request::ConversionOptions::messages_web_search`].
//!
//! For [`DeepseekV41Flash`](crate::request::ModelVersion::DeepseekV41Flash),
//! image blocks become placeholders. For
//! [`DeepseekV4Pro`](crate::request::ModelVersion::DeepseekV4Pro), they become
//! the unsupported-image text `[Unsupported Image]` without validation. Text
//! that spells the image token stays plain text.
//!
//! Use [`crate::request::ProtocolRequest::convert`] to produce a shared
//! [`crate::request::ConversationRequest`], and
//! [`crate::request::ProtocolRequest::chunk_generator`] to construct the
//! chunk generator of response events. [`crate::stream::StreamProcessor`]
//! parses backend output into typed [`response::MessagesStreamEvent`] values,
//! which can also be accumulated into a complete
//! [`response::MessagesResponse`].
//!
//! The caller supplies model inference, tool execution, prompt usage, and
//! HTTP transport.

pub mod request;
pub mod response;
