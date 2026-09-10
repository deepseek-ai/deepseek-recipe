//! Anthropic-compatible protocol types and conversion.

pub use messages::request::MessagesRequest;
pub use messages::response::{MessagesResponse, MessagesStreamEvent};

pub mod messages;
