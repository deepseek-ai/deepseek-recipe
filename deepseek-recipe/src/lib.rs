//! API protocol conversion and inference output parsing.
//!
//! [`request::ProtocolRequest::convert`] produces a shared
//! [`request::ConversationRequest`]. [`stream::StreamProcessor`] parses backend
//! output and passes it to the selected protocol's event generator. The caller
//! supplies prompt rendering, token encoding, inference, usage, and transport.

pub use protocol::anthropic;
pub use protocol::openai;

pub mod error_response;
mod protocol;
pub mod request;
pub mod response;
pub mod stream;
pub mod util;
