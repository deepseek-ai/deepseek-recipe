//! API protocol conversion and inference output parsing.
//!
//! [`request::ProtocolRequest::convert`] produces a shared
//! [`request::ConversationRequest`]. [`stream::StreamProcessor`] parses backend
//! output, passes it to the selected protocol's chunk generator
//! ([`stream::ChunkGenerator`]), and accumulates completion usage. The caller
//! renders and encodes the prompt, supplies the backend and its prompt usage,
//! and transports the protocol responses.

pub use protocol::anthropic;
pub use protocol::openai;

pub mod error_response;
mod protocol;
pub mod request;
pub mod response;
pub mod stream;
