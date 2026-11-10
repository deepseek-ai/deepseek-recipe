//! Responses request deserialization and conversion.
//!
//! [`ResponsesRequest`] supports text, images, unencrypted reasoning history
//! (`reasoning_text` content), instructions, sampling parameters, reasoning
//! effort, and JSON object and JSON Schema output. Conversion validates the
//! schema of a `json_schema` format and passes it through as
//! `ResponseFormat::JsonSchema`; rendering writes the schema into the prompt.
//! Constrained decoding belongs to the caller. Client tools include functions,
//! namespaces, the `apply_patch` custom tool, and client-executed
//! `tool_search`.
//!
//! Document content, encrypted reasoning, conversation storage, and hosted tool
//! execution are outside the supported scope. Web-search declarations and
//! choices are dropped by default; [`WebSearchBehavior::Reject`] rejects them
//! instead.
//!
//! For [`ModelVersion::DeepseekV41Flash`], `input_image` blocks become
//! placeholders. For [`ModelVersion::DeepseekV4Pro`], they become the
//! unsupported-image text `[Unsupported Image]` without validation. Text that
//! spells the image token stays plain text.
//!
//! `top_logprobs` enables output token log probabilities. Conversion
//! normalizes the parameters of a function tool with `strict: true` as strict
//! tool parameters and passes the flag through; enforcement during inference
//! belongs to the caller.
//!
//! Conversion preserves response settings, custom tool names, and namespace
//! mappings for [`ProtocolRequest::chunk_generator`].
//!
//! Some rules follow the model version that conversion resolves from the
//! request's model name through [`ConversionOptions::model_versions_by_name`].
//! For V4.1, a namespace tool is named `{namespace}::{name}` and its
//! description starts with the namespace description, and an
//! `<image_resize_notice>` developer message that directly follows a tool
//! output is placed right after that output. For V4, a namespace tool is named
//! `{namespace}__{name}`, or `{namespace}{name}` when the namespace ends with
//! or the name starts with an underscore, and `apply_patch` uses the V4
//! description. The caller supplies model inference, tool execution, and HTTP
//! transport.
//!
//! [`WebSearchBehavior::Reject`]: crate::request::WebSearchBehavior::Reject
//! [`ModelVersion::DeepseekV41Flash`]:
//!     crate::request::ModelVersion::DeepseekV41Flash
//! [`ModelVersion::DeepseekV4Pro`]: crate::request::ModelVersion::DeepseekV4Pro
//! [`ProtocolRequest::chunk_generator`]:
//!     crate::request::ProtocolRequest::chunk_generator
//! [`ConversionOptions::model_versions_by_name`]:
//!     crate::request::ConversionOptions::model_versions_by_name

pub use schema::*;

mod convert;
mod schema;
mod tools;
