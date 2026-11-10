//! Chat Completions request deserialization and conversion.
//!
//! [`ChatCompletionRequest`] accepts messages, sampling parameters, client
//! tools, JSON object, JSON Schema, and regex output, and log probabilities. An
//! explicit `thinking` setting overrides `reasoning_effort`, which overrides
//! the caller's thinking default. Conversion validates the sampling parameters.
//! It passes through `max_tokens`, `seed`, `logprobs`, `top_logprobs`, and
//! outside thinking mode the temperature, which defaults to 1.0; in thinking
//! mode the temperature is 1.0. Deserialization validates `parallel_tool_calls`
//! as a boolean, and conversion ignores it.
//! [`ModelVersion::resolve_top_p`] computes the final `top_p`. Conversion
//! rejects an `n` other than 1 and drops `n`, `presence_penalty`, and
//! `frequency_penalty` after validation.
//!
//! Text, images, and historical tool calls are supported. A tool result must
//! follow the assistant message that requested it, and tool names must be
//! unique. Named or required tool choices require thinking to be disabled when
//! tools are present. Conversion validates the schema of a `json_schema`
//! response format and passes it through as `ResponseFormat::JsonSchema`;
//! rendering writes the schema into the prompt. A `regex` response format is
//! passed through without rendering. Constrained decoding, which enforces
//! either format during inference, belongs to the caller. Conversion selects
//! the output constraint in [`InferenceOptions::output_constraint`]: a tool
//! call block when tools are present and the tool choice is not `none`,
//! otherwise the JSON, JSON Schema, or regex of `response_format`; a strict
//! `json_schema` gives the constraint its normalized schema. With such a tool
//! choice, conversion rejects tools that mix strict and non-strict definitions,
//! and any `response_format`, including `text`, beside strict tools. Conversion
//! normalizes
//! the parameters of a function tool with `strict: true` as strict tool
//! parameters and passes the flag through; enforcement during inference belongs
//! to the caller. A final assistant message with `prefix: true` is continued
//! instead of answered. Conversion resolves the model version through
//! [`ConversionOptions::model_versions_by_name`]; inference execution and tool
//! execution belong to the caller.
//!
//! `image_url` and `file` blocks become placeholders, or the unsupported-image
//! text `[Unsupported Image]` for [`ModelVersion::DeepseekV4Pro`]; text that
//! spells the image token stays plain text.
//!
//! [`ModelVersion::resolve_top_p`]: crate::request::ModelVersion::resolve_top_p
//! [`InferenceOptions::output_constraint`]:
//!     crate::request::InferenceOptions::output_constraint
//! [`ModelVersion::DeepseekV4Pro`]: crate::request::ModelVersion::DeepseekV4Pro
//! [`ConversionOptions::model_versions_by_name`]:
//!     crate::request::ConversionOptions::model_versions_by_name

pub use schema::*;

mod convert;
mod schema;
