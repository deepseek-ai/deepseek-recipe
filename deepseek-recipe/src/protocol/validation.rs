use deepseek_recipe_core::messages::InputMessage;

use crate::protocol::json_schema::{JsonSchema, normalize_strict_schema, validate_schema};
use crate::request::{ConversionError, InferenceOptions};

/// Maximum UTF-8 byte length of a tool name.
const MAX_TOOL_NAME_LEN: usize = 128;

/// The inference parameters of a protocol request before validation.
/// [`Self::into_inference_options`] checks them in one fixed order and
/// produces [`InferenceOptions`].
#[derive(Default)]
pub(super) struct RawInferenceOptions {
    pub thinking_mode: bool,
    pub n: Option<u8>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
    pub max_tokens: Option<u32>,
    /// Lower `max_tokens` above the upper bound to the bound instead of
    /// rejecting it.
    pub clamp_max_tokens: bool,
    pub seed: Option<u64>,
}

impl RawInferenceOptions {
    /// Validate the parameters and convert them into [`InferenceOptions`].
    ///
    /// Validation runs in the order `n`, `presence_penalty`,
    /// `frequency_penalty`, `temperature`, `top_p`, `top_logprobs` together
    /// with `logprobs`, `max_tokens`, and `seed`. The temperature is 1.0 in
    /// thinking mode and defaults to 1.0 otherwise.
    /// `top_p` is returned as requested; the adapter computes the final value
    /// for the model version. `max_tokens_upper_bound` is the bound of the
    /// request's thinking mode, from
    /// [`ConversionOptions::resolve_max_tokens_upper_bound`]; it limits
    /// `max_tokens` and appears in the error text.
    ///
    /// [`ConversionOptions::resolve_max_tokens_upper_bound`]:
    ///     crate::request::ConversionOptions::resolve_max_tokens_upper_bound
    pub(super) fn into_inference_options(
        self,
        max_tokens_upper_bound: u32,
    ) -> Result<InferenceOptions, ConversionError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ConversionError::bad_request(
                "Invalid n value (currently only n = 1 is supported)",
            ));
        }
        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if value.is_some_and(|value| !(-2.0..=2.0).contains(&value)) {
                return Err(ConversionError::bad_request(format!(
                    "Invalid {name} value, the valid range of {name} is [-2, 2]"
                )));
            }
        }
        if self
            .temperature
            .is_some_and(|value| !(0.0..=2.0).contains(&value))
        {
            return Err(ConversionError::bad_request(
                "Invalid temperature value, the valid range of temperature is [0, 2]",
            ));
        }
        if self
            .top_p
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
        {
            return Err(ConversionError::bad_request(
                "Invalid top_p value, the valid range of top_p is (0, 1.0]",
            ));
        }
        let logprobs = self.logprobs == Some(true);
        if let Some(top_logprobs) = self.top_logprobs {
            if !logprobs {
                return Err(ConversionError::bad_request(
                    "Invalid top_logprobs and logprobs value, logprobs must be set to true if \
                     top_logprobs is used.",
                ));
            }
            if top_logprobs > 20 {
                return Err(ConversionError::bad_request(
                    "Invalid top_logprobs value, the valid range of top_logprobs is [0, 20].",
                ));
            }
        }
        let max_tokens = self.max_tokens.map(|max_tokens| {
            if self.clamp_max_tokens {
                max_tokens.min(max_tokens_upper_bound)
            } else {
                max_tokens
            }
        });
        if max_tokens.is_some_and(|max_tokens| !(1..=max_tokens_upper_bound).contains(&max_tokens))
        {
            return Err(ConversionError::bad_request(format!(
                "Invalid max_tokens value, the valid range of max_tokens is [1, \
                 {max_tokens_upper_bound}]"
            )));
        }
        if self.seed.is_some_and(|seed| seed >= 1 << 63) {
            return Err(ConversionError::bad_request(
                "Invalid seed value, the valid range of seed is [0, 2^63)",
            ));
        }
        Ok(InferenceOptions {
            max_tokens,
            temperature: Some(if self.thinking_mode {
                1.0
            } else {
                self.temperature.unwrap_or(1.0)
            }),
            top_p: self.top_p,
            logprobs,
            top_logprobs: self.top_logprobs,
            seed: self.seed,
            output_constraint: None,
        })
    }
}

/// Validate a tool name. `path` locates the tool, such as `tools[0]`.
pub(super) fn validate_tool_name(name: &str, path: &str) -> Result<(), ConversionError> {
    if name.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "Invalid '{path}.name': empty string. Expected a string with minimum length 1, but got \
             an empty string instead."
        )));
    }
    if name.len() > MAX_TOOL_NAME_LEN {
        return Err(ConversionError::bad_request(format!(
            "Invalid '{path}.name': string too long. Expected a string with maximum length \
             {MAX_TOOL_NAME_LEN}, but got a string with length {} instead.",
            name.len()
        )));
    }
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err(ConversionError::bad_request(format!(
            "Invalid '{path}.name': string does not match pattern. Expected a string that matches \
             the pattern '^[a-zA-Z0-9_-]+$'."
        )));
    }
    Ok(())
}

/// Maximum number of stop sequences.
const MAX_STOP_SEQUENCES: usize = 16;

/// Validate a list of stop sequences: at most [`MAX_STOP_SEQUENCES`] entries,
/// none empty.
pub(super) fn validate_stop_sequences(stop_sequences: &[String]) -> Result<(), ConversionError> {
    if stop_sequences.len() > MAX_STOP_SEQUENCES {
        return Err(ConversionError::bad_request(format!(
            "Stop string array too long: {}",
            stop_sequences.len()
        )));
    }
    if let Some(index) = stop_sequences.iter().position(String::is_empty) {
        return Err(ConversionError::bad_request(format!(
            "Invalid 'stop[{index}]': empty string. Expected a string with minimum length 1, but \
             got an empty string instead."
        )));
    }
    Ok(())
}

/// Validate one stop sequence, which must not be empty.
pub(super) fn validate_stop_sequence(stop: &str) -> Result<(), ConversionError> {
    if stop.is_empty() {
        return Err(ConversionError::bad_request(
            "Invalid 'stop': empty string. Expected a string with minimum length 1, but got an \
             empty string instead.",
        ));
    }
    Ok(())
}

/// Return the error for duplicate tool names.
pub(super) fn duplicate_tool_names() -> ConversionError {
    ConversionError::bad_request("Tool names must be unique.")
}

/// Return the error for a named tool choice that names no declared tool.
pub(super) fn unknown_named_tool(name: &str) -> ConversionError {
    ConversionError::bad_request(format!(
        "Invalid value for 'tool_call': no function named '{name}' was specified in the 'tools' \
         parameter."
    ))
}

/// Normalize the parameter schema of a strict tool, or validate the schema of
/// a non-strict tool, returning the schema to store.
///
/// With `strict: true`, the schema is normalized for strict decoding: every
/// object lists all properties as required and forbids additional properties,
/// and the top level must be an object; a missing schema becomes an empty
/// object. Otherwise a present schema must be a valid JSON Schema that
/// declares `type: "object"`, and it is stored as written; an absent schema
/// stays absent.
pub(super) fn normalize_tool_parameters(
    name: &str,
    parameters: Option<serde_json::Value>,
    strict: Option<bool>,
) -> Result<Option<serde_json::Value>, ConversionError> {
    if strict == Some(true) {
        let schema = parameters.unwrap_or_else(|| {
            serde_json::to_value(JsonSchema::default()).expect("the default schema serializes")
        });
        return normalize_strict_schema(
            schema,
            true,
            |err| format!("Invalid tool parameters schema : {err}"),
            "The top level of the function parameters schema must be an object.",
        )
        .map(Some);
    }
    let Some(parameters) = parameters else {
        return Ok(None);
    };
    validate_schema(&parameters).map_err(|err| {
        ConversionError::bad_request(format!("Invalid schema for function '{name}': {err}"))
    })?;
    let schema_type = parameters.get("type").unwrap_or(&serde_json::Value::Null);
    if schema_type != "object" {
        return Err(ConversionError::bad_request(format!(
            "Invalid schema for function '{name}': schema must be a JSON Schema of 'type: \
             \"object\"', got 'type: {schema_type}'."
        )));
    }
    Ok(Some(parameters))
}

/// Validate the schema of a `json_schema` response format or Responses text
/// format.
///
/// With `strict: true`, the schema must normalize for strict decoding and have
/// an object top level; the normalized schema is returned for the output
/// constraint. Otherwise it must be a valid JSON Schema and is returned as is.
/// The schema is rendered as written either way.
pub(super) fn validate_response_schema(
    schema: &serde_json::Value,
    strict: Option<bool>,
) -> Result<serde_json::Value, ConversionError> {
    if strict == Some(true) {
        return normalize_strict_schema(
            schema.clone(),
            false,
            |err| format!("Invalid json schema: {err}"),
            "The top level of the json schema must be an object.",
        );
    }
    validate_schema(schema)
        .map_err(|err| ConversionError::bad_request(format!("Invalid json schema: {err}")))?;
    Ok(schema.clone())
}

/// Return whether the content of any message contains `json` as a substring,
/// compared ASCII case-insensitively.
pub(super) fn contains_json_instruction(messages: &[InputMessage]) -> bool {
    messages.iter().any(|message| {
        let content = match message {
            InputMessage::System { content, .. }
            | InputMessage::User { content, .. }
            | InputMessage::Tool { content, .. } => content.as_str(),
            InputMessage::Assistant { content, .. } | InputMessage::LatestReminder { content } => {
                content.as_str()
            }
        };
        content
            .as_bytes()
            .windows(4)
            .any(|part| part.eq_ignore_ascii_case(b"json"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_and_out_of_range_top_p_values_are_rejected() {
        for top_p in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.5] {
            let options = RawInferenceOptions {
                top_p: Some(top_p),
                ..RawInferenceOptions::default()
            };
            let error = options
                .into_inference_options(1024)
                .expect_err("the top_p value is rejected");
            assert_eq!(
                error.to_string(),
                "Invalid top_p value, the valid range of top_p is (0, 1.0]"
            );
        }
        let options = RawInferenceOptions {
            top_p: Some(1.0),
            ..RawInferenceOptions::default()
        };
        let inference_options = options
            .into_inference_options(1024)
            .expect("the top_p value is accepted");
        assert_eq!(inference_options.top_p, Some(1.0));
    }
}
