//! Error response bodies for conversion errors.
//!
//! [`ConversionError::into_error_response`] builds the JSON body
//! `{"error": {"message", "type", "param", "code"}}`, and
//! [`ConversionError::status_code`] returns the matching HTTP status code.

use serde::Serialize;

use crate::request::ConversionError;

/// The `error` object of an [`ErrorResponse`].
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    /// The detail of the conversion error.
    pub message: String,
    /// `invalid_request_error` for [`ConversionError::BadRequest`] and
    /// `internal_error` for [`ConversionError::Internal`].
    pub r#type: String,
    /// Always `None`, serialized as `null` to match the OpenAI error format.
    pub param: Option<String>,
    /// `invalid_request_error` for a status code below 500; otherwise the
    /// same value as `type`.
    pub code: String,
}

/// The JSON body of an error response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// The error object.
    pub error: ErrorDetail,
}

impl ConversionError {
    /// Return the HTTP status code for this error: 400 for
    /// [`ConversionError::BadRequest`] and 500 for
    /// [`ConversionError::Internal`].
    pub fn status_code(&self) -> u16 {
        match self {
            ConversionError::BadRequest { .. } => 400,
            ConversionError::Internal { .. } => 500,
        }
    }

    /// Build the error response body for this error.
    pub fn into_error_response(self) -> ErrorResponse {
        let status_code = self.status_code();
        let (type_, detail) = match self {
            ConversionError::BadRequest { detail } => ("invalid_request_error", detail),
            ConversionError::Internal { detail } => ("internal_error", detail),
        };

        ErrorResponse {
            error: ErrorDetail {
                message: detail,
                r#type: type_.to_string(),
                param: None,
                code: if status_code >= 500 {
                    type_.to_string()
                } else {
                    "invalid_request_error".to_owned()
                },
            },
        }
    }
}
