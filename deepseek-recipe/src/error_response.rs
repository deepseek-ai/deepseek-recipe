use serde::Serialize;

use crate::request::ConversionError;

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

impl ConversionError {
    /// The HTTP status code for this error.
    pub fn status_code(&self) -> u16 {
        match self {
            ConversionError::BadRequest { .. } => 400,
            ConversionError::Internal { .. } => 500,
        }
    }

    /// Build the error response body for this error.
    pub fn into_response(self) -> ErrorResponse {
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
