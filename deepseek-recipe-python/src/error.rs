//! The exception raised when a protocol request cannot be converted.

use deepseek_recipe::request::ConversionError as RustConversionError;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

pyo3::create_exception!(
    _native,
    ConversionError,
    PyException,
    "Raised when a protocol request is invalid or cannot be converted."
);

/// Build the exception for a failed conversion.
///
/// The exception message is the conversion error message. Its `status_code` and
/// `body` attributes hold the HTTP status and serialized JSON error response.
pub(crate) fn request_error(py: Python<'_>, error: RustConversionError) -> PyErr {
    let status_code = error.status_code();
    let response = error.into_response();
    let body = match serde_json::to_string(&response) {
        Ok(body) => body,
        Err(error) => return pyo3::exceptions::PyRuntimeError::new_err(error.to_string()),
    };
    let exception = ConversionError::new_err(response.error.message);
    let value = exception.value(py);
    if let Err(error) = value
        .setattr("status_code", status_code)
        .and_then(|()| value.setattr("body", body))
    {
        return error;
    }
    exception
}
