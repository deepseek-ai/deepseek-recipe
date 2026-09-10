//! Image input validation shared by the protocol adapters.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use deepseek_recipe_core::multimodal::{
    ImageDetail, ImageMediaType, ImageSource, MAX_IMAGE_URL_LEN, is_http_url,
};

use crate::request::ConversionError;

/// Text substituted for document blocks by the Messages and Responses adapters.
/// Document contents are not read or forwarded to the backend.
pub const UNSUPPORTED_DOCUMENT_PLACEHOLDER: &str = "[Unsupported Document]";

/// Error message for an image whose bytes are missing or unsupported.
pub const UNSUPPORTED_IMAGE_ERR: &str = "You have uploaded an unsupported image. Please make sure your image is valid and has one of the following formats: webp, png, jpeg, and gif.";

/// Convert an external image URL into an image source.
///
/// Checks the case-insensitive `http` prefix and a UTF-8 byte length no greater
/// than [`MAX_IMAGE_URL_LEN`]. Full URL validation belongs to the fetcher.
pub(super) fn external_image_source(
    url: String,
    detail: Option<ImageDetail>,
    path: &str,
) -> Result<ImageSource, ConversionError> {
    if !is_http_url(&url) {
        return Err(ConversionError::bad_request(format!(
            "{path}: invalid url: {url:?}"
        )));
    }
    if url.len() > MAX_IMAGE_URL_LEN {
        return Err(ConversionError::bad_request(format!(
            "{path}: external link length {} too long, max link length {MAX_IMAGE_URL_LEN}. Consider passing file data directly with \
             base64 data url instead.",
            url.len()
        )));
    }
    Ok(ImageSource::Url {
        url,
        detail: detail.unwrap_or_default(),
    })
}

/// Convert a data URL into an image source.
///
/// The body is not inspected. The resolver accepts both a base64 body and a
/// percent-encoded body.
pub(super) fn data_url_image_source(data_url: String, detail: Option<ImageDetail>) -> ImageSource {
    ImageSource::DataUrl {
        data_url,
        detail: detail.unwrap_or_default(),
    }
}

/// Convert an `image_url` value into an image source.
///
/// Values with a case-insensitive `http` prefix are external. Other values must
/// contain the literal `base64,` marker. Data URL syntax, media type, and body
/// decoding are checked later by the image resolver.
pub(super) fn image_url_source(
    url: String,
    detail: Option<ImageDetail>,
    path: &str,
) -> Result<ImageSource, ConversionError> {
    if is_http_url(&url) {
        external_image_source(url, detail, path)
    } else if url.contains("base64,") {
        Ok(data_url_image_source(url, detail))
    } else {
        Err(ConversionError::bad_request(format!(
            "{path}: Unsupported image_url format"
        )))
    }
}

/// Build a base64 data URL from a declared media type and base64 payload.
///
/// Normalizes trailing padding and validates the standard base64 encoding.
/// Empty payloads and invalid image contents are checked by the resolver and
/// preprocessor, respectively.
pub(super) fn base64_image_source(
    media_type: &str,
    data: String,
    path: &str,
) -> Result<ImageSource, ConversionError> {
    let media_type =
        ImageMediaType::from_mime(media_type).ok_or_else(|| unsupported_image(path))?;
    let data = normalize_base64_padding(data);
    if STANDARD.decode(&data).is_err() {
        return Err(ConversionError::bad_request(format!(
            "{path}: base64 decode error"
        )));
    }
    Ok(ImageSource::DataUrl {
        data_url: format!("data:{};base64,{data}", media_type.mime()),
        detail: ImageDetail::default(),
    })
}

/// Return the error for an unsupported image.
pub(super) fn unsupported_image(path: &str) -> ConversionError {
    ConversionError::bad_request(format!("{path}: {UNSUPPORTED_IMAGE_ERR}"))
}

/// Return the error for an image in a message that does not accept images.
pub(super) fn image_not_allowed(role: &str) -> ConversionError {
    ConversionError::bad_request(format!("Image in {role} message is unsupported"))
}

/// Return the error for a file identifier, which requires a file service this
/// library does not provide.
pub(super) fn file_id_unsupported(path: &str) -> ConversionError {
    ConversionError::bad_request(format!(
        "{path}: file_id is not supported. Pass the image as a base64 data url instead."
    ))
}

/// Normalize the padding of a base64 payload.
///
/// A caller may supply the wrong number of trailing `=` characters. A payload
/// carrying at least one trailing `=` has its padding dropped and replaced with
/// the canonical amount. A payload with no trailing `=` is returned unchanged,
/// because a truncated payload and an unpadded payload are indistinguishable.
/// A payload whose unpadded length leaves a single symbol is also returned
/// unchanged, so the decoder reports the invalid payload.
fn normalize_base64_padding(mut data: String) -> String {
    if !data.ends_with('=') {
        return data;
    }
    let unpadded_len = data.trim_end_matches('=').len();
    let remainder = unpadded_len % 4;
    if remainder == 1 {
        return data;
    }
    data.truncate(unpadded_len);
    data.push_str(&"=".repeat((4 - remainder) % 4));
    data
}
