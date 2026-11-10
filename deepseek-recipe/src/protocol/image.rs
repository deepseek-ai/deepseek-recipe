//! Image input validation and image source construction shared by the
//! protocol adapters.
//!
//! This module validates image URLs and data URLs, constructs the
//! [`ImageSource`] of an image block, defines the unsupported-content text,
//! and builds the errors for unsupported images and `file_id` references.
//!
//! Each constructor takes the `path` that prefixes its validation errors and
//! the `image_path` that the source records as [`ImageSource::path`]: the
//! request location of the message or block whose images are resolved
//! together, which image resolution errors report.

use std::convert::Infallible;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use data_url::DataUrl;
use deepseek_recipe_core::multimodal::{
    ImageDetail, ImageMediaType, ImageSource, MAX_IMAGE_URL_LEN, UNSUPPORTED_IMAGE_ERROR_MESSAGE,
    is_http_url,
};

use crate::request::ConversionError;

/// Text substituted for document blocks by the Messages and Responses adapters.
/// Document contents are not read or forwarded to the backend.
pub const UNSUPPORTED_DOCUMENT_PLACEHOLDER: &str = "[Unsupported Document]";

/// Text substituted for image blocks when the target model takes no image
/// input.
pub const UNSUPPORTED_IMAGE_PLACEHOLDER: &str = "[Unsupported Image]";

/// Error message for a Chat Completions `file` block with an unsupported media
/// type.
pub const UNSUPPORTED_FILE_ERROR_MESSAGE: &str = "You have uploaded an unsupported file. Please \
    make sure your file is valid and has one of the following formats: webp, png, jpeg, and gif.";

/// Convert a Messages `url` image source.
///
/// Checks the case-insensitive `http` prefix and a UTF-8 byte length no greater
/// than [`MAX_IMAGE_URL_LEN`]. Full URL validation belongs to the fetcher.
pub(super) fn external_image_source(
    url: String,
    path: &str,
    image_path: &str,
) -> Result<ImageSource, ConversionError> {
    if !is_http_url(&url) {
        return Err(ConversionError::bad_request(format!(
            "{path}: invalid url: {url:?}"
        )));
    }
    if url.len() > MAX_IMAGE_URL_LEN {
        return Err(ConversionError::bad_request(format!(
            "{path}: external link length {} too long, max link length {MAX_IMAGE_URL_LEN}. \
             Consider passing file data directly with base64 data url instead.",
            url.len()
        )));
    }
    Ok(ImageSource::Url {
        url,
        detail: ImageDetail::default(),
        path: image_path.to_string(),
    })
}

/// Convert a Chat Completions or Responses `image_url` value into an image
/// source.
///
/// Values with a case-insensitive `http` prefix are external and must not
/// exceed [`MAX_IMAGE_URL_LEN`] bytes; `too_long` formats the error from the
/// length. Other values are data URLs, checked later by
/// [`validate_data_url_sources`] for system, user, and tool content.
pub(super) fn image_url_source(
    url: String,
    detail: Option<ImageDetail>,
    image_path: &str,
    too_long: impl FnOnce(usize) -> String,
) -> Result<ImageSource, ConversionError> {
    let detail = detail.unwrap_or_default();
    let path = image_path.to_string();
    if is_http_url(&url) {
        if url.len() > MAX_IMAGE_URL_LEN {
            return Err(ConversionError::bad_request(too_long(url.len())));
        }
        Ok(ImageSource::Url { url, detail, path })
    } else {
        Ok(ImageSource::DataUrl {
            data_url: url,
            detail,
            path,
        })
    }
}

/// Validate the data URL image sources of system, user, or tool content. `path`
/// prefixes the errors.
///
/// A data URL must contain the literal `base64,` marker, parse as a data URL,
/// declare a supported image media type, and carry a decodable body.
pub(super) fn validate_data_url_sources(
    image_sources: &[ImageSource],
    path: &str,
) -> Result<(), ConversionError> {
    for source in image_sources {
        let ImageSource::DataUrl { data_url, .. } = source else {
            continue;
        };
        let unsupported_format =
            || ConversionError::bad_request(format!("{path}: Unsupported image_url format"));
        if !data_url.contains("base64,") {
            return Err(unsupported_format());
        }
        let parsed = DataUrl::process(data_url).map_err(|_| unsupported_format())?;
        if !is_supported_media_type(parsed.mime_type()) {
            return Err(unsupported_image(path));
        }
        // Conversion rejects an undecodable body without keeping the decoded
        // bytes; `ImageResolver` decodes the body again to obtain them.
        parsed
            .decode(|_| Ok::<(), Infallible>(()))
            .map_err(|_| ConversionError::bad_request(format!("{path}: Invalid base64 data")))?;
    }
    Ok(())
}

/// Validate the `file_data` of a Chat Completions `file` block, returning the
/// data URL image source. `path` prefixes the errors and is the path the
/// source records, because Chat Completions resolves the images of a message
/// under the message's path.
pub(super) fn file_data_image_source(
    file_data: String,
    path: &str,
) -> Result<ImageSource, ConversionError> {
    let invalid = || {
        ConversionError::bad_request(format!("{path}: file_data is not a valid base64 data url"))
    };
    let parsed = DataUrl::process(&file_data).map_err(|_| invalid())?;
    if !is_supported_media_type(parsed.mime_type()) {
        return Err(ConversionError::bad_request(format!(
            "{path}: {UNSUPPORTED_FILE_ERROR_MESSAGE}"
        )));
    }
    // The decoded bytes are not kept; see `validate_data_url_sources`.
    parsed
        .decode(|_| Ok::<(), Infallible>(()))
        .map_err(|_| invalid())?;
    Ok(ImageSource::DataUrl {
        data_url: file_data,
        detail: ImageDetail::default(),
        path: path.to_string(),
    })
}

/// Return whether a data URL media type is a supported image type.
fn is_supported_media_type(mime: &data_url::mime::Mime) -> bool {
    ImageMediaType::from_mime(&format!("{}/{}", mime.type_, mime.subtype)).is_some()
}

/// Construct a data URL image source from a declared media type and a base64
/// payload.
///
/// [`normalize_base64_padding`] normalizes the trailing padding, and a payload
/// that then fails to decode as standard base64 is rejected. `ImageResolver`
/// rejects an empty payload, and an `ImagePreprocessor` rejects invalid image
/// contents.
pub(super) fn base64_image_source(
    media_type: ImageMediaType,
    data: String,
    path: &str,
    image_path: &str,
) -> Result<ImageSource, ConversionError> {
    let data = normalize_base64_padding(data);
    if STANDARD.decode(&data).is_err() {
        return Err(ConversionError::bad_request(format!(
            "{path}: base64 decode error"
        )));
    }
    Ok(ImageSource::DataUrl {
        data_url: format!("data:{};base64,{data}", media_type.mime()),
        detail: ImageDetail::default(),
        path: image_path.to_string(),
    })
}

/// Return the error for an unsupported image.
pub(super) fn unsupported_image(path: &str) -> ConversionError {
    ConversionError::bad_request(format!("{path}: {UNSUPPORTED_IMAGE_ERROR_MESSAGE}"))
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
/// A request might carry the wrong number of trailing `=` characters. A
/// payload with at least one trailing `=` has its padding replaced with the
/// canonical amount. A payload with no trailing `=` is returned unchanged,
/// because a truncated payload and an unpadded payload are indistinguishable.
/// A payload whose unpadded length leaves a remainder of 1 when divided by 4
/// is also returned unchanged, so the decoder reports the invalid payload.
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
