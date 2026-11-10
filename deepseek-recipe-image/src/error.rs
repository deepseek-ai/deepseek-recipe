//! Failures while resolving, fetching, or preprocessing images.

use std::fmt;

use deepseek_recipe_core::multimodal::{CalcResizeError, UNSUPPORTED_IMAGE_ERROR_MESSAGE};

/// Failure to resolve, fetch, or preprocess images, including request limits.
///
/// The variants that report one image or one request limit carry the request
/// location of the image: `path` is the [`ImageSource::path`] of the source,
/// and `index` counts the earlier sources of the same resolve call with the
/// same path. [`ImageResolver::resolve`] fills both before returning an error,
/// so an [`ImageFetcher`] or [`ImagePreprocessor`] implementation constructs
/// its errors through the constructors of this type, which leave the location
/// empty.
///
/// The [`Display`](fmt::Display) text of these variants is the text of a
/// `400 Bad Request` body: a request-level limit is written as
/// `{path}: <text>`, and a failure of one image as `{path}.image[{index}]:
/// <text>`. An empty path omits the `{path}` part. The sizes in the limit texts
/// are whole MB, with 1 MB being 1024 × 1024 bytes and the remainder
/// truncated. A decode, resize, encode, media-type, dimension, or empty-data
/// failure reports [`UNSUPPORTED_IMAGE_ERROR_MESSAGE`], and a fetch failure
/// reports the URL without the underlying error, which [`source`] and
/// [`Debug`](fmt::Debug) retain.
///
/// [`ImageSource::path`]: deepseek_recipe_core::multimodal::ImageSource::path
/// [`ImageResolver::resolve`]: crate::ImageResolver::resolve
/// [`ImageFetcher`]: crate::ImageFetcher
/// [`ImagePreprocessor`]: crate::ImagePreprocessor
/// [`source`]: std::error::Error::source
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// An operation that the implementation in use does not provide, such as
    /// fetching through the default [`ImageFetcher::fetch`] or preprocessing
    /// through the default [`ImagePreprocessor::preprocess`].
    ///
    /// [`ImageFetcher::fetch`]: crate::ImageFetcher::fetch
    /// [`ImagePreprocessor::preprocess`]: crate::ImagePreprocessor::preprocess
    Unsupported(&'static str),
    /// The external image URL is too long.
    UrlTooLong {
        /// Request location of the image.
        path: String,
        /// Length of the rejected URL in UTF-8 bytes.
        len: usize,
        /// Maximum accepted length in UTF-8 bytes.
        max: usize,
    },
    /// The external image reference does not start with `http`, ignoring
    /// ASCII case.
    InvalidUrl {
        /// Request location of the image.
        path: String,
        /// Rejected URL.
        url: String,
    },
    /// The request carries more images than the limit allows.
    TooManyImages {
        /// Request location of the first image beyond the limit.
        path: String,
        /// Maximum number of images per request.
        max: usize,
        /// Number of image sources of the request through the last source
        /// that shares `path`, including those of earlier resolve calls.
        count: usize,
    },
    /// One image is larger than the limit allows.
    ImageTooLarge {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// Encoded size of the rejected image in bytes. When the image is
        /// rejected before it is complete, this is the size known at that
        /// point: the response `Content-Length`, the bytes received so far, or
        /// the estimated decoded length of a data URL.
        size: usize,
        /// Maximum accepted encoded size of one image in bytes.
        max: usize,
    },
    /// The images together are larger than the limit allows.
    TotalSizeTooLarge {
        /// Request location of the image whose bytes exceed the limit.
        path: String,
        /// Total encoded size in bytes that the request would reach: the bytes
        /// of earlier resolve calls, the bytes already counted in this call,
        /// and the rejected bytes.
        size: usize,
        /// Maximum accepted total encoded size of the images of one request in
        /// bytes.
        max: usize,
    },
    /// The data URL or its encoded body is malformed.
    InvalidDataUrl {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// Parser message.
        detail: String,
    },
    /// The image bytes are empty, or the decoded image has zero width or
    /// height.
    EmptyImage {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
    },
    /// The image media type is unsupported.
    UnsupportedMediaType {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// Rejected media type.
        media_type: String,
    },
    /// The image dimensions exceed the limit.
    ImageDimensionsTooLarge {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
    },
    /// The image could not be decoded.
    Decode {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// Decoder message.
        detail: String,
    },
    /// The image could not be resized.
    Resize {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// Resizer message.
        detail: String,
    },
    /// The image could not be encoded.
    Encode {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// Encoder message.
        detail: String,
    },
    /// The image encoder reported failure without an error message.
    EncodeFailed {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
    },
    /// Fitting the image into the token budget failed.
    TokenBudget(#[from] CalcResizeError),
    /// The image could not be fetched.
    Fetch {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// URL whose fetch failed.
        url: String,
        /// Underlying fetcher failure, including wrapped input or size errors.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The destination filter of the fetcher rejected the host of the image
    /// URL or of a redirect target.
    DestinationRejected {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// URL whose fetch was rejected.
        url: String,
        /// Rejected host: the host of the URL or of a redirect target.
        host: String,
    },
    /// The image server returned a non-success status.
    FetchStatus {
        /// Request location of the image.
        path: String,
        /// Index of the image among the sources with the same path.
        index: usize,
        /// URL whose response had the non-success status.
        url: String,
        /// HTTP status code of the response.
        status: u16,
    },
    /// The HTTP client could not be constructed.
    Client(String),
}

/// Number of bytes in one MB of the limit texts.
const BYTES_PER_MB: usize = 1024 * 1024;

impl ImageError {
    /// Return a [`Self::Decode`] error without a request location.
    pub fn decode(detail: impl ToString) -> Self {
        Self::Decode {
            path: String::new(),
            index: 0,
            detail: detail.to_string(),
        }
    }

    /// Return a [`Self::Resize`] error without a request location.
    pub fn resize(detail: impl ToString) -> Self {
        Self::Resize {
            path: String::new(),
            index: 0,
            detail: detail.to_string(),
        }
    }

    /// Return a [`Self::Encode`] error without a request location.
    pub fn encode(detail: impl ToString) -> Self {
        Self::Encode {
            path: String::new(),
            index: 0,
            detail: detail.to_string(),
        }
    }

    /// Return a [`Self::EncodeFailed`] error without a request location.
    pub fn encode_failed() -> Self {
        Self::EncodeFailed {
            path: String::new(),
            index: 0,
        }
    }

    /// Return an [`Self::EmptyImage`] error without a request location.
    pub fn empty_image() -> Self {
        Self::EmptyImage {
            path: String::new(),
            index: 0,
        }
    }

    /// Return an [`Self::ImageDimensionsTooLarge`] error without a request
    /// location.
    pub fn dimensions_too_large() -> Self {
        Self::ImageDimensionsTooLarge {
            path: String::new(),
            index: 0,
        }
    }

    /// Return an [`Self::UnsupportedMediaType`] error without a request
    /// location.
    pub fn unsupported_media_type(media_type: impl ToString) -> Self {
        Self::UnsupportedMediaType {
            path: String::new(),
            index: 0,
            media_type: media_type.to_string(),
        }
    }

    /// Return an [`Self::InvalidDataUrl`] error without a request location.
    pub fn invalid_data_url(detail: impl ToString) -> Self {
        Self::InvalidDataUrl {
            path: String::new(),
            index: 0,
            detail: detail.to_string(),
        }
    }

    /// Return an [`Self::ImageTooLarge`] error without a request location.
    /// Both sizes are in bytes.
    pub fn image_too_large(size: usize, max: usize) -> Self {
        Self::ImageTooLarge {
            path: String::new(),
            index: 0,
            size,
            max,
        }
    }

    /// Return a [`Self::TotalSizeTooLarge`] error without a request location.
    /// Both sizes are in bytes.
    pub fn total_size_too_large(size: usize, max: usize) -> Self {
        Self::TotalSizeTooLarge {
            path: String::new(),
            size,
            max,
        }
    }

    /// Return a [`Self::Fetch`] error without a request location.
    pub fn fetch(
        url: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Fetch {
            path: String::new(),
            index: 0,
            url: url.into(),
            source: Box::new(source),
        }
    }

    /// Return a [`Self::FetchStatus`] error without a request location.
    pub fn fetch_status(url: impl Into<String>, status: u16) -> Self {
        Self::FetchStatus {
            path: String::new(),
            index: 0,
            url: url.into(),
            status,
        }
    }

    /// Return a [`Self::DestinationRejected`] error without a request
    /// location.
    pub fn destination_rejected(url: impl Into<String>, host: impl Into<String>) -> Self {
        Self::DestinationRejected {
            path: String::new(),
            index: 0,
            url: url.into(),
            host: host.into(),
        }
    }

    /// Return the error with the request location of the image it reports:
    /// `path` and, for the variants that report one image, `index`. Variants
    /// without a location are returned unchanged.
    pub(crate) fn with_location(mut self, path: &str, index: usize) -> Self {
        match &mut self {
            Self::UrlTooLong { path: location, .. }
            | Self::InvalidUrl { path: location, .. }
            | Self::TooManyImages { path: location, .. }
            | Self::TotalSizeTooLarge { path: location, .. } => {
                location.clear();
                location.push_str(path);
            }
            Self::ImageTooLarge {
                path: location,
                index: position,
                ..
            }
            | Self::InvalidDataUrl {
                path: location,
                index: position,
                ..
            }
            | Self::EmptyImage {
                path: location,
                index: position,
            }
            | Self::UnsupportedMediaType {
                path: location,
                index: position,
                ..
            }
            | Self::ImageDimensionsTooLarge {
                path: location,
                index: position,
            }
            | Self::Decode {
                path: location,
                index: position,
                ..
            }
            | Self::Resize {
                path: location,
                index: position,
                ..
            }
            | Self::Encode {
                path: location,
                index: position,
                ..
            }
            | Self::EncodeFailed {
                path: location,
                index: position,
            }
            | Self::Fetch {
                path: location,
                index: position,
                ..
            }
            | Self::DestinationRejected {
                path: location,
                index: position,
                ..
            }
            | Self::FetchStatus {
                path: location,
                index: position,
                ..
            } => {
                location.clear();
                location.push_str(path);
                *position = index;
            }
            Self::Unsupported(_) | Self::TokenBudget(_) | Self::Client(_) => {}
        }
        self
    }

    /// Return whether another attempt of the same operation might succeed.
    ///
    /// [`Self::Fetch`] is retryable regardless of its `source` error;
    /// [`Self::FetchStatus`] is retryable for status codes of 500 or greater.
    /// Other variants are not retryable. The resolver evaluates the fetcher's
    /// error before wrapping it in [`Self::Fetch`], so a returned wrapper might
    /// have a different classification.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Fetch { .. } => true,
            Self::FetchStatus { status, .. } => *status >= 500,
            _ => false,
        }
    }
}

/// Write the `{path}: ` prefix of a request-level error; nothing for an empty
/// path.
fn write_request_location(formatter: &mut fmt::Formatter<'_>, path: &str) -> fmt::Result {
    if path.is_empty() {
        Ok(())
    } else {
        write!(formatter, "{path}: ")
    }
}

/// Write the `{path}.image[{index}]: ` prefix of a per-image error;
/// `image[{index}]: ` for an empty path.
fn write_image_location(
    formatter: &mut fmt::Formatter<'_>,
    path: &str,
    index: usize,
) -> fmt::Result {
    if path.is_empty() {
        write!(formatter, "image[{index}]: ")
    } else {
        write!(formatter, "{path}.image[{index}]: ")
    }
}

impl fmt::Display for ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(operation) => write!(formatter, "unsupported operation: {operation}"),
            Self::UrlTooLong { path, len, max } => {
                write_request_location(formatter, path)?;
                write!(
                    formatter,
                    "external link length {len} too long, max link length {max}"
                )
            }
            Self::InvalidUrl { path, url } => {
                write_request_location(formatter, path)?;
                write!(formatter, "invalid url: {url:?}")
            }
            Self::TooManyImages { path, max, count } => {
                write_request_location(formatter, path)?;
                write!(
                    formatter,
                    "Too many images: max {max} images per request, got {count}"
                )
            }
            Self::ImageTooLarge {
                path, index, max, ..
            } => {
                write_image_location(formatter, path, *index)?;
                write!(
                    formatter,
                    "image file size exceeds limit {} MB",
                    max / BYTES_PER_MB
                )
            }
            Self::TotalSizeTooLarge { path, size, max } => {
                write_request_location(formatter, path)?;
                write!(
                    formatter,
                    "Total image size exceeds the limit: max {} MB per request, got {} MB",
                    max / BYTES_PER_MB,
                    size / BYTES_PER_MB
                )
            }
            Self::InvalidDataUrl { path, index, .. }
            | Self::EmptyImage { path, index }
            | Self::UnsupportedMediaType { path, index, .. }
            | Self::ImageDimensionsTooLarge { path, index }
            | Self::Decode { path, index, .. }
            | Self::Resize { path, index, .. }
            | Self::Encode { path, index, .. }
            | Self::EncodeFailed { path, index } => {
                write_image_location(formatter, path, *index)?;
                formatter.write_str(UNSUPPORTED_IMAGE_ERROR_MESSAGE)
            }
            Self::TokenBudget(error) => fmt::Display::fmt(error, formatter),
            Self::Fetch {
                path, index, url, ..
            }
            | Self::DestinationRejected {
                path, index, url, ..
            }
            | Self::FetchStatus {
                path, index, url, ..
            } => {
                write_image_location(formatter, path, *index)?;
                write!(formatter, "Failed to download image from {url}")
            }
            Self::Client(detail) => write!(formatter, "failed to create the HTTP client: {detail}"),
        }
    }
}
