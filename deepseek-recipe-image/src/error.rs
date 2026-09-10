//! Failures while resolving or preprocessing an image.

use deepseek_recipe_core::multimodal::CalcResizeError;

/// Failure to fetch or preprocess one image.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// The image source needs a capability this implementation does not provide.
    #[error("unsupported image source: {0}")]
    Unsupported(&'static str),
    /// The external image URL is too long.
    #[error("external link length {len} too long, max link length {max}")]
    UrlTooLong {
        /// Length of the rejected URL.
        len: usize,
        /// Maximum accepted length.
        max: usize,
    },
    /// The external image reference lacks the required `http` prefix.
    #[error("invalid url: {url:?}")]
    InvalidUrl {
        /// Rejected URL.
        url: String,
    },
    /// The request carries more images than the limit allows.
    #[error("too many images: max {max} images per request, got {count}")]
    TooManyImages {
        /// Maximum number of images per request.
        max: usize,
        /// Number of images in the request.
        count: usize,
    },
    /// One image is larger than the limit allows.
    #[error("image size {size} bytes exceeds the limit of {max} bytes")]
    ImageTooLarge {
        /// Size of the rejected image.
        size: usize,
        /// Maximum accepted size.
        max: usize,
    },
    /// The images together are larger than the limit allows.
    #[error("total image size {size} bytes exceeds the limit of {max} bytes")]
    TotalSizeTooLarge {
        /// Total size of all images.
        size: usize,
        /// Maximum accepted total size.
        max: usize,
    },
    /// The data URL or its encoded body is malformed.
    #[error("invalid data url: {0}")]
    InvalidDataUrl(String),
    /// The image bytes are empty.
    #[error("input image data is empty")]
    EmptyImage,
    /// The image media type is unsupported.
    #[error("unsupported image media type: {0}")]
    UnsupportedMediaType(String),
    /// The image dimensions exceed the limit.
    #[error("image dimensions exceed the maximum allowed")]
    ImageDimensionsTooLarge,
    /// The image could not be decoded.
    #[error("failed to decode image: {0}")]
    Decode(String),
    /// The image could not be resized.
    #[error("failed to resize image: {0}")]
    Resize(String),
    /// The image could not be encoded.
    #[error("failed to encode image: {0}")]
    Encode(String),
    /// The image encoder reported failure without an error message.
    #[error("failed to encode image")]
    EncodeFailed,
    /// Fitting the image into the token budget failed.
    #[error(transparent)]
    TokenBudget(#[from] CalcResizeError),
    /// The image could not be fetched.
    #[error("failed to fetch image from {url}: {source}")]
    Fetch {
        /// Rejected URL.
        url: String,
        /// Underlying fetcher failure, including wrapped input or size errors.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The image server returned a non-success status.
    #[error("failed to fetch image from {url}: unexpected status {status}")]
    FetchStatus {
        /// Rejected URL.
        url: String,
        /// HTTP status code of the response.
        status: u16,
    },
    /// The HTTP client could not be constructed.
    #[error("failed to create the HTTP client: {0}")]
    Client(String),
}

impl ImageError {
    /// Return whether another attempt of the same operation may succeed.
    ///
    /// `Fetch` is retryable regardless of its source; `FetchStatus` is retryable
    /// for status codes at least 500. Other variants are not retryable. The
    /// resolver evaluates the fetcher's error before wrapping it in `Fetch`,
    /// so a returned wrapper may have a different classification.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Fetch { .. } => true,
            Self::FetchStatus { status, .. } => *status >= 500,
            _ => false,
        }
    }
}
