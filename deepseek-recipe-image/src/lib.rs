//! Image fetching and preprocessing for DeepSeek V4.1 model input.
//!
//! [`ImageResolver`] turns the [`ImageSource`](deepseek_recipe_core::multimodal::ImageSource)
//! values produced by protocol conversion into
//! [`MultiModalData`](deepseek_recipe_core::multimodal::MultiModalData). Fetching
//! external URLs and preprocessing are supplied by [`ImageFetcher`] and
//! [`ImagePreprocessor`] implementations.
//!
//! The default `opencv-preprocess` feature provides `OpenCvImagePreprocessor`,
//! and the default `reqwest-fetch` feature provides `ReqwestImageFetcher`. A
//! caller that supplies its own preprocessor disables default features and
//! builds without OpenCV.

use deepseek_recipe_core::multimodal::ImageInfo;

mod error;
#[cfg(feature = "reqwest-fetch")]
mod fetcher;
mod limits;
#[cfg(feature = "opencv-preprocess")]
mod opencv;
mod resolver;
mod retry;

pub use error::ImageError;
#[cfg(feature = "reqwest-fetch")]
pub use fetcher::ReqwestImageFetcher;
pub use limits::{ImageByteBudget, ImageLimits, ImageQuota, PreprocessOptions};
#[cfg(feature = "opencv-preprocess")]
pub use opencv::OpenCvImagePreprocessor;
pub use resolver::ImageResolver;
pub use retry::RetryPolicy;

use std::future::Future;

/// Fetches the bytes of an image identified by an external URL.
///
/// The default implementation rejects every URL. Supply an implementation when
/// requests may reference external images.
pub trait ImageFetcher: Send + Sync {
    /// Fetch the bytes of `url`.
    ///
    /// The resolver checks a case-insensitive `http` prefix before calling this
    /// method. The fetcher must validate the complete URL and allowed destinations.
    ///
    /// The fetches of one resolve call run concurrently, so an implementation
    /// keeps their total size within the limit by reserving every chunk it keeps
    /// from `budget` before it keeps the chunk, and it rejects a body larger than
    /// [`ImageByteBudget::max_image_bytes`]. An implementation releases the bytes
    /// it has reserved before it returns a retryable error, so the retry of that
    /// attempt reserves them once. The resolver checks the size of the returned
    /// bytes against [`ImageLimits`] as well.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when the URL cannot be fetched or the image
    /// exceeds a limit of `budget`.
    fn fetch(
        &self,
        url: &str,
        budget: &ImageByteBudget,
    ) -> impl Future<Output = Result<Vec<u8>, ImageError>> + Send {
        let _ = (url, budget);
        async {
            Err(ImageError::Unsupported(
                "image fetching requires an ImageFetcher implementation",
            ))
        }
    }
}

/// Decodes, resizes, and encodes one image for the inference backend.
///
/// The default implementation rejects every image.
pub trait ImagePreprocessor: Send + Sync {
    /// Preprocess one encoded image.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when the image is unsupported, exceeds a limit, or
    /// cannot be preprocessed.
    fn preprocess(
        &self,
        data: Vec<u8>,
        options: PreprocessOptions,
    ) -> impl Future<Output = Result<ImageInfo, ImageError>> + Send {
        let _ = (data, options);
        async {
            Err(ImageError::Unsupported(
                "image preprocessing requires an ImagePreprocessor implementation",
            ))
        }
    }
}
