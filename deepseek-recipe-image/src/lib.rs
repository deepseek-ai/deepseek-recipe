//! Image fetching and preprocessing for DeepSeek V4.1 model input.
//!
//! [`ImageResolver`] turns the [`ImageSource`] values that protocol conversion
//! produces into [`MultiModalData`]. An [`ImageFetcher`] implementation fetches
//! external URLs, and an [`ImagePreprocessor`] implementation preprocesses the
//! images.
//!
//! A failure is an [`ImageError`]. The resolver fills the request location of
//! the failing image, from [`ImageSource::path`], into the error, and the
//! error's `Display` text is the text of a `400 Bad Request` body, such as
//! `.messages[0].image[1]: Failed to download image from https://example.com/a.png`.
//! [`ImageError::Client`] and [`ImageError::TokenBudget`] report failures of the
//! server rather than of the request, so a server answers them with a `500`.
//!
//! [`ImageSource::path`]: deepseek_recipe_core::multimodal::ImageSource::path
//! The default `opencv-preprocess` feature provides `OpenCvImagePreprocessor`,
//! and the default `reqwest-fetch` feature provides `ReqwestImageFetcher`. A
//! caller that supplies its own preprocessor can disable `opencv-preprocess` to
//! build without OpenCV. `reqwest-fetch` is a separate default feature, so a
//! caller that disables the default features and uses `ReqwestImageFetcher`
//! must enable `reqwest-fetch` again.
//!
//! [`ImageSource`]: deepseek_recipe_core::multimodal::ImageSource
//! [`MultiModalData`]: deepseek_recipe_core::multimodal::MultiModalData

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
pub use fetcher::{DestinationFilter, ReqwestImageFetcher};
pub use limits::{ImageByteBudget, ImageLimits, ImageQuota, PreprocessOptions};
#[cfg(feature = "opencv-preprocess")]
pub use opencv::OpenCvImagePreprocessor;
pub use resolver::ImageResolver;
pub use retry::RetryPolicy;

use std::future::Future;

/// A component that downloads the encoded bytes of an image from an external
/// URL.
///
/// The default [`fetch`](Self::fetch) implementation rejects every URL with
/// [`ImageError::Unsupported`], so a caller whose requests can reference
/// external images needs an implementation that fetches them.
pub trait ImageFetcher: Send + Sync {
    /// Fetch the encoded image bytes of `url`.
    ///
    /// [`ImageResolver::resolve`] calls this method after checking that `url`
    /// starts with `http`, ignoring ASCII case, and that its length is within
    /// [`ImageLimits::max_url_len`]. The fetches of one resolve call run
    /// concurrently and share `budget`. The resolver repeats an attempt whose
    /// error [`ImageError::is_retryable`] classifies as retryable, according
    /// to its [`RetryPolicy`], and checks the length of the returned bytes
    /// against [`ImageLimits::max_image_bytes`] and
    /// [`ImageLimits::max_total_bytes`].
    ///
    /// # Implementor obligations
    ///
    /// An implementation must validate the complete URL. The resolver does not
    /// restrict destinations, so destination filtering, such as SSRF
    /// protection, belongs to the implementation or to the caller that supplies
    /// the URLs. [`ReqwestImageFetcher`] filters destinations with a
    /// [`DestinationFilter`] that the caller installs.
    ///
    /// An implementation must reserve every chunk it keeps from `budget` before
    /// keeping it, so that the concurrent fetches of one call stay within
    /// [`ImageLimits::max_total_bytes`]; it must reject a body larger than
    /// [`ImageByteBudget::max_image_bytes`] and must release the bytes it
    /// reserved before returning a retryable error, so that the next attempt
    /// reserves them once.
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

/// A component that decodes, resizes, and encodes one image for the inference
/// backend.
///
/// The default [`preprocess`](Self::preprocess) implementation rejects every
/// image with [`ImageError::Unsupported`].
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
