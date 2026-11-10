//! Resolve image sources into the images of one request.

use std::collections::HashMap;

use futures::StreamExt;
use futures::stream::{self, TryStreamExt};

use deepseek_recipe_core::multimodal::{ImageMediaType, ImageSource, MultiModalData, is_http_url};

use crate::error::ImageError;
use crate::limits::{ImageByteBudget, ImageLimits, ImageQuota};
use crate::retry::RetryPolicy;
use crate::{ImageFetcher, ImagePreprocessor};

/// A resolver that turns image sources into preprocessed images.
///
/// The resolver accepts external URLs, data URLs, and caller-supplied encoded
/// bytes. It applies [`ImageLimits`] and keeps the images in source order. The
/// caller supplies the fetcher and the preprocessor. At most
/// [`ImageLimits::max_concurrent_sources`] sources are resolved and
/// preprocessed concurrently.
pub struct ImageResolver<F, P> {
    fetcher: F,
    preprocessor: P,
    limits: ImageLimits,
    retry: RetryPolicy,
}

impl<F, P> ImageResolver<F, P>
where
    F: ImageFetcher,
    P: ImagePreprocessor,
{
    /// Combine an image fetcher with an image preprocessor.
    pub fn new(fetcher: F, preprocessor: P) -> Self {
        Self {
            fetcher,
            preprocessor,
            limits: ImageLimits::default(),
            retry: RetryPolicy::download(),
        }
    }

    /// Replace the default limits.
    pub fn with_limits(mut self, limits: ImageLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Replace the default retry policy of image downloads.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Resolve sources concurrently and preserve their order in the result.
    ///
    /// Pass one [`ImageQuota`] per request and reuse it across calls, so the
    /// count and total byte limits apply across calls. Dimension limits use the
    /// accumulated count including this call; earlier images are not rechecked.
    /// The call reserves the bytes it keeps from one [`ImageByteBudget`] it
    /// shares with every fetch, so the sources of the call are limited together.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when a source cannot be fetched, decoded, or
    /// preprocessed, or when the request exceeds a limit. A failed call leaves
    /// `quota` unchanged: the sources of the call are recorded once every one of
    /// them is preprocessed.
    ///
    /// Every returned error carries the request location of the image it
    /// reports, from [`ImageSource::path`]. An error of one image also carries
    /// the image's index among the sources of this call that share its path,
    /// so the sources of one message or block are numbered from zero.
    /// [`ImageError::TooManyImages`] names the path of the first source beyond
    /// [`ImageLimits::max_images`] and counts the sources through the last one
    /// that shares that path. [`ImageError::TotalSizeTooLarge`] names the path
    /// of the source whose bytes exceed [`ImageLimits::max_total_bytes`].
    pub async fn resolve(
        &self,
        sources: &[ImageSource],
        quota: &mut ImageQuota,
    ) -> Result<MultiModalData, ImageError> {
        let request_image_count = quota.image_count() + sources.len();
        if request_image_count > self.limits.max_images {
            // The first source beyond the limit names the error; the count
            // includes the rest of the sources that share its path.
            let first_beyond = self.limits.max_images.saturating_sub(quota.image_count());
            let path = sources.get(first_beyond).map_or("", ImageSource::path);
            let group_end = sources
                .iter()
                .skip(first_beyond)
                .position(|source| source.path() != path)
                .map_or(sources.len(), |offset| first_beyond + offset);
            return Err(ImageError::TooManyImages {
                path: path.to_string(),
                max: self.limits.max_images,
                count: quota.image_count() + group_end,
            });
        }
        let indices = image_indices(sources);

        // The bytes of earlier calls of this request are recorded in `quota`
        // rather than reserved again in this budget.
        let budget = ImageByteBudget::new(
            self.limits.max_image_bytes,
            self.limits.max_total_bytes,
            quota.byte_size(),
        );
        let concurrency = self.limits.max_concurrent_sources.max(1);
        // The futures are created before they are polled, so the sources are
        // resolved at most `concurrency` at a time.
        let budget = &budget;
        let resolutions: Vec<_> = sources
            .iter()
            .zip(&indices)
            .map(|(source, &index)| async move {
                self.load_encoded_bytes(source, budget)
                    .await
                    .map_err(|error| error.with_location(source.path(), index))
            })
            .collect();
        let encoded: Vec<Vec<u8>> = stream::iter(resolutions)
            .buffered(concurrency)
            .try_collect()
            .await?;

        // A fetcher might return bytes it did not reserve from `budget`, so this
        // loop applies the per-image and total limits to the returned lengths.
        let mut request_bytes = quota.byte_size();
        for ((data, source), &index) in encoded.iter().zip(sources).zip(&indices) {
            if data.len() > self.limits.max_image_bytes {
                return Err(
                    ImageError::image_too_large(data.len(), self.limits.max_image_bytes)
                        .with_location(source.path(), index),
                );
            }
            request_bytes += data.len();
            if request_bytes > self.limits.max_total_bytes {
                return Err(ImageError::total_size_too_large(
                    request_bytes,
                    self.limits.max_total_bytes,
                )
                .with_location(source.path(), index));
            }
        }
        let admitted_bytes = request_bytes - quota.byte_size();

        let preprocesses: Vec<_> = encoded
            .into_iter()
            .zip(sources)
            .zip(&indices)
            .map(|((data, source), &index)| async move {
                let options = self
                    .limits
                    .preprocess_options(source.detail(), request_image_count);
                self.preprocessor
                    .preprocess(data, options)
                    .await
                    .map_err(|error| error.with_location(source.path(), index))
            })
            .collect();
        let images = stream::iter(preprocesses)
            .buffered(concurrency)
            .try_collect()
            .await?;
        // The call records its sources only once all of them are preprocessed, so
        // a failure or a cancellation leaves the quota unchanged.
        quota.add(sources.len(), admitted_bytes);
        Ok(MultiModalData { images })
    }

    /// Load the encoded image bytes of one source: fetch an external URL with
    /// retries, decode a data URL, or copy caller-supplied bytes, reserving
    /// the bytes from `budget`. The returned error carries no request
    /// location.
    async fn load_encoded_bytes(
        &self,
        source: &ImageSource,
        budget: &ImageByteBudget,
    ) -> Result<Vec<u8>, ImageError> {
        let data = match source {
            ImageSource::DataUrl { data_url, .. } => decode_data_url(data_url, budget)?,
            ImageSource::Url { url, .. } => {
                // Sources the caller constructs without protocol conversion
                // receive the same HTTP prefix check as protocol input. The
                // fetcher validates the full URL.
                if !is_http_url(url) {
                    return Err(ImageError::InvalidUrl {
                        path: String::new(),
                        url: url.clone(),
                    });
                }
                if url.len() > self.limits.max_url_len {
                    return Err(ImageError::UrlTooLong {
                        path: String::new(),
                        len: url.len(),
                        max: self.limits.max_url_len,
                    });
                }
                self.retry
                    .execute(
                        |_| async { self.fetcher.fetch(url, budget).await },
                        ImageError::is_retryable,
                    )
                    .await
                    .map_err(|error| match error {
                        // These errors already carry the URL.
                        error @ (ImageError::Fetch { .. }
                        | ImageError::FetchStatus { .. }
                        | ImageError::DestinationRejected { .. }) => error,
                        error => ImageError::fetch(url, error),
                    })?
            }
            ImageSource::Bytes { data, .. } => {
                reserve_image_bytes(budget, data.len())?;
                data.clone()
            }
        };
        if data.is_empty() {
            return Err(ImageError::empty_image());
        }
        Ok(data)
    }
}

/// Return, for each source, the number of earlier sources with the same path:
/// the index that error messages report for the image.
fn image_indices(sources: &[ImageSource]) -> Vec<usize> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    sources
        .iter()
        .map(|source| {
            let count = seen.entry(source.path()).or_insert(0);
            let index = *count;
            *count += 1;
            index
        })
        .collect()
}

/// Reserve the bytes of one image that is already in memory.
fn reserve_image_bytes(budget: &ImageByteBudget, bytes: usize) -> Result<(), ImageError> {
    let max = budget.max_image_bytes();
    if bytes > max {
        return Err(ImageError::image_too_large(bytes, max));
    }
    budget.reserve(bytes)
}

/// Decode a base64 or percent-encoded data URL with a supported image media
/// type, reserving the decoded bytes from `budget`.
fn decode_data_url(data_url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
    let parsed = data_url::DataUrl::process(data_url).map_err(ImageError::invalid_data_url)?;
    let media_type = parsed.mime_type().to_string();
    if ImageMediaType::from_mime(&media_type).is_none() {
        return Err(ImageError::unsupported_media_type(media_type));
    }
    // A data URL carries the image in the request itself, so the estimated
    // decoded length is checked before decoding, and the decoded length is
    // checked and reserved once it is known.
    let estimated = estimated_decoded_len(data_url);
    let max_image_bytes = budget.max_image_bytes();
    if estimated > max_image_bytes {
        return Err(ImageError::image_too_large(estimated, max_image_bytes));
    }
    let (data, _) = parsed
        .decode_to_vec()
        .map_err(ImageError::invalid_data_url)?;
    reserve_image_bytes(budget, data.len())?;
    Ok(data)
}

/// Estimate the decoded byte length of a data URL from its body length.
///
/// For a header that ends with `;base64`, the estimate is three bytes for each
/// complete group of four body characters. For a body that holds only base64
/// alphabet and padding characters, the estimate differs from the decoded
/// length by at most two bytes: padding makes it larger, and omitted padding
/// makes it smaller. For any other header, the estimate counts each `%` and the
/// two characters after it as one byte, which is exact when every `%` begins a
/// two-digit hexadecimal escape. The header check is case-sensitive, while
/// decoding also accepts spellings such as `;BASE64`; a base64 body under such
/// a header receives the percent-encoded estimate, which is larger than its
/// decoded length. Characters that decoding removes or combines, such as ASCII
/// tabs and newlines, percent escapes in a base64 body, or a fragment, also
/// make the estimate larger than the decoded length.
fn estimated_decoded_len(data_url: &str) -> usize {
    let Some((header, body)) = data_url.split_once(',') else {
        return 0;
    };
    if header.ends_with(";base64") {
        body.len() / 4 * 3
    } else {
        body.len().saturating_sub(2 * body.matches('%').count())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use deepseek_recipe_core::multimodal::{ImageDetail, ImageInfo};

    use super::*;
    use crate::limits::PreprocessOptions;

    /// A fetcher that returns a body of a fixed size and ignores the budget.
    struct FixedFetcher {
        body_len: usize,
    }

    impl ImageFetcher for FixedFetcher {
        async fn fetch(
            &self,
            _url: &str,
            _budget: &ImageByteBudget,
        ) -> Result<Vec<u8>, ImageError> {
            Ok(vec![0; self.body_len])
        }
    }

    /// A fetcher that records how many of its calls run at the same time.
    #[derive(Clone, Default)]
    struct CountingFetcher {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl ImageFetcher for CountingFetcher {
        async fn fetch(&self, _url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            tokio::task::yield_now().await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            budget.reserve(1)?;
            Ok(vec![0])
        }
    }

    /// A preprocessor that rejects every image.
    struct NoPreprocessor;

    impl ImagePreprocessor for NoPreprocessor {}

    /// A preprocessor that accepts every image without inspecting its bytes.
    struct AcceptingPreprocessor;

    impl ImagePreprocessor for AcceptingPreprocessor {
        async fn preprocess(
            &self,
            data: Vec<u8>,
            _options: PreprocessOptions,
        ) -> Result<ImageInfo, ImageError> {
            Ok(ImageInfo {
                data,
                width: 1,
                height: 1,
            })
        }
    }

    fn limits(max_image_bytes: usize, max_total_bytes: usize) -> ImageLimits {
        ImageLimits {
            max_image_bytes,
            max_total_bytes,
            ..ImageLimits::default()
        }
    }

    fn bytes_source(len: usize) -> ImageSource {
        ImageSource::Bytes {
            data: vec![0; len],
            detail: ImageDetail::High,
            path: String::new(),
        }
    }

    fn url_source() -> ImageSource {
        ImageSource::Url {
            url: "https://example.com/image.webp".to_string(),
            detail: ImageDetail::High,
            path: String::new(),
        }
    }

    fn located_bytes_source(len: usize, path: &str) -> ImageSource {
        ImageSource::Bytes {
            data: vec![0; len],
            detail: ImageDetail::High,
            path: path.to_string(),
        }
    }

    #[tokio::test]
    async fn bytes_over_the_image_limit_are_rejected() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, NoPreprocessor)
            .with_limits(limits(4, 64));
        let error = resolver
            .resolve(&[bytes_source(5)], &mut ImageQuota::new())
            .await
            .expect_err("the image exceeds the image limit");

        assert!(
            matches!(
                error,
                ImageError::ImageTooLarge {
                    size: 5,
                    max: 4,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn bytes_over_the_total_limit_are_rejected() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, NoPreprocessor)
            .with_limits(limits(8, 10));
        let error = resolver
            .resolve(&[bytes_source(6), bytes_source(6)], &mut ImageQuota::new())
            .await
            .expect_err("the images exceed the total limit");

        assert!(
            matches!(
                error,
                ImageError::TotalSizeTooLarge {
                    size: 12,
                    max: 10,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_fetcher_that_ignores_the_budget_is_rejected() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 5 }, NoPreprocessor)
            .with_limits(limits(4, 64));
        let error = resolver
            .resolve(&[url_source()], &mut ImageQuota::new())
            .await
            .expect_err("the fetched image exceeds the image limit");

        assert!(
            matches!(
                error,
                ImageError::ImageTooLarge {
                    size: 5,
                    max: 4,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn data_urls_over_the_image_limit_are_rejected_without_decoding() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, NoPreprocessor)
            .with_limits(limits(4, 64));
        let source = ImageSource::DataUrl {
            data_url: format!("data:image/webp;base64,{}", "A".repeat(8)),
            detail: ImageDetail::High,
            path: String::new(),
        };
        let error = resolver
            .resolve(&[source], &mut ImageQuota::new())
            .await
            .expect_err("the data url exceeds the image limit");

        assert!(
            matches!(
                error,
                ImageError::ImageTooLarge {
                    size: 6,
                    max: 4,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[tokio::test]
    async fn sources_within_the_limits_reach_preprocessing() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 4 }, NoPreprocessor)
            .with_limits(limits(4, 64));
        let error = resolver
            .resolve(&[url_source()], &mut ImageQuota::new())
            .await
            .expect_err("the preprocessor rejects every image");

        assert!(matches!(error, ImageError::Unsupported(_)), "{error}");
    }

    #[tokio::test]
    async fn a_completed_call_records_its_sources() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, AcceptingPreprocessor)
            .with_limits(limits(8, 64));
        let mut quota = ImageQuota::new();
        resolver
            .resolve(&[bytes_source(6), bytes_source(6)], &mut quota)
            .await
            .expect("the preprocessor accepts every image");

        assert_eq!(quota.image_count(), 2);
        assert_eq!(quota.byte_size(), 12);
    }

    #[tokio::test]
    async fn a_preprocessing_failure_leaves_the_quota_unchanged() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, NoPreprocessor)
            .with_limits(limits(8, 64));
        let mut quota = ImageQuota::new();
        resolver
            .resolve(&[bytes_source(6), bytes_source(6)], &mut quota)
            .await
            .expect_err("the preprocessor rejects every image");

        assert_eq!(quota, ImageQuota::new());
    }

    #[tokio::test]
    async fn resolutions_are_limited_to_max_concurrent_sources() {
        const SOURCES: usize = 6;
        let fetcher = CountingFetcher::default();
        let resolver =
            ImageResolver::new(fetcher.clone(), NoPreprocessor).with_limits(ImageLimits {
                max_concurrent_sources: 3,
                ..limits(8, 64)
            });
        let sources = vec![url_source(); SOURCES];
        resolver
            .resolve(&sources, &mut ImageQuota::new())
            .await
            .expect_err("the preprocessor rejects every image");

        assert_eq!(
            fetcher.max_active.load(Ordering::SeqCst),
            3,
            "the resolver starts at most max_concurrent_sources fetches at a time"
        );
    }
    #[tokio::test]
    async fn errors_report_the_path_and_the_index_within_the_path() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, NoPreprocessor)
            .with_limits(limits(8, 64));
        let sources = [
            located_bytes_source(1, ".messages[0]"),
            located_bytes_source(1, ".messages[0]"),
            located_bytes_source(1, ".messages[2]"),
            located_bytes_source(9, ".messages[2]"),
        ];
        let error = resolver
            .resolve(&sources, &mut ImageQuota::new())
            .await
            .expect_err("the last image exceeds the image limit");

        assert!(
            matches!(
                &error,
                ImageError::ImageTooLarge { path, index: 1, size: 9, max: 8 }
                    if path == ".messages[2]"
            ),
            "{error:?}"
        );
        assert_eq!(
            error.to_string(),
            ".messages[2].image[1]: image file size exceeds limit 0 MB"
        );

        let error = resolver
            .resolve(&sources[..3], &mut ImageQuota::new())
            .await
            .expect_err("the preprocessor rejects every image");
        assert!(
            matches!(&error, ImageError::Unsupported(_)),
            "a preprocessing failure without a location is returned unchanged: {error:?}"
        );
    }

    #[tokio::test]
    async fn the_count_error_names_the_first_path_beyond_the_limit() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, AcceptingPreprocessor)
            .with_limits(ImageLimits {
                max_images: 3,
                ..limits(8, 64)
            });
        let mut quota = ImageQuota::new();
        resolver
            .resolve(&[located_bytes_source(1, "input[0]")], &mut quota)
            .await
            .expect("the first call is within the limit");
        let sources = [
            located_bytes_source(1, "input[1]"),
            located_bytes_source(1, "input[2]"),
            located_bytes_source(1, "input[2]"),
            located_bytes_source(1, "input[3]"),
        ];
        let error = resolver
            .resolve(&sources, &mut quota)
            .await
            .expect_err("the request exceeds the image count limit");

        assert_eq!(
            error.to_string(),
            "input[2]: Too many images: max 3 images per request, got 4"
        );
        assert_eq!(quota.image_count(), 1, "a failed call records nothing");
    }

    #[tokio::test]
    async fn the_total_size_error_names_the_crossing_source() {
        let resolver = ImageResolver::new(FixedFetcher { body_len: 0 }, NoPreprocessor)
            .with_limits(limits(8, 10));
        let sources = [
            located_bytes_source(6, "messages.0.content[0]"),
            located_bytes_source(6, "messages.0.content[1]"),
        ];
        let error = resolver
            .resolve(&sources, &mut ImageQuota::new())
            .await
            .expect_err("the images exceed the total limit");

        assert!(
            matches!(
                &error,
                ImageError::TotalSizeTooLarge { path, size: 12, max: 10 }
                    if path == "messages.0.content[1]"
            ),
            "{error:?}"
        );
    }
}
