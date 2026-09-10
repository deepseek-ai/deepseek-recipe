//! Resolve image sources into the images of one request.

use futures::StreamExt;
use futures::stream::{self, TryStreamExt};

use deepseek_recipe_core::multimodal::{ImageMediaType, ImageSource, MultiModalData, is_http_url};

use crate::error::ImageError;
use crate::limits::{ImageByteBudget, ImageLimits, ImageQuota};
use crate::retry::RetryPolicy;
use crate::{ImageFetcher, ImagePreprocessor};

/// Resolves image sources into preprocessed images.
///
/// Supports external URLs, data URLs, and encoded bytes. Applies request limits
/// and preserves source order. The caller supplies the fetcher and preprocessor.
/// At most [`ImageLimits::max_concurrent_sources`] sources are resolved and
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
    pub async fn resolve(
        &self,
        sources: &[ImageSource],
        quota: &mut ImageQuota,
    ) -> Result<MultiModalData, ImageError> {
        let request_image_count = quota.image_count() + sources.len();
        if request_image_count > self.limits.max_images {
            return Err(ImageError::TooManyImages {
                max: self.limits.max_images,
                count: request_image_count,
            });
        }

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
        let resolutions: Vec<_> = sources
            .iter()
            .map(|source| self.encoded_bytes(source, &budget))
            .collect();
        let encoded: Vec<Vec<u8>> = stream::iter(resolutions)
            .buffered(concurrency)
            .try_collect()
            .await?;

        // A fetch reserves the bytes it keeps while it reads a body. This check
        // rejects an implementation that returns more than it reserved.
        let mut request_bytes = quota.byte_size();
        for data in &encoded {
            if data.len() > self.limits.max_image_bytes {
                return Err(ImageError::ImageTooLarge {
                    size: data.len(),
                    max: self.limits.max_image_bytes,
                });
            }
            request_bytes += data.len();
            if request_bytes > self.limits.max_total_bytes {
                return Err(ImageError::TotalSizeTooLarge {
                    size: request_bytes,
                    max: self.limits.max_total_bytes,
                });
            }
        }
        let admitted_bytes = request_bytes - quota.byte_size();

        let preprocesses: Vec<_> = encoded
            .into_iter()
            .zip(sources)
            .map(|(data, source)| {
                let options = self
                    .limits
                    .preprocess_options(source.detail(), request_image_count);
                self.preprocessor.preprocess(data, options)
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

    /// Return the encoded bytes of one image source.
    async fn encoded_bytes(
        &self,
        source: &ImageSource,
        budget: &ImageByteBudget,
    ) -> Result<Vec<u8>, ImageError> {
        let data = match source {
            ImageSource::DataUrl { data_url, .. } => decode_data_url(data_url, budget)?,
            ImageSource::Url { url, .. } => {
                // Direct sources receive the same HTTP prefix check as
                // protocol input. The fetcher validates the full URL.
                if !is_http_url(url) {
                    return Err(ImageError::InvalidUrl { url: url.clone() });
                }
                if url.len() > self.limits.max_url_len {
                    return Err(ImageError::UrlTooLong {
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
                        // The fetcher already reports the rejected URL.
                        error @ (ImageError::Fetch { .. } | ImageError::FetchStatus { .. }) => {
                            error
                        }
                        error => ImageError::Fetch {
                            url: url.clone(),
                            source: Box::new(error),
                        },
                    })?
            }
            ImageSource::Bytes { data, .. } => {
                reserve_image_bytes(budget, data.len())?;
                data.clone()
            }
        };
        if data.is_empty() {
            return Err(ImageError::EmptyImage);
        }
        Ok(data)
    }
}

/// Reserve the bytes of one image that is already in memory.
fn reserve_image_bytes(budget: &ImageByteBudget, bytes: usize) -> Result<(), ImageError> {
    let max = budget.max_image_bytes();
    if bytes > max {
        return Err(ImageError::ImageTooLarge { size: bytes, max });
    }
    budget.reserve(bytes)
}

/// Decode a base64 or percent-encoded data URL with a supported image media type.
fn decode_data_url(data_url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
    let parsed = data_url::DataUrl::process(data_url)
        .map_err(|err| ImageError::InvalidDataUrl(err.to_string()))?;
    let media_type = parsed.mime_type().to_string();
    if ImageMediaType::from_mime(&media_type).is_none() {
        return Err(ImageError::UnsupportedMediaType(media_type));
    }
    // A data URL carries the image in the request itself, so the decoded size is
    // checked before the body is decoded and again once it is known.
    let estimated = estimated_decoded_len(data_url);
    let max_image_bytes = budget.max_image_bytes();
    if estimated > max_image_bytes {
        return Err(ImageError::ImageTooLarge {
            size: estimated,
            max: max_image_bytes,
        });
    }
    let (data, _) = parsed
        .decode_to_vec()
        .map_err(|err| ImageError::InvalidDataUrl(err.to_string()))?;
    reserve_image_bytes(budget, data.len())?;
    Ok(data)
}

/// Return the decoded size a data URL body reports before it is decoded.
///
/// A base64 body occupies four characters for every three bytes it carries, so
/// the result can be up to two bytes smaller than the decoded body. A
/// percent-encoded byte occupies three characters.
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
        }
    }

    fn url_source() -> ImageSource {
        ImageSource::Url {
            url: "https://example.com/image.webp".to_string(),
            detail: ImageDetail::High,
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
            matches!(error, ImageError::ImageTooLarge { size: 5, max: 4 }),
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
            matches!(error, ImageError::TotalSizeTooLarge { size: 12, max: 10 }),
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
            matches!(error, ImageError::ImageTooLarge { size: 5, max: 4 }),
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
        };
        let error = resolver
            .resolve(&[source], &mut ImageQuota::new())
            .await
            .expect_err("the data url exceeds the image limit");

        assert!(
            matches!(error, ImageError::ImageTooLarge { size: 6, max: 4 }),
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
}
