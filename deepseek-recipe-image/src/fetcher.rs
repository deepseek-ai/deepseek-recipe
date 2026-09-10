//! Image fetching over HTTP with the `reqwest` client.

use std::time::Duration;

use tokio_stream::StreamExt;

use crate::ImageFetcher;
use crate::error::ImageError;
use crate::limits::{ImageByteBudget, ImageLimits};

/// Downloads images from external URLs with `reqwest`.
///
/// Clients created by [`Self::new`] and [`Self::with_max_bytes`] follow at most
/// five redirects and use a ten-second connection timeout and a sixty-second
/// request timeout. [`Self::from_client`] preserves the supplied client's
/// settings. All constructors enforce the configured response body size limit.
/// A response body is read as a stream, and the bytes that are kept are reserved
/// from the budget of the resolve call, so a body that would exceed the total
/// limit of the request stops the download.
///
/// We strongly recommend that you implement SSRF protection, such as rejecting
/// private, loopback, and link-local addresses, when callers may supply image
/// URLs; this fetcher does not filter its targets, because we think it is beyond
/// the scope of the library.
pub struct ReqwestImageFetcher {
    client: reqwest::Client,
    max_bytes: usize,
}

impl ReqwestImageFetcher {
    /// Create a fetcher with the default image size limit.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when the HTTP client cannot be constructed.
    pub fn new() -> Result<Self, ImageError> {
        Self::with_max_bytes(ImageLimits::default().max_image_bytes)
    }

    /// Create a fetcher with an explicit size limit in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when the HTTP client cannot be constructed.
    pub fn with_max_bytes(max_bytes: usize) -> Result<Self, ImageError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|err| ImageError::Client(err.to_string()))?;
        Ok(Self { client, max_bytes })
    }

    /// Create a fetcher from an existing client.
    pub fn from_client(client: reqwest::Client, max_bytes: usize) -> Self {
        Self { client, max_bytes }
    }

    /// Return the encoded size limit this fetcher applies to one image.
    ///
    /// The resolver can apply a smaller limit through the
    /// [`ImageByteBudget`] of the resolve call.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

impl ImageFetcher for ReqwestImageFetcher {
    async fn fetch(&self, url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|err| ImageError::Fetch {
                url: url.to_string(),
                source: Box::new(err),
            })?;
        if !response.status().is_success() {
            return Err(ImageError::FetchStatus {
                url: url.to_string(),
                status: response.status().as_u16(),
            });
        }
        // The limit of the fetcher and the limit of the resolve call both apply.
        let max_bytes = self.max_bytes.min(budget.max_image_bytes());
        if let Some(length) = response.content_length()
            && length > max_bytes as u64
        {
            return Err(ImageError::ImageTooLarge {
                size: length as usize,
                max: max_bytes,
            });
        }

        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    // A response that fails while its body is read is the
                    // retryable failure that can occur after bytes were
                    // reserved, so the retry reserves them once.
                    budget.release(body.len());
                    return Err(ImageError::Fetch {
                        url: url.to_string(),
                        source: Box::new(err),
                    });
                }
            };
            let image_bytes = body.len() + chunk.len();
            if image_bytes > max_bytes {
                return Err(ImageError::ImageTooLarge {
                    size: image_bytes,
                    max: max_bytes,
                });
            }
            budget.reserve(chunk.len())?;
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }
}
