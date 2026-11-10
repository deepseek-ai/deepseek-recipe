//! Limits applied while resolving image sources.

use std::sync::atomic::{AtomicUsize, Ordering};

use deepseek_recipe_core::multimodal::{ImageDetail, MAX_IMAGE_URL_LEN};

use crate::error::ImageError;

/// Limits applied to the images of one request.
#[derive(Debug, Clone, Copy)]
pub struct ImageLimits {
    /// Maximum number of images in one request.
    pub max_images: usize,
    /// Maximum UTF-8 byte length of an external image URL.
    pub max_url_len: usize,
    /// Maximum encoded size of one image in bytes.
    pub max_image_bytes: usize,
    /// Maximum total encoded size of the images of one request in bytes.
    pub max_total_bytes: usize,
    /// Maximum number of sources processed concurrently by one resolve call.
    /// A value of zero is treated as one.
    pub max_concurrent_sources: usize,
    /// Maximum width or height in pixels when a request has fewer than
    /// [`many_images_threshold`](Self::many_images_threshold) images.
    pub max_dimension_px: u32,
    /// Maximum width or height in pixels when a request has at least
    /// [`many_images_threshold`](Self::many_images_threshold) images.
    pub max_dimension_on_many_images_px: u32,
    /// Smallest image count of a request to which
    /// [`max_dimension_on_many_images_px`] applies.
    ///
    /// [`max_dimension_on_many_images_px`]: Self::max_dimension_on_many_images_px
    pub many_images_threshold: usize,
    /// Maximum long side in pixels at low detail, applied before token-budget
    /// fitting. Token-budget fitting might then scale the image up.
    pub low_detail_max_dimension_px: u32,
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_images: 600,
            max_url_len: MAX_IMAGE_URL_LEN,
            max_image_bytes: 32 * 1024 * 1024,
            max_total_bytes: 64 * 1024 * 1024,
            max_concurrent_sources: 8,
            max_dimension_px: 8192,
            max_dimension_on_many_images_px: 4096,
            many_images_threshold: 15,
            low_detail_max_dimension_px: 512,
        }
    }
}

impl ImageLimits {
    /// Return the maximum accepted width or height in pixels for a request
    /// carrying `image_count` images.
    pub const fn max_dimension(&self, image_count: usize) -> u32 {
        if image_count < self.many_images_threshold {
            self.max_dimension_px
        } else {
            self.max_dimension_on_many_images_px
        }
    }

    /// Return the preprocessing options for a request carrying `image_count`
    /// images at the given detail level.
    pub const fn preprocess_options(
        &self,
        detail: ImageDetail,
        image_count: usize,
    ) -> PreprocessOptions {
        PreprocessOptions {
            detail,
            max_dimension_px: self.max_dimension(image_count),
            low_detail_max_dimension_px: self.low_detail_max_dimension_px,
        }
    }
}

/// The image count and encoded bytes recorded by the completed resolve calls of
/// one request.
///
/// [`ImageResolver::resolve`](crate::ImageResolver::resolve) applies
/// [`ImageLimits::max_images`] and [`ImageLimits::max_total_bytes`] to the
/// recorded usage together with the sources of the current call. To apply
/// these limits across the resolve calls of a request, the caller must create
/// one quota for the request and pass it to every call. A call records its
/// sources after preprocessing every one of them, so a failed call adds
/// nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageQuota {
    images: usize,
    bytes: usize,
}

impl ImageQuota {
    /// Return an empty quota.
    pub const fn new() -> Self {
        Self {
            images: 0,
            bytes: 0,
        }
    }

    /// Return the number of sources recorded by the completed calls.
    pub const fn image_count(&self) -> usize {
        self.images
    }

    /// Return the number of encoded bytes recorded by the completed calls.
    pub const fn byte_size(&self) -> usize {
        self.bytes
    }

    /// Record `images` preprocessed sources with `bytes` encoded bytes in total.
    pub(crate) fn add(&mut self, images: usize, bytes: usize) {
        self.images += images;
        self.bytes += bytes;
    }
}

/// The encoded bytes still available to the image sources of one resolve call.
///
/// [`ImageResolver::resolve`](crate::ImageResolver::resolve) creates one budget
/// for each call and passes it to every fetch of that call. As
/// [`ImageFetcher::fetch`](crate::ImageFetcher::fetch) requires, a fetcher
/// reserves the bytes it keeps as its body arrives, so the concurrent fetches
/// of the call stay within [`ImageLimits::max_total_bytes`]. A budget is not
/// shared between calls: it starts with the bytes that the [`ImageQuota`] of
/// the request leaves available.
#[derive(Debug)]
pub struct ImageByteBudget {
    /// Encoded bytes still available to the sources of one resolve call.
    remaining: AtomicUsize,
    /// Maximum encoded size of one image in bytes.
    max_image_bytes: usize,
    /// Maximum total encoded size of the images of one request in bytes.
    max_total_bytes: usize,
}

impl ImageByteBudget {
    /// Create the budget of one resolve call for a request whose earlier
    /// resolve calls recorded `used_bytes` encoded bytes.
    ///
    /// `max_image_bytes` limits one image, and `max_total_bytes` limits all
    /// images of the request, including `used_bytes`. Both are in bytes.
    pub const fn new(max_image_bytes: usize, max_total_bytes: usize, used_bytes: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(max_total_bytes.saturating_sub(used_bytes)),
            max_image_bytes,
            max_total_bytes,
        }
    }

    /// Return the maximum encoded size of one image in bytes.
    pub const fn max_image_bytes(&self) -> usize {
        self.max_image_bytes
    }

    /// Reserve `bytes` encoded bytes of one image.
    ///
    /// This method does not check `bytes` against
    /// [`max_image_bytes`](Self::max_image_bytes).
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::TotalSizeTooLarge`] when the reservation would
    /// make the encoded bytes of the request exceed `max_total_bytes`. Its
    /// `size` counts the bytes recorded by earlier calls, the bytes reserved so
    /// far in this call, and `bytes`; its request location is empty until the
    /// resolver fills it. A failed reservation reserves nothing.
    pub fn reserve(&self, bytes: usize) -> Result<(), ImageError> {
        let mut available = self.remaining.load(Ordering::Relaxed);
        loop {
            if available < bytes {
                let used = self.max_total_bytes.saturating_sub(available);
                return Err(ImageError::total_size_too_large(
                    used.saturating_add(bytes),
                    self.max_total_bytes,
                ));
            }
            match self.remaining.compare_exchange_weak(
                available,
                available - bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => available = current,
            }
        }
    }

    /// Release `bytes` that a failed attempt reserved, making them available
    /// again.
    pub fn release(&self, bytes: usize) {
        self.remaining.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Options for one image preprocessing call.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessOptions {
    /// Requested image detail level.
    pub detail: ImageDetail,
    /// Maximum accepted source width or height in pixels, before resizing.
    pub max_dimension_px: u32,
    /// Maximum long side in pixels at low detail, applied before token-budget
    /// fitting. Token-budget fitting might then scale the image up.
    pub low_detail_max_dimension_px: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserving_bytes_reduces_the_available_bytes() {
        let budget = ImageByteBudget::new(8, 10, 0);
        assert_eq!(budget.max_image_bytes(), 8);
        budget.reserve(6).expect("the total limit is available");

        let error = budget
            .reserve(5)
            .expect_err("the reservation exceeds the total limit");
        assert!(
            matches!(
                error,
                ImageError::TotalSizeTooLarge {
                    size: 11,
                    max: 10,
                    ..
                }
            ),
            "{error}"
        );
        budget
            .reserve(4)
            .expect("the remaining bytes are available");
    }

    #[test]
    fn a_budget_accounts_for_the_bytes_of_earlier_calls() {
        let budget = ImageByteBudget::new(8, 10, 6);

        let error = budget
            .reserve(5)
            .expect_err("earlier calls used six bytes of the total limit");
        assert!(
            matches!(
                error,
                ImageError::TotalSizeTooLarge {
                    size: 11,
                    max: 10,
                    ..
                }
            ),
            "{error}"
        );
        budget
            .reserve(4)
            .expect("the remaining bytes are available");
    }
}
