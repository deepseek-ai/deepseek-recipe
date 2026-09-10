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
    /// Maximum encoded size of one image.
    pub max_image_bytes: usize,
    /// Maximum total encoded size of the images of one request.
    pub max_total_bytes: usize,
    /// Maximum number of sources processed concurrently by one resolve call.
    /// A value of zero is treated as one.
    pub max_concurrent_sources: usize,
    /// Maximum width or height in pixels when a request has few images.
    pub max_dimension_px: u32,
    /// Maximum width or height in pixels when a request has many images.
    pub max_dimension_on_many_images_px: u32,
    /// Image count at which [`max_dimension_on_many_images_px`](Self::max_dimension_on_many_images_px)
    /// applies.
    pub many_images_threshold: usize,
    /// Maximum long side before token-budget fitting at low detail.
    /// The token specification may subsequently upscale the image.
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
    /// Return the maximum accepted width or height for a request carrying
    /// `image_count` images.
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

/// Usage accumulated over the resolve calls of one request.
///
/// The image count and total source byte limits are enforced across
/// [`resolve`](crate::ImageResolver::resolve) calls. Create one quota per
/// request and pass the same quota to every call. A call records its sources
/// after preprocessing every one of them, so a failed call adds nothing.
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

    /// Return the encoded source bytes recorded by the completed calls.
    pub const fn byte_size(&self) -> usize {
        self.bytes
    }

    /// Record `images` preprocessed sources of `bytes` encoded size.
    pub(crate) fn add(&mut self, images: usize, bytes: usize) {
        self.images += images;
        self.bytes += bytes;
    }
}

/// The encoded bytes still available to the image sources of one resolve call.
///
/// [`ImageResolver::resolve`](crate::ImageResolver::resolve) creates one budget
/// for each call and passes it to every fetch of that call. A fetch reserves the
/// bytes it keeps as its body arrives, so the fetches running concurrently
/// cannot together exceed [`ImageLimits::max_total_bytes`]. A budget is not
/// shared between calls: the images of earlier calls are accounted for through
/// the [`ImageQuota`] the call was given.
#[derive(Debug)]
pub struct ImageByteBudget {
    /// Bytes still available to the sources of one resolve call.
    remaining: AtomicUsize,
    /// Maximum encoded size of one image.
    max_image_bytes: usize,
    /// Maximum total encoded size of the images of one request.
    max_total_bytes: usize,
}

impl ImageByteBudget {
    /// Create the budget of one request that has already accumulated
    /// `used_bytes`.
    ///
    /// `max_image_bytes` limits one image, and `max_total_bytes` limits every
    /// image of the request.
    pub const fn new(max_image_bytes: usize, max_total_bytes: usize, used_bytes: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(max_total_bytes.saturating_sub(used_bytes)),
            max_image_bytes,
            max_total_bytes,
        }
    }

    /// Return the maximum encoded size of one image.
    pub const fn max_image_bytes(&self) -> usize {
        self.max_image_bytes
    }

    /// Reserve `bytes` of one image.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::TotalSizeTooLarge`] when the request would exceed
    /// its total encoded size. A failed reservation reserves nothing.
    pub fn reserve(&self, bytes: usize) -> Result<(), ImageError> {
        let mut available = self.remaining.load(Ordering::Relaxed);
        loop {
            if available < bytes {
                let used = self.max_total_bytes.saturating_sub(available);
                return Err(ImageError::TotalSizeTooLarge {
                    size: used.saturating_add(bytes),
                    max: self.max_total_bytes,
                });
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

    /// Return `bytes` reserved by an attempt that failed.
    pub fn release(&self, bytes: usize) {
        self.remaining.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Options for one image preprocessing call.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessOptions {
    /// Requested detail level.
    pub detail: ImageDetail,
    /// Maximum accepted source width or height in pixels, before resizing.
    pub max_dimension_px: u32,
    /// Maximum long side before token-budget fitting at low detail.
    /// The token specification may subsequently upscale the image.
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
            matches!(error, ImageError::TotalSizeTooLarge { size: 11, max: 10 }),
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
            matches!(error, ImageError::TotalSizeTooLarge { size: 11, max: 10 }),
            "{error}"
        );
        budget
            .reserve(4)
            .expect("the remaining bytes are available");
    }

    #[test]
    fn a_release_returns_the_bytes_of_a_failed_attempt() {
        let budget = ImageByteBudget::new(8, 10, 0);
        budget.reserve(10).expect("the total limit is available");
        budget.release(10);

        budget
            .reserve(10)
            .expect("the released bytes are available again");
    }
}
