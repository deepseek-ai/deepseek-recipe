//! Image input types and DeepSeek V4.1 image token accounting.
//!
//! A caller supplies images as [`ImageSource`] values. Resolving an image source
//! produces an [`ImageInfo`] carrying the bytes sent to the inference backend.
//! Every image contributes one [`IMAGE_SPECIAL_TOKEN`] placeholder to the prompt
//! text, in the order the images appear in the conversation.

use serde::{Deserialize, Serialize};

pub mod token_spec;

pub use token_spec::{CalcResizeError, ImageTokenSpec, ResizeResult};

/// Placeholder written into the prompt for every image.
pub const IMAGE_SPECIAL_TOKEN: &str = "<｜image｜>";

/// Token ID that the V4.1 tokenizer assigns to [`IMAGE_SPECIAL_TOKEN`]. A caller
/// that encodes the prompt into token IDs can compare against this ID.
pub const IMAGE_SPECIAL_TOKEN_ID: u32 = 129264;

/// Maximum UTF-8 byte length of an external image URL during conversion.
pub const MAX_IMAGE_URL_LEN: usize = 8192;

/// Media type accepted for image input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMediaType {
    Jpeg,
    Png,
    Gif,
    Webp,
}

impl ImageMediaType {
    /// Return the media type of this image media type.
    pub const fn mime(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }

    /// Resolve a media type string. Parameters after `;` are ignored, matching
    /// is case-insensitive, and `image/jpg` is accepted as an alias of
    /// `image/jpeg`. Returns `None` for an unsupported media type.
    pub fn from_mime(mime: &str) -> Option<Self> {
        let mime = mime.split(';').next().unwrap_or_default().trim();
        let (kind, subtype) = mime.split_once('/')?;
        if !kind.eq_ignore_ascii_case("image") {
            return None;
        }
        if subtype.eq_ignore_ascii_case("jpeg") || subtype.eq_ignore_ascii_case("jpg") {
            return Some(Self::Jpeg);
        }
        if subtype.eq_ignore_ascii_case("png") {
            return Some(Self::Png);
        }
        if subtype.eq_ignore_ascii_case("gif") {
            return Some(Self::Gif);
        }
        if subtype.eq_ignore_ascii_case("webp") {
            return Some(Self::Webp);
        }
        None
    }
}

/// Requested image detail level.
///
/// The OpenCV preprocessor applies an additional long-side limit for
/// [`Low`](Self::Low) before fitting the token budget. All levels still undergo
/// token-budget fitting and WebP encoding, which can resize the image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    #[default]
    High,
    Original,
    Auto,
}

impl ImageDetail {
    /// Return whether the additional low-detail long-side limit applies.
    pub const fn is_low(self) -> bool {
        matches!(self, Self::Low)
    }
}

/// One image supplied with a conversation, before resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    /// Data URL with a base64 or percent-encoded body.
    DataUrl {
        data_url: String,
        detail: ImageDetail,
    },
    /// External image URL. Conversion and resolution check the case-insensitive
    /// `http` prefix; the fetcher is responsible for validating the full URL.
    Url { url: String, detail: ImageDetail },
    /// Encoded image bytes supplied by the caller.
    Bytes { data: Vec<u8>, detail: ImageDetail },
}

impl ImageSource {
    /// Return the requested detail level.
    pub const fn detail(&self) -> ImageDetail {
        match self {
            Self::DataUrl { detail, .. }
            | Self::Url { detail, .. }
            | Self::Bytes { detail, .. } => *detail,
        }
    }
}

/// Return whether a value is an external image URL.
///
/// Checks whether the first four bytes spell `http`, ignoring ASCII case.
/// This classifies external image references during conversion and resolution;
/// it does not validate URL syntax, the complete scheme, or the destination.
pub fn is_http_url(url: &str) -> bool {
    url.get(..4)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("http"))
}

/// A preprocessed image sent to the inference backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInfo {
    /// Encoded image bytes.
    pub data: Vec<u8>,
    /// Width in pixels after preprocessing.
    pub width: u32,
    /// Height in pixels after preprocessing.
    pub height: u32,
}

/// Images attached to one conversation request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiModalData {
    /// Images in the order their placeholders appear in the prompt.
    pub images: Vec<ImageInfo>,
}

impl MultiModalData {
    /// Return whether no image is attached.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Return the number of prompt tokens beyond one per placeholder.
    ///
    /// Every image contributes one placeholder token to the prompt and
    /// [`ImageTokenSpec::calc_token_len`] tokens to the inference input. The
    /// adjustment is the difference over all images.
    ///
    /// # Errors
    ///
    /// Returns [`CalcResizeError`] when the token length of an image does not
    /// converge.
    pub fn image_token_adjustment(&self) -> Result<usize, CalcResizeError> {
        let spec = ImageTokenSpec::v41();
        let mut adjustment = 0usize;
        for image in &self.images {
            adjustment += spec.calc_token_len(image.width as usize, image.height as usize)? - 1;
        }
        Ok(adjustment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_adjustment_counts_one_placeholder_per_image() {
        let spec = ImageTokenSpec::v41();
        let data = MultiModalData {
            images: vec![
                ImageInfo {
                    data: Vec::new(),
                    width: 1920,
                    height: 1080,
                },
                ImageInfo {
                    data: Vec::new(),
                    width: 640,
                    height: 480,
                },
            ],
        };
        let expected =
            spec.calc_token_len(1920, 1080).unwrap() + spec.calc_token_len(640, 480).unwrap() - 2;
        assert_eq!(data.image_token_adjustment().unwrap(), expected);
        assert!(MultiModalData::default().image_token_adjustment().unwrap() == 0);
    }
}
