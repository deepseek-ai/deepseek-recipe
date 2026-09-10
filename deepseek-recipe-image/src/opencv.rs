//! The default image preprocessor for DeepSeek V4.1.

use std::io::Cursor;

use deepseek_recipe_core::multimodal::{ImageInfo, ImageMediaType, ImageTokenSpec};
use image::{ImageDecoder, Limits};
use opencv::core::{Mat, MatTraitConst, Scalar, Vector, VectorToVec};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use crate::ImagePreprocessor;
use crate::error::ImageError;
use crate::limits::PreprocessOptions;

/// Padding color (BGR) matching the model image transform mean of 0.5.
const PAD_COLOR: Scalar = Scalar::new(127.0, 127.0, 127.0, 0.0);
/// Background (BGR) applied where an image has transparency: `#FDFDFD`.
const INFERENCE_BACKGROUND: (u8, u8, u8) = (0xfd, 0xfd, 0xfd);
/// Maximum size of one dimension of a WebP image.
const WEBP_MAX_DIMENSION: i32 = 16383;
/// Largest allocation allowed while reading only an image header.
const MAX_HEADER_ALLOC: u64 = 16 * 1024 * 1024;
/// WebP quality of a preprocessed image.
const WEBP_QUALITY: i32 = 90;

/// Preprocesses images with OpenCV.
///
/// Preprocessing decodes the image, applies the low-detail limit, fits the image
/// into the V4.1 token budget, and encodes it as WebP at quality 90.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCvImagePreprocessor;

impl ImagePreprocessor for OpenCvImagePreprocessor {
    async fn preprocess(
        &self,
        data: Vec<u8>,
        options: PreprocessOptions,
    ) -> Result<ImageInfo, ImageError> {
        tokio::task::spawn_blocking(move || preprocess(&data, options))
            .await
            .map_err(|err| ImageError::Decode(err.to_string()))?
    }
}

/// Preprocess one encoded image.
fn preprocess(data: &[u8], options: PreprocessOptions) -> Result<ImageInfo, ImageError> {
    if data.is_empty() {
        return Err(ImageError::EmptyImage);
    }
    let media_type = detect_media_type(data)?;
    check_dimensions(data, options.max_dimension_px)?;
    let mut mat = decode_image_mat(data, media_type)?;
    if options.detail.is_low() {
        mat = limit_image_mat(mat, options.low_detail_max_dimension_px)?;
    }
    preprocess_mat(mat, options.max_dimension_px)
}

/// Return the media type of the image bytes, read from the file header.
fn detect_media_type(data: &[u8]) -> Result<ImageMediaType, ImageError> {
    let kind = infer::get(data)
        .ok_or_else(|| ImageError::UnsupportedMediaType("unknown media type".to_string()))?;
    let mime = kind.mime_type();
    ImageMediaType::from_mime(mime)
        .ok_or_else(|| ImageError::UnsupportedMediaType(mime.to_string()))
}

/// Reject an image whose declared dimensions exceed `max_dimension`, reading the
/// file header only.
fn check_dimensions(data: &[u8], max_dimension: u32) -> Result<(), ImageError> {
    let mut reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(max_dimension);
    limits.max_image_height = Some(max_dimension);
    limits.max_alloc = Some(MAX_HEADER_ALLOC);
    reader.limits(limits);

    let decoder = match reader.into_decoder() {
        Ok(decoder) => decoder,
        Err(err) => return check_dimensions_without_limits(err, data, max_dimension),
    };
    let (width, height) = decoder.dimensions();
    if width > max_dimension || height > max_dimension {
        return Err(ImageError::ImageDimensionsTooLarge);
    }
    Ok(())
}

/// Handle a header read that failed under the configured limits.
///
/// Decoders that do not support limits report the limit as unsupported. Their
/// dimensions are read directly instead.
fn check_dimensions_without_limits(
    err: image::ImageError,
    data: &[u8],
    max_dimension: u32,
) -> Result<(), ImageError> {
    let image::ImageError::Limits(limit_error) = &err else {
        return Err(ImageError::Decode(err.to_string()));
    };
    match limit_error.kind() {
        image::error::LimitErrorKind::DimensionError => Err(ImageError::ImageDimensionsTooLarge),
        image::error::LimitErrorKind::Unsupported { .. } => {
            let reader = image::ImageReader::new(Cursor::new(data))
                .with_guessed_format()
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            let decoder = reader
                .into_decoder()
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            let (width, height) = decoder.dimensions();
            if width > max_dimension || height > max_dimension {
                return Err(ImageError::ImageDimensionsTooLarge);
            }
            Ok(())
        }
        _ => Err(ImageError::Decode(err.to_string())),
    }
}

/// Decode one image into an 8-bit BGR matrix.
///
/// GIF images are decoded with the `image` crate because OpenCV does not support
/// them. Images with transparency are blended onto the inference background.
fn decode_image_mat(data: &[u8], media_type: ImageMediaType) -> Result<Mat, ImageError> {
    let mat = if media_type == ImageMediaType::Gif {
        gif_to_mat(data)?
    } else {
        imgcodecs::imdecode(&Vector::from_slice(data), imgcodecs::IMREAD_UNCHANGED)
            .map_err(|err| ImageError::Decode(err.to_string()))?
    };

    let size = mat
        .size()
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    if size.width == 0 || size.height == 0 {
        return Err(ImageError::EmptyImage);
    }
    normalize_decoded_mat(mat)
}

/// Reduce a decoded matrix to 8-bit BGR.
fn normalize_decoded_mat(mat: Mat) -> Result<Mat, ImageError> {
    let mat = if mat.depth() == opencv::core::CV_16U {
        let mut mat8 = Mat::default();
        mat.convert_to(&mut mat8, opencv::core::CV_8U, 1.0 / 257.0, 0.0)
            .map_err(|err| ImageError::Decode(err.to_string()))?;
        mat8
    } else {
        mat
    };

    match mat.channels() {
        3 => Ok(mat),
        4 => alpha_blend(&mat),
        1 => {
            let mut channels = Vector::<Mat>::new();
            channels.push(mat.clone());
            channels.push(mat.clone());
            channels.push(mat);
            let mut bgr = Mat::default();
            opencv::core::merge(&channels, &mut bgr)
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            Ok(bgr)
        }
        2 => {
            let mut channels = Vector::<Mat>::new();
            opencv::core::split(&mat, &mut channels)
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            let gray = channels
                .get(0)
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            let alpha = channels
                .get(1)
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            let mut bgra_channels = Vector::<Mat>::new();
            bgra_channels.push(gray.clone());
            bgra_channels.push(gray.clone());
            bgra_channels.push(gray);
            bgra_channels.push(alpha);
            let mut bgra = Mat::default();
            opencv::core::merge(&bgra_channels, &mut bgra)
                .map_err(|err| ImageError::Decode(err.to_string()))?;
            alpha_blend(&bgra)
        }
        channels => Err(ImageError::Decode(format!(
            "unsupported image channel count: {channels}"
        ))),
    }
}

/// Blend a 4-channel BGRA matrix onto the inference background.
fn alpha_blend(bgra: &Mat) -> Result<Mat, ImageError> {
    if !bgra.is_continuous() {
        return Err(ImageError::Decode(
            "image matrix is not continuous".to_string(),
        ));
    }
    let (rows, cols) = (bgra.rows(), bgra.cols());
    // SAFETY: the allocation is owned by the returned matrix and is fully
    // written by the loop below.
    let mut blended = unsafe { Mat::new_rows_cols(rows, cols, opencv::core::CV_8UC3) }
        .map_err(|err| ImageError::Decode(err.to_string()))?;

    let source = bgra
        .data_bytes()
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    let target = blended
        .data_bytes_mut()
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    let (background_b, background_g, background_r) = INFERENCE_BACKGROUND;
    let (background_b, background_g, background_r) = (
        u32::from(background_b),
        u32::from(background_g),
        u32::from(background_r),
    );

    for (source_pixel, target_pixel) in source.chunks_exact(4).zip(target.chunks_exact_mut(3)) {
        let alpha = u32::from(source_pixel[3]);
        let inverse_alpha = 255 - alpha;
        target_pixel[0] =
            ((u32::from(source_pixel[0]) * alpha + background_b * inverse_alpha + 127) / 255) as u8;
        target_pixel[1] =
            ((u32::from(source_pixel[1]) * alpha + background_g * inverse_alpha + 127) / 255) as u8;
        target_pixel[2] =
            ((u32::from(source_pixel[2]) * alpha + background_r * inverse_alpha + 127) / 255) as u8;
    }
    Ok(blended)
}

/// Scale an image down so that its long side is at most `max_dimension`.
fn limit_image_mat(mat: Mat, max_dimension: u32) -> Result<Mat, ImageError> {
    let size = mat
        .size()
        .map_err(|err| ImageError::Resize(err.to_string()))?;
    let long_side = size.width.max(size.height);
    if long_side <= max_dimension as i32 {
        return Ok(mat);
    }
    let ratio = max_dimension as f64 / long_side as f64;
    let width = ((size.width as f64 * ratio).round() as i32).max(1);
    let height = ((size.height as f64 * ratio).round() as i32).max(1);
    direct_resize(&mat, width, height)
}

/// Fit an image into the V4.1 token budget and encode it as WebP.
fn preprocess_mat(mat: Mat, max_dimension_px: u32) -> Result<ImageInfo, ImageError> {
    let size = mat
        .size()
        .map_err(|err| ImageError::Resize(err.to_string()))?;
    // `check_dimensions` reads the size the header declares. The decoded matrix
    // is the image the request carries, so it is checked against the limit too.
    let max_dimension = max_dimension_px as i32;
    if size.width > max_dimension || size.height > max_dimension {
        return Err(ImageError::ImageDimensionsTooLarge);
    }
    let (best_width, best_height) =
        fit_webp_target_size(size.width as usize, size.height as usize)?;
    if best_width <= 0 || best_height <= 0 {
        return Err(ImageError::Resize(format!(
            "invalid target size: {best_width}x{best_height}"
        )));
    }

    let resized = if size.width != best_width || size.height != best_height {
        resize_to_best(&mat, size.width, size.height, best_width, best_height)
            .or_else(|_| direct_resize(&mat, best_width, best_height))?
    } else {
        mat
    };

    let mut encoded = Vector::new();
    let params = Vector::from_slice(&[imgcodecs::IMWRITE_WEBP_QUALITY, WEBP_QUALITY]);
    // A false return value reports an encoding failure, leaving the output
    // buffer without an encoded image.
    let encode_succeeded = imgcodecs::imencode(".webp", &resized, &mut encoded, &params)
        .map_err(|err| ImageError::Encode(err.to_string()))?;
    if !encode_succeeded {
        return Err(ImageError::EncodeFailed);
    }
    Ok(ImageInfo {
        data: encoded.to_vec(),
        width: best_width as u32,
        height: best_height as u32,
    })
}

/// Return the V4.1 target size of an image of `width`×`height` pixels that WebP
/// can encode.
///
/// [`ImageTokenSpec::v41`] fits the token budget without a dimension limit, and
/// an elongated image can require a side above [`WEBP_MAX_DIMENSION`]. A size
/// outside the limit is scaled into it and placed on the patch grid, and fitting
/// is repeated for that size, so the returned size is both a size the
/// specification leaves unchanged and a size WebP encodes.
fn fit_webp_target_size(width: usize, height: usize) -> Result<(i32, i32), ImageError> {
    // The specification scales a size below its minimum area back up, so the
    // limit is reached after a few rounds that each raise the short side.
    const MAX_FIT_ROUNDS: usize = 6;

    let spec = ImageTokenSpec::v41();
    let limit = WEBP_MAX_DIMENSION as usize;
    let mut source = (width, height);
    for _ in 0..MAX_FIT_ROUNDS {
        let fitted = spec.calc_resize(source.0, source.1)?;
        if fitted.best_width <= limit && fitted.best_height <= limit {
            return Ok((fitted.best_width as i32, fitted.best_height as i32));
        }
        source = scale_into_webp_limit(spec.patch_size(), fitted.best_width, fitted.best_height);
    }
    Err(ImageError::Resize(format!(
        "no V4.1 target size within {WEBP_MAX_DIMENSION} pixels for an image of {width}x{height} pixels"
    )))
}

/// Scale a size into the WebP dimension limit.
///
/// The long side is placed on the patch grid at or below the limit, and the
/// short side is placed on the patch grid at or above its scaled value. Raising
/// the short side keeps the scaled size at or above the minimum area of the
/// specification, which otherwise scales the long side past the limit again.
fn scale_into_webp_limit(patch_size: usize, width: usize, height: usize) -> (usize, usize) {
    let limit = WEBP_MAX_DIMENSION as usize;
    let long_side = width.max(height);
    if long_side <= limit {
        return (width, height);
    }
    let scaled_long = limit / patch_size * patch_size;
    let scaled_short = (width.min(height) as f64 * scaled_long as f64 / long_side as f64).round();
    let short_side = (scaled_short as usize).max(1).div_ceil(patch_size) * patch_size;
    if width >= height {
        (scaled_long, short_side)
    } else {
        (short_side, scaled_long)
    }
}

/// Scale an image to fit `best_width`×`best_height` while preserving its aspect
/// ratio, and center it on a background of [`PAD_COLOR`].
fn resize_to_best(
    mat: &Mat,
    source_width: i32,
    source_height: i32,
    best_width: i32,
    best_height: i32,
) -> Result<Mat, ImageError> {
    let image_ratio = source_width as f64 / source_height as f64;
    let target_ratio = best_width as f64 / best_height as f64;
    let (width, height) = if image_ratio > target_ratio {
        (
            best_width,
            round_half_even(source_height as f64 / source_width as f64 * best_width as f64).max(1),
        )
    } else if image_ratio < target_ratio {
        (
            round_half_even(source_width as f64 / source_height as f64 * best_height as f64).max(1),
            best_height,
        )
    } else {
        (best_width, best_height)
    };

    let mut scaled = Mat::default();
    imgproc::resize(
        mat,
        &mut scaled,
        opencv::core::Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_CUBIC,
    )
    .map_err(|err| ImageError::Resize(err.to_string()))?;

    let left = round_half_even((best_width - width) as f64 * 0.5);
    let top = round_half_even((best_height - height) as f64 * 0.5);
    let right = best_width - width - left;
    let bottom = best_height - height - top;

    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &scaled,
        &mut padded,
        top,
        bottom,
        left,
        right,
        opencv::core::BORDER_CONSTANT,
        PAD_COLOR,
    )
    .map_err(|err| ImageError::Resize(err.to_string()))?;
    Ok(padded)
}

/// Scale an image directly to `width`×`height`.
fn direct_resize(mat: &Mat, width: i32, height: i32) -> Result<Mat, ImageError> {
    let mut resized = Mat::default();
    imgproc::resize(
        mat,
        &mut resized,
        opencv::core::Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_CUBIC,
    )
    .map_err(|err| ImageError::Resize(err.to_string()))?;
    Ok(resized)
}

/// Decode the first frame of a GIF into a 4-channel BGRA matrix.
fn gif_to_mat(data: &[u8]) -> Result<Mat, ImageError> {
    let reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    let format = reader
        .format()
        .ok_or_else(|| ImageError::Decode("unknown image format".to_string()))?;
    if format != image::ImageFormat::Gif {
        return Err(ImageError::Decode(format!("expected GIF, got {format:?}")));
    }
    let frame = reader
        .decode()
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    let (width, height) = (frame.width(), frame.height());
    if width == 0 || height == 0 {
        return Err(ImageError::EmptyImage);
    }

    // OpenCV expects BGRA; the `image` crate decodes to RGBA.
    let mut bgra = frame.to_rgba8().into_raw();
    for pixel in bgra.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }

    // SAFETY: the allocation is owned by the returned matrix and is fully
    // written from `bgra` immediately below.
    let mut mat = unsafe { Mat::new_rows_cols(height as i32, width as i32, opencv::core::CV_8UC4) }
        .map_err(|err| ImageError::Decode(err.to_string()))?;
    mat.data_bytes_mut()
        .map_err(|err| ImageError::Decode(err.to_string()))?
        .copy_from_slice(&bgra);
    Ok(mat)
}

/// Round to the nearest integer, with a value exactly between two integers
/// rounding to the even one.
fn round_half_even(value: f64) -> i32 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 || (fraction == 0.5 && (floor as i64) % 2 == 0) {
        floor as i32
    } else {
        (floor + 1.0) as i32
    }
}
