//! Python bindings for image fetching, preprocessing, and token accounting.

use std::future::Future;
use std::sync::Arc;

use deepseek_recipe_core::multimodal::{ImageInfo, MultiModalData};
use deepseek_recipe_image::{
    ImageByteBudget, ImageError as RustImageError, ImageFetcher, ImagePreprocessor, ImageQuota,
    ImageResolver, OpenCvImagePreprocessor, PreprocessOptions, ReqwestImageFetcher,
};
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::conversation::{PyImageSource, image_detail_str, parse_image_detail};

pyo3::create_exception!(
    _native,
    ImageError,
    PyException,
    "An image fetching, decoding, or preprocessing failure."
);

pyo3::create_exception!(
    _native,
    CalcResizeError,
    PyValueError,
    "The Rust image token calculation did not converge."
);

/// Preserve the Rust error variant and retry classification in Python.
fn image_error(py: Python<'_>, error: RustImageError) -> PyErr {
    let kind = match &error {
        RustImageError::Unsupported(_) => "Unsupported",
        RustImageError::UrlTooLong { .. } => "UrlTooLong",
        RustImageError::InvalidUrl { .. } => "InvalidUrl",
        RustImageError::TooManyImages { .. } => "TooManyImages",
        RustImageError::ImageTooLarge { .. } => "ImageTooLarge",
        RustImageError::TotalSizeTooLarge { .. } => "TotalSizeTooLarge",
        RustImageError::InvalidDataUrl(_) => "InvalidDataUrl",
        RustImageError::EmptyImage => "EmptyImage",
        RustImageError::UnsupportedMediaType(_) => "UnsupportedMediaType",
        RustImageError::ImageDimensionsTooLarge => "ImageDimensionsTooLarge",
        RustImageError::Decode(_) => "Decode",
        RustImageError::Resize(_) => "Resize",
        RustImageError::Encode(_) => "Encode",
        RustImageError::TokenBudget(_) => "TokenBudget",
        RustImageError::Fetch { .. } => "Fetch",
        RustImageError::FetchStatus { .. } => "FetchStatus",
        RustImageError::Client(_) => "Client",
        RustImageError::EncodeFailed => "EncodeFailed",
    };
    let exception = ImageError::new_err(error.to_string());
    let value = exception.value(py);
    if let Err(error) = value
        .setattr("kind", kind)
        .and_then(|()| value.setattr("is_retryable", error.is_retryable()))
    {
        return error;
    }
    exception
}

/// Run an asynchronous Rust image operation with the GIL released.
fn run_image_operation<T: Send>(
    py: Python<'_>,
    future: impl Future<Output = Result<T, RustImageError>> + Send,
) -> PyResult<T> {
    let result = py
        .detach(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            Ok::<_, std::io::Error>(runtime.block_on(future))
        })
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    result.map_err(|error| image_error(py, error))
}

/// Encoded image bytes and their dimensions in pixels.
#[pyclass(name = "ImageInfo", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyImageInfo {
    inner: ImageInfo,
}

#[pymethods]
impl PyImageInfo {
    #[new]
    fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            inner: ImageInfo {
                data,
                width,
                height,
            },
        }
    }

    /// Encoded image bytes.
    #[getter]
    fn data(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.inner.data).unbind()
    }

    /// Image width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Image height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }
}

/// Images in the order their placeholders appear in a prompt.
#[pyclass(name = "MultiModalData", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyMultiModalData {
    inner: MultiModalData,
}

#[pymethods]
impl PyMultiModalData {
    #[new]
    #[pyo3(signature = (images=None))]
    fn new(images: Option<Vec<Py<PyImageInfo>>>) -> Self {
        Self {
            inner: MultiModalData {
                images: images
                    .unwrap_or_default()
                    .iter()
                    .map(|image| image.get().inner.clone())
                    .collect(),
            },
        }
    }

    /// Images in placeholder order.
    #[getter]
    fn images(&self) -> Vec<PyImageInfo> {
        self.inner
            .images
            .iter()
            .map(|image| PyImageInfo {
                inner: image.clone(),
            })
            .collect()
    }

    /// Return whether no image is attached.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the number of prompt tokens beyond one per image placeholder.
    fn image_token_adjustment(&self) -> PyResult<usize> {
        self.inner
            .image_token_adjustment()
            .map_err(|error| CalcResizeError::new_err(error.to_string()))
    }
}

/// Image count and encoded size accumulated across resolve calls.
/// A call records its sources after preprocessing all of them, so a failed call
/// adds nothing.
#[pyclass(name = "ImageQuota", module = "deepseek_recipe._native")]
pub(crate) struct PyImageQuota {
    inner: ImageQuota,
}

#[pymethods]
impl PyImageQuota {
    #[new]
    fn new() -> Self {
        Self {
            inner: ImageQuota::new(),
        }
    }

    /// Return the number of sources recorded by the completed calls.
    fn image_count(&self) -> usize {
        self.inner.image_count()
    }

    /// Return the encoded source bytes recorded by the completed calls.
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }
}

/// Options passed to one image preprocessing operation.
#[pyclass(name = "PreprocessOptions", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyPreprocessOptions {
    inner: PreprocessOptions,
}

#[pymethods]
impl PyPreprocessOptions {
    #[new]
    fn new(
        detail: &str,
        max_dimension_px: u32,
        low_detail_max_dimension_px: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PreprocessOptions {
                detail: parse_image_detail(detail)?,
                max_dimension_px,
                low_detail_max_dimension_px,
            },
        })
    }

    /// Requested image detail level.
    #[getter]
    fn detail(&self) -> &'static str {
        image_detail_str(self.inner.detail)
    }

    /// Maximum accepted source width or height in pixels, before resizing.
    #[getter]
    fn max_dimension_px(&self) -> u32 {
        self.inner.max_dimension_px
    }

    /// Maximum long side before token-budget fitting at low detail.
    #[getter]
    fn low_detail_max_dimension_px(&self) -> u32 {
        self.inner.low_detail_max_dimension_px
    }
}

/// Fetches external image URLs through the Rust HTTP client.
#[derive(Clone)]
#[pyclass(
    name = "ReqwestImageFetcher",
    module = "deepseek_recipe._native",
    frozen
)]
pub(crate) struct PyReqwestImageFetcher {
    inner: Arc<ReqwestImageFetcher>,
}

impl ImageFetcher for PyReqwestImageFetcher {
    fn fetch(
        &self,
        url: &str,
        budget: &ImageByteBudget,
    ) -> impl Future<Output = Result<Vec<u8>, RustImageError>> + Send {
        self.inner.fetch(url, budget)
    }
}

#[pymethods]
impl PyReqwestImageFetcher {
    /// Construct the Rust fetcher with its default image size limit.
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        ReqwestImageFetcher::new()
            .map(|inner| Self {
                inner: Arc::new(inner),
            })
            .map_err(|error| image_error(py, error))
    }

    /// Construct the Rust fetcher with an explicit encoded size limit.
    #[staticmethod]
    fn with_max_bytes(py: Python<'_>, max_bytes: usize) -> PyResult<Self> {
        ReqwestImageFetcher::with_max_bytes(max_bytes)
            .map(|inner| Self {
                inner: Arc::new(inner),
            })
            .map_err(|error| image_error(py, error))
    }

    /// Fetch encoded image bytes with the GIL released.
    ///
    /// A standalone fetch applies the size limit of this fetcher to one image.
    fn fetch(&self, py: Python<'_>, url: &str) -> PyResult<Py<PyBytes>> {
        let max_bytes = self.inner.max_bytes();
        let budget = ImageByteBudget::new(max_bytes, max_bytes, 0);
        let data = run_image_operation(py, self.inner.fetch(url, &budget))?;
        Ok(PyBytes::new(py, &data).unbind())
    }
}

/// Preprocesses encoded images through the Rust OpenCV implementation.
#[pyclass(
    name = "OpenCvImagePreprocessor",
    module = "deepseek_recipe._native",
    frozen
)]
pub(crate) struct PyOpenCvImagePreprocessor {
    inner: OpenCvImagePreprocessor,
}

#[pymethods]
impl PyOpenCvImagePreprocessor {
    #[new]
    fn new() -> Self {
        Self {
            inner: OpenCvImagePreprocessor,
        }
    }

    /// Preprocess encoded image bytes with the GIL released.
    fn preprocess(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        options: &PyPreprocessOptions,
    ) -> PyResult<PyImageInfo> {
        run_image_operation(py, self.inner.preprocess(data, options.inner))
            .map(|inner| PyImageInfo { inner })
    }
}

/// Resolves image sources using explicit fetching and preprocessing components.
#[pyclass(name = "ImageResolver", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PyImageResolver {
    inner: ImageResolver<PyReqwestImageFetcher, OpenCvImagePreprocessor>,
}

#[pymethods]
impl PyImageResolver {
    #[new]
    fn new(fetcher: &PyReqwestImageFetcher, preprocessor: &PyOpenCvImagePreprocessor) -> Self {
        Self {
            inner: ImageResolver::new(fetcher.clone(), preprocessor.inner),
        }
    }

    /// Resolve image sources and update the caller-supplied request quota.
    /// A failed call leaves the quota unchanged.
    fn resolve(
        &self,
        py: Python<'_>,
        sources: Vec<Py<PyImageSource>>,
        mut quota: PyRefMut<'_, PyImageQuota>,
    ) -> PyResult<PyMultiModalData> {
        let sources: Vec<_> = sources
            .iter()
            .map(|source| source.get().inner.clone())
            .collect();
        run_image_operation(py, self.inner.resolve(&sources, &mut quota.inner))
            .map(|inner| PyMultiModalData { inner })
    }
}
