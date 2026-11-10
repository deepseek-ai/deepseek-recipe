//! Python bindings for the special token mapping used in rendering and parsing.

use std::sync::Arc;

use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens, SpecialTokensMap};
use pyo3::prelude::*;

/// A kind of special token.
#[derive(Clone, Copy, PartialEq, Eq)]
#[pyclass(name = "SpecialToken", module = "deepseek_recipe._native", eq, frozen)]
pub(crate) enum PySpecialToken {
    Bos,
    Eos,
    ThinkingStart,
    ThinkingEnd,
    System,
    User,
    Assistant,
    LatestReminder,
    Dsml,
    Image,
    Ref,
    RefEnd,
    Box,
    BoxEnd,
    Point,
    PointEnd,
    Polygon,
    PolygonEnd,
}

impl PySpecialToken {
    fn to_core(self) -> SpecialToken {
        match self {
            Self::Bos => SpecialToken::Bos,
            Self::Eos => SpecialToken::Eos,
            Self::ThinkingStart => SpecialToken::ThinkingStart,
            Self::ThinkingEnd => SpecialToken::ThinkingEnd,
            Self::System => SpecialToken::System,
            Self::User => SpecialToken::User,
            Self::Assistant => SpecialToken::Assistant,
            Self::LatestReminder => SpecialToken::LatestReminder,
            Self::Dsml => SpecialToken::Dsml,
            Self::Image => SpecialToken::Image,
            Self::Ref => SpecialToken::Ref,
            Self::RefEnd => SpecialToken::RefEnd,
            Self::Box => SpecialToken::Box,
            Self::BoxEnd => SpecialToken::BoxEnd,
            Self::Point => SpecialToken::Point,
            Self::PointEnd => SpecialToken::PointEnd,
            Self::Polygon => SpecialToken::Polygon,
            Self::PolygonEnd => SpecialToken::PolygonEnd,
        }
    }
}

#[pymethods]
impl PySpecialToken {
    /// Return the default literal spelling of this token.
    fn default_str(&self) -> String {
        self.to_core().default_str().to_string()
    }
}

/// A mapping from each `SpecialToken` to its spelling.
///
/// Every keyword defaults to the default spelling of its token, so only the
/// tokens being overridden need to be supplied.
#[pyclass(name = "SpecialTokens", module = "deepseek_recipe._native", frozen)]
pub(crate) struct PySpecialTokens {
    pub(crate) inner: Arc<dyn SpecialTokens>,
}

impl PySpecialTokens {
    pub(crate) fn from_core(inner: Arc<dyn SpecialTokens>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySpecialTokens {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, bos = None, eos = None, thinking_start = None, thinking_end = None,
        system = None, user = None, assistant = None, latest_reminder = None, dsml = None,
        image = None, r#ref = None, ref_end = None, r#box = None, box_end = None, point = None,
        point_end = None, polygon = None, polygon_end = None))]
    fn new(
        bos: Option<String>,
        eos: Option<String>,
        thinking_start: Option<String>,
        thinking_end: Option<String>,
        system: Option<String>,
        user: Option<String>,
        assistant: Option<String>,
        latest_reminder: Option<String>,
        dsml: Option<String>,
        image: Option<String>,
        r#ref: Option<String>,
        ref_end: Option<String>,
        r#box: Option<String>,
        box_end: Option<String>,
        point: Option<String>,
        point_end: Option<String>,
        polygon: Option<String>,
        polygon_end: Option<String>,
    ) -> Self {
        let defaults = SpecialTokensMap::default();
        Self {
            inner: Arc::new(SpecialTokensMap {
                bos: bos.unwrap_or(defaults.bos),
                eos: eos.unwrap_or(defaults.eos),
                thinking_start: thinking_start.unwrap_or(defaults.thinking_start),
                thinking_end: thinking_end.unwrap_or(defaults.thinking_end),
                system: system.unwrap_or(defaults.system),
                user: user.unwrap_or(defaults.user),
                assistant: assistant.unwrap_or(defaults.assistant),
                latest_reminder: latest_reminder.unwrap_or(defaults.latest_reminder),
                dsml: dsml.unwrap_or(defaults.dsml),
                image: image.unwrap_or(defaults.image),
                r#ref: r#ref.unwrap_or(defaults.r#ref),
                ref_end: ref_end.unwrap_or(defaults.ref_end),
                r#box: r#box.unwrap_or(defaults.r#box),
                box_end: box_end.unwrap_or(defaults.box_end),
                point: point.unwrap_or(defaults.point),
                point_end: point_end.unwrap_or(defaults.point_end),
                polygon: polygon.unwrap_or(defaults.polygon),
                polygon_end: polygon_end.unwrap_or(defaults.polygon_end),
            }),
        }
    }

    /// Return the spelling of `token` in this mapping.
    fn token(&self, token: PySpecialToken) -> String {
        self.inner.token(token.to_core()).to_string()
    }
}
