//! Python bindings for Rust request, rendering, image, and response components.
//!
//! Bindings adapt Python values and ownership to the corresponding Rust types.

use deepseek_recipe_core::multimodal::{IMAGE_SPECIAL_TOKEN, IMAGE_SPECIAL_TOKEN_ID};
use deepseek_recipe_encoding::v4::{
    ASSISTANT_SP_TOKEN, BOS_TOKEN, DSML_SP_TOKEN, EOS_TOKEN, LATEST_REMINDER_SP_TOKEN,
    SYSTEM_SP_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN, USER_SP_TOKEN,
};
use pyo3::prelude::*;

use crate::conversation::{
    PyAssistantMessage, PyBytesImageSource, PyConversation, PyDataUrlImageSource, PyImageSource,
    PyLatestReminderMessage, PyMessage, PyRenderedPrompt, PySystemMessage, PyToolCall,
    PyToolDefinition, PyToolMessage, PyUrlImageSource, PyUserMessage,
};
use crate::encoding::{PyDeepseekV4Encoding, PyDeepseekV41Encoding};
use crate::error::ConversionError;
use crate::image::{
    CalcResizeError, ImageError, PyImageInfo, PyImageQuota, PyImageResolver, PyMultiModalData,
    PyOpenCvImagePreprocessor, PyPreprocessOptions, PyReqwestImageFetcher,
};
use crate::inference::{PyInferenceChunk, PyInferenceFinishReason, PyPromptUsage};
use crate::parsing::{PyParsingOptions, PyReasoningStage};
use crate::request::{
    PyChatCompletionRequest, PyConversationRequest, PyConversionOptions, PyInferenceOptions,
    PyMessagesRequest, PyResponsesRequest, PyWebSearchBehavior,
};
use crate::response::{
    PyChatCompletionChunk, PyChatCompletionChunkGenerator, PyChatCompletionResponse,
    PyMessagesChunkGenerator, PyMessagesResponse, PyMessagesStreamEvent, PyResponsesChunkGenerator,
    PyResponsesResponse, PyResponsesStreamEvent, PyStreamProcessor,
};
use crate::tokenizer::PyTokenizer;

mod conversation;
mod encoding;
mod error;
mod image;
mod inference;
mod json;
mod parsing;
mod request;
mod response;
mod tokenizer;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add("BOS_TOKEN", BOS_TOKEN)?;
    m.add("EOS_TOKEN", EOS_TOKEN)?;
    m.add("SYSTEM_SP_TOKEN", SYSTEM_SP_TOKEN)?;
    m.add("USER_SP_TOKEN", USER_SP_TOKEN)?;
    m.add("ASSISTANT_SP_TOKEN", ASSISTANT_SP_TOKEN)?;
    m.add("LATEST_REMINDER_SP_TOKEN", LATEST_REMINDER_SP_TOKEN)?;
    m.add("THINKING_START_TOKEN", THINKING_START_TOKEN)?;
    m.add("THINKING_END_TOKEN", THINKING_END_TOKEN)?;
    m.add("DSML_SP_TOKEN", DSML_SP_TOKEN)?;
    m.add("IMAGE_SPECIAL_TOKEN", IMAGE_SPECIAL_TOKEN)?;
    m.add("IMAGE_SPECIAL_TOKEN_ID", IMAGE_SPECIAL_TOKEN_ID)?;

    m.add("ConversionError", m.py().get_type::<ConversionError>())?;
    m.add("ImageError", m.py().get_type::<ImageError>())?;
    m.add("CalcResizeError", m.py().get_type::<CalcResizeError>())?;

    m.add_class::<PyChatCompletionRequest>()?;
    m.add_class::<PyResponsesRequest>()?;
    m.add_class::<PyMessagesRequest>()?;
    m.add_class::<PyConversionOptions>()?;
    m.add_class::<PyWebSearchBehavior>()?;
    m.add_class::<PyConversationRequest>()?;
    m.add_class::<PyInferenceOptions>()?;
    m.add_class::<PyDeepseekV4Encoding>()?;
    m.add_class::<PyDeepseekV41Encoding>()?;
    m.add_class::<PyImageInfo>()?;
    m.add_class::<PyMultiModalData>()?;
    m.add_class::<PyImageQuota>()?;
    m.add_class::<PyPreprocessOptions>()?;
    m.add_class::<PyReqwestImageFetcher>()?;
    m.add_class::<PyOpenCvImagePreprocessor>()?;
    m.add_class::<PyImageResolver>()?;
    m.add_class::<PyInferenceChunk>()?;
    m.add_class::<PyPromptUsage>()?;
    m.add_class::<PyInferenceFinishReason>()?;
    m.add_class::<PyReasoningStage>()?;
    m.add_class::<PyParsingOptions>()?;
    m.add_class::<PyChatCompletionChunkGenerator>()?;
    m.add_class::<PyResponsesChunkGenerator>()?;
    m.add_class::<PyMessagesChunkGenerator>()?;
    m.add_class::<PyChatCompletionChunk>()?;
    m.add_class::<PyResponsesStreamEvent>()?;
    m.add_class::<PyMessagesStreamEvent>()?;
    m.add_class::<PyChatCompletionResponse>()?;
    m.add_class::<PyResponsesResponse>()?;
    m.add_class::<PyMessagesResponse>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyStreamProcessor>()?;
    m.add_class::<PyConversation>()?;
    m.add_class::<PyRenderedPrompt>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PySystemMessage>()?;
    m.add_class::<PyUserMessage>()?;
    m.add_class::<PyAssistantMessage>()?;
    m.add_class::<PyToolMessage>()?;
    m.add_class::<PyLatestReminderMessage>()?;
    m.add_class::<PyToolCall>()?;
    m.add_class::<PyToolDefinition>()?;
    m.add_class::<PyImageSource>()?;
    m.add_class::<PyDataUrlImageSource>()?;
    m.add_class::<PyUrlImageSource>()?;
    m.add_class::<PyBytesImageSource>()?;
    Ok(())
}
